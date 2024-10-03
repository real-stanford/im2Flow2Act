import random

import cv2
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import v2

# import diffusers
from im2flow2act.common.pointcloud import add_depth_noise, get_pointcloud
from im2flow2act.common.utility.arr import (
    complete_random_sampling,
    random_sampling,
    stratified_random_sampling,
    uniform_sampling,
)
from im2flow2act.common.utility.file import read_pickle
from im2flow2act.diffusion_policy.dataloader.diffusion_bc_dataset import (
    create_sample_indices,
    get_data_stats,
    normalize_data,
    sample_sequence,
)
from im2flow2act.diffusion_policy.dataloader.replay_buffer import (
    PointTrackingReplayBuffer,
)


def process_image(image, optional_transforms=[]):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # (3,H,W)

    transform_list = [
        v2.ToDtype(torch.float32, scale=True),  # convert to float32 and scale to [0,1]
    ]

    for transform_name in optional_transforms:
        if transform_name == "ColorJitter":
            transform_list.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.32, contrast=0.32, saturation=0.32, hue=0.08
                        )
                    ],
                    p=0.8,
                )
            )
        elif transform_name == "RandomGrayscale":
            transform_list.append(v2.RandomGrayscale(p=0.2))

    transform_list.append(
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    transforms = v2.Compose(transform_list)
    return transforms(image)


class DiffusionFlowBCCloseLoopDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs,
        num_points,
        interval_target_flow=False,
        equal_sampling=False,
        object_sampling=False,
        downsample_rate=1,
        sample_frames=64,
        sampling_type="random",
        sampling_replace=False,
        pred_horizon=16,
        target_flow_horizon=16,
        target_flow_offeset=0,
        target_till_end=True,
        obs_horizon=1,
        action_horizon=8,
        action_dim=7,
        shuffle_points=True,
        is_plan_indices=False,
        max_episode=None,
        load_camera_ids=[0],
        point_tracking_camera_id=0,
        camera_resize_shape=None,
        point_tracking_img_size=[256, 256],
        unnormal_list=[],
        optional_transforms=[],
        is_sam=True,
        max_episode_len=None,
        padding_size=None,
        herustic_filter=[],
        load_pointcloud=True,
        normalize_pointcloud=True,
        camera_intrinsic_path=None,
        camera_pose_matrix_path=None,
        load_rgb=False,
        ignore_robot_mask=False,
        depth_noisy_augmentation=False,
        gaussian_shifts=[0, 2],
        base_noise=[0, 100],
        std_noise=[0, 50],
        seed=0,
    ):
        self.buffer = PointTrackingReplayBuffer(
            data_path=data_dirs,
            load_camera_ids=load_camera_ids,
            point_tracking_camera_id=point_tracking_camera_id,
            camera_resize_shape=camera_resize_shape,
            point_tracking_img_size=point_tracking_img_size,
            downsample_rate=downsample_rate,
            max_episode=max_episode,
            max_episode_len=max_episode_len,
            padding_size=padding_size,
            is_sam=is_sam,
            load_depth=load_pointcloud,
            load_rgb=load_rgb,
        )
        self.load_rgb = load_rgb
        self.data_dirs = data_dirs
        self.point_tracking_camera_id = point_tracking_camera_id
        self.load_pointcloud = load_pointcloud
        self.camera_resize_shape = camera_resize_shape
        self.optional_transforms = optional_transforms
        self.point_tracking_img_size = point_tracking_img_size
        self.num_points = num_points
        self.shuffle_points = shuffle_points
        self.sample_frames = sample_frames
        self.equal_sampling = equal_sampling
        self.object_sampling = object_sampling
        self.is_plan_indices = is_plan_indices
        self.seed = seed
        self.set_seed(self.seed)
        self.unnormal_list = unnormal_list
        self.episode_ends = self.buffer.eps_end
        self.buffer.memory_buffer["actual_sample_indices"] = np.arange(
            len(self.buffer["action"])
        )
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        self.point_tracking_data = self.buffer["point_tracking_sequence"]
        self.moving_mask_data = self.buffer["moving_mask"]
        self.robot_mask_data = self.buffer["robot_mask"]
        # self.frame_sample_indices = self.buffer["sample_indices"]
        self.initial_frames = self.buffer["initial_frame"]
        # self.depth = self.buffer["depth"]
        self.initial_depth = self.buffer["initial_depth"]

        self.buffer.remove_key("point_tracking_sequence")
        self.buffer.remove_key("moving_mask")
        self.buffer.remove_key("robot_mask")
        # self.buffer.remove_key("sample_indices")
        self.buffer.remove_key("initial_frame")
        if self.load_pointcloud:
            # self.buffer.remove_key("depth")
            self.buffer.remove_key("initial_depth")
        # self.buffer.remove_key("actual_sample_indices")

        stats = dict()
        for key, data in self.buffer.memory_buffer.items():
            print(key)
            if key in self.unnormal_list:
                pass
            else:
                stats[key] = get_data_stats(data)

            if key in self.unnormal_list:
                pass
            else:
                print(f"Normalizing {key} with shape {data.shape}")
                self.buffer.memory_buffer[key] = normalize_data(data, stats[key])

        self.normalize_pointcloud = normalize_pointcloud
        if self.load_pointcloud and self.normalize_point_cloud:
            point_cloud_stats = stats["action"].copy()
            point_cloud_stats["min"] = point_cloud_stats["min"][:3]
            point_cloud_stats["max"] = point_cloud_stats["max"][:3]
            stats["point_cloud"] = point_cloud_stats

        self.indices = indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.sampling_type = sampling_type
        self.sampling_replace = sampling_replace
        self.interval_target_flow = interval_target_flow
        self.target_flow_horizon = target_flow_horizon
        self.target_flow_offeset = target_flow_offeset
        self.target_till_end = target_till_end
        self.herustic_filter = herustic_filter
        self.ignore_robot_mask = ignore_robot_mask
        self.depth_noisy_augmentation = depth_noisy_augmentation
        self.gaussian_shifts = gaussian_shifts
        self.base_noise = base_noise
        self.std_noise = std_noise

        if self.load_pointcloud:
            self.camera_intrinsic = read_pickle(camera_intrinsic_path)
            self.camera_pose_matrix = read_pickle(camera_pose_matrix_path)

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def normalize_points(tracking_point_sequence, H, W):
        """
        Args:
            tracking_point_sequence: (T,num_points,3) 3: (x,y,visible)
            H: height of image
            W: width of image
        """
        # TODO: check H,W
        tracking_point_sequence[:, :, 0] = tracking_point_sequence[:, :, 0] / H
        tracking_point_sequence[:, :, 1] = tracking_point_sequence[:, :, 1] / W
        return tracking_point_sequence

    @staticmethod
    def process_point_tracking_data(
        episode_point_tracking,
        episode_moving_mask,
        num_points_to_sample,
        point_tracking_img_size,
        robot_mask=None,
        equal_sampling=False,
        object_sampling=False,
        herustic_filter=[],
        ignore_robot_mask=False,
    ):
        # clip the points to the image size
        episode_point_tracking = np.clip(
            episode_point_tracking, a_min=0, a_max=point_tracking_img_size[0] - 1
        )
        condition = np.zeros_like(episode_moving_mask).astype(bool)
        # sim2real camera
        if "left" in herustic_filter:
            condition = condition | (
                (episode_point_tracking[0, :, 0] < 30)
                & (episode_point_tracking[0, :, 1] < 100)
            )
        if "right" in herustic_filter:
            condition = condition | (
                (episode_point_tracking[0, :, 0] > 100)
                & (episode_point_tracking[0, :, 1] < 100)
            )
        episode_moving_mask[condition] = False
        # extract moving points
        if robot_mask is not None and equal_sampling:
            # robot_mask = episode_moving_mask & robot_mask
            # object_mask = episode_moving_mask & ~robot_mask
            robot_mask = (
                episode_moving_mask
                & (episode_point_tracking[0, :, 1] < 85)
                & (episode_point_tracking[0, :, 0] < 195)
                & (episode_point_tracking[0, :, 0] > 115)
            )
            object_mask = episode_moving_mask & (~robot_mask)

            if (
                np.sum(robot_mask) >= num_points_to_sample // 16
                and np.sum(object_mask) >= num_points_to_sample // 16
            ):
                # print("equal_sampling")
                robot_flow = episode_point_tracking[:, robot_mask].copy()
                object_flow = episode_point_tracking[:, object_mask].copy()
                robot_flow = rearrange(robot_flow, "T N C -> N T C")
                robot_flow = complete_random_sampling(
                    robot_flow, num_points_to_sample // 2
                )
                robot_flow = rearrange(robot_flow, "N T C -> T N C")
                object_flow = rearrange(object_flow, "T N C -> N T C")
                object_flow = complete_random_sampling(
                    object_flow, num_points_to_sample // 2
                )
                object_flow = rearrange(object_flow, "N T C -> T N C")
                point_tracking_data = np.concatenate([robot_flow, object_flow], axis=1)
            else:
                # print("fail to equal sampling")
                # raise ValueError("fail to equal sampling")
                point_tracking_data = episode_point_tracking[
                    :, episode_moving_mask, :
                ].copy()
                # random sampling points
                point_tracking_data = rearrange(point_tracking_data, "T N C -> N T C")
                point_tracking_data = complete_random_sampling(
                    point_tracking_data, num_points_to_sample
                )
                point_tracking_data = rearrange(point_tracking_data, "N T C -> T N C")
        elif robot_mask is not None and object_sampling:
            # old camera
            # robot_mask = (
            #     episode_moving_mask
            #     & (episode_point_tracking[0, :, 1] < 85)
            #     & (episode_point_tracking[0, :, 0] < 195)
            #     & (episode_point_tracking[0, :, 0] > 115)
            # )
            # new camera
            if ignore_robot_mask:
                robot_mask = np.zeros_like(episode_moving_mask).astype(bool)
            else:
                # slighly chunked the upper part of the frame to avoid the robot
                robot_mask = robot_mask | (episode_point_tracking[0, :, 1] < 15)
            object_mask = episode_moving_mask & (~robot_mask)

            # if np.sum(object_mask) >= num_points_to_sample // 16:
            if np.sum(object_mask) >= 1:
                object_flow = episode_point_tracking[:, object_mask].copy()
                object_flow = rearrange(object_flow, "T N C -> N T C")
                object_flow = complete_random_sampling(
                    object_flow, num_points_to_sample
                )
                object_flow = rearrange(object_flow, "N T C -> T N C")
                point_tracking_data = object_flow

            else:
                # print("fail to equal sampling")
                # raise ValueError("fail to equal sampling")
                point_tracking_data = episode_point_tracking[
                    :, episode_moving_mask, :
                ].copy()
                # random sampling points
                point_tracking_data = rearrange(point_tracking_data, "T N C -> N T C")
                point_tracking_data = complete_random_sampling(
                    point_tracking_data, num_points_to_sample
                )
                point_tracking_data = rearrange(point_tracking_data, "N T C -> T N C")
        else:
            point_tracking_data = episode_point_tracking[
                :, episode_moving_mask, :
            ].copy()
            # random sampling points
            point_tracking_data = rearrange(point_tracking_data, "T N C -> N T C")
            point_tracking_data = complete_random_sampling(
                point_tracking_data, num_points_to_sample
            )
            point_tracking_data = rearrange(point_tracking_data, "N T C -> T N C")

        # normalize the x and y to [0,1]
        point_tracking_data = DiffusionFlowBCCloseLoopDataset.normalize_points(
            point_tracking_data, *point_tracking_img_size
        )
        return point_tracking_data

    @staticmethod
    def add_noise_to_depth(depth_img, gaussian_shifts, base_noise, std_noise):
        noisy_depth = add_depth_noise(
            depth=depth_img,
            gaussian_shifts=gaussian_shifts,
            base_noise=base_noise,
            std_noise=std_noise,
        )
        return noisy_depth

    @staticmethod
    def get_point_cloud(depth_img, color_img, cam_intr, cam_pose, flow):
        # assume flow is originally under color_img size
        depth_h, depth_w = depth_img.shape[:2]
        img_h, img_w = color_img.shape[:2]
        color_img = cv2.resize(color_img, (depth_w, depth_h))
        if flow is not None:
            # get the flow under depth size
            flow[:, 0] = flow[:, 0] / img_w * depth_w
            flow[:, 1] = flow[:, 1] / img_h * depth_h
        # get point cloud under orginal depth
        point_cloud, color_pts, _ = get_pointcloud(
            depth_img=depth_img,
            color_img=color_img,
            segmentation_img=None,
            cam_intr=cam_intr,
            cam_pose=cam_pose,
            points=flow,
        )
        return point_cloud, color_pts

    def normalize_point_cloud(self, point_cloud):
        point_cloud_stats = self.stats["point_cloud"]
        point_cloud = normalize_data(point_cloud, point_cloud_stats)

        return point_cloud

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.buffer.memory_buffer,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        # load episode-level flow and mask
        episode_idx = nsample["episode_idx"][0]
        episode_point_tracking = self.point_tracking_data[episode_idx].copy()  # (T,N,3)
        episode_moving_mask = self.moving_mask_data[episode_idx].copy()  # (N,)
        episode_robot_mask = self.robot_mask_data[episode_idx].copy()  # (N,)
        episode_initial_depth = self.initial_depth[episode_idx].copy()
        # random sample points and normalize
        episode_point_tracking = self.process_point_tracking_data(
            episode_point_tracking,
            episode_moving_mask,
            self.num_points,
            self.point_tracking_img_size,
            robot_mask=episode_robot_mask,
            equal_sampling=self.equal_sampling,
            object_sampling=self.object_sampling,
            herustic_filter=self.herustic_filter,
            ignore_robot_mask=self.ignore_robot_mask,
        )  # (T,N,3)
        # shuffle the points
        if self.shuffle_points:
            shuffled_indices = np.random.permutation(self.num_points)
            episode_point_tracking = episode_point_tracking[:, shuffled_indices, :]
        # initial flow
        initial_flow = episode_point_tracking[0].copy()
        initial_flow[:, 0] *= self.camera_resize_shape[0]
        initial_flow[:, 1] *= self.camera_resize_shape[1]
        initial_flow = initial_flow.astype(np.int32)
        initial_flow = np.clip(
            initial_flow, a_min=0, a_max=self.camera_resize_shape[0] - 1
        )  # (N,3)
        if self.depth_noisy_augmentation:
            episode_initial_depth = self.add_noise_to_depth(
                depth_img=episode_initial_depth.copy(),
                gaussian_shifts=np.random.uniform(*self.gaussian_shifts),
                base_noise=np.random.uniform(*self.base_noise),
                std_noise=np.random.uniform(*self.std_noise),
            )
        point_cloud, color_pts = self.get_point_cloud(
            depth_img=episode_initial_depth.copy(),
            # color_img=self.initial_frames[episode_idx].copy(),
            color_img=np.zeros((224, 224, 3)),
            cam_intr=self.camera_intrinsic,
            cam_pose=self.camera_pose_matrix,
            flow=initial_flow.copy(),
        )
        # (
        #     debug_point_cloud,
        #     debug_color_pts,
        # ) = self.get_point_cloud(
        #     depth_img=episode_initial_depth.copy(),
        #     # color_img=self.initial_frames[episode_idx].copy(),
        #     color_img=np.zeros((224, 224, 3)),
        #     cam_intr=self.camera_intrinsic,
        #     cam_pose=self.camera_pose_matrix,
        #     flow=None,
        # )
        point_cloud = point_cloud.astype(np.float32)
        if self.normalize_pointcloud:
            point_cloud = self.normalize_point_cloud(point_cloud)
        # episode start index and length
        episode_start_index = (
            self.episode_ends[episode_idx - 1] if episode_idx > 0 else 0
        )
        episode_length = self.episode_ends[episode_idx] - episode_start_index
        actual_idx = nsample["actual_sample_indices"][0]
        # get where the action start in the episode
        current_idx_in_episode = actual_idx - episode_start_index
        # current flow
        current_flow = episode_point_tracking[current_idx_in_episode].copy()
        current_flow[:, 0] *= self.camera_resize_shape[0]
        current_flow[:, 1] *= self.camera_resize_shape[1]
        current_flow = current_flow.astype(np.int32)
        current_flow = np.clip(
            current_flow, a_min=0, a_max=self.camera_resize_shape[0] - 1
        )
        # create plan by sampling along the time dimension
        if self.sampling_type == "random":
            episode_flow_plan, frame_sample_indices = random_sampling(
                episode_point_tracking[:episode_length],  # discard the repetive frames
                self.sample_frames,
                zero_include=True,  # must include the initial frame
                return_indices=True,
                replace=self.sampling_replace,
            )
        elif self.sampling_type == "uniform":
            episode_flow_plan, frame_sample_indices = uniform_sampling(
                episode_point_tracking[:episode_length],
                self.sample_frames,
                return_indices=True,
            )
        elif self.sampling_type == "stratified":
            episode_flow_plan, frame_sample_indices = stratified_random_sampling(
                episode_point_tracking[:episode_length],
                self.sample_frames,
                return_indices=True,
            )
        elif self.sampling_type == "complete":
            episode_flow_plan = episode_point_tracking
            frame_sample_indices = np.arange(len(episode_flow_plan)).astype(int)

        # update sample indices after sampling along the time
        # episode_sample_indices = self.frame_sample_indices[episode_idx][
        #     frame_sample_indices
        # ]
        plan_offset = (
            np.clip(
                actual_idx + self.pred_horizon,
                a_min=0,
                a_max=episode_start_index + episode_length - 1,
            )
            - episode_start_index
        )
        if self.interval_target_flow:
            if self.target_till_end:
                target_indices = np.linspace(
                    current_idx_in_episode,
                    episode_length - 1,
                    self.target_flow_horizon,
                ).astype(int)
            else:
                target_indices = np.arange(
                    current_idx_in_episode,
                    current_idx_in_episode + self.target_flow_horizon,
                ).astype(int)
            target_indices = np.clip(target_indices, a_min=0, a_max=episode_length - 1)
            target_flow = episode_point_tracking[target_indices].copy()
        else:
            target_flow = episode_point_tracking[plan_offset].copy()
        if self.load_rgb:
            images_to_transform = []
            for camera_id in self.buffer.load_camera_ids:
                images_to_transform.append(
                    torch.from_numpy(
                        nsample[f"camera_{camera_id}"][: self.obs_horizon, :]
                    )
                )
            images_to_transform.append(
                torch.from_numpy(self.initial_frames[episode_idx]).unsqueeze(0)
            )
            images_to_transform = torch.cat(images_to_transform, dim=0).permute(
                0, 3, 1, 2
            )  # (B,C,H,W)
            images_to_transform = process_image(
                images_to_transform, self.optional_transforms
            )
            split_indices = [self.obs_horizon] * len(self.buffer.load_camera_ids)
            split_indices.append(1)
            images_to_transform = torch.split(images_to_transform, split_indices)
            for i, camera_id in enumerate(self.buffer.load_camera_ids):
                nsample[f"camera_{camera_id}"] = images_to_transform[i]

            nsample["initial_frame"] = images_to_transform[-1].squeeze(0)
        else:
            nsample["camera_0"] = 0
            nsample["initial_frame"] = np.zeros((3, 224, 224))
        # nsample["target_proprioception"] = nsample["proprioception"][
        #     self.pred_horizon - 1, :
        # ]
        nsample["proprioception"] = nsample["proprioception"][: self.obs_horizon, :]
        nsample["episode_flow_plan"] = episode_flow_plan
        nsample["target_flow"] = target_flow
        nsample["initial_flow"] = initial_flow
        nsample["current_flow"] = current_flow
        nsample["initial_depth"] = episode_initial_depth
        nsample["point_cloud"] = point_cloud
        # nsample["color_pts"] = color_pts
        # nsample["debug_point_cloud"] = debug_point_cloud
        # nsample["debug_color_pts"] = debug_color_pts

        return nsample
