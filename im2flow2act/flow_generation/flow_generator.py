import numpy as np
import zarr
from einops import rearrange
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from torch import nn

from im2flow2act.common.utility.arr import (
    complete_random_sampling,
)
from im2flow2act.common.utility.viz import save_to_gif
from im2flow2act.tapnet.utility.utility import max_distance_moved
from im2flow2act.tapnet.utility.viz import (
    draw_point_tracking_sequence,
)


class FlowGenerator(nn.Module):
    def __init__(
        self,
        buffer,
        point_tracking_img_size,
        num_points,
        num_frames,
        filters=[],
        sam_area_thres=1000,
        sam_closeness=5,
        moving_threshold=15,
        mv_t=-1,
        mv_ratio=0.5,
        flow_from_sam=False,
        flow_from_bbox=False,
        flow_from_generated=False,
        generated_flow_buffer=None,
        workspace_depth=1.2,
    ) -> None:
        super().__init__()
        self.buffer = zarr.open(buffer, mode="r")
        self.point_tracking_img_size = point_tracking_img_size
        self.num_points = num_points
        self.num_frames = num_frames
        self.filters = filters
        self.sam_area_thres = sam_area_thres
        self.sam_closeness = sam_closeness
        self.moving_threshold = moving_threshold
        self.mv_t = mv_t
        self.mv_ratio = mv_ratio
        self.workspace_depth = workspace_depth
        self.flow_from_sam = flow_from_sam
        self.flow_from_bbox = flow_from_bbox
        self.flow_from_generated = flow_from_generated
        if generated_flow_buffer is not None:
            self.generated_flow_buffer = zarr.open(generated_flow_buffer, mode="r")

    def apply_filter(self, episode_point_tracking_sequence, filters=[], episode=None):
        if "left" in filters:
            condition = (episode_point_tracking_sequence[:, 0, 0] < 100) & (
                episode_point_tracking_sequence[:, 0, 1] < 100
            )
            episode_point_tracking_sequence = episode_point_tracking_sequence[
                ~condition
            ]

        if "right" in filters:
            condition = (episode_point_tracking_sequence[:, 0, 0] > 100) & (
                episode_point_tracking_sequence[:, 0, 1] < 100
            )
            episode_point_tracking_sequence = episode_point_tracking_sequence[
                ~condition
            ]

        if "mv" in filters:
            moving_mask = (
                max_distance_moved(
                    episode_point_tracking_sequence, t_threshold=self.mv_t
                )
                > self.moving_threshold
            )
            episode_point_tracking_sequence = episode_point_tracking_sequence[
                moving_mask
            ]
        if "mv_ratio" in filters:
            max_distances = max_distance_moved(
                episode_point_tracking_sequence, t_threshold=self.mv_t
            )
            threshold_index = int(self.mv_ratio * len(max_distances))
            sorted_max_distance = np.sort(max_distances)[::-1]
            moving_threshold = sorted_max_distance[threshold_index]
            moving_mask = max_distances > moving_threshold
            episode_point_tracking_sequence = episode_point_tracking_sequence[
                moving_mask
            ]

        if "sam" in filters:
            sam_mask = self.buffer[f"episode_{episode}/sam_mask"][:]
            unique_mask = np.unique(sam_mask)
            object_index = []
            initial_flow = episode_point_tracking_sequence[:, 0, :2]
            for um in unique_mask:
                area = np.where(sam_mask == um)
                area_set = np.array(list(zip(area[1], area[0])))
                for i, point in enumerate(initial_flow):
                    if (
                        len(area_set) < self.sam_area_thres
                        and np.isclose(point, area_set, atol=self.sam_closeness)
                        .all(axis=1)
                        .any()
                    ):
                        object_index.append(i)
            episode_point_tracking_sequence = episode_point_tracking_sequence[
                object_index
            ]
        if "db_scan" in filters:
            clustering = DBSCAN(eps=20, min_samples=10).fit(
                episode_point_tracking_sequence[:, 0, :2]
            )
            labels = clustering.labels_
            # Filtering points belonging to the main cluster (ignoring outliers)
            # Assume the largest cluster is the main cluster
            main_cluster = np.bincount(labels[labels >= 0]).argmax()
            episode_point_tracking_sequence = episode_point_tracking_sequence[
                labels == main_cluster
            ]

        return episode_point_tracking_sequence

    def generate_flow(self, rgb, depth, **kwargs):
        episode = kwargs.get("episode", 0)
        if self.flow_from_sam:
            episode_point_tracking_sequence = self.buffer[
                f"episode_{episode}/sam_point_tracking_sequence"
            ][:]  # (N,T,3)
            episode_moving_mask = self.buffer[f"episode_{episode}/sam_moving_mask"][:]
            episode_point_tracking_sequence = episode_point_tracking_sequence[
                episode_moving_mask
            ]
        elif self.flow_from_bbox:
            episode_point_tracking_sequence = self.buffer[
                f"episode_{episode}/bbox_point_tracking_sequence"
            ][:]
        elif self.flow_from_generated:
            # (N,T,3) [0,1] np
            print(">> getting flow generated flow dataset")
            episode_point_tracking_sequence = self.generated_flow_buffer[
                f"episode_{episode}/generated_flows"
            ][:]
            # rescale to point tracking size scale
            episode_point_tracking_sequence[:, :, :2] = (
                episode_point_tracking_sequence[:, :, :2] * self.point_tracking_img_size
            )
            episode_point_tracking_sequence = episode_point_tracking_sequence.reshape(
                32, 32, 32, 3
            )
            # down samlple
            episode_point_tracking_sequence = episode_point_tracking_sequence[::2, ::2]
            episode_point_tracking_sequence = episode_point_tracking_sequence.reshape(
                -1, 32, 3
            )
            print(
                "resize to point tracking size", episode_point_tracking_sequence.shape
            )
        condition = episode_point_tracking_sequence[:, :, 2] > 0.99
        episode_point_tracking_sequence[:, :, 2][condition] = 1
        episode_point_tracking_sequence = self.apply_filter(
            episode_point_tracking_sequence, self.filters, episode
        )
        # righ now the flow is under point tracking image size
        episode_initial_flow = episode_point_tracking_sequence[:, 0, :2]

        # convert under depth size
        depth_h, depth_w = depth.shape[:2]
        cam_size_initial_flow = episode_initial_flow.copy()
        cam_size_initial_flow[:, 0] = (
            cam_size_initial_flow[:, 0] / self.point_tracking_img_size[1] * depth_w
        )
        cam_size_initial_flow[:, 1] = (
            cam_size_initial_flow[:, 1] / self.point_tracking_img_size[0] * depth_h
        )
        cam_size_initial_flow = cam_size_initial_flow.astype(np.int32)
        # filter out zero depth
        none_zero_indices = self.get_non_zero_depth_indicies(
            depth, cam_size_initial_flow
        )
        workspace_depth_indices = self.get_workspace_depth_indices(
            depth, cam_size_initial_flow
        )
        valid_depth = set(none_zero_indices).intersection(set(workspace_depth_indices))
        valid_depth = np.array(list(valid_depth))
        plan_flow = complete_random_sampling(
            episode_point_tracking_sequence[valid_depth],
            self.num_points,
            return_indices=False,
        )
        # normalize flow
        plan_flow = self.normalize_flow(plan_flow)

        # sample frames
        plan_flow = rearrange(plan_flow, "N T C -> T N C")
        plan_flow = complete_random_sampling(plan_flow, self.num_frames)
        return plan_flow  # (num_frames,num_points,3)

    def get_non_zero_depth_indicies(self, depth, cam_size_initial_flow):
        none_zero_indices = np.where(
            depth[cam_size_initial_flow[:, 1], cam_size_initial_flow[:, 0]] != 0
        )[0]
        return none_zero_indices

    def get_workspace_depth_indices(self, depth, cam_size_initial_flow):
        workspace_depth_indices = np.where(
            depth[cam_size_initial_flow[:, 1], cam_size_initial_flow[:, 0]]
            < self.workspace_depth
        )[0]
        return workspace_depth_indices

    def normalize_flow(self, flow):
        flow[..., 0] = flow[..., 0] / self.point_tracking_img_size[1]
        flow[..., 1] = flow[..., 1] / self.point_tracking_img_size[0]
        flow = np.clip(flow, 0, 1)
        return flow

    def recover_flow(self, flow, rgb):
        flow[..., 0] = flow[..., 0] * rgb.shape[1]
        flow[..., 1] = flow[..., 1] * rgb.shape[0]
        return flow

    def plot_plans(self, rgb, plan_flow):
        img_h, img_w = rgb.shape[:2]
        figure = plt.figure()
        plt.imshow(rgb)
        plt.scatter(plan_flow[0, :, 0] * img_w, plan_flow[0, :, 1] * img_h, s=1)
        for i in range(4):
            plt.scatter(
                plan_flow[self.num_frames // 4 * i, :, 0] * img_w,
                plan_flow[self.num_frames // 4 * i, :, 1] * img_h,
                s=1,
            )
        return figure

    def generate_gif(self, rgb, plan_flow, save_path):
        viz_plan_flow = plan_flow.copy()
        viz_plan_flow = rearrange(viz_plan_flow, "T N C -> N T C")
        img_h, img_w = rgb.shape[:2]
        viz_plan_flow[..., 0] = viz_plan_flow[..., 0] * img_w
        viz_plan_flow[..., 1] = viz_plan_flow[..., 1] * img_h
        frames = []
        for j in range(viz_plan_flow.shape[1]):
            frame = draw_point_tracking_sequence(
                rgb.copy(),
                viz_plan_flow[:, :j],
                draw_line=True,
                thickness=3,
                add_alpha_channel=True,
            )
            frames.append(frame)
        save_to_gif(
            frames,
            save_path,
        )

    def get_first_frame(self, episode):
        first_frame = self.buffer[f"episode_{episode}/camera_0/rgb"][0]
        return first_frame
