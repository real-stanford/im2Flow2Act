import concurrent.futures
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import zarr
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs

register_codecs()


class ReplayBuffer:
    def __init__(
        self,
        data_path,
        load_camera_ids=[],
        camera_resize_shape=[],
        max_episode=None,
        max_workers=32,
    ) -> None:
        self.data_path = data_path
        self.load_camera_ids = load_camera_ids
        self.camera_resize_shape = camera_resize_shape
        self.max_workers = max_workers
        self.max_episode = max_episode
        self.initiate_memory_buffer()
        self.load_data_to_memory()

    def initiate_memory_buffer(self):
        self.memory_buffer = defaultdict(list)

    def load_data_to_memory(self):
        load_episode_num = 0
        for path in self.data_path:
            root = zarr.open(path, mode="a")
            episodes = list(root.group_keys())
            for episode in episodes:
                self.memory_buffer["action"].append(
                    self.load_low_dim_data(root, osp.join(episode, "action"))
                )
                self.memory_buffer["proprioception"].append(
                    self.load_low_dim_data(root, osp.join(episode, "proprioception"))
                )
                for camera_ids in self.load_camera_ids:
                    cam_name = f"camera_{camera_ids}"
                    self.memory_buffer[cam_name].append(
                        self.load_visual_data(root, osp.join(episode, cam_name, "rgb"))
                    )
                load_episode_num += 1
                if (
                    self.max_episode is not None
                    and load_episode_num >= self.max_episode
                ):
                    break

        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])
        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

    def load_low_dim_data(self, root, low_dim_path):
        return root[low_dim_path][:].astype(np.float32)

    def load_visual_data(self, root, visual_path, dim=3):
        visual_shape = root[visual_path].shape
        np_arr_shape = (
            (visual_shape[0], *self.camera_resize_shape, dim)
            if self.camera_resize_shape
            else visual_shape
        )
        np_arr = np.zeros(np_arr_shape, dtype=np.uint8)

        def load_img(zarr_arr, visual_path, index, np_arr):
            try:
                if self.camera_resize_shape:
                    np_arr[index] = cv2.resize(
                        zarr_arr[visual_path][index], self.camera_resize_shape
                    )
                else:
                    np_arr[index] = zarr_arr[visual_path][index]
                return True
            except Exception as e:
                print(e)
                return False

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = set()
            for i in range(visual_shape[0]):
                futures.add(executor.submit(load_img, root, visual_path, i, np_arr))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to load image!")

        return np_arr

    def __repr__(self) -> str:
        rep = ""
        for k, v in self.memory_buffer.items():
            rep += f"{k}, {v.shape}\n"
        rep += f"eps_end, {self.eps_end}\n"
        return rep

    def __getitem__(self, key):
        return self.memory_buffer[key]

    def remove_key(self, key):
        del self.memory_buffer[key]


class PointTrackingReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        data_path,
        point_tracking_img_size,
        load_camera_ids=[],
        point_tracking_camera_id=0,
        camera_resize_shape=[],
        max_episode=None,
        downsample_rate=1,
        max_workers=32,
        max_episode_len=None,
        padding_size=160,
        is_sam=False,
        load_depth=False,
        load_rgb=False,
    ) -> None:
        self.point_tracking_camera_id = point_tracking_camera_id
        self.point_tracking_img_size = point_tracking_img_size
        self.downsample_rate = downsample_rate
        self.max_episode_len = max_episode_len
        self.padding_size = padding_size
        self.is_sam = is_sam
        self.load_depth = load_depth
        self.load_rgb = load_rgb
        self.data_path_offset = [0]
        super().__init__(
            data_path, load_camera_ids, camera_resize_shape, max_episode, max_workers
        )

    def load_data_to_memory(self):
        load_episode_num = 0
        for path in self.data_path:
            print(">> Loading data from: ", path)
            root = zarr.open(path, mode="a")
            episodes = list(root.group_keys())
            episodes = sorted(episodes, key=lambda x: int(x.split("_")[1]))
            print(f">> {len(episodes)} episodes num from {path} ")
            if len(episodes) == 0:
                raise ValueError("No episodes found!")
            path_load_episode_num = 0
            for i, episode in tqdm(enumerate(episodes)):
                episode_length = len(
                    self.load_low_dim_data(root, osp.join(episode, "action"))[
                        :: self.downsample_rate
                    ]
                )
                if self.max_episode_len and episode_length > self.max_episode_len:
                    print(f"episode {episode} exceeds max episode length")

                self.memory_buffer["action"].append(
                    self.load_low_dim_data(root, osp.join(episode, "action"))[
                        :: self.downsample_rate
                    ][: self.max_episode_len]
                )
                self.memory_buffer["proprioception"].append(
                    self.load_low_dim_data(root, osp.join(episode, "proprioception"))[
                        :: self.downsample_rate
                    ][: self.max_episode_len]
                )
                if self.load_rgb:
                    for camera_ids in self.load_camera_ids:
                        cam_name = f"camera_{camera_ids}"
                        self.memory_buffer[cam_name].append(
                            self.load_visual_data(
                                root, osp.join(episode, cam_name, "rgb")
                            )[:: self.downsample_rate][: self.max_episode_len]
                        )
                    self.memory_buffer["initial_frame"].append(
                        self.memory_buffer[f"camera_{self.point_tracking_camera_id}"][
                            -1
                        ][0][None, :]
                    )
                if self.load_depth:
                    initial_depth = root[
                        osp.join(
                            episode, f"camera_{self.point_tracking_camera_id}/depth"
                        )
                    ][0].astype(np.float32)
                    self.memory_buffer["initial_depth"].append(initial_depth[None, :])
                self.memory_buffer["point_tracking_sequence"].append(
                    self.load_point_tracking_data(
                        root,
                        (
                            osp.join(episode, "sam_point_tracking_sequence")
                            if self.is_sam
                            else osp.join(episode, "point_tracking_sequence")
                        ),
                    )[: self.max_episode_len]
                )
                self.memory_buffer["moving_mask"].append(
                    self.load_moving_mask_data(
                        root,
                        (
                            osp.join(episode, "sam_moving_mask")
                            if self.is_sam
                            else osp.join(episode, "moving_mask")
                        ),
                    )
                )
                try:
                    self.memory_buffer["robot_mask"].append(
                        self.load_mask_data(root, osp.join(episode, "robot_mask"))
                    )
                except:
                    print("no robot_mask found!")
                    self.memory_buffer["robot_mask"].append(
                        np.ones_like(self.memory_buffer["moving_mask"][-1])
                    )
                episode_length = len(self.memory_buffer["action"][-1])
                self.memory_buffer["episode_idx"].append(
                    np.array([load_episode_num] * episode_length)
                )
                load_episode_num += 1
                path_load_episode_num += 1
                if (
                    self.max_episode is not None
                    and path_load_episode_num >= self.max_episode
                ):
                    break
            print("load_episode_num", load_episode_num)
            self.data_path_offset.append(load_episode_num)
        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])

        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

        print(f">> Loaded {load_episode_num} episodes!")
        print(f">> self.data_path_offset {self.data_path_offset}")

    def load_point_tracking_data(self, root, point_tracking_path):
        point_tracking_sequence = np.transpose(
            root[point_tracking_path][:], [1, 0, 2]
        )  # (T,N,3)
        if self.padding_size is not None:
            last_time_step = point_tracking_sequence[-1, :, :]
            padding = np.repeat(
                last_time_step[np.newaxis, :, :],
                self.padding_size - point_tracking_sequence.shape[0],
                axis=0,
            )
            return np.concatenate([point_tracking_sequence, padding], axis=0)[
                None, :
            ]  # (1,padding_size,N,3)
        else:
            return point_tracking_sequence[None, :]

    def load_moving_mask_data(self, root, moving_mask_path):
        return root[moving_mask_path][:][None, :]

    def load_mask_data(self, root, mask_path):
        return root[mask_path][:][None, :]

    def load_depth_data(self, root, depth_path):
        depth_shape = root[depth_path].shape
        np_arr_shape = (
            (depth_shape[0], *self.camera_resize_shape)
            if self.camera_resize_shape
            else depth_shape
        )
        np_arr = np.zeros(np_arr_shape, dtype=np.float32)

        def load_img(zarr_arr, visual_path, index, np_arr):
            try:
                if self.camera_resize_shape:
                    np_arr[index] = cv2.resize(
                        zarr_arr[visual_path][index],
                        self.camera_resize_shape,
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    np_arr[index] = zarr_arr[visual_path][index]
                return True
            except Exception as e:
                print(e)
                return False

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = set()
            for i in range(depth_shape[0]):
                futures.add(executor.submit(load_img, root, depth_path, i, np_arr))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to load image!")

        return np_arr
