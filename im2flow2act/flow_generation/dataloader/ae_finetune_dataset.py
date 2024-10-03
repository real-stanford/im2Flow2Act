import random

import numpy as np
import torch
import zarr
from einops import rearrange, repeat
from torchvision.transforms import v2
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.tapnet.utility.utility import get_buffer_size

register_codecs()


class AeFinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_pathes,
        grid_size=32,
        frame_resize_shape=[224, 224],
        point_tracking_img_size=[256, 256],
        diff_flow=True,
        max_episode=None,
        start_episode=0,
        seed=0,
    ):
        self.data_pathes = data_pathes
        self.grid_size = grid_size
        self.frame_resize_shape = frame_resize_shape
        self.point_tracking_img_size = point_tracking_img_size
        self.diff_flow = diff_flow
        self.max_episode = max_episode
        self.start_episode = start_episode
        self.set_seed(seed)
        self.train_data = []
        self.construct_dataset()

        self.flow_transforms = v2.Compose(
            [
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def construct_dataset(self):
        for i, data_path in enumerate(self.data_pathes):
            buffer = zarr.open(data_path, mode="r")
            num_episodes = get_buffer_size(buffer)
            data_path_discard_num = 0
            data_path_append_num = 0
            for j in tqdm(range(num_episodes), desc=f"Loading {data_path}"):
                if (
                    self.max_episode is not None
                    and data_path_append_num >= self.max_episode
                ):
                    break
                episode = buffer[f"episode_{j}"]
                if "bbox_point_tracking_sequence" not in episode:
                    data_path_discard_num += 1
                    continue
                # load point tracking sequence
                point_tracking_sequence = episode["bbox_point_tracking_sequence"][
                    :
                ].copy()  # (N^2, T, 3)
                data_grid_size = np.sqrt(point_tracking_sequence.shape[0]).astype(int)
                # if no sufficient point, simply repeat the first sample
                if data_grid_size < self.grid_size:
                    num_samples = (
                        self.grid_size * self.grid_size
                        - point_tracking_sequence.shape[0]
                    )
                    slack_point_tracking_sequence = repeat(
                        point_tracking_sequence[0], "t c -> s t c", s=num_samples
                    )
                    point_tracking_sequence = np.concatenate(
                        [point_tracking_sequence, slack_point_tracking_sequence], axis=0
                    )
                    data_grid_size = np.sqrt(point_tracking_sequence.shape[0]).astype(
                        int
                    )
                assert data_grid_size == self.grid_size
                point_tracking_sequence = rearrange(
                    point_tracking_sequence,
                    "(N1 N2) T C -> T C N1 N2",
                    N1=self.grid_size,
                )  # (T, 3, grid_size, grid_size)
                if self.diff_flow:
                    # (n_sample_frames, 3, grid_size, grid_size)
                    diff_point_tracking_sequence = point_tracking_sequence.copy()[1:]
                    diff_point_tracking_sequence[:, :2, :, :] = (
                        diff_point_tracking_sequence[:, :2, :, :]
                        - point_tracking_sequence[:1, :2, :, :]
                    )
                    # This make sure the flow is in the range of [0,1]
                    diff_point_tracking_sequence[:, :2, :, :] = (
                        diff_point_tracking_sequence[:, :2, :, :]
                        + self.point_tracking_img_size[0]
                    ) / (2 * self.point_tracking_img_size[0])
                    point_tracking_sequence = diff_point_tracking_sequence
                else:
                    # discard the first frame points
                    point_tracking_sequence = point_tracking_sequence[1:]
                    # normalize the flow
                    point_tracking_sequence[:, :2, :, :] = (
                        point_tracking_sequence[:, :2, :, :]
                        / self.point_tracking_img_size[0]
                    )  # assume the image size is square

                point_tracking_sequence = point_tracking_sequence.astype(np.float32)

                data_path_append_num += 1

                self.train_data.extend(point_tracking_sequence)

            print(f">> Loaded {data_path_append_num} episodes from {data_path}")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        data = self.train_data[idx]
        point_tracking_sequence = torch.from_numpy(data)
        point_tracking_sequence = self.flow_transforms(point_tracking_sequence)

        sample = dict(
            point_tracking_sequence=point_tracking_sequence,
        )
        return sample
