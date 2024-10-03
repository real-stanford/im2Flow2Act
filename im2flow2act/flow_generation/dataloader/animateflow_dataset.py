import random

import cv2
import numpy as np
import omegaconf
import torch
import zarr
from einops import rearrange, repeat
from torchvision.transforms import v2
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.common.utility.arr import (
    random_sampling,
    stratified_random_sampling,
    uniform_sampling,
)
from im2flow2act.tapnet.utility.utility import get_buffer_size

register_codecs()


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
        v2.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
    )

    transforms = v2.Compose(transform_list)
    return transforms(image)


class AnimateFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_pathes,
        n_sample_frames=16,
        grid_size=32,
        frame_sampling_method="uniform",
        frame_resize_shape=[224, 224],
        point_tracking_img_size=[256, 256],
        diff_flow=True,
        optional_transforms=[],
        max_episode=None,
        start_episode=0,
        seed=0,
    ):
        self.data_pathes = data_pathes
        self.n_sample_frames = n_sample_frames
        self.grid_size = grid_size
        self.frame_sampling_method = frame_sampling_method
        self.frame_resize_shape = frame_resize_shape
        self.point_tracking_img_size = point_tracking_img_size
        self.diff_flow = diff_flow
        self.optional_transforms = optional_transforms
        self.max_episode = (
            max_episode
            if isinstance(max_episode, (list, omegaconf.listconfig.ListConfig))
            else [max_episode] * len(data_pathes)
        )
        print(">> max_episode", self.max_episode)
        self.start_episode = start_episode
        self.set_seed(seed)
        self.train_data = []
        self.construct_dataset()

        self.flow_transforms = v2.Compose(
            [
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def sample_point_tracking_frame(
        point_tracking_sequence,
        frame_sampling_method="uniform",
        n_sample_frames=16,
    ):
        sequence = point_tracking_sequence.copy()
        if frame_sampling_method == "uniform":
            sampled_sequence = uniform_sampling(sequence, n_sample_frames)
        elif frame_sampling_method == "random":
            sampled_sequence = random_sampling(
                sequence, n_sample_frames, zero_include=True, replace=False
            )
        elif frame_sampling_method == "stratified_random":
            sampled_sequence = stratified_random_sampling(sequence, n_sample_frames)
        return sampled_sequence

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
            for j in tqdm(
                range(self.start_episode, num_episodes), desc=f"Loading {data_path}"
            ):
                if (
                    self.max_episode[i] is not None
                    and data_path_append_num >= self.max_episode[i]
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
                point_tracking_sequence = np.clip(
                    point_tracking_sequence,
                    a_min=0,
                    a_max=self.point_tracking_img_size[0] - 1,
                )
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
                )
                point_tracking_sequence = self.sample_point_tracking_frame(
                    point_tracking_sequence,
                    self.frame_sampling_method,
                    self.n_sample_frames + 1,
                )  # (n_sample_frames+1, 3, grid_size, grid_size)
                first_frame_point_uv = point_tracking_sequence[0, :2, :, :].copy()
                first_frame_point_uv = rearrange(
                    first_frame_point_uv, "C N1 N2 -> (N1 N2) C"
                )  # (N^2, 2)
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
                global_image = episode["rgb_arr"][0].copy()
                # resize the global image
                global_image = cv2.resize(global_image, self.frame_resize_shape)
                # get u,v under the resize shape
                first_frame_point_uv = (
                    first_frame_point_uv
                    / self.point_tracking_img_size[0]
                    * self.frame_resize_shape[0]
                ).astype(int)
                first_frame_point_uv = np.clip(
                    first_frame_point_uv, a_min=0, a_max=self.frame_resize_shape[0] - 1
                )
                text = episode["task_description"][0]

                data_path_append_num += 1

                self.train_data.append(
                    {
                        "global_image": global_image,
                        "point_tracking_sequence": point_tracking_sequence,
                        "first_frame_point_uv": first_frame_point_uv,
                        "text": text,
                    }
                )
            print(f">> Loaded {data_path_append_num} data from {data_path}")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        data = self.train_data[idx]
        global_image = data["global_image"]
        global_image = process_image(
            global_image, optional_transforms=self.optional_transforms
        )
        point_tracking_sequence = torch.from_numpy(data["point_tracking_sequence"])
        point_tracking_sequence = self.flow_transforms(point_tracking_sequence)
        first_frame_point_uv = data["first_frame_point_uv"]
        text = data["text"]

        sample = dict(
            global_image=global_image,
            point_tracking_sequence=point_tracking_sequence,
            first_frame_point_uv=first_frame_point_uv,
            text=text,
        )
        return sample
