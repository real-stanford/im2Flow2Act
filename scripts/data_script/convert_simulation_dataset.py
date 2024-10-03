import os

import cv2
import hydra
import numcodecs
import numpy as np
import omegaconf
import zarr
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import JpegXl, register_codecs
from im2flow2act.common.utility.zarr import parallel_saving
from im2flow2act.tapnet.utility.utility import get_buffer_size

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
register_codecs()


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="convert_simulation_dataset",
)
def main(cfg: omegaconf.DictConfig):
    # store destination
    store_path = cfg.store_path
    group = zarr.open(store_path, mode="a")
    embodiment_group = group.require_groups(cfg.dataset)[0]

    # read in data from buffer
    data_buffer = zarr.open(cfg.data_buffer_path, mode="a")
    if cfg.episode_start is None or cfg.episode_end is None:
        n_episode = get_buffer_size(data_buffer)
        episode_arr = np.arange(n_episode)
    else:
        episode_arr = np.arange(cfg.episode_start, cfg.episode_end)
    for i in tqdm(episode_arr, desc="Constructing dataset"):
        images = data_buffer[f"episode_{i}/camera_{cfg.camera_id}/rgb"][:]
        episode_data = embodiment_group.require_group(f"episode_{i}")
        images = np.array(images, dtype=np.uint8)
        images = np.array(
            [
                cv2.resize(image, (cfg.resize_shape, cfg.resize_shape))
                for image in images
            ]
        )
        n, h, w, c = images.shape
        if cfg.drop_front_ratio is not None:
            drop_front = int(n * cfg.drop_front_ratio)
            images = images[drop_front:]
        if cfg.drop_tail_ratio is not None:
            drop_tail = int(n * cfg.drop_tail_ratio)
            images = images[:-drop_tail]

        if cfg.downsample_ratio is not None:
            images = images[:: cfg.downsample_ratio]

        n, h, w, c = images.shape
        if cfg.n_sample_frame is not None:
            n_sample_frame = cfg.n_sample_frame
            # if n >= cfg.n_sample_frame:
            #     sample_indices = np.round(
            #         np.linspace(0, n - 1, cfg.n_sample_frame)
            #     ).astype(int)
            # else:
            sample_indices = np.arange(cfg.n_sample_frame)
            sample_indices = np.clip(sample_indices, 0, n - 1)
            images = images[sample_indices]
        else:
            n_sample_frame = n
            sample_indices = np.arange(n)

        sample_indices_array = episode_data.create_dataset(
            "sample_indices",
            shape=(n_sample_frame,),
            dtype=np.int32,
            overwrite=True,
        )
        sample_indices_array[:] = sample_indices

        n, h, w, c = images.shape
        this_compressor = JpegXl(level=80, numthreads=1)
        parallel_saving(
            group=episode_data,
            array_name="rgb_arr",
            shape=(n, h, w, c),
            chunks=(1, h, w, c),
            dtype=np.uint8,
            overwrite=True,
            arr_to_save=images,
            compressor=this_compressor,
        )
        if cfg.add_task_description:
            task_description = data_buffer[f"episode_{i}/info"][0]["task_description"]
            task_description_array = episode_data.create_dataset(
                "task_description",
                shape=(1,),  # Since it's a single string
                dtype=object,
                object_codec=numcodecs.VLenUTF8(),
                overwrite=True,
            )
            task_description_array[0] = task_description
        else:
            pass

        if cfg.max_episode is not None and i >= cfg.max_episode:
            break


if __name__ == "__main__":
    main()
