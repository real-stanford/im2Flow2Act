from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import hydra
import numpy as np
import omegaconf
import zarr
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.tapnet.utility.utility import get_buffer_size, max_distance_moved

register_codecs()


def get_moving_mask(
    buffer_path, index, moving_threshold, is_sam, simulation_herustic_filter=[]
):
    try:
        buffer = zarr.open(buffer_path, mode="r+")
        tracking_point_sequence = (
            buffer[f"episode_{index}/point_tracking_sequence"][:]
            if not is_sam
            else buffer[f"episode_{index}/sam_point_tracking_sequence"][:]
        )
        moving_mask = max_distance_moved(tracking_point_sequence) > moving_threshold
        if "left" in simulation_herustic_filter:
            # sim2real camera
            condition = (tracking_point_sequence[:, 0, 0] < 40) & (
                tracking_point_sequence[:, 0, 1] < 100
            )
            moving_mask[condition] = False
        if "right" in simulation_herustic_filter:
            condition = (tracking_point_sequence[:, 0, 0] > 110) & (
                tracking_point_sequence[:, 0, 1] < 100
            )
            moving_mask[condition] = False
        if "bbox" in simulation_herustic_filter:
            frame = buffer[f"episode_{index}/rgb_arr"][0]
            rgb = buffer[f"episode_{index}/camera_0/rgb"][0]
            camera_h, camera_w = rgb.shape[:2]
            frame_resize_shape = frame.shape[1]
            rgb = cv2.resize(rgb, (frame_resize_shape, frame_resize_shape))
            bboxes = buffer[f"episode_{index}/bbox"][:]
            inside_conditions = []
            for bbox in bboxes:
                min_x, min_y, max_x, max_y = bbox
                min_x = min_x / camera_w * frame_resize_shape
                min_y = min_y / camera_h * frame_resize_shape
                max_x = max_x / camera_w * frame_resize_shape
                max_y = max_y / camera_h * frame_resize_shape
                # Set moving mask to False for points outside the bounding box
                condition = (
                    (tracking_point_sequence[:, 0, 0] > min_x)
                    & (tracking_point_sequence[:, 0, 0] < max_x)
                    & (tracking_point_sequence[:, 0, 1] > min_y)
                    & (tracking_point_sequence[:, 0, 1] < max_y)
                )
                inside_conditions.append(condition)
            # Find indices where all arrays are False
            condition = np.logical_or(*inside_conditions)
            indices = np.where(~condition)
            moving_mask[indices] = False
        if is_sam:
            buffer[f"episode_{index}/sam_moving_mask"] = moving_mask
        else:
            buffer[f"episode_{index}/moving_mask"] = moving_mask
        return True
    except Exception as e:
        print(e)
        return False


def moving_threshold(cfg):
    print(cfg)

    for i, data_path in tqdm(
        enumerate(cfg.data_buffer_pathes), desc="Generating moving mask"
    ):
        point_moving_threshold = cfg.moving_threshold_args.point_move_thresholds[i]
        num_episodes = get_buffer_size(zarr.open(data_path))
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = [
                executor.submit(
                    get_moving_mask,
                    data_path,
                    episode_index,
                    point_moving_threshold,
                    cfg.moving_threshold_args.is_sam,
                    cfg.moving_threshold_args.simulation_herustic_filter,
                )
                for episode_index in range(num_episodes)
            ]
            for future in as_completed(futures):
                if not future.result():
                    raise RuntimeError("Failed to create moving mask!")


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="point_selection",
)
def main(cfg: omegaconf.DictConfig):
    import time

    start = time.time()
    if cfg.criterior == "moving_threshold":
        moving_threshold(cfg)
    print("Time taken: ", time.time() - start)


if __name__ == "__main__":
    main()
