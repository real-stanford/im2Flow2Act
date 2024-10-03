import concurrent.futures

import hydra
import numpy as np
import omegaconf
import zarr
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.tapnet.utility.utility import get_buffer_size, max_distance_moved

register_codecs()


def annotate_robot_mask(cfg):
    def get_robot_mask(
        buffer,
        index,
        moving_threshold,
        t_threshold,
        is_sam=False,
        simulation_herustic=False,
        simulation_herustic_patial=False,
        zero_robot_mask=False,
        real_herustic_filter_type=None,
    ):
        try:
            tracking_point_sequence = (
                buffer[f"episode_{index}/point_tracking_sequence"][:]
                if not is_sam
                else buffer[f"episode_{index}/sam_point_tracking_sequence"][:]
            )
            if simulation_herustic:
                moving_mask = buffer[f"episode_{index}/sam_moving_mask"][:]
                # sim2real
                robot_mask = (
                    (tracking_point_sequence[:, 0, 1] < 65)
                    & (tracking_point_sequence[:, 0, 0] < 140)
                    & (tracking_point_sequence[:, 0, 0] > 35)
                    & moving_mask
                )
            elif simulation_herustic_patial:
                moving_mask = buffer[f"episode_{index}/sam_moving_mask"][:]
                # sim2real
                robot_mask = (
                    (tracking_point_sequence[:, 0, 1] < 65)
                    & (tracking_point_sequence[:, 0, 0] < 140)
                    & (tracking_point_sequence[:, 0, 0] > 80)
                    & moving_mask
                )
            elif zero_robot_mask:
                robot_mask = np.zeros(tracking_point_sequence.shape[0], dtype=bool)
            else:
                moving_mask = buffer[f"episode_{index}/moving_mask"][:]
                robot_mask = (
                    max_distance_moved(tracking_point_sequence, t_threshold)
                    > moving_threshold
                )
                robot_mask = robot_mask & moving_mask

            buffer[f"episode_{index}/robot_mask"] = robot_mask
            return True
        except Exception as e:
            print(e)
            return False

    for i, data_path in tqdm(
        enumerate(cfg.data_buffer_pathes), desc="Generating robot mask"
    ):
        buffer = zarr.open(data_path)
        point_moving_threshold = cfg.robot_mask_args.point_move_thresholds[i]
        t_threshold = cfg.robot_mask_args.t_thresholds[i]
        num_episodes = get_buffer_size(buffer)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cfg.max_workers
        ) as executor:
            futures = set()
            for episode_index in range(num_episodes):
                futures.add(
                    executor.submit(
                        get_robot_mask,
                        buffer,
                        episode_index,
                        point_moving_threshold,
                        t_threshold,
                        cfg.robot_mask_args.is_sam,
                        cfg.robot_mask_args.simulation_herustic,
                        cfg.robot_mask_args.simulation_herustic_patial,
                        cfg.robot_mask_args.zero_robot_mask,
                        cfg.robot_mask_args.real_herustic_filter_type,
                    )
                )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to create robot mask!")


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="point_annotation",
)
def main(cfg: omegaconf.DictConfig):
    print(cfg)
    if cfg.mask == "robot_mask":
        annotate_robot_mask(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
