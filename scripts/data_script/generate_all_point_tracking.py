import os
import subprocess

import hydra
import omegaconf
import zarr

from im2flow2act.common.utility.parallel import assign_task_bounds_to_gpus
from im2flow2act.tapnet.utility.utility import get_buffer_size


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="generate_point_tracking",
)
def main(cfg: omegaconf.DictConfig):
    print(cfg)
    avaliable_gpu = cfg.avaliable_gpu
    num_gpu = len(avaliable_gpu)
    data_buffer = zarr.open(cfg.data_buffer_path, mode="a")
    num_episode = get_buffer_size(data_buffer)
    task_bounds = assign_task_bounds_to_gpus(num_episode, num_gpu)
    processes = []
    for i, (start, end) in enumerate(task_bounds):
        # Start a new process for each task range
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(avaliable_gpu[i])
        process = subprocess.Popen(
            [
                "python",
                "generate_point_tracking.py",
                f"episode_start={start}",
                f"episode_end={end}",
                f"data_buffer_path={cfg.data_buffer_path}",
                f"num_points={cfg.num_points}",
                f"sam_iterative={cfg.sam_iterative}",
                f"sam_iterative_additional_kwargs.from_grid={cfg.sam_iterative_additional_kwargs.from_grid}",
                f"background_filter={cfg.background_filter}",
                f"from_bbox={cfg.from_bbox}",
                f"simulation_herustic_filter={cfg.simulation_herustic_filter}",
                f"dbscan_bbox={cfg.dbscan_bbox}",
                f"dbscan_additional_kwargs.dbscan_use_sam={cfg.dbscan_additional_kwargs.dbscan_use_sam}",
                f"dbscan_additional_kwargs.dbscan_epsilon={cfg.dbscan_additional_kwargs.dbscan_epsilon}",
                f"dbscan_additional_kwargs.dbscan_sam_area_thres={cfg.dbscan_additional_kwargs.dbscan_sam_area_thres}",
                f"dbscan_additional_kwargs.dbscan_sam_closeness={cfg.dbscan_additional_kwargs.dbscan_sam_closeness}",
                f"dbscan_additional_kwargs.dbscan_min_samples={cfg.dbscan_additional_kwargs.dbscan_min_samples}",
                f"dbscan_additional_kwargs.dbscan_bbox_padding={cfg.dbscan_additional_kwargs.dbscan_bbox_padding}",
            ],
            env=env,
        )
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
