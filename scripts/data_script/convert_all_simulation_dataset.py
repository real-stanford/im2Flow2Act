import subprocess

import hydra
import omegaconf
import zarr

from im2flow2act.common.utility.parallel import assign_task_bounds_to_process
from im2flow2act.tapnet.utility.utility import get_buffer_size


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="convert_simulation_dataset",
)
def main(cfg: omegaconf.DictConfig):
    print(cfg)
    data_buffer = zarr.open(cfg.data_buffer_path, mode="a")
    num_episode = get_buffer_size(data_buffer)
    task_bounds = assign_task_bounds_to_process(num_episode, cfg.max_processes)
    processes = []
    for i, (start, end) in enumerate(task_bounds):
        # Start a new process for each task range
        process = subprocess.Popen(
            [
                "python",
                "convert_simulation_dataset.py",
                f"episode_start={start}",
                f"episode_end={end}",
                f"data_buffer_path={cfg.data_buffer_path}",
                f"store_path={cfg.store_path}",
                f"dataset={cfg.dataset}",
                f"downsample_ratio={cfg.downsample_ratio}",
                f"n_sample_frame={cfg.n_sample_frame}",
            ],
        )
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
