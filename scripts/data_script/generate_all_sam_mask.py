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
    config_name="generate_sam_mask",
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
                "generate_sam_mask.py",
                f"episode_start={start}",
                f"episode_end={end}",
                f"data_buffer_path={cfg.data_buffer_path}",
            ],
            env=env,
        )
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
