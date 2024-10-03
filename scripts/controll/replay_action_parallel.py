import subprocess

import hydra
import omegaconf
import zarr

from im2flow2act.common.utility.parallel import assign_task_bounds_to_process


@hydra.main(
    version_base=None,
    config_path="../../config/diffusion_policy",
    config_name="replay_action",
)
def main(cfg: omegaconf.DictConfig):
    processes = []
    data_buffer = zarr.open(cfg.data_buffer_path, mode="r")
    task_bounds = assign_task_bounds_to_process(len(data_buffer), cfg.num_processes)
    print("task_bounds:", task_bounds)
    for i, (replay_offset, replay_end) in enumerate(task_bounds):
        # Start a new process for each task range
        num_samples = replay_end - replay_offset
        process = subprocess.Popen(
            [
                "python",
                "replay_action.py",
                f"replay_offset={replay_offset}",
                f"num_samples={num_samples}",
            ],
        )
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
