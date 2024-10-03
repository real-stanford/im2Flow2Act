import hydra
import os
from im2flow2act.diffusion_policy.utility.evaluation import (
    replay_action,
)
import zarr
import torch


@hydra.main(
    version_base=None,
    config_path="../../config/diffusion_policy",
    config_name="replay_action",
)
def replay(cfg):
    data_buffer = zarr.open(cfg.data_buffer_path, mode="r")
    if cfg.num_samples is None:
        cfg.num_samples = len(data_buffer)
    with torch.no_grad():
        replay_save_path = cfg.replay_save_path
        os.makedirs(replay_save_path, exist_ok=True)
        replay_action(
            replay_offset=cfg.replay_offset,
            num_samples=cfg.num_samples,
            env_cfg=cfg.env_cfg,
            data_buffer=data_buffer,
            result_save_path=replay_save_path,
        )


if __name__ == "__main__":
    replay()
