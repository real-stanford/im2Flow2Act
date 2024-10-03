import os

import hydra
import torch
import zarr

from im2flow2act.common.utility.diffusion import (
    build_eval_dataset,
    load_flow_diffusion_model,
)
from im2flow2act.common.utility.file import read_pickle
from im2flow2act.common.utility.model import load_config
from im2flow2act.diffusion_policy.utility.evaluation import (
    evaluate_flow_diffusion_policy,
)


@hydra.main(
    version_base=None,
    config_path="../../config/diffusion_policy",
    config_name="evaluate_flow_diffusion_policy",
)
def eval(cfg):
    model_cfg = load_config(cfg.model_path)
    model, noise_scheduler = load_flow_diffusion_model(
        cfg.model_path, cfg.ckpt, use_ema=model_cfg.training.use_ema
    )
    stats = read_pickle(os.path.join(cfg.model_path, "stats.pickle"))
    # gt flows
    gt_dataset_args = (
        model_cfg.evaluation.gt_eval_dataset_args
        if model_cfg.evaluation.gt_eval_dataset_args is not None
        else {}
    )
    gt_eval_dataset = build_eval_dataset(
        cfg.model_path,
        max_episode=model_cfg.evaluation.gt_eval_num,
        optional_transforms=[],
        **gt_dataset_args,
    )
    gt_data_buffers = [
        zarr.open(data_dir, mode="r") for data_dir in gt_eval_dataset.data_dirs
    ]
    with torch.no_grad():
        gt_result_save_path = os.path.join(
            cfg.model_path, "evaluation", f"epoch_{cfg.ckpt}", "gt_flow"
        )
        os.makedirs(gt_result_save_path, exist_ok=True)
        evaluate_flow_diffusion_policy(
            model=model,
            noise_scheduler=noise_scheduler,
            num_inference_steps=model_cfg.num_inference_steps,
            stats=stats,
            num_samples=model_cfg.evaluation.gt_eval_num,
            env_cfg=model_cfg.evaluation.gt_env_cfg,
            eval_dataset=gt_eval_dataset,
            data_buffers=gt_data_buffers,
            result_save_path=gt_result_save_path,
            seed=cfg.seed,
        )


if __name__ == "__main__":
    eval()
