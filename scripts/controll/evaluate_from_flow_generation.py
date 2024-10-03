import os

import hydra
import torch

from im2flow2act.common.utility.diffusion import load_flow_diffusion_model
from im2flow2act.common.utility.file import read_pickle
from im2flow2act.common.utility.model import load_config
from im2flow2act.diffusion_policy.utility.evaluation import (
    evaluate_flow_diffusion_policy_from_generated_flow,
)


@hydra.main(
    version_base=None,
    config_path="../../config/diffusion_policy",
    config_name="evaluate_from_flow_generation",
)
def eval(cfg):
    model_cfg = load_config(cfg.model_path)
    model, noise_scheduler = load_flow_diffusion_model(
        cfg.model_path, cfg.ckpt, use_ema=model_cfg.training.use_ema
    )
    stats = read_pickle(os.path.join(cfg.model_path, "stats.pickle"))
    camera_intrinsic = read_pickle(cfg.camera_intrinsic_path)
    camera_pose_matrix = read_pickle(cfg.camera_pose_matrix_path)
    with torch.no_grad():
        gt_result_save_path = os.path.join(
            cfg.model_path,
            "evaluation",
            f"epoch_{cfg.ckpt}",
            "generated_flow",
        )
        os.makedirs(gt_result_save_path, exist_ok=True)
        evaluate_flow_diffusion_policy_from_generated_flow(
            model=model,
            noise_scheduler=noise_scheduler,
            num_inference_steps=model_cfg.num_inference_steps,
            stats=stats,
            num_samples=20,
            env_cfg=model_cfg.evaluation.gt_env_cfg,
            data_dirs=cfg.data_dirs,
            num_points=model_cfg.dataset.num_points,
            point_tracking_img_size=model_cfg.dataset.point_tracking_img_size,
            resize_shape=model_cfg.dataset.camera_resize_shape,
            action_dim=model_cfg.action_dim,
            obs_horizon=model_cfg.dataset.obs_horizon,
            action_horizon=model_cfg.dataset.action_horizon,
            target_flow_horizon=model_cfg.dataset.target_flow_horizon,
            pred_horizon=model_cfg.dataset.pred_horizon,
            point_tracking_camera_id=0,
            camera_intrinsic=camera_intrinsic,
            camera_pose_matrix=camera_pose_matrix,
            normalize_pointcloud=model_cfg.dataset.normalize_pointcloud,
            result_save_path=gt_result_save_path,
            flow_generator_additional_args=cfg.flow_generator_additional_args,
            seed=cfg.seed,
        )


if __name__ == "__main__":
    eval()
