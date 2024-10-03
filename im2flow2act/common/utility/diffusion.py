import os

import hydra
import torch

from im2flow2act.common.utility.model import load_config


def load_flow_diffusion_model(
    model_path, ckpt, load_pretrain_weight=True, use_ema=False, **kwargs
):
    model_cfg = load_config(model_path)
    model = hydra.utils.instantiate(model_cfg.model)
    if use_ema:
        print("Loading ema model checkpoints!")
        ckpt_path = os.path.join(model_path, "checkpoints", f"ema_epoch_{ckpt}.ckpt")
    else:
        ckpt_path = os.path.join(model_path, "checkpoints", f"epoch_{ckpt}.ckpt")
    if load_pretrain_weight:
        model.load_state_dict(torch.load(ckpt_path))
    model.to("cuda")
    model.eval()
    noise_scheduler = hydra.utils.instantiate(model_cfg.noise_scheduler)

    return model, noise_scheduler


def build_eval_dataset(model_path, **args):
    model_cfg = load_config(model_path)
    eval_dataset_cfg = model_cfg["dataset"]
    eval_dataset = hydra.utils.instantiate(eval_dataset_cfg, **args)
    return eval_dataset
