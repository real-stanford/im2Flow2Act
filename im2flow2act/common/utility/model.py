import os

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass


def load_hydra_config(cfg_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)
    cfg = OmegaConf.create(cfg)
    return cfg


def load_config(model_path):
    cfg = omegaconf.OmegaConf.load(os.path.join(model_path, ".hydra/config.yaml"))
    cfg = OmegaConf.create(cfg)
    return cfg


def load_model(model_path, ckpt):
    cfg = load_config(model_path)
    model = hydra.utils.instantiate(cfg.model)
    loadpath = os.path.join(model_path, f"epoch={ckpt}.ckpt")
    checkpoint = torch.load(loadpath, map_location="cuda:0")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda()
    model.eval()
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def enable_gradient(model):
    for param in model.parameters():
        param.requires_grad = True
