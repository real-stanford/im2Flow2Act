import numpy as np
import torch
from torch import nn
from transformers import CLIPVisionModel

from im2flow2act.common.utility.model import freeze_model


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class AnimateFlow(nn.Module):
    def __init__(
        self,
        unet,
        clip_model,
        global_image_size,
        freeze_visual_encoder=True,
        global_condition_type="cls_token",
        emb_dim=768,
    ) -> None:
        super().__init__()
        self.freeze_visual_encoder = freeze_visual_encoder
        self.global_condition_type = global_condition_type
        self.visual_encoder = CLIPVisionModel.from_pretrained(clip_model)
        if self.freeze_visual_encoder:
            freeze_model(self.visual_encoder)

        if not self.freeze_visual_encoder and (
            self.global_condition_type != "cls_token"
            or self.global_condition_type != "all"
        ):
            self.visual_encoder.vision_model.post_layernorm = nn.Identity()

        self.unet = unet

        with torch.no_grad():
            reference_image = torch.zeros(1, 3, *global_image_size)
            reference_last_hidden = self.visual_encoder(reference_image)[0]
            token_num, clip_dim = reference_last_hidden.shape[-2:]

        self.vit_grid_size = np.sqrt(token_num).astype(int)
        self.global_projection_head = nn.Linear(clip_dim, emb_dim)
        self.local_projection_head = nn.Linear(clip_dim, emb_dim)

        zero_module(self.global_projection_head)
        zero_module(self.local_projection_head)

        pos_2d = posemb_sincos_2d(*global_image_size, clip_dim).reshape(
            *global_image_size, clip_dim
        )
        self.register_buffer("pos_2d", pos_2d)

    def prepare_condition(self, global_image, point_uv):
        B, _, H, W = global_image.shape
        vision_output = self.visual_encoder(global_image)
        last_hidden_states = vision_output["last_hidden_state"]
        vit_patch_embedding = last_hidden_states[:, 1:]
        # get global feature
        if self.global_condition_type == "cls_token":
            vit_cls_token = vision_output["pooler_output"]
            global_condition = self.global_projection_head(vit_cls_token).unsqueeze(
                1
            )  # (B,1,C)
        elif self.global_condition_type == "patch":
            global_condition = self.global_projection_head(
                vit_patch_embedding
            )  # (B,P^2,C)
        elif self.global_condition_type == "all":
            vit_cls_token = vision_output["pooler_output"]
            global_condition = self.global_projection_head(
                torch.cat([vit_cls_token.unsqueeze(1), vit_patch_embedding], axis=1)
            )  # (B,1+P^2,C)
        else:
            raise ValueError(
                f"global_condition_type {self.global_condition_type} not supported"
            )
        uv_discriptor = self.pos_2d[point_uv[:, :, 1], point_uv[:, :, 0]]
        local_condition = self.local_projection_head(uv_discriptor)  # (B,N,C)
        return global_condition, local_condition

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        global_image,
        point_uv,
    ):
        global_condition, local_condition = self.prepare_condition(
            global_image, point_uv
        )
        # concat with text feature
        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                global_condition,
                local_condition,
            ],
            axis=1,
        )  # (B,77+1+N,C)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return model_pred

    def load_model(self, path):
        print("Loading complete model...")
        self.load_state_dict(torch.load(path))
        print(">> loaded model")
