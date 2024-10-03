import math
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange, reduce, repeat
from im2flow2act.diffusion_policy.model.network import GaussianMlp, Mlp
from torch import nn
from torchvision.ops import roi_align


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


class CNN(nn.Module):
    def __init__(self, out_size) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Linear(6400, out_size)

    def forward(self, images):
        return self.linear(self.cnn(images))


class CNN_v3(nn.Module):
    def __init__(self, out_size) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Linear(8960, out_size)

    def forward(self, images):
        return self.linear(self.cnn(images))


class SimpleCNN(nn.Module):
    def __init__(
        self,
        state_size,
        n_frames,
        out_size,
        net_arch,
        nmb_prototypes=0,
        normalize=False,
        use_group_norm=False,
        use_spectral_norm=False,
        use_batch_norm=False,
        encode_distribution=False,
        transformer_encoder=None,
        temporal_position_encoding=None,
        use_classification_head=False,
    ) -> None:
        super().__init__()
        self.encode_distribution = encode_distribution
        self.state_size = state_size
        self.state_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Linear(6400, self.state_size)
        self.transformer_encoder = transformer_encoder
        if use_classification_head:
            self.classification_head = nn.Linear(6400, 1)
        if encode_distribution:
            self.net = GaussianMlp(
                int(state_size * n_frames), out_size, net_arch, use_batch_norm
            )
        else:
            self.net = Mlp(
                int(state_size * n_frames),
                out_size,
                net_arch,
                use_batch_norm,
                use_group_norm=use_group_norm,
                use_spectral_norm=use_spectral_norm,
            )

        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
            self.normalize = normalize
        else:
            self.prototypes = None

    def forward(self, images, bbox):
        """
        x: trajectory (b,t,c,h,w)
        bbox: bounding box (b,t,o,4) where o is the number of the object. Note, the box has been normalized
        """
        state_representation = self.get_state_representation(images, bbox)  # (b,t,f)
        # projection
        traj_representation = self.get_traj_representation(state_representation)
        if self.prototypes is not None:
            if self.normalize:
                traj_representation = nn.functional.normalize(
                    traj_representation, dim=1, p=2
                )
            return self.prototypes(traj_representation)
        return traj_representation

    def classification(self, images):
        out = self.state_net(images)

        return self.classification_head(out)  # (b*t,1)

    def get_traj_representation(self, state_representation):
        if self.transformer_encoder is not None:
            traj_representation = self.transformer_encoder(
                state_representation
            )  # (b,t,f)
        else:
            state_representation = rearrange(state_representation, "b t f -> b (t f)")
            traj_representation = self.net(state_representation)
        return traj_representation

    def get_state_representation(self, images, bbox):
        b, t, c, h, w = images.shape
        images = rearrange(images, "b t c h w -> (b t) c h w")
        out = self.linear(self.state_net(images))  # (b*t,f)
        out = rearrange(out, "(b t) f-> b t f", b=b)
        return out


class VisualMotionEncoder(nn.Module):
    def __init__(
        self,
        vision_encoder,
        nmb_prototypes,
        state_size,
        out_size,
        vision_only=False,
        normalize=False,
        start_end=False,
        goal_condition=False,
        temporal_transformer_encoder=None,
    ) -> None:
        super().__init__()

        self.vision_encoder = vision_encoder
        self.state_size = state_size
        self.out_size = out_size
        self.vision_only = vision_only
        self.goal_condition = goal_condition

        self.temporal_transformer_encoder = temporal_transformer_encoder
        self.start_end = start_end
        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
            self.normalize = normalize
        else:
            self.prototypes = None

    def forward(self, image, state):
        """
        image: trajectory (b,t,c,h,w)
        state: bounding box (b,t,f)
        """
        state_representation = self.get_state_representation(image, state)  # (b,t,f)
        traj_representation = self.get_traj_representation(state_representation)

        # projection
        if self.prototypes is not None:
            if self.normalize:
                traj_representation = nn.functional.normalize(
                    traj_representation, dim=1, p=2
                )
            return self.prototypes(traj_representation)
        return traj_representation

    def get_traj_representation(self, state_representation):
        traj_representation = self.temporal_transformer_encoder(
            state_representation
        )  # (b,t,f)

        return traj_representation

    def get_state_representation(self, image, state):
        b, t, c, h, w = image.shape
        if self.start_end:
            if self.goal_condition:
                image = image[:, [0, -2, -1]]
            else:
                image = image[:, [0, -1]]
        image = rearrange(image, "b t c h w  -> (b t) c h w")
        image_encoding = self.vision_encoder(image)
        image_encoding = rearrange(image_encoding, "(b t) f -> b t f", b=b)

        return image_encoding


class VisualMotionPrior(nn.Module):
    def __init__(
        self,
        vision_encoder,
        out_size,
        vision_only=False,
        nmb_prototypes=0,
        normalize=False,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.nmb_prototypes = nmb_prototypes
        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
        self.normalize = normalize
        self.vision_only = vision_only

    def forward(self, image, state):
        """
        x: trajectory (b,t,c,h,w)
        state: bounding box (b,t,f)
        """
        state_representation = self.get_state_representation(image, state)  # (b,f)
        if self.normalize:
            state_representation = nn.functional.normalize(
                state_representation, dim=1, p=2
            )

        if self.nmb_prototypes > 0:
            return self.prototypes(state_representation)
        else:
            return state_representation

    def get_state_representation(self, image, state):
        b, t, c, h, w = image.shape
        image = rearrange(image, "b t c h w  -> (b t) c h w")
        image_encoding = self.vision_encoder(image)
        image_encoding = rearrange(image_encoding, "(b t) f -> b (t f)", b=b)

        return image_encoding


class AblationVisualMotionEncoder(nn.Module):
    def __init__(
        self,
        vision_encoder,
        nmb_prototypes,
        state_size,
        out_size,
        vision_only=False,
        normalize=False,
        start_end=False,
        goal_condition=False,
        temporal_transformer_encoder=None,
    ) -> None:
        super().__init__()
        """
        ablation module for no prototype learning. 1) Turn on bias 2) Turn off normalization
        Lazy implemetation such that the interface do not change
        """

        self.vision_encoder = vision_encoder
        self.state_size = state_size
        self.out_size = out_size
        self.vision_only = vision_only
        self.goal_condition = goal_condition

        self.temporal_transformer_encoder = temporal_transformer_encoder
        self.start_end = start_end
        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=True)
            self.normalize = normalize
        else:
            self.prototypes = None

    def forward(self, image, state):
        """
        image: trajectory (b,t,c,h,w)
        state: bounding box (b,t,f)
        """
        state_representation = self.get_state_representation(image, state)  # (b,t,f)
        traj_representation = self.get_traj_representation(state_representation)

        # projection
        if self.prototypes is not None:
            if self.normalize:
                traj_representation = nn.functional.normalize(
                    traj_representation, dim=1, p=2
                )
            return self.prototypes(traj_representation)
        return traj_representation

    def get_traj_representation(self, state_representation):
        traj_representation = self.temporal_transformer_encoder(
            state_representation
        )  # (b,t,f)

        return traj_representation

    def get_state_representation(self, image, state):
        b, t, c, h, w = image.shape
        if self.start_end:
            if self.goal_condition:
                image = image[:, [0, -2, -1]]
            else:
                image = image[:, [0, -1]]
        image = rearrange(image, "b t c h w  -> (b t) c h w")
        image_encoding = self.vision_encoder(image)
        image_encoding = rearrange(image_encoding, "(b t) f -> b t f", b=b)

        return image_encoding


class TCNVisualMotionEncoder(nn.Module):
    def __init__(self, vision_encoder, state_net, state_size, out_size) -> None:
        super().__init__()

        self.vision_encoder = vision_encoder
        self.state_net = state_net
        self.state_size = state_size
        self.out_size = out_size

    def forward(self, image, state):
        """
        image: trajectory (b,c,h,w)
        state: bounding box (b,t,f)
        """
        state_representation = self.get_state_representation(image, state)  # (b,f)
        state_representation = self.state_net(state_representation)
        return state_representation

    def get_state_representation(self, image, state):
        image_encoding = self.vision_encoder(image)
        return image_encoding


class MixinEncoder(nn.Module):
    def __init__(
        self,
        vision_encoder,
        state_encoder,
        mix_net,
        nmb_prototypes,
        state_size,
        out_size,
        vision_only=False,
        normalize=False,
        start_end=False,
        goal_condition=False,
        temporal_transformer_encoder=None,
    ) -> None:
        super().__init__()

        self.vision_encoder = vision_encoder
        self.state_encoder = state_encoder
        self.mix_net = mix_net
        self.state_size = state_size
        self.out_size = out_size
        self.vision_only = vision_only
        self.goal_condition = goal_condition

        self.temporal_transformer_encoder = temporal_transformer_encoder
        self.start_end = start_end
        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
            self.normalize = normalize
        else:
            self.prototypes = None

    def forward(self, image, state):
        """
        image: trajectory (b,t,c,h,w)
        state: bounding box (b,t,f)
        """
        state_representation = self.get_state_representation(image, state)  # (b,t,f)
        traj_representation = self.get_traj_representation(state_representation)

        # projection
        if self.prototypes is not None:
            if self.normalize:
                traj_representation = nn.functional.normalize(
                    traj_representation, dim=1, p=2
                )
            return self.prototypes(traj_representation)
        return traj_representation

    def get_traj_representation(self, state_representation):
        # if self.start_end:
        #     if self.goal_condition:
        #         state_representation = state_representation[:,[0,-2,-1],:] #(b,3,f)
        #     else:
        #         state_representation = state_representation[:,[0,-1],:] #(b,2,f)

        traj_representation = self.temporal_transformer_encoder(
            state_representation
        )  # (b,t,f)

        return traj_representation

    def get_state_representation(self, image, state):
        b, t, c, h, w = image.shape
        if self.start_end:
            if self.goal_condition:
                image = image[:, [0, -2, -1]]
                state = state[:, [0, -2, -1]]
            else:
                image = image[:, [0, -1]]
                state = state[:, [0, -1]]
        image = rearrange(image, "b t c h w  -> (b t) c h w")
        state = rearrange(state, "b t f -> (b t) f")
        image_encoding = self.vision_encoder(image)
        if self.vision_only:
            mixin_encoding = image_encoding
        elif self.state_encoder is None:
            mixin_encoding = torch.cat([image_encoding, state], dim=-1)
        else:
            state_encoding = self.state_encoder(state)
            mixin_encoding = torch.cat([image_encoding, state_encoding], dim=-1)

        mix_out = self.mix_net(mixin_encoding)
        mix_out = rearrange(mix_out, "(b t) f -> b t f", b=b)

        return mix_out


class MixinMotionPrior(nn.Module):
    def __init__(
        self,
        vision_encoder,
        state_encoder,
        mix_net,
        out_size,
        vision_only=False,
        nmb_prototypes=0,
        normalize=False,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.state_encoder = state_encoder
        self.mix_net = mix_net
        self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
        self.normalize = normalize
        self.vision_only = vision_only

    def forward(self, image, state):
        """
        x: trajectory (b,t,c,h,w)
        state: bounding box (b,t,f)
        """
        state_representation = self.get_state_representation(image, state)  # (b,f)
        if self.normalize:
            state_representation = nn.functional.normalize(
                state_representation, dim=1, p=2
            )

        return self.prototypes(state_representation)

    def get_state_representation(self, image, state):
        b, t, c, h, w = image.shape
        image = rearrange(image, "b t c h w  -> (b t) c h w")
        state = rearrange(state, "b t f -> (b t) f")
        image_encoding = self.vision_encoder(image)
        if self.vision_only:
            mixin_encoding = image_encoding
        elif self.state_encoder is None:
            mixin_encoding = torch.cat([image_encoding, state], dim=-1)
        else:
            state_encoding = self.state_encoder(state)
            mixin_encoding = torch.cat([image_encoding, state_encoding], dim=-1)

        mix_out = self.mix_net(mixin_encoding)
        mix_out = rearrange(mix_out, "(b t) f -> b (t f)", b=b)

        return mix_out


class BboxEncoder(nn.Module):
    def __init__(
        self,
        state_size,
        n_frames,
        n_objects,
        out_size,
        net_arch,
        start_end=False,
        append_distance=False,
        append_center=False,
        nmb_prototypes=0,
        normalize=False,
        use_group_norm=False,
        use_spectral_norm=False,
        use_batch_norm=False,
        encode_distribution=False,
        spatial_transformer_encoder=None,
        temporal_transformer_encoder=None,
        bbox_embedding=None,
    ) -> None:
        super().__init__()
        self.encode_distribution = encode_distribution
        self.state_size = state_size
        if append_distance:
            self.state_net = Mlp(
                int(n_objects * (4 + 2 + n_objects)),
                self.state_size,
                net_arch=[128, 128, 128],
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                use_spectral_norm=use_spectral_norm,
            )
        elif append_center:
            self.state_net = Mlp(
                int(n_objects * (4 + 2)),
                self.state_size,
                net_arch=[128, 128, 128],
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                use_spectral_norm=use_spectral_norm,
            )
        else:
            self.state_net = Mlp(
                int(n_objects * 4),
                self.state_size,
                net_arch=[128, 128, 128],
                use_batch_norm=use_batch_norm,
                use_group_norm=use_group_norm,
                use_spectral_norm=use_spectral_norm,
            )
        self.spatial_transformer_encoder = spatial_transformer_encoder
        self.temporal_transformer_encoder = temporal_transformer_encoder
        self.bbox_embedding = bbox_embedding
        self.start_end = start_end
        if self.start_end:
            n_frames = 2
        self.net = Mlp(
            int(state_size * n_frames),
            out_size,
            net_arch,
            use_batch_norm,
            use_group_norm=use_group_norm,
            use_spectral_norm=use_spectral_norm,
        )

        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
            self.normalize = normalize
        else:
            self.prototypes = None

    def forward(self, images, bbox):
        """
        x: trajectory (b,t,c,h,w)
        bbox: bounding box (b,t,o,4) where o is the number of the object. Note, the box has been normalized
        """
        state_representation = self.get_state_representation(images, bbox)  # (b,t,f)
        # projection
        traj_representation = self.get_traj_representation(state_representation)
        if self.prototypes is not None:
            if self.normalize:
                traj_representation = nn.functional.normalize(
                    traj_representation, dim=1, p=2
                )
            return self.prototypes(traj_representation)
        return traj_representation

    def get_traj_representation(self, state_representation):
        if self.start_end:
            state_representation = state_representation[:, [0, -1], :]  # (b,2,f)
        if self.temporal_transformer_encoder is not None:
            traj_representation = self.temporal_transformer_encoder(
                state_representation
            )  # (b,t,f)
        else:
            state_representation = rearrange(state_representation, "b t f -> b (t f)")
            traj_representation = self.net(state_representation)
        return traj_representation

    def get_state_representation(self, images, bbox):
        b, t, o, _ = bbox.shape
        if self.spatial_transformer_encoder is not None:
            if self.bbox_embedding is not None:
                bbox = rearrange(bbox, "b t o d -> (b t o) d")
                bbox_emb = self.bbox_embedding(bbox)
                bbox_emb = rearrange(bbox_emb, "(b t o) d -> (b t) o d", b=b, t=t, o=o)
                out = self.spatial_transformer_encoder(bbox_emb)
            else:
                bbox = rearrange(bbox, "b t o d -> (b t) o d")
                out = self.spatial_transformer_encoder(bbox)
        else:
            bbox = rearrange(bbox, "b t o d -> (b t) (o d)")
            out = self.state_net(bbox)  # (b*t,f)

        out = rearrange(out, "(b t) f-> b t f", b=b)
        return out


class BboxMotionPrior(nn.Module):
    def __init__(
        self,
        out_size,
        nmb_prototypes=0,
        normalize=False,
        spatial_transformer_encoder=None,
    ) -> None:
        super().__init__()
        self.spatial_transformer_encoder = spatial_transformer_encoder
        self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
        self.normalize = normalize

    def forward(self, images, bbox):
        """
        x: trajectory (b,t,c,h,w)
        bbox: bounding box (b,t,o,4) where o is the number of the object. Note, the box has been normalized
        """
        state_representation = self.get_state_representation(images, bbox)  # (b,t,f)
        if self.normalize:
            state_representation = nn.functional.normalize(
                state_representation, dim=1, p=2
            )

        return self.prototypes(state_representation)

    def get_state_representation(self, images, bbox):
        b, t, o, _ = bbox.shape
        bbox = rearrange(bbox, "b t o d -> (b t) o d")
        out = self.spatial_transformer_encoder(bbox)
        out = rearrange(out, "(b t) f-> b (t f)", b=b)
        return out


class DiffBboxEncoder(nn.Module):
    def __init__(
        self,
        state_size,
        n_frames,
        n_objects,
        out_size,
        net_arch,
        nmb_prototypes=0,
        normalize=False,
        use_group_norm=False,
        use_spectral_norm=False,
        use_batch_norm=False,
        encode_distribution=False,
        spatial_transformer_encoder=None,
        temporal_transformer_encoder=None,
    ) -> None:
        super().__init__()
        self.net = Mlp(
            int(4 * n_objects * (n_frames - 1)),
            out_size,
            net_arch,
            use_batch_norm,
            use_group_norm=use_group_norm,
            use_spectral_norm=use_spectral_norm,
        )

        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
            self.normalize = normalize
        else:
            self.prototypes = None

    def forward(self, images, bbox):
        """
        x: trajectory (b,t,c,h,w)
        bbox: bounding box (b,t,o,4) where o is the number of the object. Note, the box has been normalized
        """
        diff_bbox = bbox[:, 1:] - bbox[:, :-1]
        diff_bbox = rearrange(diff_bbox.squeeze(1), "b o d -> b (o d)")

        # projection
        traj_representation = self.net(diff_bbox)
        if self.prototypes is not None:
            if self.normalize:
                traj_representation = nn.functional.normalize(
                    traj_representation, dim=1, p=2
                )
            return self.prototypes(traj_representation)
        return traj_representation


class DebugEncoder(torch.nn.Module):
    def __init__(self, in_size, out_size, net_arch, use_batch_norm=False) -> None:
        super().__init__()
        self.net = Mlp(in_size, out_size, net_arch, use_batch_norm)

    def forward(self, images, bbox):
        """
        x: trajectory (b,t,c,h,w)
        bbox: bounding box (b,t,o,4) where o is the number of the object. Note, the box has been normalized
        """
        state_representation = self.get_state_representation(images, bbox)  # (b,t,f)
        traj_representation = self.get_traj_representation(state_representation)
        return traj_representation

    def get_traj_representation(self, state_representation):
        state_representation = rearrange(state_representation, "b t f -> b (t f)")
        traj_representation = self.net(state_representation)
        return traj_representation

    def get_state_representation(self, images, bbox):
        return rearrange(bbox, "b t o n -> b t (o n)")


class StateDebugEncoder(torch.nn.Module):
    def __init__(
        self, in_size, out_size, net_arch, use_batch_norm=False, use_gaussian=False
    ) -> None:
        super().__init__()
        if use_gaussian:
            self.net = GaussianMlp(in_size, out_size, net_arch, use_batch_norm)
        else:
            self.net = Mlp(in_size, out_size, net_arch, use_batch_norm)

    def forward(self, images, bbox):
        """
        x: trajectory (b,t,c,h,w)
        state: bounding box (b,t,17)
        """
        state_representation = self.get_state_representation(images, bbox)  # (b,t,f)
        traj_representation = self.get_traj_representation(state_representation)
        return traj_representation

    def get_traj_representation(self, state_representation):
        state_representation = rearrange(state_representation, "b t f -> b (t f)")
        traj_representation = self.net(state_representation)
        return traj_representation

    def get_state_representation(self, images, bbox):
        return bbox


class MlpEncoder(torch.nn.Module):
    def __init__(self, in_size, out_size, net_arch, use_batch_norm=False) -> None:
        super().__init__()
        self.net = Mlp(in_size, out_size, net_arch, use_batch_norm)

    def forward(self, images, bbox):
        """_summary_

        Args:
            features (_type_): _description_
            bbox (_type_): (b,t,o,4)

        Returns:
            _type_: _description_
        """
        tensor_in = rearrange(bbox, "b t o f -> b (t o f)")
        return self.net(tensor_in)


class Conv3DEncoder(torch.nn.Module):
    def __init__(
        self,
        out_size,
        net_arch,
        nmb_prototypes=None,
        normalize=False,
        use_batch_norm=False,
    ) -> None:
        super().__init__()
        self.conv3d = Conv3D(None)
        self.net = Mlp(128, out_size, net_arch, use_batch_norm)

        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(out_size, nmb_prototypes, bias=False)
            self.normalize = normalize
        else:
            self.prototypes = None

    def forward(self, images, bbox):
        """_summary_

        Args:
            images (_type_): b,t,3,h,w
            bbox (_type_): _description_

        Returns:
            _type_: _description_
        """
        images = rearrange(images, "b t c h w -> b c t h w")
        con3d_out = self.conv3d(images)
        traj_representation = self.net(con3d_out)
        if self.prototypes is not None:
            if self.normalize:
                traj_representation = nn.functional.normalize(
                    traj_representation, dim=1, p=2
                )
            return self.prototypes(traj_representation)
        return traj_representation

    def get_state_representation(self, images, bbox):
        images = rearrange(images, "b t c h w -> b c t h w")
        con3d_out = self.conv3d(images)
        traj_representation = self.net(con3d_out)
        return traj_representation


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        state_size,
        n_frames,
        out_size,
        net_arch,
        use_batch_norm,
        remove_layer_num=1,
        encode_distribution=False,
        use_spatial_softmax=False,
        use_group_norm=False,
        transformer_encoder=None,
        temperature=0.1,
    ) -> None:
        super().__init__()

        self.state_net = ResnetConv(
            embedding_size=state_size,
            remove_layer_num=remove_layer_num,
            no_stride=False,
            use_spatial_softmax=use_spatial_softmax,
            temperature=temperature,
        )
        if use_group_norm:
            replace_submodules(
                root_module=self.state_net,
                predicate=lambda x: isinstance(x, nn.BatchNorm1d)
                or isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
        self.encode_distribution = encode_distribution
        self.transformer_encoder = transformer_encoder
        if encode_distribution:
            self.traj_net = GaussianMlp(
                int(state_size * n_frames),
                out_size,
                net_arch,
                use_batch_norm,
                use_group_norm,
            )
        else:
            self.traj_net = Mlp(
                int(state_size * n_frames),
                out_size,
                net_arch,
                use_batch_norm,
                use_group_norm,
            )

    def forward(self, x, bbox):
        """
        x: trajectory (b,t,c,h,w)
        bbox: bounding box (b,t,o,4) where o is the number of the object. Note, the box has been normalized
        """
        state_representation = self.get_state_representation(x, bbox)
        traj_representation = self.get_traj_representation(state_representation)

        return traj_representation  # tra: (b,f) _state_representation:(b,t,f)

    def get_traj_representation(self, state_representation):
        if self.transformer_encoder is not None:
            # (b,t,f)
            traj_representation = self.transformer_encoder(state_representation)
        else:
            state_representation = rearrange(state_representation, "b t f -> b (t f)")
            traj_representation = self.traj_net(state_representation)

        return traj_representation

    def get_state_representation(self, images, bbox):
        b, t, c, h, w = images.shape
        images = rearrange(images, "b t c h w -> (b t) c h w")
        out = self.state_net(images)  # (b*t,f)
        out = rearrange(out, "(b t) f-> b t f", b=b)
        return out


class SpatialSoftmaxCNN(nn.Module):
    def __init__(self, temperature) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.spatial_softmax = SpatialSoftmax(temperature=temperature)

    def forward(self, images, bbox):
        """_summary_

        Args:
            images (_type_): (b t c h w)
            bbox (_type_): _description_

        Returns:
            _type_: _description_
        """
        b, t, c, h, w = images.shape
        images = rearrange(images, "b t c h w -> (b t) c h w")
        out = self.net(images)
        out = self.spatial_softmax(out)

        out = rearrange(out, "(b t) f-> b (t f)", b=b)
        return out


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature, normalized_coordinates=True) -> None:
        super().__init__()
        self.normalized_coordinates = normalized_coordinates
        self.temperature = temperature

    def forward(self, x):
        """
        in: (batch x channel x Height x Width)
        out:(batch x 2*channel)
        """
        # create coordinates grid
        pos_y, pos_x = self.create_meshgrid(x, self.normalized_coordinates)

        pos_x = rearrange(pos_x, "h w ->(h w)")
        pos_y = rearrange(pos_y, "h w ->(h w)")

        batch, channel, _, _ = x.shape
        x = rearrange(x, "b c h w -> (b c)(h w)")
        softmax_attention = torch.softmax(x / self.temperature, dim=-1)

        expected_x = torch.sum(pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(pos_y * softmax_attention, dim=1, keepdim=True)

        output = torch.cat([expected_x, expected_y], dim=-1)
        output = output.reshape(batch, -1)
        assert output.shape == (batch, 2 * channel)

        return output

    def create_meshgrid(self, x, normalized_coordinates=True):
        _, _, height, width = x.shape
        _device, _dtype = x.device, x.dtype
        if normalized_coordinates:
            xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
            ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
        else:
            xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
            ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
        return torch.meshgrid(ys, xs)  # pos_y, pos_x


class ResnetConv(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        pretrained=None,
        no_training=False,
        remove_layer_num=1,
        img_c=3,
        no_stride=False,
        use_group_norm=False,
        feature_per_group=16,
    ):
        super().__init__()

        # assert remove_layer_num ==1
        layers = list(torchvision.models.resnet18(weights=pretrained).children())[
            :-remove_layer_num
        ]
        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1
        self.resnet18_embeddings = torch.nn.Sequential(*layers)

        if use_group_norm:
            replace_submodules(
                root_module=self.resnet18_embeddings,
                predicate=lambda x: isinstance(x, nn.BatchNorm1d)
                or isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // feature_per_group,
                    num_channels=x.num_features,
                ),
            )

        if no_training:
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        self.remove_layer_num = remove_layer_num
        if self.remove_layer_num == 1:
            self.encoder = nn.Linear(512, embedding_size)
        elif self.remove_layer_num == 2:
            # no pooling
            self.encoder = nn.Linear(8192, embedding_size)

    def forward(self, x):
        h = self.resnet18_embeddings(x)
        feats_flat = torch.flatten(h, 1)
        embs = self.encoder(feats_flat)
        return embs


class Conv3D(nn.Module):
    """
    - A 3D CNN with 11 layers.
    - Kernel size is kept 3 for all three dimensions - (time, H, W)
      except the first layer has kernel size of (3, 5, 5)
    - Time dimension is preserved with `padding=1` and `stride=1`, and is
      averaged at the end
    Arguments:
    - Input: a (batch_size, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(self, column_units):
        super(Conv3D, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(
                3,
                32,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                dilation=(1, 1, 1),
                padding=(1, 2, 2),
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(
                32,
                64,
                kernel_size=(3, 3, 3),
                stride=1,
                dilation=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                64,
                128,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                dilation=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(
                128,
                128,
                kernel_size=(3, 3, 3),
                stride=1,
                dilation=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                128,
                128,
                kernel_size=(3, 3, 3),
                stride=1,
                dilation=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                128,
                128,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                dilation=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout3d(p=0.2),
        )

        # self.block3 = nn.Sequential(
        #     nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout3d(p=0.2),
        # )
        # self.block4 = nn.Sequential(
        #     nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout3d(p=0.2),
        # )

        # self.block5 = nn.Sequential(
        #     nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        # get convolution column features

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)

        # averaging features in time dimension
        x = x.mean(-1).mean(-1).mean(-1)
        return x


class ObjectsCrops(nn.Module):
    def __init__(self, video_hw, output_size):
        """
        video_size: original boxes scale
        spatial_slots: output spatial size
        """
        super().__init__()
        self.aligned = True
        self.sampling_ratio = -1
        self.video_hw = video_hw
        self.output_size = output_size

    def prepare_outdim(self, outdim):
        if isinstance(outdim, (int, float)):
            outdim = [outdim, outdim]
        assert len(outdim) == 2
        return tuple(outdim)

    def forward(self, features, boxes):
        """
        boxes: [BS, O, 4]
        features: [BS, d, H=12, W=12] where d is channel dim
        """
        BS, d, H, W = features.shape
        Horig, Worig = self.video_hw
        O = boxes.size(1)
        spatial_scale = [H / Horig, W / Worig][0]
        ret = roi_align(
            features,
            list(boxes.float()),
            self.output_size,
            spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )  # [BS * T * O, d, H, W]
        ret = ret.reshape(BS, O, d, *self.output_size)  # [BS, O, d, H, W]
        return ret
