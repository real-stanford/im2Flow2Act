import numpy as np
import timm
import torch
from einops import rearrange
from im2flow2act.common.utility.arr import stratified_random_sampling, uniform_sampling
from im2flow2act.common.utility.model import freeze_model
from torch import nn


class FlowDiffusionPolicy(nn.Module):
    def __init__(
        self,
        flow_encoder,
        state_encoder,
        plan_encoder,
        time_alignment_transformer,
        discriptor_projection,
        flow_proj_in,
        diffusion_policy,
        alignment_condition_on_prop=True,
        plan_encoder_condition_on_current_state=False,
        policy_condition_on_proprioception_proj=True,
        point_encoder=None,
        alignment_detach=True,
        proprioception_proj_in=None,
        vision_transformer_0_kwargs=None,
        sampling_method="stratified",
        sampling_frame=48,
        proprioception_predictor=None,
        freeze_vit=False,
        plan_condition_type="initial",
        target_condition_type="initial",
        target_plan_drop_prob=0.0,
    ) -> None:
        super().__init__()
        self.sampling_method = sampling_method
        self.sampling_frame = sampling_frame
        self.alignment_detach = alignment_detach
        self.freeze_vit = freeze_vit
        self.proprioception_predictor = proprioception_predictor
        if vision_transformer_0_kwargs is not None:
            self.vision_transformer_0 = timm.create_model(**vision_transformer_0_kwargs)
            if self.freeze_vit:
                freeze_model(self.vision_transformer_0)
        else:
            self.vision_transformer_0 = None
        self.flow_encoder = flow_encoder
        self.state_encoder = state_encoder
        self.point_encoder = point_encoder
        self.plan_encoder = plan_encoder
        self.time_alignment_transformer = time_alignment_transformer
        self.flow_proj_in = flow_proj_in
        self.proprioception_proj_in = proprioception_proj_in
        self.alignment_condition_on_prop = alignment_condition_on_prop
        self.plan_encoder_condition_on_current_state = (
            plan_encoder_condition_on_current_state
        )
        self.policy_condition_on_proprioception_proj = (
            policy_condition_on_proprioception_proj
        )
        self.prop_detach_in_alignment = (
            True if policy_condition_on_proprioception_proj else False
        )
        self.discriptor_projection = discriptor_projection
        self.diffusion_policy = diffusion_policy
        self.plan_condition_type = plan_condition_type
        self.target_condition_type = target_condition_type
        self.target_plan_drop = nn.Dropout(target_plan_drop_prob)

        assert self.plan_condition_type in ["initial", "current", "none"]
        assert self.target_condition_type in ["initial", "current", "none"]

    def forward(
        self,
        noisy_actions,
        timesteps,
        initial_frames,
        frames_0,
        frames_1,
        proprioception,
        flows,
        initial_flows,
        current_flows,
        target_flows,
        initial_point_cloud,
    ):
        # flows: BxTxNx3; frames: Bxobs_horzonx3xHxW; proprioception: Bxobs_horizonxk
        B, T, N, _ = flows.shape
        if self.sampling_method == "stratified":
            target_horizon = target_flows.shape[1]
            reference_arr = np.arange(target_horizon)
            _, sample_indices = stratified_random_sampling(
                reference_arr, self.sampling_frame, return_indices=True
            )
            target_flows = target_flows[:, sample_indices].clone()  # (B,T',N,3)
        elif self.sampling_method == "uniform":
            target_horizon = target_flows.shape[1]
            reference_arr = np.arange(target_horizon)
            _, sample_indices = uniform_sampling(
                reference_arr, self.sampling_frame, return_indices=True
            )
            target_flows = target_flows[:, sample_indices].clone()  # (B,T',N,3)
        else:
            target_flows = target_flows
        target_horizon = target_flows.shape[1]
        sample_flows = flows
        B, T, N, _ = sample_flows.shape
        _, _, H, W = initial_frames.shape
        initial_flows = initial_flows.long()  # (B,N,3)
        current_flows = current_flows.long()  # (B,N,3)
        if self.vision_transformer_0 is not None:
            initial_frame_features = self.vision_transformer_0(
                initial_frames
            )  # (B,num_patches*num_patches+1,D)
            initial_frames_patch_embedding = initial_frame_features[:, 1:]
            initial_frames_patch_embedding = rearrange(
                initial_frames_patch_embedding,
                "b (h w) d -> b d h w ",
                h=self.vision_transformer_0.patch_embed.grid_size[0],
            )  # (B,D,num_patches,num_patches)
            initial_frames_patch_embedding = nn.functional.interpolate(
                initial_frames_patch_embedding,
                size=initial_frames.shape[2:],
                mode="bicubic",
            )  # (B,D,H,W)
            initial_frames_patch_embedding = rearrange(
                initial_frames_patch_embedding, "b d h w -> b h w d"
            )
            # initial_points: BxNx2
            initial_discriptors = initial_frames_patch_embedding[
                torch.arange(B)[:, None],
                initial_flows[:, :, 1],
                initial_flows[:, :, 0],
            ]  # (B,N,D)
            # initial_points: BxNx2
            point_representation = self.point_encoder(initial_point_cloud)  # (B,N,k')
            discriptors = torch.cat(
                [initial_discriptors, point_representation], axis=-1
            )
        else:
            discriptors = initial_point_cloud
        # proj discriptor
        discriptors = self.discriptor_projection(discriptors)  # (B,N,K)

        current_flows = current_flows.float()
        current_flows[:, :, 0] = current_flows[:, :, 0] / H
        current_flows[:, :, 1] = current_flows[:, :, 1] / W
        initial_flows = initial_flows.float()
        initial_flows[:, :, 0] = initial_flows[:, :, 0] / H
        initial_flows[:, :, 1] = initial_flows[:, :, 1] / W

        # proj flow
        sample_flows = self.flow_proj_in(sample_flows)  # (B,T,N,K)
        current_flows = self.flow_proj_in(current_flows)  # (B,N,K)
        initial_flows = self.flow_proj_in(initial_flows)  # (B,N,K)
        # prepare input
        plan_repeated_initial_discriptors = discriptors.unsqueeze(1).expand(
            -1, T, -1, -1
        )  # (B,T,N,K)
        target_repeated_initial_discriptors = discriptors.unsqueeze(1).expand(
            -1, target_horizon, -1, -1
        )  # (B,T',N,K)
        target_repeated_initial_discriptors = rearrange(
            target_repeated_initial_discriptors, "B T N K -> (B T) N K"
        )
        # get initial flow embedding
        initial_flow_embedding = self.flow_encoder(
            discriptors, initial_flows
        )  # (B,N,K)
        # Get current flow embedding
        current_flow_embedding = self.flow_encoder(
            discriptors,
            current_flows,
        )  # (B,N,K)
        # get plan flow embedding
        flow_embedding = self.flow_encoder(
            plan_repeated_initial_discriptors, sample_flows
        )  # (B,T,N,K)
        flow_embedding = rearrange(flow_embedding, "B T N K -> (B T) N K", B=B)

        # cross-attend with the initial flow to extract motion
        current_state_embedding = self.state_encoder(
            current_flow_embedding, initial_flow_embedding
        )

        # each frame in plan cross-attend with the initial flow to extract per frame motion
        if self.plan_condition_type == "initial":
            repeated_initial_flow_embedding = initial_flow_embedding.unsqueeze(
                1
            ).expand(-1, T, -1, -1)
            repeated_initial_flow_embedding = rearrange(
                repeated_initial_flow_embedding, "B T N K -> (B T) N K"
            )
            plan_embedding = self.state_encoder(
                flow_embedding, repeated_initial_flow_embedding
            )  # (B*T,K)
        elif self.plan_condition_type == "current":
            repeated_current_flow_embedding = current_flow_embedding.unsqueeze(
                1
            ).expand(-1, T, -1, -1)
            repeated_current_flow_embedding = rearrange(
                repeated_current_flow_embedding, "B T N K -> (B T) N K"
            )
            plan_embedding = self.state_encoder(
                flow_embedding, repeated_current_flow_embedding
            )  # (B*T,K)
        elif self.plan_condition_type == "none":
            plan_embedding = self.state_encoder(flow_embedding)  # (B*T,K)
        else:
            raise NotImplementedError
        plan_embedding = rearrange(plan_embedding, "(B T) K -> B T K", B=B)

        if self.proprioception_proj_in is not None:
            proprioception_proj = self.proprioception_proj_in(
                proprioception[
                    :,
                    -1:,
                ]
            )
        if self.alignment_condition_on_prop:
            predict_plan, _ = self.time_alignment_transformer(
                torch.cat(
                    [
                        (
                            current_state_embedding.unsqueeze(1).clone().detach()
                            if self.alignment_detach
                            else current_state_embedding.unsqueeze(1)
                        ),
                        (
                            proprioception_proj.clone().detach()
                            if self.prop_detach_in_alignment
                            else proprioception_proj
                        ),
                        (
                            plan_embedding.clone().detach()
                            if self.alignment_detach
                            else plan_embedding
                        ),
                    ],
                    dim=1,
                ),
                return_cls_token=True,
            )
        else:
            predict_plan, _ = self.time_alignment_transformer(
                torch.cat(
                    [
                        (
                            current_state_embedding.unsqueeze(1).clone().detach()
                            if self.alignment_detach
                            else current_state_embedding.unsqueeze(1)
                        ),
                        (
                            plan_embedding.clone().detach()
                            if self.alignment_detach
                            else plan_embedding
                        ),
                    ],
                    dim=1,
                ),
                return_cls_token=True,
            )

        if self.proprioception_predictor is not None:
            proprioception_prediction = self.proprioception_predictor(
                current_state_embedding
            )

        if self.training:
            target_flows = self.flow_proj_in(target_flows)  # (B,T,N,K)
            target_flows = rearrange(target_flows, "B T N K -> (B T) N K")
            target_flow_embedding = self.flow_encoder(
                target_repeated_initial_discriptors, target_flows
            )  # (B*T,N,K)

            # Summarize N points current frame.
            if self.target_condition_type == "initial":
                target_repeated_initial_flow_embedding = (
                    initial_flow_embedding.unsqueeze(1).expand(
                        -1, target_horizon, -1, -1
                    )
                )
                target_repeated_initial_flow_embedding = rearrange(
                    target_repeated_initial_flow_embedding, "B T N K -> (B T) N K"
                )
                target_plan = self.state_encoder(
                    target_flow_embedding, target_repeated_initial_flow_embedding
                )
            elif self.target_condition_type == "current":
                target_repeated_current_flow_embedding = (
                    current_flow_embedding.unsqueeze(1).expand(
                        -1, target_horizon, -1, -1
                    )
                )
                target_repeated_current_flow_embedding = rearrange(
                    target_repeated_current_flow_embedding, "B T N K -> (B T) N K"
                )
                target_plan = self.state_encoder(
                    target_flow_embedding, target_repeated_current_flow_embedding
                )
            elif self.target_condition_type == "none":
                target_plan = self.state_encoder(target_flow_embedding)
            else:
                raise NotImplementedError
            target_plan = rearrange(target_plan, "(B T) K -> B T K", B=B)
            target_plan_condition = [target_plan]
            if self.plan_encoder_condition_on_current_state:
                target_plan_condition.insert(0, current_state_embedding.unsqueeze(1))
            target_plan_condition = torch.cat(target_plan_condition, dim=1)
            target_plan, _ = self.plan_encoder(
                target_plan_condition,
                return_cls_token=True,
            )
        else:
            target_plan = predict_plan

        condition = [
            current_state_embedding,
            self.target_plan_drop(target_plan),
            (
                proprioception_proj.flatten(start_dim=1)
                if self.proprioception_proj_in is not None
                and self.policy_condition_on_proprioception_proj
                else proprioception.flatten(start_dim=1)
            ),
        ]
        condition = torch.cat(condition, dim=1)
        prediction = self.diffusion_policy(
            noisy_actions, timesteps, global_cond=condition
        )
        return (
            prediction,
            predict_plan,
            target_plan.clone().detach(),
            (
                proprioception_prediction
                if self.proprioception_predictor is not None
                else None
            ),
        )
