import os

import numpy as np
import torch
from einops import rearrange

from im2flow2act.common.utility.viz import convert_tensor_to_image, save_to_gif
from im2flow2act.flow_generation.inference import viz_generated_flow


def ae_flow_inference(cfg, vae, dataset, wandb_log=False, prefix=""):
    vae.eval()
    for i in range(cfg.num_samples):
        data = dataset[i]
        point_tracking_sequence = data["point_tracking_sequence"].cuda()  # (T,3,N1,N2)
        show_image = convert_tensor_to_image(data["global_image"])
        with torch.no_grad():
            recover_flows = []
            for step in range(len(point_tracking_sequence)):
                sample_flow_image = point_tracking_sequence[step]
                latents = vae.encode(
                    sample_flow_image.unsqueeze(0)
                ).latent_dist.sample()
                recover_flow_image = vae.decode(latents).sample
                recover_flows.append(recover_flow_image.detach().cpu().numpy())

        recover_flows = np.concatenate(
            recover_flows, axis=0
        )  # (n_sample_frames, 3, grid_size, grid_size)
        recover_flows = rearrange(recover_flows, "T C N1 N2 -> (N1 N2) T C")
        frames = viz_generated_flow(
            recover_flows,
            draw_line=False,
            initial_frame_uv=None,
            initial_frame=show_image,
            diff_flow=cfg.diff_flow,
            is_gt=True,
        )
        save_to_gif(
            frames,
            os.path.join(cfg.evaluation_save_path, f"sample_{i}_decoder_flow.gif"),
        )
        if wandb_log:
            import wandb

            frames = np.array(frames)
            frames = np.transpose(frames, [0, 3, 1, 2])
            wandb.log(
                {f"{prefix}/eval_video_{i}": wandb.Video(frames, fps=5, format="gif")}
            )
    vae.train()
