import os

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from im2flow2act.common.utility.arr import uniform_sampling
from im2flow2act.common.utility.model import load_config
from im2flow2act.common.utility.viz import convert_tensor_to_image, save_to_gif
from im2flow2act.flow_generation.animatediff.models.unet import UNet3DConditionModel
from im2flow2act.flow_generation.AnimateFlow import AnimateFlow
from im2flow2act.flow_generation.dataloader.animateflow_dataset import process_image
from im2flow2act.tapnet.utility.viz import draw_point_tracking_sequence


def load_model(cfg):
    model_cfg = load_config(cfg.model_path)

    noise_scheduler = DDIMScheduler(**model_cfg.noise_scheduler_kwargs)
    print(f"loading vae from {model_cfg.vae_pretrained_model_path}")
    vae = AutoencoderKL.from_pretrained(
        model_cfg.vae_pretrained_model_path, subfolder="vae"
    ).to("cuda")
    print(f"loading tokenizer and text_encoder from {model_cfg.pretrained_model_path}")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_cfg.pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_cfg.pretrained_model_path, subfolder="text_encoder"
    ).to("cuda")
    print(f"loading unet from {model_cfg.pretrained_model_path}")
    unet = UNet3DConditionModel.from_pretrained_2d(
        model_cfg.pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs=model_cfg.unet_additional_kwargs,
    )
    # load the model
    model = AnimateFlow(unet=unet, **model_cfg.animateflow_kwargs)
    model.load_model(
        os.path.join(
            cfg.model_path,
            "checkpoints",
            f"epoch_{cfg.model_ckpt}",
            f"epoch_{cfg.model_ckpt}.ckpt",
        )
    )
    model = model.to("cuda")
    model.eval()
    return vae, text_encoder, tokenizer, noise_scheduler, model


def viz_generated_flow(
    flows,
    initial_frame_uv,
    initial_frame,
    diff_flow=True,
    draw_line=True,
    viz_n_points=-1,
    is_gt=False,
):
    # flows: (N,T,3)
    frame_shape = initial_frame.shape[0]
    if is_gt:
        flows = np.clip(flows / 2 + 0.5, a_min=0, a_max=1)
    video_length = flows.shape[1]
    if diff_flow:
        for i in range(video_length):
            step_sequence = (
                initial_frame_uv + flows[:, i, :2] * (2 * frame_shape) - frame_shape
            )
            flows[:, i, :2] = step_sequence
    else:
        flows[:, :, :2] = flows[:, :, :2] * frame_shape
    flows = np.round(flows).astype(np.int32)
    flows = np.clip(flows, 0, frame_shape - 1)
    frames = []
    if viz_n_points == -1:
        viz_indicies = np.arange(flows.shape[0])
    else:
        _, viz_indicies = uniform_sampling(flows, viz_n_points, return_indices=True)
    for j in range(flows.shape[1]):
        frame = draw_point_tracking_sequence(
            initial_frame.copy(),
            flows[viz_indicies, :j],
            draw_line=draw_line,
        )
        frames.append(frame)
    return frames


def inference_from_dataset(
    pipeline,
    evalulation_datasets,
    evaluation_save_path,
    num_inference_steps,
    num_samples,
    guidance_scale,
    viz_n_points,
    draw_line=True,
    wandb_log=False,
):
    for k, evalulation_dataset in enumerate(evalulation_datasets):
        dataset_result_save_path = os.path.join(evaluation_save_path, f"dataset_{k}")
        video_length = evalulation_dataset.n_sample_frames
        os.makedirs(dataset_result_save_path, exist_ok=True)
        for i in range(num_samples):
            sample = evalulation_dataset[i]
            text = sample["text"]
            global_image = sample["global_image"].unsqueeze(0).cuda()
            point_uv = (
                torch.from_numpy(sample["first_frame_point_uv"]).unsqueeze(0).cuda()
            )
            gt_flow = sample["point_tracking_sequence"].unsqueeze(0).cuda()
            flows = pipeline(
                prompt=text,
                global_image=global_image,
                point_uv=point_uv,
                video_length=video_length,
                height=evalulation_dataset.grid_size,
                width=evalulation_dataset.grid_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="numpy",
            )  # (B,T,C,N1,N2) [0,1]
            flows = flows[0]  # (T,3,N,N)
            flows = rearrange(flows, "T C N1 N2 -> (N1 N2) T C")
            gt_flow = gt_flow[0].cpu().numpy()
            gt_flow = rearrange(gt_flow, "T C N1 N2 -> (N1 N2) T C")
            frames = viz_generated_flow(
                flows=gt_flow,
                initial_frame_uv=sample["first_frame_point_uv"],
                initial_frame=convert_tensor_to_image(global_image[0]),
                diff_flow=evalulation_dataset.diff_flow,
                draw_line=draw_line,
                viz_n_points=viz_n_points,
                is_gt=True,
            )
            save_to_gif(
                frames,
                os.path.join(dataset_result_save_path, f"gt_flow_{i}.gif"),
            )

            frames = viz_generated_flow(
                flows=flows,
                initial_frame_uv=sample["first_frame_point_uv"],
                initial_frame=convert_tensor_to_image(global_image[0]),
                diff_flow=evalulation_dataset.diff_flow,
                draw_line=draw_line,
                viz_n_points=viz_n_points,
            )
            save_to_gif(
                frames,
                os.path.join(dataset_result_save_path, f"generated_flow_{i}.gif"),
            )
            if wandb_log:
                import wandb

                git_name = f"eval/dataset_{k}/sample_{i}"
                frames = np.array(frames)
                frames = np.transpose(frames, [0, 3, 1, 2])
                wandb.log(
                    {git_name: wandb.Video(frames, fps=5, format="gif", caption=text)}
                )


def inference(
    pipeline,
    global_image,
    point_uv,
    text,
    height,
    width,
    video_length,
    diff_flow,
    num_inference_steps,
    guidance_scale,
    evaluation_save_path=None,
    gif_name=None,
    viz_n_points=-1,
    draw_line=True,
    wandb_log=False,
):
    point_uv = point_uv.astype(int)
    initial_flow = point_uv.copy()
    initial_flow = initial_flow / global_image.shape[0]
    initial_flow = np.concatenate(
        [initial_flow, np.ones((initial_flow.shape[0], 1))], axis=-1
    )
    initial_flow = initial_flow[:, None, :]  # (N1*N2,1,3)
    global_image = process_image(global_image).unsqueeze(0).cuda()
    point_uv_tensor = torch.from_numpy(point_uv).unsqueeze(0).cuda()
    flows = pipeline(
        prompt=text,
        global_image=global_image,
        point_uv=point_uv_tensor,
        video_length=video_length,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="numpy",
    )  # (B,T,C,N1,N2) [0,1]
    flows = flows[0]  # (T,3,N,N)
    flows = rearrange(flows, "T C N1 N2 -> (N1 N2) T C")
    print("final flows shape", flows.shape)
    if evaluation_save_path is not None and gif_name is not None:
        frames = viz_generated_flow(
            flows=flows.copy(),
            initial_frame_uv=point_uv,
            initial_frame=convert_tensor_to_image(global_image[0]),
            diff_flow=diff_flow,
            draw_line=draw_line,
            viz_n_points=viz_n_points,
        )
        save_to_gif(
            frames,
            os.path.join(evaluation_save_path, gif_name),
        )
        if wandb_log:
            import wandb

            frames = np.array(frames)
            frames = np.transpose(frames, [0, 3, 1, 2])
            wandb.log(
                {gif_name: wandb.Video(frames, fps=5, format="gif", caption=text)}
            )

    return flows  # (N,T,3)
