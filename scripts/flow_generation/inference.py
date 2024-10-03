import os

import cv2
import hydra
import numpy as np
import zarr
from einops import repeat

from im2flow2act.flow_generation.AnimationFlowPipeline import AnimationFlowPipeline
from im2flow2act.flow_generation.inference import (
    inference,
    load_model,
)
from im2flow2act.tapnet.utility.utility import (
    create_uniform_grid_from_bbox,
)


def is_single_list(lst):
    return all(not isinstance(i, list) for i in lst)


@hydra.main(
    version_base=None,
    config_path="../../config/flow_generation",
    config_name="inference",
)
def inference_animateflow(cfg):
    vae, text_encoder, tokenizer, noise_scheduler, model = load_model(cfg)
    pipeline = AnimationFlowPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        model=model,
        scheduler=noise_scheduler,
    )
    evaluation_save_path = cfg.evaluation_save_path
    os.makedirs(evaluation_save_path, exist_ok=True)
    data = zarr.open(cfg.dataset_path, mode="a")
    for i in range(cfg.num_samples):
        episode = data[f"episode_{i}"]
        rgb = episode["camera_0/rgb"][30]
        camera_h, camera_w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (cfg.frame_resize_shape, cfg.frame_resize_shape))
        text = episode["info"][0]["task_description"]
        print("text condition", text)
        bboxes = episode["bbox"][:]
        if is_single_list(bboxes):
            bboxes = [bboxes]
        bboxes_area = []
        # get mutiple bbox area
        for bbox in bboxes:
            min_x, min_y, max_x, max_y = bbox
            area = (max_x - min_x) * (max_y - min_y)
            assert area > 0
            bboxes_area.append(area)
        bboxes_area = np.array(bboxes_area)
        # calculate sample number based on the area
        sample_proportion = bboxes_area / np.sum(bboxes_area)
        sample_num = np.round(sample_proportion * cfg.height * cfg.width).astype(int)
        print("splitting sample number", sample_num)
        sample_num_grid = np.sqrt(sample_num).astype(int)
        inital_keypoints = []
        for n_bbox, bbox in enumerate(bboxes):
            min_x, min_y, max_x, max_y = bbox
            min_x = min_x / camera_w * cfg.frame_resize_shape
            min_y = min_y / camera_h * cfg.frame_resize_shape
            max_x = max_x / camera_w * cfg.frame_resize_shape
            max_y = max_y / camera_h * cfg.frame_resize_shape
            bbox = [min_x, min_y, max_x, max_y]
            # bbox is under point tracking image size scale
            sampled_grid_points = create_uniform_grid_from_bbox(
                bbox, (sample_num_grid[n_bbox], sample_num_grid[n_bbox])
            )
            sampled_grid_points = np.clip(
                sampled_grid_points, a_min=0, a_max=cfg.frame_resize_shape - 1
            )
            # TODO: check here whether need to flip the x and y
            inital_keypoints.append(sampled_grid_points)

        inital_keypoints = np.concatenate(inital_keypoints, axis=0)
        if inital_keypoints.shape[0] < cfg.height * cfg.width:
            num_samples = cfg.height * cfg.width - inital_keypoints.shape[0]
            print("padding", num_samples)
            slack_inital_keypoints = repeat(
                inital_keypoints[0], "c -> s c", s=num_samples
            )
            print("slack_inital_keypoints", slack_inital_keypoints.shape)
            inital_keypoints = np.concatenate(
                [inital_keypoints, slack_inital_keypoints], axis=0
            )
            print("inital_keypoints", inital_keypoints.shape)
        else:
            print("skip padding")

        generated_flows = inference(
            pipeline=pipeline,
            global_image=rgb,
            point_uv=inital_keypoints,
            text=text,
            height=cfg.height,
            width=cfg.width,
            video_length=cfg.video_length,
            diff_flow=cfg.diff_flow,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            evaluation_save_path=evaluation_save_path,
            gif_name=f"episode_{i}.gif",
            viz_n_points=cfg.viz_n_points,
            draw_line=cfg.draw_line,
        )  # (N,T,3) [0,1] np
        episode["generated_flows"] = zarr.array(generated_flows)


if __name__ == "__main__":
    inference_animateflow()
