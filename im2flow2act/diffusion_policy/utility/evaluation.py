import collections
import json
import os
import random

import cv2
import hydra
import numpy as np
import torch
import zarr
from tqdm import tqdm

from im2flow2act.common.pointcloud import meshwrite
from im2flow2act.common.utility.arr import (
    random_sampling,
    stratified_random_sampling,
    uniform_sampling,
)
from im2flow2act.diffusion_policy.dataloader.diffusion_bc_dataset import (
    normalize_data,
    unnormalize_data,
)
from im2flow2act.diffusion_policy.dataloader.diffusion_flow_bc_dataset import (
    DiffusionFlowBCCloseLoopDataset,
    process_image,
)
from im2flow2act.flow_generation.flow_generator import FlowGenerator
from im2flow2act.simulation_env.environment.wrapper.record import RecordWrapper
from im2flow2act.tapnet.online_point_tracking import (
    build,
    construct_initial_features_and_state,
    inference,
)
from im2flow2act.tapnet.utility.viz import viz_point_tracking_flow


def build_evaluation_environment(env_cfg, info, qpos, qvel):
    print(info)
    if "env" in info and info["env"] == "MultiPour":
        env_cfg.env._target_ = (
            "im2flow2act.simulation_env.environment.task.pouring.Pouring"
        )
        env = hydra.utils.instantiate(
            env_cfg.env,
        )
    elif "env" in info and info["env"] == "RealPick":
        env_cfg.env._target_ = (
            "im2flow2act.simulation_env.environment.task.pickNplace.PickNPlace"
        )
        env = hydra.utils.instantiate(
            env_cfg.env,
        )
    elif "env" in info and info["env"] == "RealOpen":
        env_cfg.env._target_ = "im2flow2act.simulation_env.environment.task.open.Open"
        env = hydra.utils.instantiate(
            env_cfg.env,
            drawer_pos=info["drawer_pos"],
            drawer_quat=info["drawer_quat"],
            drawer_type=int(info["drawer_type"]),
        )
    elif "env" in info and info["env"] == "RealDeformable":
        env_cfg.env._target_ = (
            "im2flow2act.simulation_env.environment.task.folding.Folding"
        )
        env = hydra.utils.instantiate(
            env_cfg.env,
            deformable_simulation=True,
        )
    elif "env" in info and info["env"] == "OpenAndPlace":
        env_cfg.env._target_ = (
            "im2flow2act.simulation_env.environment.task.open_and_place.OpenAndPlace"
        )
        env = hydra.utils.instantiate(
            env_cfg.env,
            drawer_pos=info["drawer_pos"],
            drawer_quat=info["drawer_quat"],
            drawer_type=int(info["drawer_type"]),
        )

    env = RecordWrapper(
        env=env,
        render_height=env_cfg.eval_render_res[0],
        render_width=env_cfg.eval_render_res[1],
        render_fps=env_cfg.eval_render_fps,
        camera_ids=env_cfg.eval_camera_ids,
        store_path=env_cfg.eval_store_path,
        save_mj_physics=False,
    )
    env.reset()
    print("loading buffer state...")
    env = load_env_state(env, qpos, qvel)
    if env_cfg.env._target_ == "im2flow2act.simulation_env.environment.task.open.Open":
        env.set_initial_handle_pos()  # to check success
    print("loaded buffer state")
    return env


def load_env_state(env, qpos, qvel):
    assert len(env.mj_physics.data.qpos) == len(qpos)
    assert len(env.mj_physics.data.qvel) == len(qvel)
    for i in range(len(qpos)):
        env.mj_physics.data.qpos[i] = qpos[i]
    for i in range(len(qvel)):
        env.mj_physics.data.qvel[i] = qvel[i]
    env.mj_physics.forward()
    return env


def process_env_visual_observation(visual_observation, resize_shape):
    visual_observation = cv2.resize(visual_observation, resize_shape)
    visual_observation = process_image(visual_observation)
    return visual_observation


def evaluate_flow_diffusion_policy(
    model,
    noise_scheduler,
    num_inference_steps,
    stats,
    num_samples,
    env_cfg,
    eval_dataset,
    data_buffers,
    result_save_path,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.eval()
    eval_dataset.stats = stats
    data_path_offset = eval_dataset.buffer.data_path_offset
    print(">> data_path_offset", data_path_offset)
    resize_shape = eval_dataset.camera_resize_shape
    action_dim = eval_dataset.action_dim
    obs_horizon = eval_dataset.obs_horizon
    action_horizon = eval_dataset.action_horizon
    pred_horizon = eval_dataset.pred_horizon
    point_tracking_data = eval_dataset.point_tracking_data
    moving_mask_data = eval_dataset.moving_mask_data
    robot_mask_data = eval_dataset.robot_mask_data
    episode_ends = eval_dataset.episode_ends
    online_predict_apply, online_init_apply, online_predict, online_init = build(
        num_points=eval_dataset.num_points,
        img_size=eval_dataset.point_tracking_img_size,
    )
    if result_save_path is not None:
        os.makedirs(result_save_path, exist_ok=True)
    print("Evaluation store at:", result_save_path)
    success_count_dict = {}
    for db, data_buffer in enumerate(data_buffers):
        buffer_result_save_path = os.path.join(result_save_path, f"buffer_{db}")
        os.makedirs(buffer_result_save_path, exist_ok=True)
        env_cfg.eval_store_path = buffer_result_save_path
        buffer_offset = data_path_offset[db]
        success_count = 0
        for i in tqdm(range(num_samples)):
            # this is the index in the dataloader
            # use this index to access the episode flow plan
            episode_index = buffer_offset + i
            episode_start_index = (
                episode_ends[episode_index - 1] if episode_index > 0 else 0
            )
            episode_length = episode_ends[episode_index] - episode_start_index
            # load and process flow
            episode_point_tracking = point_tracking_data[
                episode_index
            ].copy()  # (T,N,3)
            episode_moving_mask = moving_mask_data[episode_index].copy()  # (N,)
            episode_robot_mask = robot_mask_data[episode_index].copy()  # (N,)
            episode_point_tracking = eval_dataset.process_point_tracking_data(
                episode_point_tracking=episode_point_tracking,
                episode_moving_mask=episode_moving_mask,
                num_points_to_sample=eval_dataset.num_points,
                point_tracking_img_size=eval_dataset.point_tracking_img_size,
                robot_mask=episode_robot_mask,
                equal_sampling=eval_dataset.equal_sampling,
                object_sampling=eval_dataset.object_sampling,
                herustic_filter=eval_dataset.herustic_filter,
                ignore_robot_mask=eval_dataset.ignore_robot_mask,
            )
            initial_flow = episode_point_tracking[0].copy()

            query_points = np.array(
                [
                    [
                        0,
                        p[1] * eval_dataset.point_tracking_img_size[0],
                        p[0] * eval_dataset.point_tracking_img_size[0],
                    ]
                    for p in initial_flow
                ]
            )
            query_points = query_points.astype(np.float32)

            initial_flow[:, 0] *= eval_dataset.camera_resize_shape[0]
            initial_flow[:, 1] *= eval_dataset.camera_resize_shape[1]
            initial_flow = initial_flow.astype(np.int32)
            initial_flow = np.clip(
                initial_flow, a_min=0, a_max=eval_dataset.camera_resize_shape[0] - 1
            )
            initial_flow_depth = initial_flow.copy()
            initial_flow = torch.tensor(initial_flow).unsqueeze(0).cuda()
            eval_dataset.sampling_type = "uniform"
            if eval_dataset.sampling_type == "random":
                episode_flow_plan, episode_plan_indices = random_sampling(
                    episode_point_tracking[:episode_length],
                    eval_dataset.sample_frames,
                    zero_include=True,
                    return_indices=True,
                )
            elif eval_dataset.sampling_type == "uniform":
                episode_flow_plan, episode_plan_indices = uniform_sampling(
                    episode_point_tracking[:episode_length],
                    eval_dataset.sample_frames,
                    return_indices=True,
                )
            elif eval_dataset.sampling_type == "stratified":
                episode_flow_plan, episode_plan_indices = stratified_random_sampling(
                    episode_point_tracking[:episode_length],
                    eval_dataset.sample_frames,
                    return_indices=True,
                )
            elif eval_dataset.sampling_type == "complete":
                episode_flow_plan = episode_point_tracking
                episode_plan_indices = np.arange(episode_length).astype(int)

            episode_flow_plan = (
                torch.from_numpy(episode_flow_plan).unsqueeze(0).cuda()
            )  # (1,T,N,3)
            # load env state and build env
            info = data_buffer[f"episode_{i}"]["info"][0]
            qpos, qvel = (
                data_buffer[f"episode_{i}"]["qpos"][0],
                data_buffer[f"episode_{i}"]["qvel"][0],
            )
            env = build_evaluation_environment(env_cfg, info, qpos, qvel)
            proprioceptive_observation = env.get_proprioceptive_observation()
            visual_observation = env.get_visual_observation()
            # get depth
            initial_depth = env.render_depth(
                camera_ids=[eval_dataset.point_tracking_camera_id],
                height=env.camera_res[0],
                width=env.camera_res[1],
            )[eval_dataset.point_tracking_camera_id]
            if eval_dataset.depth_noisy_augmentation:
                initial_depth = eval_dataset.add_noise_to_depth(
                    depth_img=initial_depth,
                    gaussian_shifts=np.random.uniform(*eval_dataset.gaussian_shifts),
                    base_noise=np.random.uniform(*eval_dataset.base_noise),
                    std_noise=np.random.uniform(*eval_dataset.std_noise),
                )
            point_clouds_debug, color_pts_debug = eval_dataset.get_point_cloud(
                depth_img=initial_depth,
                color_img=visual_observation["camera_0"],
                cam_intr=eval_dataset.camera_intrinsic,
                cam_pose=eval_dataset.camera_pose_matrix,
                flow=None,
            )
            meshwrite(
                filename=os.path.join(
                    os.path.join(buffer_result_save_path, f"episode_{i}_debug_pts.ply")
                ),
                verts=point_clouds_debug,
                colors=color_pts_debug,
            )
            point_clouds, color_pts = eval_dataset.get_point_cloud(
                depth_img=initial_depth,
                color_img=cv2.resize(visual_observation["camera_0"], resize_shape),
                cam_intr=eval_dataset.camera_intrinsic,
                cam_pose=eval_dataset.camera_pose_matrix,
                flow=initial_flow_depth.copy(),
            )
            meshwrite(
                filename=os.path.join(
                    os.path.join(buffer_result_save_path, f"episode_{i}_vis_pts.ply")
                ),
                verts=point_clouds,
                colors=color_pts,
            )
            point_clouds = point_clouds.astype(np.float32)
            if eval_dataset.normalize_pointcloud:
                point_clouds = eval_dataset.normalize_point_cloud(point_clouds)
            point_clouds = torch.tensor(point_clouds).unsqueeze(0).cuda()

            img_obs_deque_0 = collections.deque(
                [
                    process_env_visual_observation(
                        visual_observation["camera_0"], resize_shape
                    )
                ]
                * obs_horizon,
                maxlen=obs_horizon,
            )
            # img_obs_deque_1 = collections.deque(
            #     [
            #         process_env_visual_observation(
            #             visual_observation["camera_6"], resize_shape
            #         )
            #     ]
            #     * obs_horizon,
            #     maxlen=obs_horizon,
            # )
            prop_deque = collections.deque(
                [
                    normalize_data(
                        proprioceptive_observation.reshape(1, -1),
                        stats=stats["proprioception"],
                    )
                ]
                * obs_horizon,
                maxlen=obs_horizon,
            )
            B = 1
            online_point_tracking = []
            point_tracking_viz_frames = []
            initial_frame = (
                process_env_visual_observation(
                    visual_observation["camera_0"], resize_shape
                )
                .unsqueeze(0)
                .cuda()
            )
            point_tracking_frame = cv2.resize(
                visual_observation["camera_0"], eval_dataset.point_tracking_img_size
            )
            query_features, causal_state = construct_initial_features_and_state(
                query_points=query_points,
                initial_frame=point_tracking_frame,
                online_init_apply=online_init_apply,
            )
            current_flow, causal_state = inference(
                query_features=query_features,
                causal_state=causal_state,
                current_frame=point_tracking_frame,
                online_predict_apply=online_predict_apply,
            )  # (N,1,3)
            current_flow = np.clip(
                current_flow, a_min=0, a_max=eval_dataset.point_tracking_img_size[0] - 1
            )
            online_point_tracking.append(current_flow.astype(np.int32))
            point_tracking_viz_frames.append(point_tracking_frame)
            current_flow[:, :, :2] = (
                current_flow[:, :, :2]
                / eval_dataset.point_tracking_img_size[0]
                * eval_dataset.camera_resize_shape[0]
            ).astype(np.int32)
            current_flow = np.clip(
                current_flow, a_min=0, a_max=eval_dataset.camera_resize_shape[0] - 1
            )
            # squeeze the time dim
            current_flow = (
                torch.tensor(current_flow[:, 0, :]).cuda().unsqueeze(0)
            )  # (1,N,3)
            target_flow = torch.zeros((1, eval_dataset.target_flow_horizon))
            for step in range(20):
                prop_seq = (
                    torch.from_numpy(np.concatenate(prop_deque, axis=0))
                    .unsqueeze(0)
                    .cuda()
                )
                visual_seq_0 = torch.stack(list(img_obs_deque_0)).unsqueeze(0).cuda()
                # visual_seq_1 = torch.stack(list(img_obs_deque_1)).unsqueeze(0).cuda()
                # print(prop_seq.shape, visual_seq_0.shape, visual_seq_1.shape)
                noise = torch.randn(B, pred_horizon, action_dim).cuda()
                input = noise
                noise_scheduler.set_timesteps(num_inference_steps)
                for t in noise_scheduler.timesteps:
                    with torch.no_grad():
                        model_out = model(
                            input,
                            t.unsqueeze(0).cuda(),
                            initial_frame,
                            visual_seq_0,
                            # visual_seq_1,
                            None,
                            prop_seq,
                            episode_flow_plan,
                            initial_flow,
                            current_flow,
                            target_flow,  # dummy flow
                            point_clouds,
                        )
                        noisy_residual = model_out[0]
                    previous_noisy_sample = noise_scheduler.step(
                        noisy_residual, t, input
                    ).prev_sample
                    input = previous_noisy_sample
                input = input.detach().to("cpu").numpy()
                # (B, pred_horizon, action_dim)
                # print(input.shape)
                naction = input[0]
                action_pred = unnormalize_data(naction, stats=stats["action"])
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end, :]
                action[:, 2] = np.clip(
                    action[:, 2],
                    a_min=env.action_z_lower_bound,
                    a_max=100,
                )
                for k in range(len(action)):
                    obs, reward, done, _, info = env.step(action[k])
                    img_obs_deque_0.append(
                        process_env_visual_observation(obs["camera_0"], resize_shape)
                    )
                    # img_obs_deque_1.append(
                    #     process_env_visual_observation(obs["camera_6"], resize_shape)
                    # )
                    prop_deque.append(
                        normalize_data(
                            obs["proprioceptive"].reshape(1, -1),
                            stats=stats["proprioception"],
                        )
                    )
                    point_tracking_frame = cv2.resize(
                        obs["camera_0"], eval_dataset.point_tracking_img_size
                    )
                    current_flow, causal_state = inference(
                        query_features=query_features,
                        causal_state=causal_state,
                        current_frame=point_tracking_frame,
                        online_predict_apply=online_predict_apply,
                    )
                    current_flow = np.clip(
                        current_flow,
                        a_min=0,
                        a_max=eval_dataset.point_tracking_img_size[0] - 1,
                    )
                    # logging
                    online_point_tracking.append(current_flow.astype(np.int32))
                    point_tracking_viz_frames.append(point_tracking_frame)
                    # next model inference input
                    current_flow[:, :, :2] = (
                        current_flow[:, :, :2]
                        / eval_dataset.point_tracking_img_size[0]
                        * eval_dataset.camera_resize_shape[0]
                    ).astype(np.int32)
                    current_flow = np.clip(
                        current_flow,
                        a_min=0,
                        a_max=eval_dataset.camera_resize_shape[0] - 1,
                    )
                    current_flow = (
                        torch.tensor(current_flow[:, 0, :]).cuda().unsqueeze(0)
                    )  # (1,N,3)
                if env.check_success():
                    print("task success!")
                    success_count += 1
                    break

            env.flush()
            online_point_tracking = np.concatenate(online_point_tracking, axis=1)
            point_tracking_viz_frames = np.array(point_tracking_viz_frames)
            viz_point_tracking_flow(
                point_tracking_viz_frames,
                online_point_tracking.copy(),
                point_per_key=len(online_point_tracking),
                output_path=os.path.join(
                    buffer_result_save_path, f"episode_{i}_online_tracking.gif"
                ),
            )
            np.save(
                os.path.join(
                    buffer_result_save_path,
                    f"episode_{i}_online_point_tracking_sequence.npy",
                ),
                online_point_tracking,
            )
        success_count_dict[f"buffer_{db}"] = success_count
        with open(
            os.path.join(buffer_result_save_path, "success_count.json"), "w"
        ) as f:
            json.dump({"success_count": success_count}, f)
        success_count_dict[f"buffer_{db}"] = success_count

    model.train()
    # dump the success count dict into json and save

    with open(os.path.join(result_save_path, "success_count.json"), "w") as f:
        json.dump(success_count_dict, f)


def evaluate_flow_diffusion_policy_from_generated_flow(
    model,
    noise_scheduler,
    num_inference_steps,
    stats,
    num_samples,
    env_cfg,
    data_dirs,
    num_points,
    point_tracking_img_size,
    resize_shape,
    action_dim,
    obs_horizon,
    action_horizon,
    target_flow_horizon,
    pred_horizon,
    point_tracking_camera_id,
    camera_intrinsic,
    camera_pose_matrix,
    normalize_pointcloud,
    result_save_path,
    flow_generator_additional_args,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.eval()
    online_predict_apply, online_init_apply, online_predict, online_init = build(
        num_points=num_points,
        img_size=point_tracking_img_size,
    )
    if result_save_path is not None:
        os.makedirs(result_save_path, exist_ok=True)
    print("Evaluation store at:", result_save_path)
    success_count_dict = {}
    for db, data_dir in enumerate(data_dirs):
        buffer_result_save_path = os.path.join(
            result_save_path, f"buffer_{os.path.basename(data_dir)}"
        )
        os.makedirs(buffer_result_save_path, exist_ok=True)
        env_cfg.eval_store_path = buffer_result_save_path
        flow_generator = FlowGenerator(
            buffer=data_dir,
            point_tracking_img_size=[256, 256],
            flow_from_sam=False,
            flow_from_bbox=False,
            flow_from_generated=True,
            generated_flow_buffer=data_dir,
            **flow_generator_additional_args,
        )
        success_count = 0
        data_buffer = zarr.open(data_dir, mode="r")
        for i in tqdm(range(num_samples)):
            # load env state and build env
            info = data_buffer[f"episode_{i}"]["info"][0]
            qpos, qvel = (
                data_buffer[f"episode_{i}"]["qpos"][0],
                data_buffer[f"episode_{i}"]["qvel"][0],
            )
            env = build_evaluation_environment(env_cfg, info, qpos, qvel)
            ####################### get flow generator here ######################
            # get depth
            initial_depth = env.render_depth(
                camera_ids=[point_tracking_camera_id],
                height=env.camera_res[0],
                width=env.camera_res[1],
            )[point_tracking_camera_id]
            proprioceptive_observation = env.get_proprioceptive_observation()
            visual_observation = env.get_visual_observation()
            inference_rgb = cv2.resize(visual_observation["camera_0"], resize_shape)
            episode_generated_flow = flow_generator.generate_flow(
                rgb=inference_rgb,
                depth=initial_depth,
                episode=i,
            )  # normalized version: (T, N, 3)
            initial_flow = episode_generated_flow[0].copy()
            flow_generator.generate_gif(
                visual_observation["camera_0"],
                episode_generated_flow,
                os.path.join(
                    buffer_result_save_path, f"episode_{i}_generated_flow.gif"
                ),
            )
            ####################### inference flow generator here #######################
            episode_flow_plan = (
                torch.from_numpy(episode_generated_flow).unsqueeze(0).cuda()
            )  # (1,T,N,3)

            query_points = np.array(
                [
                    [
                        0,
                        p[1] * point_tracking_img_size[0],
                        p[0] * point_tracking_img_size[0],
                    ]
                    for p in initial_flow
                ]
            )
            query_points = query_points.astype(np.float32)

            initial_flow[:, 0] *= resize_shape[0]
            initial_flow[:, 1] *= resize_shape[1]
            initial_flow = initial_flow.astype(np.int32)
            initial_flow = np.clip(initial_flow, a_min=0, a_max=resize_shape[0] - 1)
            initial_flow_depth = initial_flow.copy()
            initial_flow = torch.tensor(initial_flow).unsqueeze(0).cuda()
            point_clouds_debug, color_pts_debug = (
                DiffusionFlowBCCloseLoopDataset.get_point_cloud(
                    depth_img=initial_depth,
                    color_img=visual_observation["camera_0"],
                    cam_intr=camera_intrinsic,
                    cam_pose=camera_pose_matrix,
                    flow=None,
                )
            )
            meshwrite(
                filename=os.path.join(
                    os.path.join(buffer_result_save_path, f"episode_{i}_debug_pts.ply")
                ),
                verts=point_clouds_debug,
                colors=color_pts_debug,
            )
            point_clouds, color_pts = DiffusionFlowBCCloseLoopDataset.get_point_cloud(
                depth_img=initial_depth,
                color_img=cv2.resize(visual_observation["camera_0"], resize_shape),
                cam_intr=camera_intrinsic,
                cam_pose=camera_pose_matrix,
                flow=initial_flow_depth.copy(),
            )
            meshwrite(
                filename=os.path.join(
                    os.path.join(buffer_result_save_path, f"episode_{i}_vis_pts.ply")
                ),
                verts=point_clouds,
                colors=color_pts,
            )
            point_clouds = point_clouds.astype(np.float32)
            if normalize_pointcloud:
                point_clouds = normalize_data(point_clouds, stats["point_cloud"])
            point_clouds = torch.tensor(point_clouds).unsqueeze(0).cuda()
            img_obs_deque_0 = collections.deque(
                [
                    process_env_visual_observation(
                        visual_observation["camera_0"], resize_shape
                    )
                ]
                * obs_horizon,
                maxlen=obs_horizon,
            )
            prop_deque = collections.deque(
                [
                    normalize_data(
                        proprioceptive_observation.reshape(1, -1),
                        stats=stats["proprioception"],
                    )
                ]
                * obs_horizon,
                maxlen=obs_horizon,
            )
            B = 1
            online_point_tracking = []
            point_tracking_viz_frames = []
            initial_frame = (
                process_env_visual_observation(
                    visual_observation["camera_0"], resize_shape
                )
                .unsqueeze(0)
                .cuda()
            )
            point_tracking_frame = cv2.resize(
                visual_observation["camera_0"], point_tracking_img_size
            )
            query_features, causal_state = construct_initial_features_and_state(
                query_points=query_points,
                initial_frame=point_tracking_frame,
                online_init_apply=online_init_apply,
            )
            current_flow, causal_state = inference(
                query_features=query_features,
                causal_state=causal_state,
                current_frame=point_tracking_frame,
                online_predict_apply=online_predict_apply,
            )
            current_flow = np.clip(
                current_flow, a_min=0, a_max=point_tracking_img_size[0] - 1
            )
            online_point_tracking.append(current_flow.astype(np.int32))
            point_tracking_viz_frames.append(point_tracking_frame)
            current_flow[:, :, :2] = (
                current_flow[:, :, :2] / point_tracking_img_size[0] * resize_shape[0]
            ).astype(np.int32)
            current_flow = np.clip(current_flow, a_min=0, a_max=resize_shape[0] - 1)
            # squeeze the time dim
            current_flow = (
                torch.tensor(current_flow[:, 0, :]).cuda().unsqueeze(0)
            )  # (1,N,3)
            target_flow = torch.zeros((1, target_flow_horizon))
            for step in range(25):
                prop_seq = (
                    torch.from_numpy(np.concatenate(prop_deque, axis=0))
                    .unsqueeze(0)
                    .cuda()
                )
                visual_seq_0 = torch.stack(list(img_obs_deque_0)).unsqueeze(0).cuda()
                # visual_seq_1 = torch.stack(list(img_obs_deque_1)).unsqueeze(0).cuda()
                # print(prop_seq.shape, visual_seq_0.shape, visual_seq_1.shape)
                noise = torch.randn(B, pred_horizon, action_dim).cuda()
                input = noise
                noise_scheduler.set_timesteps(num_inference_steps)
                for t in noise_scheduler.timesteps:
                    with torch.no_grad():
                        model_out = model(
                            input,
                            t.unsqueeze(0).cuda(),
                            initial_frame,
                            visual_seq_0,
                            # visual_seq_1,
                            None,
                            prop_seq,
                            episode_flow_plan,
                            initial_flow,
                            current_flow,
                            target_flow,  # dummy flow
                            point_clouds,
                        )
                        noisy_residual = model_out[0]
                    previous_noisy_sample = noise_scheduler.step(
                        noisy_residual, t, input
                    ).prev_sample
                    input = previous_noisy_sample
                input = input.detach().to("cpu").numpy()
                # (B, pred_horizon, action_dim)
                # print(input.shape)
                naction = input[0]
                action_pred = unnormalize_data(naction, stats=stats["action"])
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end, :]
                action[:, 2] = np.clip(
                    action[:, 2],
                    a_min=env.action_z_lower_bound,
                    a_max=100,
                )
                for k in range(len(action)):
                    obs, reward, done, _, info = env.step(action[k])
                    img_obs_deque_0.append(
                        process_env_visual_observation(obs["camera_0"], resize_shape)
                    )
                    # img_obs_deque_1.append(
                    #     process_env_visual_observation(obs["camera_6"], resize_shape)
                    # )
                    prop_deque.append(
                        normalize_data(
                            obs["proprioceptive"].reshape(1, -1),
                            stats=stats["proprioception"],
                        )
                    )
                    point_tracking_frame = cv2.resize(
                        obs["camera_0"], point_tracking_img_size
                    )
                    current_flow, causal_state = inference(
                        query_features=query_features,
                        causal_state=causal_state,
                        current_frame=point_tracking_frame,
                        online_predict_apply=online_predict_apply,
                    )
                    current_flow = np.clip(
                        current_flow,
                        a_min=0,
                        a_max=point_tracking_img_size[0] - 1,
                    )
                    # logging
                    online_point_tracking.append(current_flow.astype(np.int32))
                    point_tracking_viz_frames.append(point_tracking_frame)
                    # next model inference input
                    current_flow[:, :, :2] = (
                        current_flow[:, :, :2]
                        / point_tracking_img_size[0]
                        * resize_shape[0]
                    ).astype(np.int32)
                    current_flow = np.clip(
                        current_flow,
                        a_min=0,
                        a_max=resize_shape[0] - 1,
                    )
                    current_flow = (
                        torch.tensor(current_flow[:, 0, :]).cuda().unsqueeze(0)
                    )  # (1,N,3)
                if env.check_success():
                    print("task success!")
                    success_count += 1
                    break

            env.flush()
            online_point_tracking = np.concatenate(online_point_tracking, axis=1)
            point_tracking_viz_frames = np.array(point_tracking_viz_frames)
            viz_point_tracking_flow(
                point_tracking_viz_frames,
                online_point_tracking,
                point_per_key=len(online_point_tracking),
                output_path=os.path.join(
                    buffer_result_save_path, f"episode_{i}_online_tracking.gif"
                ),
            )
            np.save(
                os.path.join(
                    buffer_result_save_path,
                    f"episode_{i}_online_point_tracking_sequence.npy",
                ),
                online_point_tracking,
            )
        with open(
            os.path.join(buffer_result_save_path, "success_count.json"), "w"
        ) as f:
            json.dump({"success_count": success_count}, f)
        success_count_dict[f"buffer_{os.path.basename(data_dir)}"] = success_count

    model.train()
    # dump the success count dict into json and save
    with open(os.path.join(result_save_path, "success_count.json"), "w") as f:
        json.dump(success_count_dict, f)


def replay_action(replay_offset, num_samples, env_cfg, data_buffer, result_save_path):
    env_cfg.eval_store_path = result_save_path
    for i in tqdm(range(replay_offset, replay_offset + num_samples)):
        info = data_buffer[f"episode_{i}"]["info"][0]
        qpos, qvel = (
            data_buffer[f"episode_{i}"]["qpos"][0],
            data_buffer[f"episode_{i}"]["qvel"][0],
        )
        env = build_evaluation_environment(env_cfg, info, qpos, qvel)
        episode_actions = data_buffer[f"episode_{i}"]["action"][:]
        action_len = len(episode_actions)
        for j in range(action_len):
            action = episode_actions[j]
            obs, reward, done, _, info = env.step(action)
        env.flush(store_eps=i)


def render_depth(num_samples, env_cfg, data_buffer):
    for i in tqdm(range(num_samples)):
        info = data_buffer[f"episode_{i}"]["info"][0]
        qpos, qvel = (
            data_buffer[f"episode_{i}"]["qpos"][0],
            data_buffer[f"episode_{i}"]["qvel"][0],
        )
        env = build_evaluation_environment(env_cfg, info, qpos, qvel)
        initial_depth = env.render_depth(
            camera_ids=[env_cfg.point_tracking_camera_id],
            height=env.camera_res[0],
            width=env.camera_res[1],
        )[env_cfg.point_tracking_camera_id]
        data_buffer[f"episode_{i}/initial_depth"] = zarr.array(initial_depth)
