import cv2
import numpy as np
import zarr
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.common.utility.arr import uniform_sampling
from im2flow2act.tapnet.tap import inference
from im2flow2act.tapnet.utility.utility import (
    calculate_min_bbox,
    create_uniform_grid,
    create_uniform_grid_from_bbox,
)

register_codecs()


def generate_point_tracking_sequence(
    data_buffer_path,
    episode_start,
    episode_end,
    num_points,
    sam_iterative=False,
    dbscan_bbox=False,
    simulation_herustic_filter=False,
    background_filter=False,
    from_bbox=False,
    **kwargs,
):
    data_buffer = zarr.open(data_buffer_path, mode="a")
    print(
        "Generating point tracking sequence from episode {} to {}".format(
            episode_start, episode_end
        )
    )
    print(f"sam_iterative:{sam_iterative}|dbscan_bbox:{dbscan_bbox}")
    for episode_idx in tqdm(
        range(episode_start, episode_end), desc="Generating point tracking sequence"
    ):
        frames = data_buffer[f"episode_{episode_idx}/rgb_arr"][:]
        if sam_iterative:
            from_grid = kwargs.get("from_grid", False)
            sam_area_thres = kwargs.get("sam_area_thres", 20)
            sam_new_point_num = kwargs.get("sam_new_point_num", 1024)
            sam_closeness = kwargs.get("sam_closeness", 5)
            point_tracking_sequence = data_buffer[
                f"episode_{episode_idx}/point_tracking_sequence"
            ][:]
            moving_mask = data_buffer[f"episode_{episode_idx}/moving_mask"][:]
            if from_grid:
                if len(point_tracking_sequence) == sam_new_point_num:
                    data_buffer[
                        f"episode_{episode_idx}/sam_point_tracking_sequence"
                    ] = point_tracking_sequence
                elif len(point_tracking_sequence) > sam_new_point_num:
                    sampled_point_tracking_sequence = uniform_sampling(
                        point_tracking_sequence.copy(), sam_new_point_num
                    )
                    data_buffer[
                        f"episode_{episode_idx}/sam_point_tracking_sequence"
                    ] = sampled_point_tracking_sequence
                else:
                    raise ValueError(
                        "The number of points in the point tracking sequence is less than the desired sam points"
                    )
            else:
                sam_mask = data_buffer[f"episode_{episode_idx}/sam_mask"][:]
                try:
                    if "left" in simulation_herustic_filter:
                        # sim2real camera
                        condition = (point_tracking_sequence[:, 0, 0] < 40) & (
                            point_tracking_sequence[:, 0, 1] < 100
                        )
                        moving_mask[condition] = False
                    if "right" in simulation_herustic_filter:
                        condition = (point_tracking_sequence[:, 0, 0] > 110) & (
                            point_tracking_sequence[:, 0, 1] < 100
                        )
                        moving_mask[condition] = False

                    moving_point = point_tracking_sequence[moving_mask]
                    initial_flow = moving_point[:, 0, :2].astype(int)
                    unique_mask = np.unique(sam_mask)
                    sam_filter_points = []
                    new_sampling_points = []
                    new_sampling_area = []
                    for um in unique_mask:
                        area = np.where(sam_mask == um)
                        area_set = np.array(list(zip(area[1], area[0])))
                        in_area_points = []
                        for point in initial_flow:
                            if (
                                np.isclose(point, area_set, atol=sam_closeness)
                                .all(axis=1)
                                .any()
                            ):
                                in_area_points.append(point)
                        if len(in_area_points) > 1 and len(area_set) < sam_area_thres:
                            sam_filter_points.append(in_area_points)
                            new_sampling_area.append(area_set)
                    each_area_count = np.array([len(p) for p in sam_filter_points])
                    each_area_new_point = (
                        each_area_count / each_area_count.sum() * sam_new_point_num
                    ).astype(int)
                    for k, area in enumerate(new_sampling_area):
                        if k == len(new_sampling_area) - 1:
                            current_new_point_num = sum(
                                [len(p) for p in new_sampling_points]
                            )
                            sampling_index = np.linspace(
                                0,
                                len(area) - 1,
                                sam_new_point_num - current_new_point_num,
                            ).astype(int)
                        else:
                            sampling_index = np.linspace(
                                0, len(area) - 1, each_area_new_point[k]
                            ).astype(int)
                        new_sampling_points.append(area[sampling_index])
                    sam_filter_points = np.concatenate(sam_filter_points)
                    new_sampling_points = np.concatenate(new_sampling_points)
                    grid = new_sampling_points[:, [1, 0]]
                    tracking_point = np.concatenate(
                        [np.zeros([grid.shape[0], 1]), grid], axis=1
                    )
                    tracks, visibles = inference(frames, tracking_point)
                    tracking_point_sequence = np.concatenate(
                        [tracks, np.expand_dims(visibles, axis=-1).astype(np.float32)],
                        axis=-1,
                    )
                    data_buffer[
                        f"episode_{episode_idx}/sam_point_tracking_sequence"
                    ] = tracking_point_sequence
                except Exception as e:
                    print(e)
                    print(f"epsidoe {episode_idx} failed")
                    data_buffer[
                        f"episode_{episode_idx}/sam_point_tracking_sequence"
                    ] = point_tracking_sequence
        elif dbscan_bbox:
            dbscan_epsilon = kwargs.get("dbscan_epsilon", 20)
            dbscan_min_samples = kwargs.get("dbscan_min_samples", 5)
            dbscan_use_sam = kwargs.get("dbscan_use_sam", True)
            sam_area_thres = kwargs.get("dbscan_sam_area_thres", 2000)
            sam_closeness = kwargs.get("dbscan_sam_closeness", 5)
            dbscan_bbox_padding = kwargs.get("dbscan_bbox_padding", 0)
            print("dbscan_use_sam:", dbscan_use_sam)
            if dbscan_use_sam:
                point_tracking_sequence = data_buffer[
                    f"episode_{episode_idx}/sam_point_tracking_sequence"
                ][:]
                moving_mask = data_buffer[f"episode_{episode_idx}/sam_moving_mask"][:]
            else:
                point_tracking_sequence = data_buffer[
                    f"episode_{episode_idx}/point_tracking_sequence"
                ][:]
                moving_mask = data_buffer[f"episode_{episode_idx}/moving_mask"][:]
            robot_mask = data_buffer[f"episode_{episode_idx}/robot_mask"][:]
            sam_mask = data_buffer[f"episode_{episode_idx}/sam_mask"][:]
            if "left" in simulation_herustic_filter:
                # sim2real camera
                condition = (point_tracking_sequence[:, 0, 0] < 40) & (
                    point_tracking_sequence[:, 0, 1] < 100
                )
                moving_mask[condition] = False
            if "right" in simulation_herustic_filter:
                condition = (point_tracking_sequence[:, 0, 0] > 110) & (
                    point_tracking_sequence[:, 0, 1] < 100
                )
                moving_mask[condition] = False

            moving_points = point_tracking_sequence[moving_mask & ~robot_mask]
            if background_filter:
                initial_flow = moving_points[:, 0, :2].astype(int)
                unique_mask = np.unique(sam_mask)
                object_point_indices = []
                for um in unique_mask:
                    area = np.where(sam_mask == um)
                    area_set = np.array(list(zip(area[1], area[0])))
                    for k, p in enumerate(initial_flow):
                        if (
                            np.isclose(p, area_set, atol=sam_closeness)
                            .all(axis=1)
                            .any()
                        ):
                            if len(area_set) < sam_area_thres:
                                object_point_indices.append(k)
                moving_points = moving_points[object_point_indices]
            try:
                min_bbox, _, _ = calculate_min_bbox(
                    moving_points[:, 0, :2],
                    epsilon=dbscan_epsilon,
                    min_samples=dbscan_min_samples,
                    padding=dbscan_bbox_padding,
                )
                grid = create_uniform_grid_from_bbox(min_bbox, (num_points, num_points))
                # grid = grid[:, [1, 0]]
                tracking_point = np.concatenate(
                    [np.zeros([grid.shape[0], 1]), grid[:, [1, 0]]], axis=1
                )
                tracks, visibles = inference(frames, tracking_point)
                tracking_point_sequence = np.concatenate(
                    [tracks, np.expand_dims(visibles, axis=-1).astype(np.float32)],
                    axis=-1,
                )
                # use sam mask to get precise object points and background points label
                unique_mask = np.unique(sam_mask)
                object_points = []
                for um in unique_mask:
                    area = np.where(sam_mask == um)
                    area_set = np.array(list(zip(area[1], area[0])))
                    in_area_points = []
                    for point in grid:
                        if (
                            np.isclose(point, area_set, atol=sam_closeness)
                            .all(axis=1)
                            .any()
                        ):
                            in_area_points.append(point.tolist())
                    if len(in_area_points) > 1 and len(area_set) < sam_area_thres:
                        object_points.extend(in_area_points)
                object_mask = [
                    True if p.tolist() in object_points else False for p in grid
                ]
                # save
                data_buffer[f"episode_{episode_idx}/bbox_point_tracking_sequence"] = (
                    tracking_point_sequence
                )
                # # also save as sam
                # data_buffer[f"episode_{episode_idx}/sam_point_tracking_sequence"] = (
                #     tracking_point_sequence
                # )
                data_buffer[f"episode_{episode_idx}/bbox_object_mask"] = object_mask
                data_buffer[f"episode_{episode_idx}/bbox"] = min_bbox

            except Exception:
                print(f"epsidoe {episode_idx} failed")
                # exit()
                # data_buffer[f"episode_{episode_idx}/bbox_point_tracking_sequence"] = (
                #     point_tracking_sequence
                # )
        elif from_bbox:
            rgb = data_buffer[f"episode_{episode_idx}/camera_0/rgb"][0]
            camera_h, camera_w = rgb.shape[:2]
            frame_resize_shape = frames.shape[1]
            rgb = cv2.resize(rgb, (frame_resize_shape, frame_resize_shape))
            bboxes = data_buffer[f"episode_{episode_idx}/bbox"][:]
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
            sample_num = np.round(sample_proportion * num_points * num_points).astype(
                int
            )
            sample_num_grid = np.sqrt(sample_num).astype(int)
            tracking_point = []
            for n_bbox, bbox in enumerate(bboxes):
                min_x, min_y, max_x, max_y = bbox
                min_x = min_x / camera_w * frame_resize_shape
                min_y = min_y / camera_h * frame_resize_shape
                max_x = max_x / camera_w * frame_resize_shape
                max_y = max_y / camera_h * frame_resize_shape
                bbox = [min_x, min_y, max_x, max_y]
                grid = create_uniform_grid_from_bbox(
                    bbox, (sample_num_grid[n_bbox], sample_num_grid[n_bbox])
                )
                tracking_point.append(
                    np.concatenate(
                        [np.zeros([grid.shape[0], 1]), grid[:, [1, 0]]], axis=1
                    )
                )
            tracking_point = np.concatenate(tracking_point, axis=0)
            tracks, visibles = inference(frames, tracking_point)
            tracking_point_sequence = np.concatenate(
                [tracks, np.expand_dims(visibles, axis=-1).astype(np.float32)],
                axis=-1,
            )
            # save
            # also save as sam
            # data_buffer[f"episode_{episode_idx}/sam_point_tracking_sequence"] = (
            #     tracking_point_sequence
            # )
            data_buffer[f"episode_{episode_idx}/bbox_point_tracking_sequence"] = (
                tracking_point_sequence
            )
        else:
            grid = create_uniform_grid(
                h=frames.shape[1],
                w=frames.shape[2],
                h_points=num_points,
                w_points=num_points,
            )
            tracking_point = np.concatenate(
                [np.zeros([grid.shape[0], 1]), grid], axis=1
            )
            tracks, visibles = inference(frames, tracking_point)
            tracking_point_sequence = np.concatenate(
                [tracks, np.expand_dims(visibles, axis=-1).astype(np.float32)],
                axis=-1,
            )
            data_buffer[f"episode_{episode_idx}/point_tracking_sequence"] = (
                tracking_point_sequence
            )


def track_from_frames(frames, initial_points, **kwargs):
    grid = initial_points[:, [1, 0]]
    tracking_point = np.concatenate([np.zeros([grid.shape[0], 1]), grid], axis=1)
    tracks, visibles = inference(frames, tracking_point)
    tracking_point_sequence = np.concatenate(
        [tracks, np.expand_dims(visibles, axis=-1).astype(np.float32)],
        axis=-1,
    )
    return tracking_point_sequence
