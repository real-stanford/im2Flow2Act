import colorsys

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def draw_point_tracking_sequence(
    image, sequence, draw_line=True, thickness=1, radius=3, add_alpha_channel=False
):
    # sequence: (num_points,T,3)
    frame = image.copy()

    def draw_point_flow(frame, tracking_data, color, draw_line):
        # tracking_data: (T,3)
        for i in range(len(tracking_data) - 1):
            visible = tracking_data[i][2]
            if visible == 1:
                start_point = (int(tracking_data[i][0]), int(tracking_data[i][1]))
                end_point = (int(tracking_data[i + 1][0]), int(tracking_data[i + 1][1]))
                if draw_line:
                    cv2.line(
                        frame,
                        start_point,
                        end_point,
                        color,
                        thickness=thickness,
                        lineType=16,
                    )  # Adjust the thickness hesre
                if i == len(tracking_data) - 2:
                    cv2.circle(frame, end_point, radius, color, -1, lineType=16)

    num_points = len(sequence)

    for i in range(num_points):
        color_map = cm.get_cmap("jet")
        color = np.array(color_map(i / max(1, float(num_points - 1)))[:3]) * 255
        color_alpha = 1
        hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        color = colorsys.hsv_to_rgb(hsv[0], hsv[1] * color_alpha, hsv[2])
        if add_alpha_channel:
            color = (color[0], color[1], color[2], 255.0)
        draw_point_flow(frame, sequence[i], color, draw_line)

    return frame


def draw_key_points(image, key_points, radius=3, add_alpha_channel=False):
    frame = image.copy()

    def draw_point(frame, point, color):
        cv2.circle(
            frame, (int(point[0]), int(point[1])), radius, color, -1, lineType=16
        )

    num_points = len(key_points)
    for i in range(num_points):
        color_map = cm.get_cmap("jet")
        color = np.array(color_map(i / max(1, float(num_points - 1)))[:3]) * 255
        color_alpha = 1
        hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        color = colorsys.hsv_to_rgb(hsv[0], hsv[1] * color_alpha, hsv[2])
        if add_alpha_channel:
            color = (color[0], color[1], color[2], 255.0)
        draw_point(frame, key_points[i], color)
    return frame


def viz_point_tracking_flow(
    frames,
    point_tracking_sequences,
    output_path,
    viz_key=[1],
    point_per_key=-1,
    viz_horizon=-1,
    draw_line=True,
    thickness=1,
    radius=3,
    add_alpha_channel=False,
):
    viz_points = []
    if isinstance(point_tracking_sequences, dict):
        for key, value in point_tracking_sequences.items():
            if key in viz_key:
                for _ in range(point_per_key):
                    viz_points.append(value[np.random.randint(len(value))])
        viz_points = np.array(viz_points)
    else:
        if point_per_key == -1:
            viz_points = point_tracking_sequences
        else:
            viz_points = point_tracking_sequences[
                np.random.randint(len(point_tracking_sequences), size=point_per_key)
            ]

    # print(viz_points.shape)

    episode_length = frames.shape[0]
    if viz_horizon == -1:
        viz_horizon = episode_length
    viz_gif = []
    for i in range(1, episode_length):
        frame = draw_point_tracking_sequence(
            frames[i],
            viz_points[:, max(0, i + 1 - viz_horizon) : i + 1],
            draw_line,
            thickness,
            radius,
            add_alpha_channel,
        )
        viz_gif.append(frame)
    if output_path is not None:
        imageio.mimsave(output_path, viz_gif, duration=5)
    else:
        return viz_gif


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
