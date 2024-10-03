import argparse
import os

import numpy as np
import zarr
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.tapnet.utility.utility import max_distance_moved
from im2flow2act.tapnet.utility.viz import viz_point_tracking_flow

register_codecs()


def get_embodiment_name(path):
    return os.path.basename(path)


def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--data_buffer_path", type=str)
    parser.add_argument("--viz_num", type=int, default=1)
    parser.add_argument("--viz_threshold", type=float, default=0)
    parser.add_argument("--viz_offset", type=int, default=0)
    parser.add_argument("--viz_num_point", type=int, default=-1)
    parser.add_argument("--viz_save_path", type=str)
    parser.add_argument("--draw_line", action="store_true")
    parser.add_argument("--viz_sam", action="store_true")
    parser.add_argument("--viz_bbox", action="store_true")

    # Parse the arguments
    args = parser.parse_args()
    data_buffer_path = args.data_buffer_path
    embodiment_name = get_embodiment_name(data_buffer_path)
    embodiment_save_path = os.path.join(
        parser.parse_args().viz_save_path, f"{embodiment_name}_{args.viz_threshold}"
    )
    os.makedirs(
        embodiment_save_path,
        exist_ok=True,
    )
    buffer = zarr.open(data_buffer_path)
    for i in tqdm(range(args.viz_num)):
        try:
            if args.viz_bbox:
                tracking_point_sequence = buffer[
                    f"episode_{i+args.viz_offset}/bbox_point_tracking_sequence"
                ][:]
            elif args.viz_sam:
                tracking_point_sequence = buffer[
                    f"episode_{i+args.viz_offset}/sam_point_tracking_sequence"
                ][:]
            else:
                tracking_point_sequence = buffer[
                    f"episode_{i+args.viz_offset}/point_tracking_sequence"
                ][:]
            frames = buffer[f"episode_{i+args.viz_offset}/rgb_arr"][:]
            if args.viz_threshold >= 0:
                moving_mask = (
                    max_distance_moved(tracking_point_sequence) > args.viz_threshold
                )
            else:
                moving_mask = (
                    buffer[f"episode_{i+args.viz_offset}/moving_mask"][:]
                    if not args.viz_sam
                    else buffer[f"episode_{i+args.viz_offset}/sam_moving_mask"][:]
                )
                # try:
            viz_seq = tracking_point_sequence[moving_mask]
            if args.viz_num_point != -1:
                viz_mask = np.round(
                    np.linspace(0, len(viz_seq) - 1, args.viz_num_point)
                ).astype(int)
                viz_seq = viz_seq[viz_mask]

            viz_point_tracking_flow(
                frames,
                viz_seq,
                os.path.join(
                    embodiment_save_path,
                    f"episode_{i+args.viz_offset}.gif",
                ),
                viz_key=[1],
                point_per_key=len(viz_seq),
                viz_horizon=-1,
                draw_line=args.draw_line,
            )
        except Exception as E:
            # Handle the exception here
            print(f"An error occurred: {E} at {embodiment_name}")


if __name__ == "__main__":
    main()
