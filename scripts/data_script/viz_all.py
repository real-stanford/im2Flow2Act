import os
import subprocess


def main():
    dev_path = os.getenv("DEV_PATH")
    viz_pathes = [
        f"{dev_path}/im2flow2act/data/simulated_play/rigid",
        f"{dev_path}/im2flow2act/data/simulated_play/articulated",
        f"{dev_path}/im2flow2act/data/simulated_play/deformable",
    ]
    print(viz_pathes)
    # viz_thresholds = [0]  # use it for viz_bbox for realworld dataset
    viz_thresholds = [-1] * len(viz_pathes)  # use it for viz_sam for sim dataset
    processes = []
    for v in viz_pathes:
        for thresh in viz_thresholds:
            process = subprocess.Popen(
                [
                    "python",
                    "viz_point_tracking.py",
                    "--viz_num",
                    str(20),
                    "--data_buffer_path",
                    v,  # Split into two separate strings
                    "--viz_threshold",
                    str(thresh),  # Also split, and convert thresh to string
                    "--viz_num_point",
                    str(-1),
                    # "--draw_line",
                    "--viz_offset",
                    str(0),
                    "--viz_save_path",
                    f"{dev_path}/im2flow2act/experiment/visual_pt",
                    "--viz_sam",
                    # "--viz_bbox",
                ],
            )
            processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
