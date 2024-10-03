import subprocess

import hydra
import omegaconf


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="data_pipeline",
)
def main(cfg: omegaconf.DictConfig):
    avaliable_gpu = f"'{cfg.avaliable_gpu}'"
    simulation_herustic_filter = f"'{cfg.simulation_herustic_filter}'"

    # Define the commands
    commands = [
        f"python convert_all_simulation_dataset.py "
        f"data_buffer_path={cfg.data_buffer_path} "
        f"store_path={cfg.store_path} "
        f"dataset={cfg.dataset} "
        f"downsample_ratio={cfg.downsample_ratio} "
        f"n_sample_frame={cfg.n_sample_frame}",
        ###########################################
        f"python generate_all_point_tracking.py "
        f"avaliable_gpu={avaliable_gpu} "
        f"data_buffer_path={cfg.data_buffer_path} "
        f"num_points={cfg.num_points} "
        f"sam_iterative=False "
        f"dbscan_bbox=False",
        ###########################################
        f"python point_selection.py "
        f'data_buffer_pathes="[{cfg.data_buffer_path}]" '
        f"moving_threshold_args.is_sam=False "
        f"moving_threshold_args.point_move_thresholds=[{cfg.point_move_thresholds}]",
        # ##########################################
        f"python generate_all_sam_mask.py "
        f"avaliable_gpu={avaliable_gpu} "
        f"data_buffer_path={cfg.data_buffer_path}",
        # ###########################################
        f"python generate_all_point_tracking.py "
        f"avaliable_gpu={avaliable_gpu} "
        f"data_buffer_path={cfg.data_buffer_path} "
        f"sam_iterative=True "
        f'"sam_iterative_additional_kwargs.from_grid={cfg.from_grid}" '
        f"from_bbox={cfg.from_bbox} "
        f"simulation_herustic_filter={simulation_herustic_filter}",
        ##########################################
        f"python point_selection.py "
        f'data_buffer_pathes="[{cfg.data_buffer_path}]" '
        f"moving_threshold_args.is_sam=True "
        f"moving_threshold_args.point_move_thresholds=[{cfg.point_move_thresholds}] "
        f"moving_threshold_args.simulation_herustic_filter={simulation_herustic_filter}",
        ###########################################
        f"python point_annotation.py "
        f'data_buffer_pathes="[{cfg.data_buffer_path}]" '
        f'mask="robot_mask" '
        f'"robot_mask_args.zero_robot_mask={cfg.zero_robot_mask}" '
        f"robot_mask_args.is_sam=True "
        f'"robot_mask_args.simulation_herustic={cfg.simulation_herustic}" '
        f'"robot_mask_args.simulation_herustic_patial={cfg.simulation_herustic_patial}"',
    ]

    # Execute each command
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
