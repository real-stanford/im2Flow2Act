import subprocess

import hydra
import omegaconf


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="flow_generation_data_pipeline",
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
        ###########################
        # f"python generate_all_sam_mask.py "
        # f"avaliable_gpu={avaliable_gpu} "
        # f"data_buffer_path={cfg.data_buffer_path}",
        #############################
        f"python generate_all_point_tracking.py "
        f"avaliable_gpu={avaliable_gpu} "
        f"num_points={cfg.num_points} "
        f"data_buffer_path={cfg.data_buffer_path} "
        f"sam_iterative=False "
        f"simulation_herustic_filter={simulation_herustic_filter} "
        f"real_herustic_filter_type={cfg.real_herustic_filter_type} "
        f"background_filter={cfg.background_filter} "
        f"from_bbox={cfg.from_bbox} "
        f"dbscan_bbox=False "
        f'"dbscan_additional_kwargs.dbscan_use_sam={cfg.dbscan_use_sam}" '
        f'"dbscan_additional_kwargs.dbscan_epsilon={cfg.dbscan_epsilon}" '
        f'"dbscan_additional_kwargs.dbscan_min_samples={cfg.dbscan_min_samples}" '
        f'"dbscan_additional_kwargs.dbscan_sam_closeness={cfg.dbscan_sam_closeness}" '
        f'"dbscan_additional_kwargs.dbscan_sam_area_thres={cfg.dbscan_sam_area_thres}" '
        f'"dbscan_additional_kwargs.dbscan_bbox_padding={cfg.dbscan_bbox_padding}"',
    ]

    # Execute each command
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
