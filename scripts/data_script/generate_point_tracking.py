import hydra
import omegaconf

from im2flow2act.tapnet.realworld_point_tracking import generate_point_tracking_sequence


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="generate_point_tracking",
)
def main(cfg: omegaconf.DictConfig):
    generate_point_tracking_sequence(
        cfg.data_buffer_path,
        cfg.episode_start,
        cfg.episode_end,
        cfg.num_points,
        cfg.sam_iterative,
        cfg.dbscan_bbox,
        cfg.simulation_herustic_filter,
        cfg.background_filter,
        cfg.from_bbox,
        **cfg.dbscan_additional_kwargs,
        **cfg.sam_iterative_additional_kwargs,
    )


if __name__ == "__main__":
    main()
