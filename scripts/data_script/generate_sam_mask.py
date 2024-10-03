import cv2
import hydra
import numpy as np
import zarr
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

from im2flow2act.common.imagecodecs_numcodecs import register_codecs

register_codecs()


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="generate_sam_mask",
)
def main(cfg):
    sam = sam_model_registry[cfg.model_type](checkpoint=cfg.sam_checkpoint)
    sam.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=cfg.points_per_side,
        pred_iou_thresh=cfg.pred_iou_thresh,
        stability_score_thresh=cfg.stability_score_thresh,
        crop_n_layers=cfg.crop_n_layers,
        crop_n_points_downscale_factor=cfg.crop_n_points_downscale_factor,
        min_mask_region_area=cfg.min_mask_region_area,
    )

    def get_sam_mask(buffer, index):
        try:
            initial_frame = buffer[f"episode_{index}/rgb_arr"][0]
            if cfg.resize_shape is not None:
                initial_frame = cv2.resize(initial_frame, cfg.resize_shape)
            masks = mask_generator.generate(initial_frame)
            sam_mask = np.zeros(cfg.resize_shape)
            sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
            for i, mask in enumerate(sorted_masks):
                seg = mask["segmentation"]
                sam_mask[seg == True] = i
            buffer[f"episode_{index}/sam_mask"] = sam_mask
            return True
        except Exception as e:
            print(e)
            return False

    buffer = zarr.open(cfg.data_buffer_path)
    for episode_index in tqdm(range(cfg.episode_start, cfg.episode_end)):
        get_sam_mask(buffer, episode_index)


if __name__ == "__main__":
    main()
