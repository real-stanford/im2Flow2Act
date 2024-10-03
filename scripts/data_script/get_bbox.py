import os

import hydra
import numpy as np
import requests
import torch
import zarr
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from im2flow2act.common.imagecodecs_numcodecs import register_codecs
from im2flow2act.common.utility.viz import show_box

register_codecs()


def get_object_bbox(initial_frame, text, device):
    model_id = "IDEA-Research/grounding-dino-base"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    image = Image.fromarray(initial_frame)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )
    return (results[0]["boxes"].detach().cpu().numpy()[0]).tolist()


@hydra.main(
    version_base=None,
    config_path="../../config/data",
    config_name="get_bbox",
)
def main(cfg):
    buffer = zarr.open(cfg.buffer_path, mode="a")
    object = cfg.object
    if cfg.episode_start is None:
        n_epispde = len(buffer)
        episode_start = 0
        episode_end = n_epispde
    else:
        episode_start = cfg.episode_start
        episode_end = cfg.episode_end
    print("episode_start|episode_end", episode_start, episode_end)
    for i in tqdm(range(episode_start, episode_end)):
        # try:
        episode = buffer[f"episode_{i}"]
        initial_frame = episode["camera_0/rgb"][0]
        bboxes = []
        plt.imshow(initial_frame)
        for obj in object:
            bbox = get_object_bbox(initial_frame, obj, "cuda")
            show_box(bbox, plt.gca())
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        print("bboxes", bboxes, bboxes.shape)
        episode["bbox"] = zarr.array(bboxes)

        plt.savefig(os.path.join(cfg.buffer_path, f"episode_{i}_bbox.png"))
        plt.axis("off")
        plt.close()
        # except:
        #     print(f"Error in episode {i}")
        #     continue


if __name__ == "__main__":
    try:
        device = "cuda"
        model_id = "IDEA-Research/grounding-dino-base"

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)
        # Check for cats and remote controls
        text = "a cat. a remote control."

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )
    except:
        print("Test")
    main()
