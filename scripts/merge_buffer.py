import os
import shutil

import zarr
from tqdm import tqdm

store_pathes = []

merge_path = ""
root = zarr.open(
    merge_path,
    mode="a",
)
episode_counter = 0
for store_path in tqdm(store_pathes, desc="merging store pathes"):
    episodes = [
        d
        for d in os.listdir(store_path)
        if os.path.isdir(os.path.join(store_path, d)) and d.startswith("episode")
    ]
    episodes = sorted(episodes, key=lambda x: int(x.split("_")[1]))

    for episode in tqdm(episodes, desc="merging episodes"):
        episode_path = os.path.join(store_path, episode)
        destination_path = os.path.join(merge_path, f"episode_{episode_counter}")
        episode_counter += 1
        shutil.copytree(episode_path, destination_path)
