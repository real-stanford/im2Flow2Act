import concurrent.futures
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import imageio
import numcodecs
import numpy as np
import zarr

from im2flow2act.common.imagecodecs_numcodecs import (
    JpegXl,
    register_codecs,
)

register_codecs()


@dataclass
class VideoData:
    rgb: any
    depth: Optional[any] = None
    segmentation: Optional[any] = None
    camera_id: Optional[int] = None

    @property
    def length(self) -> int:
        return len(self.rgb)

    @classmethod
    def stack(cls, video_data_list: List["VideoData"]) -> "VideoData":
        # Concatenate rgb
        stacked_rgb = np.stack([video_data.rgb for video_data in video_data_list])

        # Concatenate depth
        if all(video_data.depth is not None for video_data in video_data_list):
            stacked_depth = np.stack(
                [video_data.depth for video_data in video_data_list]
            )
        else:
            stacked_depth = None

        # Concatenate segmentation
        if all(video_data.segmentation is not None for video_data in video_data_list):
            stacked_segmentation = np.stack(
                [video_data.segmentation for video_data in video_data_list]
            )
        else:
            stacked_segmentation = None

        camera_id = video_data_list[0].camera_id

        return cls(
            rgb=stacked_rgb,
            depth=stacked_depth,
            segmentation=stacked_segmentation,
            camera_id=camera_id,
        )

    def to_mp4(self, path: str, fps: int = 30):
        imageio.mimwrite(path, self.rgb, fps=fps)


@dataclass
class EpisodeData:
    camera_datas: List[VideoData]
    action: Optional[any] = None
    proprioception: Optional[any] = None
    low_dim_state: Optional[any] = None
    qpos: Optional[any] = None
    qvel: Optional[any] = None
    info: Optional[any] = None

    @property
    def length(self) -> int:
        return len(self.action)


class EpisodeDataBuffer:
    def __init__(self, store_path, camera_ids, max_workers=32, mode="a") -> None:
        self.store_path = store_path
        self.root = zarr.open(self.store_path, mode=mode)
        self.camera_ids = camera_ids
        self.curr_eps = self.find_max_eps(self.root) + 1
        self.max_workers = max_workers

        print("current eps", self.curr_eps)

    def reset(self):
        keys = list(self.root.group_keys())
        for key in keys:
            del self.root[key]

    def find_max_eps(self, root):
        keys = list(root.group_keys())
        # print(keys)
        if len(keys) == 0:
            return -1
        else:
            return max([int(re.findall(r"\d+", key)[0]) for key in keys])

    def append(
        self,
        visual_obervations: Dict[int, VideoData],
        action: Optional[any] = None,
        proprioception: Optional[any] = None,
        low_dim_state: Optional[any] = None,
        qpos: Optional[any] = None,
        qvel: Optional[any] = None,
        info: Optional[any] = None,
        save_video=True,
        render_fps=30,
        store_eps=None,
    ):
        episode_index = self.curr_eps if store_eps is None else store_eps
        episode_data = self.root.create_group(f"episode_{episode_index}")
        for camera_id in self.camera_ids:
            camera_data = episode_data.create_group(f"camera_{camera_id}")
            this_compressor = JpegXl(level=80, numthreads=1)
            n, h, w, c = visual_obervations[camera_id].rgb.shape
            rgb_arr = camera_data.require_dataset(
                "rgb",
                shape=(n, h, w, c),
                chunks=(1, h, w, c),
                dtype=np.uint8,
                compressor=this_compressor,
            )

            def img_copy(zarr_arr, zarr_idx, np_array, np_idx):
                # print(zarr_arr.shape, np_array.shape, zarr_arr.dtype,
                #       np_array.dtype, zarr_idx, np_idx)
                try:
                    zarr_arr[zarr_idx] = np_array[np_idx]
                    # make sure we can successfully decode
                    _ = zarr_arr[zarr_idx]
                    return True
                except Exception as e:
                    print(e)
                    return False

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = set()
                for i in range(n):
                    futures.add(
                        executor.submit(
                            img_copy, rgb_arr, i, visual_obervations[camera_id].rgb, i
                        )
                    )

                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")

            if visual_obervations[camera_id].depth is not None:
                camera_data["depth"] = zarr.array(visual_obervations[camera_id].depth)

            if visual_obervations[camera_id].segmentation is not None:
                camera_data["segmentation"] = zarr.array(
                    visual_obervations[camera_id].segmentation
                )

            else:
                # camera_data["segmentation"] = None
                pass

            if save_video:
                visual_obervations[camera_id].to_mp4(
                    f"{self.store_path}/episode_{episode_index}/camera_{camera_id}.mp4",
                    fps=render_fps,
                )

        if action is not None:
            episode_data["action"] = zarr.array(action)

        if proprioception is not None:
            episode_data["proprioception"] = zarr.array(proprioception)

        if low_dim_state is not None:
            episode_data["low_dim_state"] = zarr.array(low_dim_state)

        if qpos is not None:
            episode_data["qpos"] = zarr.array(qpos)

        if qvel is not None:
            episode_data["qvel"] = zarr.array(qvel)

        if info is not None:
            # info_data = episode_data.create_group("info")
            info_arr = episode_data.create_dataset(
                "info",
                shape=(len(info),),
                chunks=(1,),
                dtype="object",
                object_codec=numcodecs.JSON(),
            )
            for i, info_dict in enumerate(info):
                info_arr[i] = info_dict
        self.curr_eps += 1

    def remove(self, episode_indices):
        for i in episode_indices:
            del self.root[f"episode_{i}"]

    def __len__(self):
        return len(list(self.root.group_keys()))

    def __getitem__(self, index):
        episode_data = self.root[f"episode_{index}"]
        action = episode_data["action"][:]
        proprioception = episode_data["proprioception"][:]
        low_dim_state = episode_data["low_dim_state"][:]
        qpos = episode_data["qpos"][:]
        qvel = episode_data["qvel"][:]
        # info = episode_data["info"][:]
        info = None
        camera_datas = []
        for camera_id in self.camera_ids:
            camera_data = episode_data[f"camera_{camera_id}"]
            rgb = camera_data["rgb"][:]
            depth = camera_data["depth"][:]
            segmentation = camera_data["segmentation"][:]
            camera_datas.append(
                VideoData(
                    rgb=rgb, depth=depth, segmentation=segmentation, camera_id=camera_id
                )
            )
        return EpisodeData(
            camera_datas=camera_datas,
            action=action,
            proprioception=proprioception,
            low_dim_state=low_dim_state,
            qpos=qpos,
            qvel=qvel,
            info=info,
        )

    def __delitem__(self, index):
        del self.root[f"episode_{index}"]

    def __repr__(self) -> str:
        return str(self.root.tree())
