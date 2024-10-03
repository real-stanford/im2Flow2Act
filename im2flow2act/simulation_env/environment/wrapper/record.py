import os

import gymnasium as gym

from im2flow2act.common.data import EpisodeDataBuffer, VideoData
from im2flow2act.simulation_env.utility.file import save_data_as_pickle


class RecordWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        render_width,
        render_height,
        render_fps,
        camera_ids,
        store_path,
        max_workers=64,
        save_mj_physics=False,
        record_low_dim_state=False,
    ) -> None:
        self.render_width = render_width
        self.render_height = render_height
        self.render_fps = render_fps
        self.camera_ids = camera_ids
        self.store_path = store_path
        self.recoder_buffer = EpisodeDataBuffer(
            store_path=store_path, camera_ids=camera_ids, max_workers=max_workers
        )
        self.save_mj_physics = save_mj_physics
        self.record_low_dim_state = record_low_dim_state
        super().__init__(env)

    def reset(self, **kwargs):
        self.clear_buffer()
        return self.env.reset(**kwargs)

    def clear_buffer(self):
        self.episode_video_buffer = {camera_id: [] for camera_id in self.camera_ids}
        self.episode_action_buffer = []
        self.episode_proprioception_buffer = []
        self.episode_low_dim_state_buffer = []
        self.episode_qpos = []
        self.episode_qvel = []
        self.episode_info = []

    def step(self, action):
        proprioceptive_observation = self.env.get_proprioceptive_observation()
        self.episode_proprioception_buffer.append(proprioceptive_observation)
        if self.record_low_dim_state:
            low_dim_state = self.env.get_low_dim_observation()
            self.episode_low_dim_state_buffer.append(low_dim_state)
        self.episode_qpos.append(self.env.mj_physics.data.qpos.copy())
        self.episode_qvel.append(self.env.mj_physics.data.qvel.copy())
        step_visual_obervations = self.env.render_all(
            height=self.render_height, width=self.render_width
        )
        for camera_id in self.camera_ids:
            self.episode_video_buffer[camera_id].append(
                step_visual_obervations[camera_id]
            )

        obs, reward, done, truncated, info = self.env.step(action)
        self.episode_action_buffer.append(action)
        self.episode_info.append(info)

        return obs, reward, done, truncated, info

    def flush(self, save_video=True, store_eps=None):
        flushed_visual_obervations_buffer = {}
        for camera_id in self.camera_ids:
            episode_video_data = VideoData.stack(self.episode_video_buffer[camera_id])
            flushed_visual_obervations_buffer[camera_id] = episode_video_data
        self.recoder_buffer.append(
            flushed_visual_obervations_buffer,
            self.episode_action_buffer,
            self.episode_proprioception_buffer,
            self.episode_low_dim_state_buffer if self.record_low_dim_state else None,
            self.episode_qpos,
            self.episode_qvel,
            self.episode_info,
            save_video=save_video,
            render_fps=self.render_fps,
            store_eps=store_eps,
        )
        if self.save_mj_physics:
            save_data_as_pickle(
                self.env.mj_physics,
                os.path.join(
                    self.store_path,
                    f"eps_{self.recoder_buffer.curr_eps}_mj_physics.pkl",
                ),
            )
        self.clear_buffer()
