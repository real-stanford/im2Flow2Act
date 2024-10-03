import os

import numpy as np
from dm_control import mjcf
from scipy.spatial.transform import Rotation as R

from im2flow2act.simulation_env.environment.mujoco.mujocoEnv import (
    TableTopUR5WSG50FinrayEnv,
)
from im2flow2act.simulation_env.environment.utlity.robot_utlity import (
    quat2euler,
)

file_dir = os.path.dirname(os.path.abspath(__file__))


class Open(TableTopUR5WSG50FinrayEnv):
    def __init__(
        self,
        drawer_pos=None,
        drawer_quat=None,
        drawer_location="left",
        drawer_type=None,
        **kwargs,
    ) -> None:
        self.drawer_pos = np.array(drawer_pos) if drawer_pos is not None else None
        self.drawer_quat = np.array(drawer_quat) if drawer_quat is not None else None
        self.drawer_location = drawer_location
        self.drawer_type = int(drawer_type) if drawer_type is not None else None
        super().__init__(sea_backgroud=True, **kwargs)
        self.task_description = "Open the drawer."
        self.action_z_lower_bound = 0.03
        self.action_z_upper_bound = 0.5

    def update_task_description(self, task_description):
        self.task_description = task_description

    def setup_model(self):
        world_model = super().setup_model()
        return world_model

    def setup_objs(self, world_model):
        drawer_variation = [
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/custom/single_drawer_color_big_handle_c1.xml",
                )
            ),
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/custom/single_drawer_color_big_handle_c2.xml",
                )
            ),
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/custom/single_drawer_color_big_handle_c3.xml",
                )
            ),
        ]
        if self.drawer_type is None:
            drawer_type = np.random.choice(np.arange(len(drawer_variation)))
            self.drawer_type = drawer_type
        print("select drawer type:", self.drawer_type)
        object_model = mjcf.from_file(
            drawer_variation[self.drawer_type],
            model_dir=f"{file_dir}/../mujoco/asset/custom/",
        )
        object_model.model = "drawer"
        if self.drawer_pos is None or self.drawer_quat is None:
            if self.drawer_location == "left":
                drawer_pos = np.array(
                    [
                        np.random.uniform(0.45, 0.62),
                        np.random.uniform(-0.4, -0.32),
                        np.random.uniform(0.05, 0.18),
                    ]
                )
                if drawer_pos[0] >= 0.6:
                    drawer_quat = R.from_euler(
                        "z",
                        np.random.uniform(np.pi * 3 / 8 + np.pi / 16, np.pi * 4 / 8),
                    ).as_quat()[[3, 0, 1, 2]]
                else:
                    drawer_quat = R.from_euler(
                        "z",
                        np.random.uniform(np.pi * 3 / 8, np.pi * 4 / 8),
                    ).as_quat()[[3, 0, 1, 2]]
            else:
                raise NotImplementedError
        else:
            drawer_pos = self.drawer_pos
            drawer_quat = self.drawer_quat

        self.add_obj_from_model(
            world_model,
            object_model,
            drawer_pos,
            quat=drawer_quat,
        )
        self.drawer_pos = drawer_pos
        self.drawer_quat = drawer_quat
        self.selected_object = "drawer"

    def reset(self, seed: int = None, **kwargs):
        obs, _ = super().reset(seed, **kwargs)
        return obs, {}

    def set_initial_handle_pos(self):
        self.handle_name = "drawer/pull_site_0"  # to check success
        self.initial_handle_pos = self.mj_physics.data.site(
            self.handle_name
        ).xpos.copy()

    def add_obj_from_model(
        self, world_model, obj_model, position, quat=None, add_freejoint=False
    ):
        object_site = world_model.worldbody.add(
            "site",
            name=f"{obj_model.model}_site",
            pos=position,
            quat=quat,
            group=3,
        )
        if add_freejoint:
            object_site.attach(obj_model).add("freejoint")
        else:
            object_site.attach(obj_model)

    def step(self, action):
        _, _, _, _, control_info = super().step(action)
        _, obj_link_states, obj_link_contacts = self.get_state()
        observation = self.get_observation(obj_link_states, obj_link_contacts)
        truncated = False
        info = {
            "env": "RealOpen",
            "task_description": self.task_description,
            "drawer_pos": self.drawer_pos.tolist(),
            "drawer_quat": self.drawer_quat.tolist(),
            "drawer_type": int(self.drawer_type),
        }
        info = info | control_info
        return observation, 0, False, truncated, info

    def check_success(self):
        current_handle_pos = self.mj_physics.data.site(self.handle_name).xpos.copy()
        success_distance = 0.10
        if (
            np.linalg.norm(current_handle_pos - self.initial_handle_pos)
            >= success_distance
        ):
            return True
        return False

    def get_observation(self, obj_link_states, obj_link_contacts):
        proprioceptive_observation = self.get_proprioceptive_observation()
        if self.visual_observation:
            env_observation = self.get_visual_observation()
            return {**env_observation, "proprioceptive": proprioceptive_observation}
        else:
            env_observation = self.get_low_dim_observation(
                obj_link_states, obj_link_contacts
            )
            return np.concatenate([proprioceptive_observation, env_observation])

    def get_reward(self, obj_link_states, obj_link_contacts):
        return 0

    def get_distance(self, position_a, position_b):
        return np.linalg.norm(position_a - position_b)

    def get_low_dim_observation(self, obj_link_states=None, obj_link_contacts=None):
        if obj_link_contacts is None or obj_link_contacts is None:
            _, obj_link_states, obj_link_contacts = self.get_state()
        return np.concatenate(
            [
                self.get_object_qpos(self.selected_object, obj_link_states),
                self.get_object_quant(self.selected_object, obj_link_states),
                # [int(self.check_contact(obj_link_contacts))],
            ]
        ).astype(np.float32)

    # @profile
    def get_object_qpos(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_qpos = []
        for k, v in object_link_state.items():
            object_links_qpos.append(v.pose.position)
        return object_links_qpos[-1].astype(np.float32)

    def get_object_quant(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_quat = []
        for k, v in object_link_state.items():
            object_links_quat.append(v.pose.orientation)
        return object_links_quat[-1].astype(np.float32)

    def get_object_euler(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_quat = []
        for k, v in object_link_state.items():
            object_links_quat.append(v.pose.orientation)
        return quat2euler(object_links_quat[-1].astype(np.float32))
