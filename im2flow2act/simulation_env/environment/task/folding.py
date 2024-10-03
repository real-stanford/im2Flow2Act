import os
from typing import Optional

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


class Folding(TableTopUR5WSG50FinrayEnv):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(sea_backgroud=True, **kwargs)
        self.task_description = "Fold the cloth"
        self.action_z_lower_bound = 0.016
        self.action_z_upper_bound = 0.5

    # @profile
    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed)
        _, obj_link_states, obj_link_contacts = self.get_state()
        return self.get_observation(obj_link_states, obj_link_contacts), {}

    def update_task_description(self, task_description):
        self.task_description = task_description

    def setup_model(self):
        world_model = super().setup_model()
        return world_model

    def setup_objs(self, world_model):
        # real #1
        # layout = np.random.choice([0, 1, 2, 3])
        layout = np.random.choice([0, 1])
        if layout == 0:
            handle_x_bound = [0.32, 0.44]
            handle_y_bound = [0.10, 0.3]
            handle_pos = (
                np.random.uniform(*handle_x_bound),
                np.random.uniform(*handle_y_bound),
                0.051,
            )
            handle_rot = R.from_euler(
                "z", np.random.uniform(-np.pi / 8, np.pi / 8)
            ).as_quat()[[3, 0, 1, 2]]
        elif layout == 1:
            handle_x_bound = [0.32, 0.44]
            handle_y_bound = [-0.3, -0.10]
            handle_pos = (
                np.random.uniform(*handle_x_bound),
                np.random.uniform(*handle_y_bound),
                0.051,
            )
            handle_rot = R.from_euler(
                "z", np.random.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
            ).as_quat()[[3, 0, 1, 2]]
        elif layout == 2:
            handle_x_bound = [0.44, 0.46]
            handle_y_bound = [-0.4, 0.0]
            handle_pos = (
                np.random.uniform(*handle_x_bound),
                np.random.uniform(*handle_y_bound),
                0.051,
            )
            handle_rot = R.from_euler(
                "z", np.random.uniform(np.pi / 2 + np.pi / 8, np.pi / 2 + np.pi / 4)
            ).as_quat()[[3, 0, 1, 2]]
        elif layout == 3:
            handle_x_bound = [0.44, 0.46]
            handle_y_bound = [-0.0, 0.4]
            handle_pos = (
                np.random.uniform(*handle_x_bound),
                np.random.uniform(*handle_y_bound),
                0.051,
            )
            handle_rot = R.from_euler(
                "z",
                np.random.uniform(
                    2 * np.pi - np.pi / 4, 2 * np.pi - np.pi / 4 + np.pi / 8
                ),
            ).as_quat()[[3, 0, 1, 2]]

        object_model = mjcf.from_file(
            os.path.normpath(
                os.path.join(
                    file_dir,
                    "../mujoco/asset/custom/real_cloth.xml",
                )
            ),
            model_dir=f"{file_dir}/../mujoco/asset/custom/",
        )
        object_model.model = "cloth"
        self.add_obj_from_model(
            world_model,
            object_model,
            handle_pos,
            quat=handle_rot,
            add_freejoint=True,
        )
        self.selected_object = "cloth"

    def is_overlapping(self, obj1_pos, obj2_pos, min_distance=0.1):
        x1, y1, _ = obj1_pos
        x2, y2, _ = obj2_pos
        if np.linalg.norm((x1 - x2, y1 - y2)) <= (min_distance):
            return True
        return False

    def get_random_z_rotation(self):
        return R.from_euler("z", np.random.uniform(0, 2 * np.pi)).as_quat()[
            [3, 0, 1, 2]
        ]

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
        _, _, _, _, _ = super().step(action)
        _, obj_link_states, obj_link_contacts = self.get_state()
        observation = self.get_observation(obj_link_states, obj_link_contacts)
        truncated = False
        info = {
            "env": "RealDeformable",
            "task_description": self.task_description,
        }
        return observation, 0, False, truncated, info

    def check_success(self):
        # need to check manually
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