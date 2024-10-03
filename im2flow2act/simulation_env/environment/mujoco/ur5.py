import os
from typing import List, Optional

import numpy as np
from dm_control.mujoco.engine import Physics

from im2flow2act.simulation_env.environment.mujoco.mujocoRobot import MujocoRobot


class Ur5(MujocoRobot):
    def __init__(
        self, mj_physics, init_joints, bodyid, prefix, override_home_ctrl_qpos=None
    ) -> None:
        self.override_home_ctrl_qpos = override_home_ctrl_qpos
        super().__init__(mj_physics, init_joints, bodyid, prefix)

    @property
    def joint_names(self) -> List[str]:
        return [
            os.path.join(self.prefix, joint_name)
            for joint_name in [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        ]

    @property
    def end_effector_site_name(self) -> str:
        # TODO make sure self.prefix is frozen
        return os.path.join(self.prefix, "wsg50", "end_effector")

    @property
    def gripper_joint_name(self) -> str:
        return [
            os.path.join(self.prefix, "wsg50", "right_driver_joint"),
            os.path.join(self.prefix, "wsg50", "left_driver_joint"),
        ]

    @property
    def gripper_actuator_name(self) -> str:
        return os.path.join(self.prefix, "wsg50", "gripper")

    @property
    def gripper_close_ctrl_val(self) -> float:
        return 0.0

    @property
    def gripper_open_ctrl_val(self) -> float:
        return 0.055

    @property
    def home_joint_ctrl_val(self):
        return (
            [0, -1.5708, -1.5708, -1.5708, +1.5708, 0]
            if self.override_home_ctrl_qpos is None
            else self.override_home_ctrl_qpos
        )

    @property
    def home_ctrl_qpos(self) -> np.ndarray:
        return np.concatenate(
            (
                self.home_joint_ctrl_val,
                np.array([self.gripper_open_ctrl_val, self.gripper_open_ctrl_val]),
            ),
            axis=0,
        )

    @property
    def curr_gripper_qpos(self, physics: Optional[Physics] = None) -> float:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        joint = physics.model.joint(self.gripper_joint_name[0])
        assert len(joint.qposadr) == 1
        qpos = physics.data.qpos[joint.qposadr[0]]
        return qpos

    @property
    def gripper_qos_indices(self):
        return [
            self.mj_physics.model.joint(gripper_joint_name).qposadr[0]
            for gripper_joint_name in self.gripper_joint_name
        ]

    def reset_robot_home(self):
        self.mj_physics.data.qpos[self.joint_qpos_indices] = self.home_joint_ctrl_val
        self.mj_physics.data.qpos[self.gripper_qos_indices] = self.gripper_open_ctrl_val
        self.mj_physics.forward()

    def reset_random_robot_home(self):
        self.mj_physics.data.qpos[self.joint_qpos_indices] = (
            self.home_joint_ctrl_val + np.random.uniform(-0.15, 0.15, size=(6,))
        )
        self.mj_physics.data.qpos[self.gripper_qos_indices] = self.gripper_open_ctrl_val
        self.mj_physics.forward()

    # def reset_robot_to(self, qos):
    #     self.mj_physics.data.qpos[self.joint_qpos_indices] = qos
    #     self.mj_physics.data.qpos[self.gripper_qos_indices] = self.gripper_open_ctrl_val
    #     self.mj_physics.forward()

    @property
    def toggle_ctrl_val(self) -> float:
        return (
            self.gripper_close_ctrl_val
            if self.curr_gripper_qpos > 0.05
            else self.gripper_open_ctrl_val
        )
