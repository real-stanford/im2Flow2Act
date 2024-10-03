import gc
import logging
import os
from abc import abstractmethod

# import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial.transform as st
from dm_control import mjcf
from dm_control.mujoco.engine import Physics
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from im2flow2act.common.data import VideoData
from im2flow2act.simulation_env.environment.mujoco.ur5 import Ur5
from im2flow2act.simulation_env.environment.utlity.env_utility import (
    JointState,
    LinkState,
    Velocity,
    get_part_path,
    parse_contact_data,
)
from im2flow2act.simulation_env.environment.utlity.robot_utlity import (
    ControlCmd,
    Pose,
    euler2quat,
)

file_dir = os.path.dirname(os.path.abspath(__file__))


class MujocoEnv:
    def __init__(
        self, robot_cls, prefix, verbose, in_place_ik=True, controll_frequency=None
    ) -> None:
        self.verbose = verbose
        self.in_place_ik = in_place_ik
        self.mj_physics = self.setup()
        self.robot = robot_cls(self.mj_physics, None, None, prefix)
        if controll_frequency is None:
            self.controll_frequency = 1 / self.mj_physics.timestep()
        else:
            self.controll_frequency = controll_frequency

    def reset(self, **kwargs):
        del self.mj_physics
        gc.collect()
        mj_physics = self.setup()
        self.update_mj_physics(mj_physics)
        gc.collect()
        self.mj_physics.forward()
        self._time_counter = 0

    def load_mj_physics(self, mj_physics):
        self.mj_physics = mj_physics
        self.mj_physics.forward()

    def load_from_vector(self, qpos, qvel):
        self.mj_physics.data.qpos[:] = qpos
        self.mj_physics.data.qvel[:] = qvel
        # crutial step
        self.mj_physics.forward()

    @property
    def dt(self):
        return 1 / self.controll_frequency

    @property
    def current_time(self):
        return self.dt * self._time_counter

    def update_mj_physics(self, mj_physics: Physics):
        self.robot.mj_physics = mj_physics
        self.mj_physics = mj_physics

    @abstractmethod
    def setup_model(self):
        raise NotImplementedError()

    @abstractmethod
    def setup_objs(self, world_model):
        pass

    def setup(self) -> Tuple[Physics, int]:
        world_model = self.setup_model()
        self.setup_objs(world_model=world_model)
        mj_physics = mjcf.Physics.from_mjcf_model(world_model)

        return mj_physics

    def step_until_stable(
        self,
    ):
        pass

    # @profile
    def compute_waypoint(self, ctrl_cmd: ControlCmd):
        current_robot_pose = self.robot.get_end_effector_pose()
        target_robot_pose = ctrl_cmd.pose
        n_waypoint = int((ctrl_cmd.time - self.current_time) / self.dt)
        current_orientation = R.from_quat(current_robot_pose.orientation[[1, 2, 3, 0]])
        target_orientation = R.from_quat(target_robot_pose.orientation[[1, 2, 3, 0]])
        orientation_slerp = Slerp(
            times=[self.current_time, ctrl_cmd.time],
            rotations=R.concatenate([current_orientation, target_orientation]),
        )

        # Interploation
        ctrl_cmd_waypoints = []
        for i in range(1, n_waypoint + 1):
            alpha = i / n_waypoint
            waypoint_timestamp = self.current_time + i * self.dt
            position_waypoints = (
                alpha * target_robot_pose.position
                + (1 - alpha) * current_robot_pose.position
            )
            # gripper_waypoints = (
            #     alpha * ctrl_cmd.gripper_ctrl
            #     + (1 - alpha) * self.robot.curr_gripper_qpos
            # )
            gripper_waypoints = (
                alpha * ctrl_cmd.gripper_ctrl
                + (1 - alpha) * self.mj_physics.data.ctrl[-1]
            )
            orientation_waypoints = (
                orientation_slerp([waypoint_timestamp])[0]
            ).as_quat()[[3, 0, 1, 2]]
            ctrl_cmd_waypoints.append(
                ControlCmd(
                    pose=Pose(
                        position=position_waypoints, orientation=orientation_waypoints
                    ),
                    gripper_ctrl=gripper_waypoints,
                    time=waypoint_timestamp,
                )
            )
            # print(waypoint_timestamp,orientation_slerp([waypoint_timestamp])[0].as_euler('xyz',degrees=True))
        return ctrl_cmd_waypoints

    def compute_n_waypoint(
        self,
        ctrl_cmd: ControlCmd,
        n_waypoint,
        initial_pose=None,
        initial_gripper_pos=None,
    ):
        if initial_pose is None:
            initial_pose = self.robot.get_end_effector_pose()
            initial_gripper_pos = self.mj_physics.data.ctrl[-1]
        target_robot_pose = ctrl_cmd.pose
        current_orientation = R.from_quat(initial_pose.orientation[[1, 2, 3, 0]])
        target_orientation = R.from_quat(target_robot_pose.orientation[[1, 2, 3, 0]])
        orientation_slerp = Slerp(
            times=[0, 1],
            rotations=R.concatenate([current_orientation, target_orientation]),
        )
        # Interploation
        ctrl_cmd_waypoints = []
        for i in range(1, n_waypoint + 1):
            alpha = i / n_waypoint
            waypoint_timestamp = i * 1 / n_waypoint
            position_waypoints = (
                alpha * target_robot_pose.position + (1 - alpha) * initial_pose.position
            )
            gripper_waypoints = (
                alpha * ctrl_cmd.gripper_ctrl + (1 - alpha) * initial_gripper_pos
            )
            orientation_waypoints = (
                orientation_slerp([waypoint_timestamp])[0]
            ).as_quat()[[3, 0, 1, 2]]
            ctrl_cmd_waypoints.append(
                ControlCmd(
                    pose=Pose(
                        position=position_waypoints, orientation=orientation_waypoints
                    ),
                    gripper_ctrl=gripper_waypoints,
                    time=waypoint_timestamp,
                )
            )
            # print(waypoint_timestamp,orientation_slerp([waypoint_timestamp])[0].as_euler('xyz',degrees=True))
        return ctrl_cmd_waypoints

    # @profile
    def step(self, ctrl_cmd: ControlCmd):
        target_timestamp = self.current_time + 0.1
        ctrl_cmd = ControlCmd(
            pose=ctrl_cmd.pose,
            gripper_ctrl=ctrl_cmd.gripper_ctrl,
            time=target_timestamp,
        )
        ctrl_cmd_waypoints = self.compute_waypoint(ctrl_cmd)
        control_info = {"IK failed": False}
        for ctrl_cmd in ctrl_cmd_waypoints:
            # print(ctrl_cmd.pose.orientation)

            target_joint_qpos = self.robot.inverse_kinematics(
                ctrl_cmd.pose, inplace=self.in_place_ik
            )
            if target_joint_qpos is None:
                if self.verbose:
                    print("IK failed")
                logging.info("IK failed!")
                control_info["IK failed"] = True
                target_joint_qpos = self.robot.get_joint_positions()

            self.mj_physics.data.ctrl[:] = np.append(
                target_joint_qpos, ctrl_cmd.gripper_ctrl
            )
            self.mj_physics.step()
            # if self.gui_handle is not None:
            #     self.gui_handle.sync()
            # self.mj_physics.forward()
            self._time_counter += 1

        return control_info

    ###@profile
    def render_rgb(self, camera_ids: List[int], height=480, width=640):
        rgbs = {}
        for camera_id in camera_ids:
            rgbs[camera_id] = self.mj_physics.render(
                camera_id=camera_id,
                height=height,
                width=width,
            )
        return rgbs

    def render_depth(self, camera_ids: List[int], height=480, width=640):
        depths = {}
        for camera_id in camera_ids:
            depths[camera_id] = self.mj_physics.render(
                camera_id=camera_id, height=height, width=width, depth=True
            )
        return depths

    def render_segmentation(self, camera_ids: List[int], height=480, width=640):
        segmentations = {}
        for camera_id in camera_ids:
            segmentations[camera_id] = self.mj_physics.render(
                camera_id=camera_id, height=height, width=width, segmentation=True
            )
        return segmentations

    def render_all(
        self, camera_ids: List[int] = [], height=480, width=640, view="camera"
    ):
        if len(camera_ids) == 0:
            camera_ids = np.arange(self.mj_physics.model.ncam)
        rgbs = self.render_rgb(camera_ids=camera_ids, height=height, width=width)
        depths = self.render_depth(camera_ids=camera_ids, height=height, width=width)
        segmentations = self.render_segmentation(
            camera_ids=camera_ids, height=height, width=width
        )
        if view == "type":
            return rgbs, depths, segmentations
        elif view == "camera":
            return {
                camera_id: VideoData(
                    rgb=rgbs[camera_id],
                    depth=depths[camera_id],
                    segmentation=segmentations[camera_id],
                    camera_id=camera_id,
                )
                for camera_id in camera_ids
            }
        else:
            raise ValueError("view should be type or camera")

    def render_camera(self, camera_id, height=480, width=640):
        img_arr = self.mj_physics.render(
            camera_id=camera_id,
            height=height,
            width=width,
        )
        return img_arr

    def get_state(self):
        obj_link_states: Dict[str, Dict[str, LinkState]] = {}
        obj_joint_states: Dict[str, Dict[str, JointState]] = {}
        model = self.mj_physics.model
        data = self.mj_physics.data

        obj_link_contacts = parse_contact_data(physics=self.mj_physics)
        for bodyid in range(model.nbody):
            body_model = model.body(bodyid)
            body_data = data.body(bodyid)  # type: ignore
            pose = Pose(
                position=body_data.xpos.copy(),
                orientation=body_data.xquat.copy(),
            )
            root_name = model.body(body_model.rootid).name
            if root_name not in obj_link_states:
                obj_link_states[root_name] = {}
            if root_name not in obj_joint_states:
                obj_joint_states[root_name] = {}
            part_path = get_part_path(self.mj_physics.model, body_model)
            obj_link_states[root_name][part_path] = LinkState(
                link_path=part_path,
                obj_name=root_name,
                pose=pose,
                velocity=Velocity(
                    linear_velocity=body_data.cvel[3:].copy(),
                    angular_velocity=body_data.cvel[:3].copy(),
                ),
                contacts=(
                    obj_link_contacts[root_name][part_path]
                    if (
                        root_name in obj_link_contacts
                        and part_path in obj_link_contacts[root_name]
                    )
                    else set()
                ),
            )
        return obj_joint_states, obj_link_states, obj_link_contacts


class MujocoUR5WSG50FinrayEnv(MujocoEnv):
    def __init__(
        self,
        prefix="ur5e",
        random_init_robot=False,
        deformable_simulation=False,
        robot_initial_qpos=None,
        sea_backgroud=False,
        **kwargs,
    ) -> None:
        self.random_init_robot = random_init_robot
        self.deformable_simulation = deformable_simulation
        self.robot_initial_qpos = robot_initial_qpos
        self.sea_backgroud = sea_backgroud
        super().__init__(
            robot_cls=Ur5,
            prefix=prefix,
            **kwargs,
        )

    def setup_model(self):
        if self.deformable_simulation:
            if self.sea_backgroud:
                world_model = mjcf.from_path(
                    os.path.normpath(
                        os.path.join(
                            file_dir, "asset/custom/deformable_sea_sim_groud.xml"
                        )
                    )
                )
            else:
                world_model = mjcf.from_path(
                    os.path.normpath(
                        os.path.join(file_dir, "asset/custom/deformable_sim_groud.xml")
                    )
                )
        else:
            if self.sea_backgroud:
                world_model = mjcf.from_path(
                    os.path.normpath(os.path.join(file_dir, "asset/sea.xml"))
                )
            else:
                world_model = mjcf.from_path(
                    os.path.normpath(os.path.join(file_dir, "asset/ground.xml"))
                )

        robot_model = mjcf.from_path(
            os.path.normpath(
                os.path.join(
                    file_dir, "asset/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
                )
            )
        )
        del robot_model.keyframe
        robot_model.worldbody.light.clear()
        attachment_site = robot_model.find("site", "attachment_site")
        assert attachment_site is not None
        gripper = mjcf.from_path(
            os.path.normpath(
                os.path.join(file_dir, "asset/wsg50/wsg50_finray_milibar.xml")
            )
        )
        attachment_site.attach(gripper)
        robit_site = world_model.worldbody.add(
            "site",
            name="robot_site",
            pos=(0.0, 0.0, 0),
            quat=R.from_euler("z", np.pi / 2).as_quat()[[3, 0, 1, 2]],
        )
        robit_site.attach(robot_model)

        return world_model

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(**kwargs)
        if self.random_init_robot:
            self.robot.reset_random_robot_home()
        else:
            if self.robot_initial_qpos is not None:
                self.robot.override_home_ctrl_qpos = self.robot_initial_qpos
            self.robot.reset_robot_home()


class TableTopUR5WSG50FinrayEnv(MujocoUR5WSG50FinrayEnv):
    def __init__(
        self,
        camera_ids: List[int] = [0],
        camera_res: List[int] = [480, 640],
        euler_gripper_rotation=True,
        delta_space=True,
        delta_orientation_limit=0.05,
        delta_position_limit=0.05,
        TabelTopConstraints: List[int] = [
            -1,
            1,
            -1,
            1,
            0.0,
            1,
        ],
        visual_observation: bool = True,
        **kwargs,
    ) -> None:
        self.camera_ids = camera_ids
        self.camera_res = camera_res
        self.visual_observation = visual_observation
        super().__init__(**kwargs)
        self.euler_gripper_rotation = euler_gripper_rotation
        self.delta_space = delta_space
        self.delta_orientation_limit = delta_orientation_limit
        self.delta_position_limit = delta_position_limit
        (
            self.x_lower_limit,
            self.x_upper_limit,
            self.y_lower_limit,
            self.y_upper_limit,
            self.z_lower_limit,
            self.z_upper_limit,
        ) = TabelTopConstraints

    @property
    def euler_observation_scaler(self):
        return 1 / (2 * np.pi)

    def setup_model(self):
        world_model = super().setup_model()
        table_model = mjcf.from_file(
            os.path.normpath(os.path.join(file_dir, "asset/custom/table.xml")),
            model_dir=f"{file_dir}/../mujoco/asset/custom/",
        )
        table_attachment_site = world_model.worldbody.add(
            "site", name="table_attachment_site", pos=(0.58, -0.04, 0)
        )
        table_attachment_site.attach(table_model)
        return world_model

    # @profile
    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed)
        _, obj_link_states, obj_link_contacts = self.get_state()
        for i in range(100):
            self.mj_physics.data.ctrl[:] = np.append(
                self.robot.home_joint_ctrl_val, self.robot.curr_gripper_qpos
            )
            self.mj_physics.step()
        return self.get_observation(obj_link_states, obj_link_contacts), {}

    # @profile
    def step(self, action):
        # process gripper
        if self.euler_gripper_rotation:
            if self.delta_space:
                current_position = self.robot.get_end_effector_position()
                current_orientation = self.robot.get_end_effector_rotation()
                delta_position = action[:3]
                delta_rot = st.Rotation.from_euler("xyz", action[3:6])
                target_rotation_in_quat = (delta_rot * current_orientation).as_quat()[
                    [3, 0, 1, 2]
                ]
                target_position = current_position + delta_position
                control_action = np.concatenate(
                    [target_position, target_rotation_in_quat, [action[-1]]]
                )
            else:
                euler_rotation = action[3:6]
                target_rotation_in_quat = euler2quat(euler_rotation, degrees=False)
                control_action = np.concatenate(
                    [action[:3], target_rotation_in_quat, [action[-1]]]
                )
        else:
            control_action = action.copy()

        control_action = self.clip_control_action(control_action)

        ctrl_cmd = ControlCmd.from_flattened(control_action)
        control_info = super().step(ctrl_cmd)
        return None, 0, False, False, control_info

    def clip_control_action(self, control_action):
        control_action[0] = np.clip(
            control_action[0], self.x_lower_limit, self.x_upper_limit
        )
        control_action[1] = np.clip(
            control_action[1], self.y_lower_limit, self.y_upper_limit
        )
        control_action[2] = np.clip(
            control_action[2], self.z_lower_limit, self.z_upper_limit
        )
        return control_action

    # @profile
    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def get_low_dim_observation(self):
        pass

    # @profile
    def get_visual_observation(self):
        visual_observation = self.render_rgb(
            camera_ids=self.camera_ids,
            height=self.camera_res[0],
            width=self.camera_res[1],
        )
        visual_observation = {
            "camera_{}".format(k): v for k, v in visual_observation.items()
        }
        return visual_observation

    # @profile
    def get_proprioceptive_observation(self):
        gripper_position = self.robot.curr_gripper_qpos
        if self.euler_gripper_rotation:
            ee_position = self.robot.get_end_effector_position()
            euler_rotation = self.robot.get_end_effector_rotation().as_euler(
                "xyz", degrees=False
            )
            normalized_euler_rotation = euler_rotation * self.euler_observation_scaler
            return np.concatenate(
                [ee_position, normalized_euler_rotation, [gripper_position]]
            ).astype(np.float32)
        else:
            end_effector_pose = self.robot.get_end_effector_pose().flattened
            return np.concatenate([end_effector_pose, [gripper_position]]).astype(
                np.float32
            )
