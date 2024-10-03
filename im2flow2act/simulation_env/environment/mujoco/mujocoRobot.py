# from transforms3d import quaternions
import os
from abc import abstractproperty

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from scipy.spatial.transform import Rotation as R

from im2flow2act.simulation_env.environment.utlity.robot_utlity import Pose


class MujocoRobot:
    def __init__(self, mj_physics, init_joints, bodyid, prefix) -> None:
        self.mj_physics = mj_physics
        self.ik_mj_physics = mj_physics.copy(share_model=True)
        self.bodyid = bodyid
        self.prefix = prefix

    @abstractproperty
    def joint_names(self):
        raise NotImplementedError()

    @property
    def joints(self):
        return [
            self.mj_physics.model.joint(joint_name) for joint_name in self.joint_names
        ]

    @property
    def joint_qpos_indices(self):
        return [joint.qposadr[0] for joint in self.joints]

    @property
    def joint_qvel_indices(self):
        return [joint.dofadr[0] for joint in self.joints]

    def get_joint_velocities(self):
        joint_velocities = self.mj_physics.data.qvel[self.joint_qvel_indices].copy()
        return joint_velocities

    def get_joint_positions(self):
        joint_positions = self.mj_physics.data.qpos[self.joint_qpos_indices].copy()
        return joint_positions

    @property
    def end_effector_site_name(self):
        return os.path.join(self.prefix, "end_effector")

    def get_end_effector_pose(self):
        # SCIPY using x, y, z, w
        quat_xyzw = self.get_end_effector_rotation().as_quat()
        # MUJOCO, TRANSFORM3D using w, x, y, z
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return Pose(position=self.get_end_effector_position(), orientation=quat_wxyz)

    def get_end_effector_position(self):
        return self.mj_physics.data.site(self.end_effector_site_name).xpos.copy()

    def get_end_effector_rotation(self):
        ee_xmt = (
            self.mj_physics.data.site(self.end_effector_site_name)
            .xmat.copy()
            .reshape((3, 3))
        )
        r = R.from_matrix(ee_xmt)
        return r

    # @profile
    def inverse_kinematics(self, pose: Pose, inplace=False):
        pose = Pose(pose.position, pose.orientation)
        if inplace:
            self.ik_mj_physics.data.qpos[:] = self.mj_physics.data.qpos[:].copy()
            result = qpos_from_site_pose(
                physics=self.ik_mj_physics,
                site_name=self.end_effector_site_name,
                joint_names=self.joint_names,
                target_pos=pose.position,
                target_quat=pose.orientation,
                tol=1e-7,
                max_steps=100,
                inplace=inplace,
            )
        else:
            result = qpos_from_site_pose(
                physics=self.mj_physics,
                site_name=self.end_effector_site_name,
                target_pos=pose.position,
                target_quat=pose.orientation,
                joint_names=self.joint_names,
                tol=1e-7,
                max_steps=100,
                inplace=inplace,
            )
        if not result.success:
            return None
        return result.qpos[self.joint_qpos_indices].copy()
