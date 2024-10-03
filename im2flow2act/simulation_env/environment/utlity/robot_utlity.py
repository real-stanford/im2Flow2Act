from dataclasses import dataclass
import numpy as np
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class Pose:
    position: np.ndarray  # shape: (3, )
    orientation: np.ndarray  # shape: (4, ), quaternion

    @property
    def flattened(self) -> List[float]:
        return np.concatenate([self.position, self.orientation])

    @classmethod
    def from_flattened(cls, flattened_list):
        # Extract position and orientation from the flattened list
        position = np.array(flattened_list[:3])
        orientation = np.array(flattened_list[3:])

        # Create a new Pose instance
        return cls(position=position, orientation=orientation)

    def __str__(self):
        return f"position:{np.round(self.position,2)},orientation:{np.round(self.orientation,2)}"


def rotvec2quat(rotvec):
    r = R.from_rotvec(rotvec)
    return r.as_quat()


# SCIPY using x, y, z, w
# MUJOCO, TRANSFORM3D using w, x, y, z
def quat2euler(quat, degrees=False):
    r = R.from_quat(quat[[1, 2, 3, 0]])
    return r.as_euler("xyz", degrees=degrees)


def euler2quat(euler, degrees=False):
    r = R.from_euler("xyz", euler, degrees=degrees)
    return r.as_quat()[[3, 0, 1, 2]]


def euler_to_matrix(euler_angles, degrees=False):
    # Assuming 'ZYX' order for Euler angles
    return R.from_euler("xyz", euler_angles, degrees=degrees).as_matrix()


def matrix2quat(matrix):
    r = R.from_matrix(matrix)
    return r.as_quat()[[3, 0, 1, 2]]


def construct_homogeneous_matrix(rotation_matrix, translation_vector):
    # Create the homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector
    return T


def transform_gripper_to_object(
    gripper_position, gripper_euler, object_position, object_euler
):
    # Convert Euler angles to rotation matrices
    gripper_rotation = euler_to_matrix(gripper_euler)
    object_rotation = euler_to_matrix(object_euler)

    # Construct the homogeneous transformation matrices
    gripper_T = construct_homogeneous_matrix(gripper_rotation, gripper_position)
    object_T = construct_homogeneous_matrix(object_rotation, object_position)

    # Compute the inverse of the object's transformation matrix
    object_T_inv = np.linalg.inv(object_T)

    # Multiply the inverted object's transformation matrix by the gripper's transformation matrix
    gripper_in_object_T = np.dot(object_T_inv, gripper_T)

    return gripper_in_object_T


# Function to convert the transformation matrix back into Euler angles and position vector
def matrix_to_euler_and_position(transformation_matrix, degrees=False):
    # Extract the rotation matrix and position vector from the homogeneous transformation matrix
    rotation_matrix = transformation_matrix[:3, :3]
    position_vector = transformation_matrix[:3, 3]

    # Convert the rotation matrix to Euler angles
    euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)

    return position_vector, euler_angles


@dataclass(frozen=True)
class ControlCmd:
    pose: Pose
    gripper_ctrl: float
    time: float

    @property
    def flattened(self):
        return np.concatenate([self.pose.flattened, [self.gripper_ctrl]])

    @classmethod
    def from_flattened(cls, flattened_array):
        # Extract the elements from the flattened array
        pose_elements = flattened_array[:-1]  # All elements except the last one
        gripper_ctrl = flattened_array[-1]  # The last element

        # Reconstruct the Pose object from the flattened elements
        # Assuming that the Pose class has a method to create an instance from flattened data
        pose = Pose.from_flattened(pose_elements)

        # Create a new ControlCmd instance
        return cls(pose=pose, gripper_ctrl=gripper_ctrl, time=0.0)


@dataclass(frozen=True)
class TSControlCmd:
    pose: Pose
    gripper_ctrl: float
    time: float

    @property
    def flattened(self):
        return np.concatenate([self.pose.flattened, [self.gripper_ctrl]])

    @classmethod
    def from_flattened(cls, flattened_array, has_gripper=True):
        # Extract the elements from the flattened array
        if has_gripper:
            pose_elements = flattened_array[:-1]  # All elements except the last one
            gripper_ctrl = flattened_array[-1]  # The last element
        else:
            pose_elements = flattened_array
            gripper_ctrl = []

        # Reconstruct the Pose object from the flattened elements
        # Assuming that the Pose class has a method to create an instance from flattened data
        pose = Pose.from_flattened(pose_elements)

        # Create a new ControlCmd instance
        return cls(pose=pose, gripper_ctrl=gripper_ctrl, time=0.0)


@dataclass(frozen=True)
class JSControlCmd:
    jpos: np.ndarray
    gripper_ctrl: float

    @property
    def flattened(self):
        return np.concatenate([self.jpos, [self.gripper_ctrl]])

    @classmethod
    def from_flattened(cls, flattened_array, has_gripper=True):
        # Extract the elements from the flattened array
        if has_gripper:
            jpos = flattened_array[:-1]  # All elements except the last one
            gripper_ctrl = flattened_array[-1]  # The last element
        else:
            jpos = flattened_array
            gripper_ctrl = []

        # Create a new ControlCmd instance
        return cls(jpos=jpos, gripper_ctrl=gripper_ctrl)
