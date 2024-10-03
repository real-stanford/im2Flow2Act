import lzma
import os
import pickle
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import brotli
import dill
import mgzip
import numpy as np
import torch
from pydantic import dataclasses, validator
from zarr import blosc

from im2flow2act.simulation_env.environment.utlity.robot_utlity import Pose

Point3D = Tuple[float, float, float]
LINK_SEPARATOR_TOKEN = "|"


class AllowArbitraryTypes:
    # TODO look into numpy.typing.NDArray
    # https://numpy.org/devdocs/reference/typing.html#numpy.typing.NDArray
    arbitrary_types_allowed = True


def limit_threads(n: int = 1):
    blosc.set_nthreads(n)
    if n == 1:
        blosc.use_threads = False
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


class Picklable:
    def dump(
        self,
        path: str,
        protocol: Any = mgzip.open,
        pickled_data_compressor: Any = None,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
        compressor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if protocol_kwargs is None:
            protocol_kwargs = {}
            if protocol == mgzip.open:
                protocol_kwargs["thread"] = 8
        if pickled_data_compressor is not None:
            if compressor_kwargs is None:
                compressor_kwargs = {}
            with open(path, "wb", **protocol_kwargs) as f:
                pickled_data = dill.dumps(self)
                compressed_pickled = pickled_data_compressor(
                    pickled_data, **compressor_kwargs
                )
                f.write(compressed_pickled)
        else:
            with protocol(path, "wb", **protocol_kwargs) as f:
                dill.dump(self, f)

    @classmethod
    def load(
        cls,
        path: str,
        protocol: Any = mgzip.open,
        pickled_data_decompressor: Any = None,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if protocol_kwargs is None:
            protocol_kwargs = {}
            if protocol == mgzip.open:
                protocol_kwargs["thread"] = 8
        if pickled_data_decompressor is not None:
            with open(path, "rb", **protocol_kwargs) as f:
                compressed_pickle = f.read()
                try:
                    decompressed_pickle = pickled_data_decompressor(compressed_pickle)
                except brotli.Error as e:
                    raise ValueError("Invalid encoder format") from e
                obj = pickle.loads(decompressed_pickle)
        else:
            with protocol(path, "rb", **protocol_kwargs) as f:
                try:
                    obj = pickle.load(f)
                except lzma.LZMAError as e:
                    raise ValueError("Invalid encoder format") from e
        if type(obj) != cls:
            raise ValueError(f"Pickle file {path} is not a `{cls.__name__}`")
        return obj


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class Contact(Picklable):
    other_link: str
    other_name: str
    self_link: str
    position: Point3D
    normal: Point3D

    @validator("normal")
    @classmethod
    def contact_normal_normalized(cls, v: Point3D):
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("normal can't be zero")
        return tuple(np.array(v) / norm)

    def __hash__(self) -> int:
        return hash(
            (
                self.other_link,
                self.other_name,
                self.self_link,
                self.position,
                self.normal,
            )
        )

    def __eq__(self, other) -> bool:
        return all(
            [
                self.other_link == other.other_link,
                self.other_name == other.other_name,
                self.self_link == other.self_link,
                self.position == other.position,
                self.normal == other.normal,
            ]
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            "Contact("
            + f"other_link={self.other_link}, "
            + f"other_name={self.other_name}, "
            + f"self_link={self.self_link})"
        )


def get_part_path(model, body) -> str:
    rootid = body.rootid
    path = ""
    while True:
        path = body.name + path
        currid = body.id
        if currid == rootid:
            return path
        body = model.body(body.parentid)
        path = LINK_SEPARATOR_TOKEN + path


def parse_contact_data(physics) -> Dict[str, Dict[str, Set[Contact]]]:
    obj_link_contacts: Dict[str, Dict[str, Set[Contact]]] = {}
    data = physics.data
    model = physics.model
    for contact_idx in range(len(data.contact.geom1)):
        geom1 = data.contact.geom1[contact_idx]
        geom2 = data.contact.geom2[contact_idx]
        link1 = model.body(model.geom(geom1).bodyid)
        link1name = get_part_path(model, link1)
        link2 = model.body(model.geom(geom2).bodyid)
        link2name = get_part_path(model, link2)

        contact_pos = data.contact.pos[contact_idx].astype(float)
        position: Point3D = (contact_pos[0], contact_pos[1], contact_pos[2])
        normal = data.contact.frame[contact_idx][:3]

        obj1name = link1name.split(LINK_SEPARATOR_TOKEN)[0]
        obj2name = link2name.split(LINK_SEPARATOR_TOKEN)[0]
        if obj1name not in obj_link_contacts:
            obj_link_contacts[obj1name] = {}
        if link1name not in obj_link_contacts[obj1name]:
            obj_link_contacts[obj1name][link1name] = set()
        obj_link_contacts[obj1name][link1name].add(
            Contact(
                other_link=link2name,
                other_name=obj2name,
                self_link=link1name,
                position=position,
                normal=(normal[0], normal[1], normal[2]),
            )
        )
        if obj2name not in obj_link_contacts:
            obj_link_contacts[obj2name] = {}
        if link2name not in obj_link_contacts[obj2name]:
            obj_link_contacts[obj2name][link2name] = set()
        obj_link_contacts[obj2name][link2name].add(
            Contact(
                other_link=link1name,
                other_name=obj1name,
                self_link=link2name,
                position=position,
                normal=(-normal[0], -normal[1], -normal[2]),
            )
        )
    return obj_link_contacts


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class Velocity(Picklable):
    linear_velocity: np.ndarray  # shape: (3, )
    angular_velocity: np.ndarray  # shape: (3, )

    @validator("linear_velocity")
    @classmethod
    def linear_velocity_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("linear_velocity must be 3D")
        return v

    @validator("angular_velocity")
    @classmethod
    def angular_velocity_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("angular_velocity must be 3D")
        return v

    @property
    def flattened(self) -> List[float]:
        return list(self.linear_velocity) + list(self.angular_velocity)

    def __hash__(self) -> int:
        return hash(
            (
                *self.linear_velocity.tolist(),
                *self.angular_velocity.tolist(),
            )
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class LinkState(Picklable):
    link_path: str
    obj_name: str
    pose: Pose
    velocity: Velocity
    contacts: Set[Contact]
    # aabbs: List[AABB]

    @property
    def flattened(self) -> List[float]:
        return self.pose.flattened + self.velocity.flattened

    def __hash__(self) -> int:
        return hash(
            (self.link_path, self.obj_name, self.pose, self.velocity, *self.contacts)
        )

    def get_contacts_with(self, other) -> Set[Contact]:
        return set(
            filter(
                lambda c: c.other_link == other.link_path
                and c.other_name == other.obj_name,
                self.contacts,
            )
        )


Enum = None

from enum import IntEnum


class JointType(IntEnum):
    REVOLUTE = 0
    PRISMATIC = 1
    FREE = 2


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class JointState(Picklable):
    name: str
    joint_type: JointType
    min_value: float
    max_value: float
    current_value: float  # DOF value
    axis: Tuple[float, float, float]  # axis direction in world frame
    position: Point3D  # origin of joint in world frame
    orientation: np.ndarray
    parent_link: str
    child_link: str
