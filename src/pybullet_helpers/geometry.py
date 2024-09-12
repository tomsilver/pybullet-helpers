"""PyBullet helper class for geometry utilities."""

from __future__ import annotations

from typing import Iterator, NamedTuple

import numpy as np
import numpy.typing as npt
import pybullet as p
from pybullet_utils.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_from_matrix,
)
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp

Pose3D = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
RollPitchYaw = tuple[float, float, float]


class Pose(NamedTuple):
    """Pose which is a position (translation) and rotation.

    We use a NamedTuple as it supports retrieving by integer indexing
    and most closely follows the PyBullet API.
    """

    # Cartesian (x, y, z) position
    position: Pose3D
    # Quaternion in (x, y, z, w) representation
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_rpy(cls, translation: Pose3D, rpy: RollPitchYaw) -> Pose:
        """Create a Pose from translation and Euler roll-pitch-yaw angles."""
        return cls(translation, quaternion_from_euler(*rpy))

    @classmethod
    def from_matrix(cls, matrix: npt.NDArray) -> Pose:
        """Create a Pose from a 4x4 homogeneous matrix."""
        return Pose(
            position=tuple(matrix[:3, 3]), orientation=quat_from_matrix(matrix[:3, :3])
        )

    @property
    def rpy(self) -> RollPitchYaw:
        """Get the Euler roll-pitch-yaw representation."""
        return euler_from_quaternion(self.orientation)

    @classmethod
    def identity(cls) -> Pose:
        """Unit pose."""
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def to_matrix(self) -> npt.NDArray:
        """Get the 4x4 homogenous matrix representation."""
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = matrix_from_quat(self.orientation)
        matrix[:3, 3] = self.position
        return matrix

    def multiply(self, *poses: Pose) -> Pose:
        """Multiplies poses (i.e., rigid transforms) together."""
        return multiply_poses(self, *poses)

    def invert(self) -> Pose:
        """Invert the pose (i.e., transform)."""
        pos, quat = p.invertTransform(self.position, self.orientation)
        return Pose(pos, quat)

    def allclose(self, other: Pose, atol: float = 1e-6) -> bool:
        """Return whether this pose is close enough to another pose."""
        return np.allclose(self.position, other.position, atol=atol) and np.allclose(
            self.orientation, other.orientation, atol=atol
        )


def multiply_poses(*poses: Pose) -> Pose:
    """Multiplies poses (which are essentially transforms) together."""
    pose = poses[0]
    for next_pose in poses[1:]:
        pybullet_pose = p.multiplyTransforms(
            pose.position, pose.orientation, next_pose.position, next_pose.orientation
        )
        pose = Pose(pybullet_pose[0], pybullet_pose[1])
    return pose


def matrix_from_quat(quat: Quaternion) -> npt.NDArray[np.float64]:
    """Get 3x3 rotation matrix from quaternion (xyzw)."""
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)


def quat_from_matrix(matrix: npt.NDArray[np.float64]) -> Quaternion:
    """Get quaternion (xyzw) from 3x3 rotation matrix."""
    M = np.identity(4)
    M[:3, :3] = matrix
    return tuple(quaternion_from_matrix(M))


def get_pose(body: int, physics_client_id: int) -> Pose:
    """Get the pose of a body."""
    pybullet_pose = p.getBasePositionAndOrientation(
        body, physicsClientId=physics_client_id
    )
    return Pose(pybullet_pose[0], pybullet_pose[1])


def interpolate_quats(
    q1: Quaternion,
    q2: Quaternion,
    num_interp: int = 10,
    include_start: bool = True,
) -> Iterator[Quaternion]:
    """Interpolate quaternions using slerp."""
    slerp = Slerp([0, num_interp], ScipyRotation.from_quat([q1, q2]))
    time_start = 0 if include_start else 1
    times = list(range(time_start, num_interp + 1))
    for t in times:
        yield tuple(slerp(t).as_quat())


def interpolate_pose3ds(
    p1: Pose3D,
    p2: Pose3D,
    num_interp: int = 10,
    include_start: bool = True,
) -> Iterator[Pose3D]:
    """Interpolate positions."""
    time_start = 0 if include_start else 1
    times = list(range(time_start, num_interp + 1))
    positions = np.linspace(p1, p2, num=(num_interp + 1), endpoint=True)
    for t in times:
        yield tuple(positions[t])


def interpolate_poses(
    p1: Pose,
    p2: Pose,
    num_interp: int = 10,
    include_start: bool = True,
) -> Iterator[Pose]:
    """Interpolate between two poses in pose space."""
    # Determine the number of interpolation steps.
    pose3d_gen = interpolate_pose3ds(
        p1.position, p2.position, num_interp=num_interp, include_start=include_start
    )
    quat_gen = interpolate_quats(
        p1.orientation,
        p2.orientation,
        num_interp=num_interp,
        include_start=include_start,
    )
    for position, orientation in zip(pose3d_gen, quat_gen, strict=True):
        yield Pose(position, orientation)
