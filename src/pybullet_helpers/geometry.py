"""PyBullet helper class for geometry utilities."""

from __future__ import annotations

from typing import Iterator, NamedTuple
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Pose:
    """A position and orientation in a frame."""

    # All poses exist in some frame.
    frame: Frame

    # Cartesian (x, y, z) position.
    position: Pose3D

    # Quaternion in (x, y, z, w) representation.
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Frame:
    """A position and axes in SO(3)."""

    # The name of the frame.
    name: str
    
    # Cartesian (x, y, z) position in world.
    origin: Pose3D
    
    # Orthogonal unit vectors.
    x_axis: Pose3D
    y_axis: Pose3D
    z_axis: Pose3D

    def __post_init__(self) -> None:
        # Verify that axes are orthogonal unit vectors.
        assert np.isclose(np.linalg.norm(self.x_axis), 1.0)
        assert np.isclose(np.linalg.norm(self.y_axis), 1.0)
        assert np.isclose(np.linalg.norm(self.z_axis), 1.0)
        assert np.isclose(np.dot(self.x_axis, self.y_axis), 0.0)
        assert np.isclose(np.dot(self.x_axis, self.z_axis), 0.0)
        assert np.isclose(np.dot(self.y_axis, self.z_axis), 0.0)
    

@dataclass(frozen=True)
class Transform:
    """Transform between frames."""

    source: Frame
    target: Frame
    _source_to_target_translation: Pose3D = field(init=False)
    _source_to_target_rotation: Quaternion = field(init=False)

    def __post_init__(self) -> None:
        # Compute translation from source to target.
        self._source_to_target_translation = (
            self.target.origin[0] - self.source.origin[0],
            self.target.origin[1] - self.source.origin[1],
            self.target.origin[2] - self.source.origin[2]
        )
        # Compute rotation from source to target.
        source_rotation_matrix = np.column_stack([self.source.x_axis, self.source.y_axis, self.source.z_axis])
        target_rotation_matrix = np.column_stack([self.target.x_axis, self.target.y_axis, self.target.z_axis])
        relative_rotation_matrix = target_rotation_matrix @ np.linalg.inv(source_rotation_matrix)
        relative_rotation = ScipyRotation.from_matrix(relative_rotation_matrix).as_quat()
        self._source_to_target_rotation = tuple(relative_rotation)

    def transform(self, pose: Pose) -> Pose:
        """Transform a pose in the source frame to the target frame."""
        assert pose.frame == self.source
        pybullet_pose = p.multiplyTransforms(
            pose.position,
            pose.orientation,
            self._source_to_target_translation,
            self._source_to_target_rotation
        )
        return Pose(self.target, pybullet_pose[0], pybullet_pose[1])


class TFUtil:
    """Utilities for transforms."""
    def __init__(self) -> None:
        self._name_to_frame: dict[str, Frame] = {}

    def get_transform(self, from_frame: str, to_frame: str) -> Transform:
        """Get transform from one frame to another."""
        return Transform(self._name_to_frame[from_frame], self._name_to_frame[to_frame])

    def get_pose_in_frame(self, pose: Pose, new_frame: str) -> Pose:
        """Transform a pose into a new frame."""
        old_to_new = self.get_transform(pose.frame, new_frame)
        return old_to_new.transform(pose)

    def add(self, frame: Frame) -> None:
        """Add or change a frame."""
        self._name_to_frame[frame.name] = frame

    def update(self, frames: list[Frame]) -> None:
        """Add or change multiple frames."""
        for frame in frames:
            self.add(frame)


def orientations_allclose(
    quat1: Quaternion, quat2: Quaternion, atol: float = 1e-6
) -> bool:
    """Check whether two quaternion orientations are close, accounting for
    double coverage.

    Note that this should not be used to check if two rotations are
    equal, e.g., in the context of slerp.

    For example, see
    https://gamedev.stackexchange.com/questions/75072/.
    """
    return np.allclose(quat1, quat2, atol=atol) or np.allclose(
        quat1, -1 * np.array(quat2), atol=atol
    )


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
    return Pose(pybullet_pose[0], pybullet_pose[1], frame="world")


def set_pose(body: int, pose: Pose, physics_client_id: int) -> None:
    """Set the pose of a body."""
    assert pose.frame == "world"
    p.resetBasePositionAndOrientation(
        body,
        pose.position,
        pose.orientation,
        physicsClientId=physics_client_id,
    )


def iter_between_quats(
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


def iter_between_pose3ds(
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


def iter_between_poses(
    p1: Pose,
    p2: Pose,
    num_interp: int = 10,
    include_start: bool = True,
) -> Iterator[Pose]:
    """Interpolate between two poses in pose space."""
    # Determine the number of interpolation steps.
    assert p1.frame == p2.frame
    frame = p1.frame
    pose3d_gen = iter_between_pose3ds(
        p1.position, p2.position, num_interp=num_interp, include_start=include_start
    )
    quat_gen = iter_between_quats(
        p1.orientation,
        p2.orientation,
        num_interp=num_interp,
        include_start=include_start,
    )
    for position, orientation in zip(pose3d_gen, quat_gen, strict=True):
        yield Pose(position, orientation, frame=frame)


def interpolate_quats(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """Interpolate between q1 and q2 given 0 <= t <= 1."""
    assert 0 <= t <= 1
    slerp = Slerp([0, 1], ScipyRotation.from_quat([q1, q2]))
    return tuple(slerp(t).as_quat())


def interpolate_pose3d(p1: Pose3D, p2: Pose3D, t: float) -> Pose3D:
    """Interpolate between p1 and p2 given 0 <= t <= 1."""
    dists_arr = np.subtract(p2, p1)
    return tuple(np.add(p1, t * dists_arr))


def interpolate_poses(
    p1: Pose,
    p2: Pose,
    t: float,
) -> Pose:
    """Interpolate between p1 and p2 given 0 <= t <= 1."""
    assert p1.frame == p2.frame
    frame = p1.frame
    assert 0 <= t <= 1
    position = interpolate_pose3d(p1.position, p2.position, t)
    quat = interpolate_quats(p1.orientation, p2.orientation, t)
    return Pose(position, quat, frame=frame)


def get_half_extents_from_aabb(
    body_id: int,
    physics_client_id: int,
    link_id: int | None = None,
) -> tuple[float, float, float]:
    """Get box half extents based on AABB."""
    if link_id is None:
        aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=physics_client_id)
    else:
        aabb_min, aabb_max = p.getAABB(
            body_id, linkIndex=link_id, physicsClientId=physics_client_id
        )
    return (
        (aabb_max[0] - aabb_min[0]) / 2,
        (aabb_max[1] - aabb_min[1]) / 2,
        (aabb_max[2] - aabb_min[2]) / 2,
    )
