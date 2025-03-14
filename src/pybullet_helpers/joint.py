"""PyBullet helper class for joint utilities."""

from functools import partial
from typing import Iterator, NamedTuple, Sequence

import numpy as np
import pybullet as p
from tomsutils.utils import get_signed_angle_distance, wrap_angle

from pybullet_helpers.geometry import Pose3D, Quaternion

# Joint Positions (i.e., angles) of each joint in the body.
# Not to be conflated with PyBullet joint states which include
# positions, velocities and forces.
JointPositions = list[float]
JointVelocities = list[float]


class JointInfo(NamedTuple):
    """Joint Information to match the output of the PyBullet getJointInfo API.

    We use a NamedTuple as it supports retrieving by integer indexing
    and most closely follows the PyBullet API.
    """

    jointIndex: int
    jointName: str
    jointType: int
    qIndex: int
    uIndex: int
    flags: int
    jointDamping: float
    jointFriction: float
    jointLowerLimit: float
    jointUpperLimit: float
    jointMaxForce: float
    jointMaxVelocity: float
    linkName: str
    jointAxis: Sequence[float]
    parentFramePos: Pose3D
    parentFrameOrn: Quaternion
    parentIndex: int

    @property
    def is_circular(self) -> bool:
        """Whether the joint is circular or not."""
        if self.is_fixed:
            return False
        # For continuous/circular joints, the PyBullet parser will give
        # us an upper limit (-1.0) that is lower than the lower limit (0.0).
        return self.jointUpperLimit < self.jointLowerLimit

    @property
    def is_movable(self) -> bool:
        """Whether the joint is movable or not."""
        return not self.is_fixed

    @property
    def is_fixed(self) -> bool:
        """Whether the joint is fixed or not."""
        return self.jointType == p.JOINT_FIXED

    def violates_limit(self, value: float, tol: float = 0.1) -> bool:
        """Whether the given value violates the joint's limits."""
        if self.is_circular:
            return False
        return self.jointLowerLimit > value - tol or value + tol > self.jointUpperLimit


class JointState(NamedTuple):
    """Joint Information to match the output of the PyBullet getJointState API.

    We use a NamedTuple as it supports retrieving by integer indexing
    and most closely follows the PyBullet API.
    """

    jointPosition: float
    jointVelocity: float
    jointReactionForces: Sequence[float]
    appliedJointMotorTorque: float


def get_num_joints(body: int, physics_client_id: int) -> int:
    """Get the number of joints for a body."""
    return p.getNumJoints(body, physicsClientId=physics_client_id)


def get_joints(body: int, physics_client_id: int) -> list[int]:
    """Get joint indices for a body."""
    return list(range(get_num_joints(body, physics_client_id)))


def get_joint_info(body: int, joint: int, physics_client_id: int) -> JointInfo:
    """Get the info for the given joint for a body."""
    raw_joint_info: list = list(
        p.getJointInfo(body, joint, physicsClientId=physics_client_id)
    )
    # Decode the byte strings for joint name and link name
    raw_joint_info[1] = raw_joint_info[1].decode("UTF-8")
    raw_joint_info[12] = raw_joint_info[12].decode("UTF-8")

    joint_info = JointInfo(*raw_joint_info)
    return joint_info


def get_joint_infos(
    body: int, joints: list[int], physics_client_id: int
) -> list[JointInfo]:
    """Get the infos for the given joints for a body."""
    return [get_joint_info(body, joint_id, physics_client_id) for joint_id in joints]


def get_joint_limits(
    body: int, joints: list[int], physics_client_id: int
) -> tuple[list[float], list[float]]:
    """Get the joint limits for the given joints for a body. Circular joints do
    not have limits (represented by ±np.inf).

    We return a Tuple where the first element is the list of lower
    limits, and the second element is the list of upper limits.
    """
    joint_infos = get_joint_infos(body, joints, physics_client_id)
    lower_limits = [
        joint.jointLowerLimit if not joint.is_circular else -np.inf
        for joint in joint_infos
    ]
    upper_limits = [
        joint.jointUpperLimit if not joint.is_circular else np.inf
        for joint in joint_infos
    ]
    return lower_limits, upper_limits


def get_joint_lower_limits(
    body: int, joints: list[int], physics_client_id: int
) -> list[float]:
    """Get the lower joint limits for the given joints for a body."""
    return get_joint_limits(body, joints, physics_client_id)[0]


def get_joint_upper_limits(
    body: int, joints: list[int], physics_client_id: int
) -> list[float]:
    """Get the upper joint limits for the given joints for a body."""
    return get_joint_limits(body, joints, physics_client_id)[1]


def get_kinematic_chain(
    body: int, end_effector: int, physics_client_id: int
) -> list[int]:
    """Get all the free joints from robot body base to end effector.

    Includes the end effector.
    """
    kinematic_chain = []
    while end_effector > -1:
        joint_info = get_joint_info(body, end_effector, physics_client_id)
        if joint_info.qIndex > -1:
            kinematic_chain.append(end_effector)
        end_effector = joint_info[-1]
    return kinematic_chain


def get_joint_states(
    body: int, joints: list[int], physics_client_id: int
) -> list[JointState]:
    """Get the joint states for the given joints for a body."""

    if len(joints) == 0:
        joint_states = []
    else:
        joint_states = [
            JointState(*raw_joint_state)
            for raw_joint_state in p.getJointStates(
                body, joints, physicsClientId=physics_client_id
            )
        ]

    return joint_states


def get_joint_positions(
    body: int, joints: list[int], physics_client_id: int
) -> JointPositions:
    """Get the joint positions for the given joints for a body."""
    return [
        joint_state.jointPosition
        for joint_state in get_joint_states(body, joints, physics_client_id)
    ]


def get_joint_velocities(
    body: int, joints: list[int], physics_client_id: int
) -> JointVelocities:
    """Get the joint velocities for the given joints for a body."""
    return [
        joint_state.jointVelocity
        for joint_state in get_joint_states(body, joints, physics_client_id)
    ]


def get_jointwise_difference(
    joint_infos: list[JointInfo],
    q2: JointPositions,
    q1: JointPositions,
) -> JointPositions:
    """Determine the difference between two joint positions.

    Returns a function that takes two joint positions and returns the
    difference between them.
    """

    if not len(q2) == len(q1) == len(joint_infos):
        raise ValueError("q2, q1, and joint infos must be the same length")
    diff: JointPositions = []
    for v1, v2, joint_info in zip(q1, q2, joint_infos):
        if joint_info.is_circular:
            joint_diff = get_signed_angle_distance(wrap_angle(v2), wrap_angle(v1))
        else:
            joint_diff = v2 - v1
        diff.append(joint_diff)
    return diff


def interpolate_joints(
    joint_infos: list[JointInfo],
    q1: JointPositions,
    q2: JointPositions,
    t: float,
) -> JointPositions:
    """Interpolate between q1 and q2 given 0 <= t <= 1."""
    assert 0 <= t <= 1.0
    dists_arr = np.array(get_jointwise_difference(joint_infos, q2, q1))
    q1_arr = np.array(q1)
    return list(q1_arr + t * dists_arr)


def iter_between_joint_positions(
    joint_infos: list[JointInfo],
    q1: JointPositions,
    q2: JointPositions,
    num_interp_per_unit: int = 10,
    include_start: bool = True,
) -> Iterator[JointPositions]:
    """Iterate points between two given joint positions."""
    # Determine the number of interpolation steps.
    interpolator = partial(interpolate_joints, joint_infos, q1, q2)
    dists_arr = np.array(np.array(get_jointwise_difference(joint_infos, q2, q1)))
    num = int(np.ceil(max(abs(dists_arr)))) * num_interp_per_unit + 1
    for t in np.linspace(0, 1, num=num, endpoint=True):
        if t == 0 and not include_start:
            continue
        yield interpolator(t)
