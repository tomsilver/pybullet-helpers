"""Tests for joint PyBullet helper utilities."""

import pybullet as p

from pybullet_helpers.joint import JointInfo, get_kinematic_chain
from pybullet_helpers.robots.fetch import FetchPyBulletRobot


def test_joint_info():
    """Tests for JointInfo()."""

    fixed_joint_info = JointInfo(
        jointIndex=0,
        jointName="fake-fixed-joint",
        jointType=p.JOINT_FIXED,
        qIndex=0,
        uIndex=0,
        flags=0,
        jointDamping=0.1,
        jointFriction=0.1,
        jointLowerLimit=0.0,
        jointUpperLimit=1.0,
        jointMaxForce=1.0,
        jointMaxVelocity=1.0,
        linkName="fake-link",
        jointAxis=(0.0, 0.0, 0.0),
        parentFramePos=(0.0, 0.0, 0.0),
        parentFrameOrn=(0.0, 0.0, 0.0, 1.0),
        parentIndex=-1,
    )

    assert fixed_joint_info.is_fixed
    assert not fixed_joint_info.is_circular
    assert not fixed_joint_info.is_movable
    assert not fixed_joint_info.violates_limit(0.5)
    assert fixed_joint_info.violates_limit(1.1)

    circular_joint_info = JointInfo(
        jointIndex=0,
        jointName="fake-circular-joint",
        jointType=p.JOINT_REVOLUTE,
        qIndex=0,
        uIndex=0,
        flags=0,
        jointDamping=0.1,
        jointFriction=0.1,
        jointLowerLimit=1.0,
        jointUpperLimit=0.0,
        jointMaxForce=1.0,
        jointMaxVelocity=1.0,
        linkName="fake-link",
        jointAxis=(0.0, 0.0, 0.0),
        parentFramePos=(0.0, 0.0, 0.0),
        parentFrameOrn=(0.0, 0.0, 0.0, 1.0),
        parentIndex=-1,
    )

    assert not circular_joint_info.is_fixed
    assert circular_joint_info.is_circular
    assert circular_joint_info.is_movable
    assert not circular_joint_info.violates_limit(9999.0)
    assert not circular_joint_info.violates_limit(0.0)


def test_get_kinematic_chain(physics_client_id):
    """Tests for get_kinematic_chain()."""
    robot = FetchPyBulletRobot(physics_client_id)
    arm_joints = get_kinematic_chain(
        robot.robot_id,
        robot.end_effector_id,
        physics_client_id=physics_client_id,
    )
    # Fetch arm has 7 DOF.
    assert len(arm_joints) == 7
