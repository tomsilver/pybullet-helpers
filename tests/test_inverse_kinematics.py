"""Tests for inverse_kinematics.py."""

import numpy as np
import pytest

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
)
from pybullet_helpers.robots.fetch import FetchPyBulletRobot
from pybullet_helpers.robots.panda import PandaPyBulletRobot


def test_pybullet_inverse_kinematics(physics_client_id):
    """Tests for pybullet_inverse_kinematics()."""
    robot = FetchPyBulletRobot(physics_client_id, control_mode="reset")
    assert robot.ikfast_info() is None
    initial_end_effector_pose = robot.get_end_effector_pose()

    # With validate = False, one call to IK is not good enough.
    target_position = np.add(initial_end_effector_pose.position, (0.1, 0.0, 0.0))
    target_orientation = initial_end_effector_pose.orientation
    target_pose = Pose(target_position, target_orientation)
    inverse_kinematics(robot, target_pose, validate=False, set_joints=True)
    recovered_end_effector_pose = robot.get_end_effector_pose()
    assert not np.allclose(
        recovered_end_effector_pose.position, target_position, atol=1e-3
    )
    # With validate = True, IK does work.
    robot.go_home()
    inverse_kinematics(robot, target_pose, validate=True, set_joints=True)
    recovered_end_effector_pose = robot.get_end_effector_pose()
    assert np.allclose(recovered_end_effector_pose.position, target_position, atol=1e-3)
    # With validate = True, if the position is impossible to reach, an error
    # is raised.
    target_position = (
        target_position[0],
        target_position[1],
        target_position[2] + 100.0,
    )
    target_pose = Pose(target_position, target_orientation)
    robot.go_home()
    with pytest.raises(Exception) as e:
        inverse_kinematics(robot, target_pose, validate=True, set_joints=True)
    assert "Inverse kinematics failed to converge." in str(e)


def test_ikfast_inverse_kinematics(physics_client_id):
    """Tests for IKFast inverse kinematics."""
    robot = PandaPyBulletRobot(physics_client_id, control_mode="reset")
    assert robot.ikfast_info() is not None
    initial_end_effector_pose = robot.get_end_effector_pose()

    # Try a few different poses, all should work fine.
    transforms = [
        Pose((0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        Pose((0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 1.0)),
        Pose((0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0)),
        Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        Pose((0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        Pose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ]
    for tf in transforms:
        target_pose = multiply_poses(initial_end_effector_pose, tf)
        robot.go_home()
        inverse_kinematics(robot, target_pose, validate=False, set_joints=True)
        recovered_end_effector_pose = robot.get_end_effector_pose()
        assert np.allclose(
            recovered_end_effector_pose.position, target_pose.position, atol=1e-3
        )
        assert np.allclose(
            recovered_end_effector_pose.orientation, target_pose.orientation, atol=1e-3
        )
    # With validate = True, if the position is impossible to reach, an error
    # is raised.
    impossible_tf = Pose((0.0, 0.0, 100.0), (0.0, 0.0, 0.0, 1.0))
    impossible_target = multiply_poses(initial_end_effector_pose, impossible_tf)
    robot.go_home()
    with pytest.raises(Exception) as e:
        inverse_kinematics(robot, impossible_target, validate=True, set_joints=True)
    assert "No IK solution found" in str(e)