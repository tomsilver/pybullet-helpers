"""Tests for Stretch robot."""

import numpy as np

from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.robots.stretch import StretchPyBulletRobot
from pybullet_helpers.geometry import Pose


def test_stretch_pybullet_robot(physics_client_id):
    """Tests for StretchPyBulletRobot()."""
    robot = StretchPyBulletRobot(physics_client_id)
    assert robot.get_name() == "stretch"
    assert not robot.fixed_base
    assert len(robot.default_home_joint_positions) == 10

    # Test inverse kinematics.
    current_ee_pose = robot.get_end_effector_pose()
    target_ee_pose = Pose(np.add(current_ee_pose.position, (0, 0, -0.3)), current_ee_pose.orientation)
    joint_positions = inverse_kinematics(robot, target_ee_pose)
    assert np.allclose(joint_positions, robot.get_joint_positions())
