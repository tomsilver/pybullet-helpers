"""Tests for Kinova robots."""

from pybullet_helpers.robots.stretch import StretchPyBulletRobot


def test_stretch_pybullet_robot(physics_client_id):
    """Tests for StretchPyBulletRobot()."""
    robot = StretchPyBulletRobot(physics_client_id)
    assert robot.get_name() == "stretch"
    assert not robot.fixed_base
    assert len(robot.default_home_joint_positions) == 10
