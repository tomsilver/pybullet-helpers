"""Tests for spot.py."""

import numpy as np

from pybullet_helpers.robots.spot import (
    SpotPyBulletRobot,
)


def test_spot_pybullet_robot(physics_client_id):
    """Tests for SpotPyBulletRobot()."""
    robot = SpotPyBulletRobot(
        physics_client_id,
    )
    assert robot.get_name() == "spot"
    assert robot.arm_joint_names == [
        "arm_sh0",
        "arm_sh1",
        "arm_el0",
        "arm_el1",
        "arm_wr0",
        "arm_wr1",
        "arm_f1x",
    ]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)
