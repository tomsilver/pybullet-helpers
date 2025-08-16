"""Tests for spot.py."""

import numpy as np

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
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

    # Tests for IK. These values were found through the joint visualizer script.
    # NOTE: we will probably update these values soon if we change the EE link.
    retract_ee_pose = Pose(
        position=(0.35817310214042664, 6.862916052341461e-06, 0.9498429298400879),
        orientation=(
            -0.00015761292888782918,
            0.0087896678596735,
            -0.009454049170017242,
            0.999916672706604,
        ),
    )
    raised_ee_pose = Pose(
        position=(0.422219455242157, 0.0025123022496700287, 1.0768202543258667),
        orientation=(
            0.012197185307741165,
            0.030934520065784454,
            -0.14907421171665192,
            0.9882667660713196,
        ),
    )
    right_ee_pose = Pose(
        position=(0.43226855993270874, -0.1964583396911621, 1.114035725593567),
        orientation=(
            0.04372112825512886,
            0.1726122498512268,
            -0.39378824830055237,
            0.9017894864082336,
        ),
    )

    for name, pose in [
        ("retract", retract_ee_pose),
        ("raised", raised_ee_pose),
        ("right", right_ee_pose),
    ]:
        joint_positions = inverse_kinematics(
            robot, end_effector_pose=pose, validate=True
        )
        robot.set_joints(joint_positions)
        recovered_pose = robot.forward_kinematics(joint_positions)
        # IK is analytic, so this should be extremely close.
        assert np.allclose(
            recovered_pose.position, pose.position, atol=1e-6
        ), f"IK failed for {name}"
