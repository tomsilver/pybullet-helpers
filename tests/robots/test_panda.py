"""Tests for PandaPyBullet Robot."""

from unittest.mock import patch

import numpy as np
import pytest

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.joint import get_joint_infos, get_joints
from pybullet_helpers.robots.panda import PandaPyBulletRobot


@pytest.fixture(scope="function", name="panda")
def _panda_fixture(physics_client_id) -> PandaPyBulletRobot:
    """Get a PandaPyBulletRobot instance."""
    # Use reset control, so we can see effects of actions without stepping.
    panda = PandaPyBulletRobot(physics_client_id, control_mode="reset")
    assert panda.get_name() == "panda"
    assert panda.physics_client_id == physics_client_id
    # Panda must have IKFast
    assert panda.ikfast_info() is not None

    return panda


def test_panda_pybullet_robot_initial_configuration(panda):
    """Check initial configuration matches expected position."""
    # Check get_state
    assert isinstance(panda, PandaPyBulletRobot)
    pose = panda.get_end_effector_pose()
    assert np.allclose(pose.position, (0.5, 0.0, 0.5), atol=1e-3)
    finger_state = panda.get_finger_state()
    assert np.isclose(finger_state, panda.open_fingers_state)


def test_panda_pybullet_robot_links(panda):
    """Test link utilities on PandaPyBulletRobot."""
    # Tool link is last link in Panda URDF
    num_links = len(panda.joint_infos)
    assert panda.tool_link_id == num_links - 1
    assert panda.tool_link_name == "tool_link"

    # Check base link
    assert panda.base_link_name == "panda_link0"

    with pytest.raises(ValueError):
        # Non-existent link
        panda.link_from_name("non_existent_link")


def test_panda_pybullet_robot_joints(panda):
    """Test joint utilities on PandaPyBulletRobot."""
    # Check joint limits match action space
    assert np.allclose(panda.action_space.low, panda.joint_lower_limits)
    assert np.allclose(panda.action_space.high, panda.joint_upper_limits)

    # Check joint infos match expected
    panda_joints = get_joints(panda.robot_id, panda.physics_client_id)
    assert panda.joint_infos == get_joint_infos(
        panda.robot_id, panda_joints, panda.physics_client_id
    )

    # Check getting joints
    assert panda.joint_info_from_name("panda_joint5").jointName == "panda_joint5"
    assert (
        panda.joint_from_name("panda_joint5")
        == panda.joint_info_from_name("panda_joint5").jointIndex
    )

    # Check Panda joints - 7 joints for arm + 2 fingers
    assert panda.arm_joints == [0, 1, 2, 3, 4, 5, 6, 9, 10]

    with pytest.raises(ValueError):
        panda.joint_from_name("non_existent_joint")
    with pytest.raises(ValueError):
        panda.joint_info_from_name("non_existent_joint")


def test_panda_movable_base_inverse_kinematics(physics_client_id):
    """Test IK when panda base can move."""
    # Need to create a separate movable base robot for this test.
    panda = PandaPyBulletRobot(
        physics_client_id, control_mode="reset", fixed_base=False
    )
    # Set the robot base to be very far from default.
    panda.set_base(Pose((100, 100, 0)))
    # Run IK for a pose that should be reachable.
    pose = Pose((100.25, 100.25, 0.25), (0.7071, 0.7071, 0.0, 0.0))
    joint_positions = inverse_kinematics(panda, end_effector_pose=pose, validate=True)
    recovered_pose = panda.forward_kinematics(joint_positions)
    assert np.allclose(recovered_pose.position, pose.position)


def test_panda_pybullet_robot_inverse_kinematics_no_solutions(panda):
    """Test when IKFast returns no solutions."""
    # Impossible target pose with no solutions
    pose = Pose((999.0, 99.0, 999.0), (0.7071, 0.7071, 0.0, 0.0))
    with pytest.raises(ValueError):
        inverse_kinematics(panda, end_effector_pose=pose, validate=True)


def test_panda_pybullet_robot_inverse_kinematics_incorrect_solution(panda):
    """Test when IKFast returns an incorrect solution.

    Note that this doesn't happen in reality, but we need to check we
    validate correctly).
    """
    pose = Pose((0.25, 0.25, 0.25), (0.7071, 0.7071, 0.0, 0.0))
    # Note: the ikfast_closest_inverse_kinematics import happens
    # in the single_arm.py module, not the panda.py module.
    with patch(
        "pybullet_helpers.inverse_kinematics.ikfast_closest_inverse_kinematics"
    ) as ikfast_mock:
        # Patch return value of IKFast to be an incorrect solution
        ikfast_mock.return_value = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

        # If validate=False, error shouldn't be raised
        inverse_kinematics(panda, end_effector_pose=pose, validate=False)

        # If validate=True, error should be raised as solution doesn't match
        # desired end effector pose
        with pytest.raises(ValueError):
            inverse_kinematics(panda, end_effector_pose=pose, validate=True)


def test_panda_pybullet_robot_inverse_kinematics(panda):
    """Test IKFast normal functionality on PandaPyBulletRobot."""
    pose = Pose((0.25, 0.25, 0.25), (0.7071, 0.7071, 0.0, 0.0))
    joint_positions = inverse_kinematics(panda, end_effector_pose=pose, validate=True)
    recovered_pose = panda.forward_kinematics(joint_positions)
    assert np.allclose(recovered_pose.position, pose.position)
