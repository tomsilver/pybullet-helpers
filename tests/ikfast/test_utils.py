"""Test for IKFast utilities module."""

from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest

from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast.utils import (
    IKFastHyperparameters,
    get_base_from_ee,
    get_ikfast_joints,
    get_jointwise_difference,
    get_ordered_ancestors,
    ikfast_closest_inverse_kinematics,
    ikfast_inverse_kinematics,
    violates_joint_limits,
)


@pytest.fixture(scope="module", name="robot_with_no_ikfast_info")
def _robot_with_no_ikfast_info_fixture():
    robot = Mock()
    robot.ikfast_info.return_value = None
    return robot


def test_get_jointwise_difference():
    """Test for get_jointwise_difference."""
    joint_infos = 7 * [Mock(is_circular=False)]
    joint_vals1 = np.random.rand(7)
    joint_vals2 = np.random.rand(7)
    expected_difference = joint_vals1 - joint_vals2
    difference_fn = partial(get_jointwise_difference, joint_infos)
    assert np.allclose(
        difference_fn(list(joint_vals1), list(joint_vals2)), expected_difference
    )

    # Joint values are different lengths
    with pytest.raises(ValueError):
        difference_fn(list(joint_vals1), list(joint_vals2[:5]))

    # Joint values are same length but different to number of joint infos
    with pytest.raises(ValueError):
        difference_fn([1, 2, 3], [4, 5, 6])


def test_violates_joint_limits_raises_error():
    """Test violated_joint_limits raises error if length of joint positions
    doesn't match number of joints."""
    joint_infos = 7 * [Mock()]
    with pytest.raises(ValueError):
        violates_joint_limits(joint_infos, [1, 2, 3, 4, 5, 6])


def test_violates_joint_limits():
    """Test for violated_joint_limits."""

    def _violates_limits(val):
        # Anything negative violates limits in our test case
        return val < 0

    joint_info = Mock()
    joint_info.violates_limit = _violates_limits
    joint_infos = 7 * [joint_info]

    # All positive values don't violate limits
    assert not violates_joint_limits(joint_infos, [1, 2, 3, 4, 5, 6, 7])

    # Some negative values violate limits
    assert violates_joint_limits(joint_infos, [-1, 2, 3, 4, 5, 6, 7])

    # All negative values violate limits
    assert violates_joint_limits(joint_infos, [-1, -2, -3, -4, -5, -6, -7])


def test_get_ordered_ancestors_raises_error(robot_with_no_ikfast_info):
    """Test get_ordered_ancestors raises error if no IKFastInfo in robot."""
    with pytest.raises(ValueError):
        get_ordered_ancestors(robot_with_no_ikfast_info, 1)


def test_get_base_from_ee_raises_error(robot_with_no_ikfast_info):
    """Test get_base_from_ee raises error if no IKFastInfo in robot."""
    with pytest.raises(ValueError):
        get_base_from_ee(robot_with_no_ikfast_info, Pose((0, 0, 0)))


def test_get_ikfast_joints_raises_error(robot_with_no_ikfast_info):
    """Test get_ikfast_joints raises error if no IKFastInfo in robot."""
    with pytest.raises(ValueError):
        get_ikfast_joints(robot_with_no_ikfast_info)


def test_ikfast_inverse_kinematics_raises_error(robot_with_no_ikfast_info):
    """Test ikfast_inverse_kinematics raises error if no IKFastInfo in
    robot."""
    with pytest.raises(ValueError):
        rng = np.random.default_rng(123)
        generator = ikfast_inverse_kinematics(
            robot_with_no_ikfast_info,
            Pose((0, 0, 0)),
            max_time=1.0,
            max_distance=999.9,
            max_attempts=100,
            norm=2,
            rng=rng,
        )
        list(generator)


def test_ikfast_closest_inverse_kinematics_raises_error():
    """Test ikfast_closest_inverse_kinematics raises error if max time,
    candidates or attempts is infinite."""
    hyperparameters = IKFastHyperparameters(
        max_time=np.inf, max_attempts=np.inf, max_candidates=np.inf
    )
    with pytest.raises(ValueError):
        ikfast_closest_inverse_kinematics(
            Mock(), Pose.identity(), hyperparameters=hyperparameters
        )
