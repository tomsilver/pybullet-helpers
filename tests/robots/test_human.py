"""Tests for Human()."""

from pybullet_helpers.robots.human import Human
from pybullet_helpers.geometry import Pose
import numpy as np
from pybullet_helpers.math_utils import sample_within_sphere


def test_human(physics_client_id):
    """Tests for Human()."""

    # TODO remove
    from pybullet_helpers.gui import create_gui_connection
    physics_client_id = create_gui_connection()

    right_leg_kwargs = {
        "home_joint_positions": [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    left_leg_kwargs = {
        "home_joint_positions": [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    base_pose = Pose.from_rpy((0.1, 0.0, 0.0), (0.0, 0.0, np.pi / 2))
    human = Human(physics_client_id, base_pose=base_pose, right_leg_kwargs=right_leg_kwargs,
                  left_leg_kwargs=left_leg_kwargs)
    
    # Test IK for the right arm.
    right_arm = human.right_arm

    # We want it to be possible to reach positions in a small sphere around
    # the end effector.
    resting_pose = right_arm.get_end_effector_pose()
    center = resting_pose.position
    orientation = resting_pose.orientation
    radius = 0.001
    rng = np.random.default_rng(123)
    for _ in range(10):
        position = sample_within_sphere(center, 0.0, radius, rng)
        pose = Pose(position, orientation)

        # TODO remove
        from pybullet_helpers.gui import visualize_pose
        visualize_pose(pose, physics_client_id)


    import pybullet as p
    while True:
        p.getMouseEvents(human.physics_client_id)
