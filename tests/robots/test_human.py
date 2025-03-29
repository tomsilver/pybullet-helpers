"""Tests for Human()."""

from pybullet_helpers.robots.human import Human
from pybullet_helpers.geometry import Pose
import numpy as np

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

    import pybullet as p
    while True:
        p.getMouseEvents(human.physics_client_id)
