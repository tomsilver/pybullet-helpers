"""Tests for math_utils.py."""

from pybullet_helpers.math_utils import get_poses_facing_axis


def test_get_poses_facing_axis():
    """Tests for get_poses_facing_axis()."""

    axis = (0.0, 0.0, 1.0)
    radius = 1.0
    poses = get_poses_facing_axis(axis, radius, num_points=4)

    # Uncomment to debug.
    # import pybullet as p
    # from pybullet_helpers.gui import create_gui_connection, visualize_pose

    # physics_client_id = create_gui_connection()
    # p.addUserDebugLine(
    #     lineFromXYZ=(0.0, 0.0, 0.0),
    #     lineToXYZ=axis,
    #     lineColorRGB=(1.0, 1.0, 1.0),
    #     lifeTime=0,
    #     physicsClientId=physics_client_id,
    # )
    # for pose in poses:
    #     visualize_pose(pose, physics_client_id)
    # while True:
    #     p.stepSimulation(physicsClientId=physics_client_id)
