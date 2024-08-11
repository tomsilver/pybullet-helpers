"""Tests for math_utils.py."""

from pybullet_helpers.geometry import Pose
from pybullet_helpers.math_utils import get_poses_facing_line


def test_get_poses_facing_line():
    """Tests for get_poses_facing_line()."""

    axis = (0.0, 0.0, 1.0)
    point_on_line = (0.0, 0.0, 0.0)
    radius = 1.0
    poses = get_poses_facing_line(axis, point_on_line, radius, num_points=4)
    expected = [
        Pose(position=(1.0, 0.0, 0.0), orientation=(0.5, 0.5, -0.5, -0.5)),
        Pose(position=(0.0, 1.0, 0.0), orientation=(0.0, 0.7071, -0.7071, 0.0)),
        Pose(position=(-1.0, 0.0, 0.0), orientation=(-0.5, 0.5, -0.5, 0.5)),
        Pose(position=(0.0, -1.0, -0.0), orientation=(-0.7071, 0.0, 0.0, 0.7071)),
    ]
    for p, e in zip(poses, expected, strict=True):
        assert p.allclose(e)

    axis = (0.0, 0.0, 1.0)
    point_on_line = (10.0, 0.0, 0.0)
    radius = 2.0
    poses = get_poses_facing_line(axis, point_on_line, radius, num_points=4)
    expected = [
        Pose(position=(12.0, 0.0, 0.0), orientation=(0.5, 0.5, -0.5, -0.5)),
        Pose(position=(10.0, 2.0, 0.0), orientation=(0.0, 0.7071, -0.7071, 0.0)),
        Pose(position=(8.0, 0.0, 0.0), orientation=(-0.5, 0.5, -0.5, 0.5)),
        Pose(position=(10.0, -2.0, -0.0), orientation=(-0.7071, 0.0, 0.0, 0.7071)),
    ]
    for p, e in zip(poses, expected, strict=True):
        assert p.allclose(e)

    axis = (1.0, 0.0, 0.0)
    point_on_line = (0.0, 0.0, 0.0)
    radius = 1.0
    poses = get_poses_facing_line(axis, point_on_line, radius, num_points=4)
    expected = [
        Pose(position=(0.0, -1.0, 0.0), orientation=(0.5, -0.5, -0.5, -0.5)),
        Pose(position=(0.0, 0.0, -1.0), orientation=(0.0, 0.0, 0.707107, 0.707107)),
        Pose(position=(0.0, 1.0, 0.0), orientation=(0.5, -0.5, 0.5, 0.5)),
        Pose(position=(0.0, 0.0, 1.0), orientation=(0.707107, -0.707107, 0.0, 0.0)),
    ]
    for p, e in zip(poses, expected, strict=True):
        assert p.allclose(e)

    # Uncomment to debug.
    # import pybullet as p
    # import numpy as np
    # from pybullet_helpers.gui import create_gui_connection, visualize_pose

    # physics_client_id = create_gui_connection(camera_target=point_on_line)
    # p.addUserDebugLine(
    #     lineFromXYZ=point_on_line,
    #     lineToXYZ=np.add(point_on_line, axis),
    #     lineColorRGB=(1.0, 1.0, 1.0),
    #     lifeTime=0,
    #     physicsClientId=physics_client_id,
    # )
    # for pose in poses:
    #     visualize_pose(pose, physics_client_id)
    # while True:
    #     p.stepSimulation(physicsClientId=physics_client_id)
