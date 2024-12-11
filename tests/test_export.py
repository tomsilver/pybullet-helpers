"""Tests for export.py."""

import tempfile
import pybullet as p
from pybullet_helpers.export import create_urdf_from_body_id
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.geometry import get_half_extents_from_aabb, get_pose, set_pose, Pose
import numpy as np


def test_write_urdf_from_body_id():
    """Tests for write_urdf_from_body_id()."""

    physics_client_id = p.connect(p.DIRECT)

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection()

    # # Test for a block.
    # block_id = create_pybullet_block(
    #     color=(0.1, 0.5, 0.9, 1.0),
    #     half_extents=(1.0, 2.0, 0.5),
    #     physics_client_id=physics_client_id,
    #     mass=1.0,
    #     friction=0.25)
    # set_pose(block_id, Pose.identity(), physics_client_id)
    # original_pose = get_pose(block_id, physics_client_id)
    # original_half_extents = get_half_extents_from_aabb(block_id, physics_client_id)
    # original_mass = p.getDynamicsInfo(block_id, -1, physicsClientId=physics_client_id)[0]

    # urdf = create_urdf_from_body_id(block_id, physics_client_id)
    
    # urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    # with open(urdf_file, mode="w", encoding="utf-8") as f:
    #     f.write(urdf)

    # p.removeBody(block_id, physicsClientId=physics_client_id)

    # # Recreate and compare.
    # recreated_block_id = p.loadURDF(
    #     urdf_file,
    #     (0, 0, 0),
    #     (0, 0, 0, 1),
    #     physicsClientId=physics_client_id
    # )
    # recovered_pose = get_pose(recreated_block_id, physics_client_id)
    # recovered_half_extents = get_half_extents_from_aabb(recreated_block_id, physics_client_id)
    # recovered_mass = p.getDynamicsInfo(recreated_block_id, -1, physicsClientId=physics_client_id)[0]

    # assert original_pose.allclose(recovered_pose, atol=1e-6)
    # # NOTE: bounding box may be slightly enlarged in loading URDF, probably
    # # because of some conservative thing that pybullet is doing.
    # assert np.allclose(
    #     original_half_extents,
    #     recovered_half_extents,
    #     atol=1e-2
    # )
    # assert np.isclose(original_mass, recovered_mass)

    # # Test for a cylinder.
    # cylinder_id = create_pybullet_cylinder(
    #     color=(0.1, 0.5, 0.9, 1.0),
    #     radius=0.2,
    #     length=0.8,
    #     physics_client_id=physics_client_id,
    #     mass=1.0,
    #     friction=0.25)
    # set_pose(cylinder_id, Pose.identity(), physics_client_id)
    # original_pose = get_pose(cylinder_id, physics_client_id)
    # original_half_extents = get_half_extents_from_aabb(cylinder_id, physics_client_id)
    # original_mass = p.getDynamicsInfo(cylinder_id, -1, physicsClientId=physics_client_id)[0]

    # urdf = create_urdf_from_body_id(cylinder_id, physics_client_id)
    # urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    # with open(urdf_file, mode="w", encoding="utf-8") as f:
    #     f.write(urdf)

    # p.removeBody(cylinder_id, physicsClientId=physics_client_id)

    # # Recreate and compare.
    # recreated_cylinder_id = p.loadURDF(
    #     urdf_file,
    #     (0, 0, 0),
    #     (0, 0, 0, 1),
    #     physicsClientId=physics_client_id
    # )
    # recovered_pose = get_pose(recreated_cylinder_id, physics_client_id)
    # recovered_half_extents = get_half_extents_from_aabb(recreated_cylinder_id, physics_client_id)
    # recovered_mass = p.getDynamicsInfo(recreated_cylinder_id, -1, physicsClientId=physics_client_id)[0]

    # assert original_pose.allclose(recovered_pose, atol=1e-6)
    # # NOTE: bounding box may be slightly enlarged in loading URDF, probably
    # # because of some conservative thing that pybullet is doing.
    # assert np.allclose(
    #     original_half_extents,
    #     recovered_half_extents,
    #     atol=1e-2
    # )
    # assert np.isclose(original_mass, recovered_mass)

    # Test for a two-link arm (joints but no meshes).
    robot = create_pybullet_robot("two-link", physics_client_id)
    original_pose = get_pose(robot.robot_id, physics_client_id)

    urdf = create_urdf_from_body_id(robot.robot_id, physics_client_id)
    # urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    urdf_file = "test.urdf"
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(robot.robot_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_robot_id = p.loadURDF(
        urdf_file,
        (0, 0, 0),
        (0, 0, 0, 1),
        physicsClientId=physics_client_id
    )
    recovered_pose = get_pose(recreated_robot_id, physics_client_id)
