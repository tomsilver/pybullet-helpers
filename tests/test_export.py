"""Tests for export.py."""

import tempfile
import pybullet as p
from pybullet_helpers.export import create_urdf_from_body_id
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder
from pybullet_helpers.geometry import get_half_extents_from_aabb, get_pose, set_pose, Pose
import numpy as np


def test_write_urdf_from_body_id():
    """Tests for write_urdf_from_body_id()."""

    physics_client_id = p.connect(p.GUI)

    # Test create / save / load for a block.
    # block_id = create_pybullet_block(
    #     color=(0.1, 0.5, 0.9, 1.0),
    #     half_extents=(1.0, 2.0, 0.5),
    #     physics_client_id=physics_client_id,
    #     mass=1.0,
    #     friction=0.25)
    # set_pose(block_id, Pose.identity(), physics_client_id)

    # urdf = create_urdf_from_body_id(block_id, physics_client_id)
    
    # urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    # with open(urdf_file, mode="w", encoding="utf-8") as f:
    #     f.write(urdf)

    # # Recreate and compare.
    # recreated_block_id = p.loadURDF(
    #     urdf_file,
    #     (0, 0, 0),
    #     (0, 0, 0, 1),
    #     physicsClientId=physics_client_id
    # )
    # original_pose = get_pose(block_id, physics_client_id)
    # recovered_pose = get_pose(recreated_block_id, physics_client_id)
    # assert original_pose.allclose(recovered_pose, atol=1e-6)
    # assert np.allclose(
    #     get_half_extents_from_aabb(block_id, physics_client_id),
    #     get_half_extents_from_aabb(recreated_block_id, physics_client_id),
    #     atol=1e-6
    # )

    # Test create / save / load for a cylinder.
    cylinder_id = create_pybullet_cylinder(
        color=(0.1, 0.5, 0.9, 1.0),
        radius=0.2,
        length=0.8,
        physics_client_id=physics_client_id,
        mass=1.0,
        friction=0.25)
    set_pose(cylinder_id, Pose.identity(), physics_client_id)

    urdf = create_urdf_from_body_id(cylinder_id, physics_client_id)
    
    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    # Recreate and compare.
    recreated_cylinder_id = p.loadURDF(
        urdf_file,
        (0, 0, 0),
        (0, 0, 0, 1),
        physicsClientId=physics_client_id
    )
    original_pose = get_pose(cylinder_id, physics_client_id)
    recovered_pose = get_pose(recreated_cylinder_id, physics_client_id)
    assert original_pose.allclose(recovered_pose, atol=1e-6)
    assert np.allclose(
        get_half_extents_from_aabb(cylinder_id, physics_client_id),
        get_half_extents_from_aabb(recreated_cylinder_id, physics_client_id),
        atol=1e-6
    )


    # import ipdb; ipdb.set_trace()
    # p.removeBody(cylinder_id, physicsClientId=physics_client_id)
    # import ipdb; ipdb.set_trace()

    # while True:
    #     p.stepSimulation(physics_client_id)
