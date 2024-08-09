"""Tests for PyBullet motion planning."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    run_motion_planning,
)
from pybullet_helpers.robots.fetch import FetchPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, get_assets_path

USE_GUI = False


def test_run_motion_planning(physics_client_id):
    """Tests for run_motion_planning()."""
    base_pose = Pose(position=(0.75, 0.7441, 0.0))
    seed = 123
    robot = FetchPyBulletRobot(physics_client_id, base_pose=base_pose)
    joint_initial = robot.get_joint_positions()
    # Should succeed with a path of length 2.
    joint_target = list(joint_initial)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies=set(),
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert len(path) == 2
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should succeed, no collisions.
    ee_pose = robot.get_end_effector_pose()
    ee_target_position = np.add(ee_pose.position, (0.0, 0.0, -0.05))
    ee_target = Pose(ee_target_position, ee_pose.orientation)
    joint_target = inverse_kinematics(robot, ee_target, validate=True)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies=set(),
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should fail because the target collides with the table.
    table_pose = (1.35, 0.75, 0.0)
    table_orientation = [0.0, 0.0, 0.0, 1.0]
    table_urdf_path = get_assets_path() / "urdf" / "table.urdf"
    table_id = p.loadURDF(
        str(table_urdf_path), useFixedBase=True, physicsClientId=physics_client_id
    )
    p.resetBasePositionAndOrientation(
        table_id, table_pose, table_orientation, physicsClientId=physics_client_id
    )
    ee_pose = robot.get_end_effector_pose()
    ee_target_position = np.add(ee_pose.position, (0.0, 0.0, -0.6))
    ee_target = Pose(ee_target_position, ee_pose.orientation)
    joint_target = inverse_kinematics(robot, ee_target, validate=True)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies={table_id},
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert path is None
    # Should fail because the initial state collides with the table.
    path = run_motion_planning(
        robot,
        joint_target,
        joint_initial,
        collision_bodies={table_id},
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert path is None
    # Should succeed, but will need to move the arm up to avoid the obstacle.
    block_pose = (1.35, 0.6, 0.5)
    block_orientation = [0.0, 0.0, 0.0, 1.0]
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block_id, block_pose, block_orientation, physicsClientId=physics_client_id
    )
    ee_target_position = (1.35, 0.4, 0.6)
    ee_target = Pose(ee_target_position, ee_pose.orientation)
    joint_target = inverse_kinematics(robot, ee_target, validate=True)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies={table_id, block_id},
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert path is not None
    p.removeBody(block_id, physicsClientId=physics_client_id)
    # Should fail because the hyperparameters are too limited.
    hyperparameters = MotionPlanningHyperparameters(
        birrt_num_attempts=1, birrt_num_iters=1
    )
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block_id, block_pose, block_orientation, physicsClientId=physics_client_id
    )
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies={table_id, block_id},
        seed=seed,
        physics_client_id=physics_client_id,
        hyperparameters=hyperparameters,
    )
    assert path is None
    p.removeBody(block_id, physicsClientId=physics_client_id)
