"""Tests for PyBullet motion planning."""

from functools import partial

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
    sample_joints_from_task_space_bounds,
)
from pybullet_helpers.joint import get_joint_infos
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    get_joint_positions_distance,
    run_motion_planning,
    select_shortest_motion_plan,
)
from pybullet_helpers.robots.fetch import FetchPyBulletRobot
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot
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


def test_motion_planning_additional_constraint(physics_client_id):
    """Tests for run_motion_planning with an additional state constraint."""
    initial_joints = [2.5, -1.5, -2.1, 1.9, 2.9, 1.5, -2.6, 0.0, 0.0]
    target_joints = list(initial_joints)
    target_joints[0] = -0.5
    robot = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id)
    robot.set_joints(target_joints)
    ee_target = robot.get_end_effector_pose()
    robot.set_joints(initial_joints)
    ee_initial = robot.get_end_effector_pose()
    seed = 123
    initial_roll = p.getEulerFromQuaternion(ee_initial.orientation)[0]
    target_roll = p.getEulerFromQuaternion(ee_target.orientation)[0]
    assert np.isclose(initial_roll, target_roll)
    roll_tolerance = 0.5

    # Add blocks to prevent direct movement.
    block1_pose = (0.0, 0.5, 0.1)
    block1_orientation = (0.0, 0.0, 0.0, 1.0)
    block1_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block1_id, block1_pose, block1_orientation, physicsClientId=physics_client_id
    )
    block2_pose = (0.0, -0.5, 0.1)
    block2_orientation = (0.0, 0.0, 0.0, 1.0)
    block2_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block2_id, block2_pose, block2_orientation, physicsClientId=physics_client_id
    )
    collision_ids = {block1_id, block2_id}

    def _check_hand_orientation(joint_positions):
        robot.set_joints(joint_positions)
        ee_pose = robot.get_end_effector_pose()
        # Prevent rotating the hand by too much.
        roll = p.getEulerFromQuaternion(ee_pose.orientation)[0]
        return abs(roll - initial_roll) < roll_tolerance

    # Running motion planning WITHOUT the constraint creates a path that
    # violates it.
    robot.set_joints(initial_joints)
    path = run_motion_planning(
        robot,
        initial_joints,
        target_joints,
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert any(not _check_hand_orientation(s) for s in path)

    # Running motion planning WITH the constraint creates a path that works.
    robot.set_joints(initial_joints)
    path = run_motion_planning(
        robot,
        initial_joints,
        target_joints,
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=physics_client_id,
        additional_state_constraint_fn=_check_hand_orientation,
    )
    assert all(_check_hand_orientation(s) for s in path)


def test_task_space_motion_planning(physics_client_id):
    """Tests for run_motion_planning with a custom sampling function."""
    initial_joints = [2.5, -1.5, -2.1, 1.9, 2.9, 1.5, -2.6, 0.0, 0.0]
    target_joints = list(initial_joints)
    target_joints[0] = -0.5
    robot = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id)
    robot.set_joints(target_joints)
    ee_target = robot.get_end_effector_pose()
    robot.set_joints(initial_joints)
    ee_initial = robot.get_end_effector_pose()
    seed = 123
    initial_roll = p.getEulerFromQuaternion(ee_initial.orientation)[0]
    target_roll = p.getEulerFromQuaternion(ee_target.orientation)[0]
    assert np.isclose(initial_roll, target_roll)

    # Add blocks to prevent direct movement.
    block1_pose = (0.0, 0.5, 0.1)
    block1_orientation = (0.0, 0.0, 0.0, 1.0)
    block1_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block1_id, block1_pose, block1_orientation, physicsClientId=physics_client_id
    )
    block2_pose = (0.0, -0.5, 0.1)
    block2_orientation = (0.0, 0.0, 0.0, 1.0)
    block2_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block2_id, block2_pose, block2_orientation, physicsClientId=physics_client_id
    )
    collision_ids = {block1_id, block2_id}

    # Create task space sampler that constrains roll.
    rng = np.random.default_rng(123)

    def _sample_fn(current_joints):
        del current_joints  # not used
        return sample_joints_from_task_space_bounds(
            rng,
            robot,
            -0.75,
            0.75,
            -0.75,
            0.75,
            -0.75,
            0.75,
            initial_roll - 1e-6,
            initial_roll + 1e-6,
            -np.pi,
            np.pi,
            -np.pi,
            np.pi,
        )
    
    from pybullet_helpers.gui import visualize_pose
    
    def _extend_fn(pt1, pt2):
        ee1 = robot.forward_kinematics(pt1)
        ee2 = robot.forward_kinematics(pt2)
        assert np.isclose(p.getEulerFromQuaternion(ee1.orientation)[0], initial_roll)
        assert np.isclose(p.getEulerFromQuaternion(ee2.orientation)[0], initial_roll)
        # TODO
        num = 10
        robot.set_joints(pt1)
        for i in range(1, num + 1):
            position = tuple(np.array(ee1.position) * (1 - i / num) + np.array(ee2.position) * i / num)
            orientation = tuple(np.array(ee1.orientation) * (1 - i / num) + np.array(ee2.orientation) * i / num)
            pose = Pose(position, orientation)
            joints = inverse_kinematics(robot, pose)
            yield joints


    # Running motion planning WITH the constraint creates a path that works.
    robot.set_joints(initial_joints)
    path = run_motion_planning(
        robot,
        initial_joints,
        target_joints,
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=physics_client_id,
        sampling_fn=_sample_fn,
        extend_fn=_extend_fn,
        hyperparameters=MotionPlanningHyperparameters(birrt_num_attempts=5000),
    )
    assert path is not None

    for s in path:
        robot.set_joints(s)
        import time; time.sleep(0.1)


def test_select_shortest_motion_plan(physics_client_id):
    """Tests for select_shortest_motion_plan()."""

    robot = FetchPyBulletRobot(physics_client_id)
    joint_initial = robot.get_joint_positions()
    joint_space = robot.action_space
    joint_space.seed(123)
    joint_perturbed = joint_space.sample()

    longer_plan = [joint_initial, joint_perturbed]
    shorter_plan = [joint_initial, joint_initial]

    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )
    weights = [1.0] * len(joint_infos)
    dist_fn = partial(
        get_joint_positions_distance,
        robot,
        joint_infos,
        metric="weighted_joints",
        weights=weights,
    )

    ret_plan = select_shortest_motion_plan([shorter_plan, longer_plan], dist_fn)
    assert ret_plan is shorter_plan
