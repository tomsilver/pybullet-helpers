"""Motion Planning in PyBullet."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Collection, Iterable, Iterator, Optional, Sequence

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from tomsutils.motion_planning import BiRRT

from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    get_joint_infos,
    get_jointwise_difference,
)
from pybullet_helpers.link import get_link_state
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
    SingleArmTwoFingerGripperPyBulletRobot,
)


@dataclass(frozen=True)
class MotionPlanningHyperparameters:
    """Hyperparameters for motion planning."""

    birrt_extend_num_interp: int = 10
    birrt_num_attempts: int = 10
    birrt_num_iters: int = 100
    birrt_smooth_amt: int = 50


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_positions: JointPositions,
    target_positions: JointPositions,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: int | None = None,
    base_link_to_held_obj: NDArray | None = None,
    hyperparameters: MotionPlanningHyperparameters | None = None,
    additional_state_constraint_fn: Callable[[JointPositions], bool] | None = None,
) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.

    If additional_state_constraint_fn is provided, the collision
    checking is augmented so that additional_state_constraint_fn() =
    False behaves as if a collision check failed. For example, if you
    want to make sure that a held object is not rotated beyond some
    threshold, you could use additional_state_constraint_fn to enforce
    that.
    """
    if hyperparameters is None:
        hyperparameters = MotionPlanningHyperparameters()

    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)
    joint_infos = get_joint_infos(robot.robot_id, robot.arm_joints, physics_client_id)
    num_interp = hyperparameters.birrt_extend_num_interp

    _collision_fn: Callable[[JointPositions], bool] = partial(
        check_collisions_with_held_object,
        robot,
        collision_bodies,
        physics_client_id,
        held_object,
        base_link_to_held_obj,
    )
    if additional_state_constraint_fn is not None:
        _collision_fn = lambda x: _collision_fn(
            x
        ) or not additional_state_constraint_fn(x)

    _distance_fn = partial(
        get_joint_positions_distance, robot, joint_infos, metric="end_effector"
    )

    def _sample_fn(pt: JointPositions) -> JointPositions:
        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        if isinstance(robot, SingleArmTwoFingerGripperPyBulletRobot):
            new_pt[robot.left_finger_joint_idx] = pt[robot.left_finger_joint_idx]
            new_pt[robot.right_finger_joint_idx] = pt[robot.right_finger_joint_idx]
        return new_pt

    def _extend_fn(
        pt1: JointPositions, pt2: JointPositions
    ) -> Iterator[JointPositions]:
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    birrt = BiRRT(
        _sample_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=hyperparameters.birrt_num_attempts,
        num_iters=hyperparameters.birrt_num_iters,
        smooth_amt=hyperparameters.birrt_smooth_amt,
    )

    return birrt.query(initial_positions, target_positions)


def filter_collision_free_joint_generator(
    generator: Iterator[JointPositions],
    robot: SingleArmPyBulletRobot,
    collision_bodies: Collection[int],
    physics_client_id: int,
    held_object: int | None = None,
    base_link_to_held_obj: NDArray | None = None,
) -> Iterator[JointPositions]:
    """Given a generator of joint positions, yield only those that pass
    collision checks.

    The typical use case is that we want to explore the null space of a
    target end effector position to find joint positions that have no
    collisions before then calling motion planning.
    """

    _collision_fn = partial(
        check_collisions_with_held_object,
        robot,
        collision_bodies,
        physics_client_id,
        held_object,
        base_link_to_held_obj,
    )

    for candidate in generator:
        if not _collision_fn(candidate):
            yield candidate


def select_shortest_motion_plan(
    motion_plans: Iterable[list[JointPositions]],
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
) -> list[JointPositions]:
    """Return the motion plan that has the least cumulative distance."""

    shortest_motion_plan: list[JointPositions] | None = None
    shortest_length = np.inf

    for motion_plan in motion_plans:
        mp_dist = 0.0
        for t in range(len(motion_plan) - 1):
            q1, q2 = motion_plan[t], motion_plan[t + 1]
            dist = joint_distance_fn(q2, q1)
            mp_dist += dist
        if mp_dist < shortest_length:
            shortest_motion_plan = motion_plan
            shortest_length = mp_dist

    assert shortest_motion_plan is not None, "motion_plans was empty"
    return shortest_motion_plan


def set_robot_joints_with_held_object(
    robot: SingleArmPyBulletRobot,
    physics_client_id: int,
    held_object: int | None,
    base_link_to_held_obj: NDArray | None,
    joint_state: JointPositions,
) -> None:
    """Set a robot's joints and apply a transform to a held object."""

    robot.set_joints(joint_state)
    if held_object is not None:
        assert base_link_to_held_obj is not None
        world_to_base_link = get_link_state(
            robot.robot_id,
            robot.end_effector_id,
            physics_client_id=physics_client_id,
        ).com_pose
        world_to_held_obj = p.multiplyTransforms(
            world_to_base_link[0],
            world_to_base_link[1],
            base_link_to_held_obj[0],
            base_link_to_held_obj[1],
        )
        p.resetBasePositionAndOrientation(
            held_object,
            world_to_held_obj[0],
            world_to_held_obj[1],
            physicsClientId=physics_client_id,
        )


def check_collisions_with_held_object(
    robot: SingleArmPyBulletRobot,
    collision_bodies: Collection[int],
    physics_client_id: int,
    held_object: int | None,
    base_link_to_held_obj: NDArray | None,
    joint_state: JointPositions,
) -> bool:
    """Check if robot or a held object are in collision with certain bodies."""
    set_robot_joints_with_held_object(
        robot, physics_client_id, held_object, base_link_to_held_obj, joint_state
    )
    p.performCollisionDetection(physicsClientId=physics_client_id)
    for body in collision_bodies:
        if p.getContactPoints(robot.robot_id, body, physicsClientId=physics_client_id):
            return True
        if held_object is not None and p.getContactPoints(
            held_object, body, physicsClientId=physics_client_id
        ):
            return True
    return False


def get_joint_positions_distance(
    robot: SingleArmPyBulletRobot,
    joint_infos: list[JointInfo],
    q1: JointPositions,
    q2: JointPositions,
    metric: str = "end_effector",
    **kwargs,
):
    """Get the distance between two joint positions."""

    if metric == "end_effector":
        return _get_end_effector_joint_positions_distance(robot, q1, q2, **kwargs)

    if metric == "weighted_joints":
        return _get_weighted_joint_positions_distance(joint_infos, q1, q2, **kwargs)

    raise NotImplementedError(f"Unrecognized metric: {metric}")


def _get_end_effector_joint_positions_distance(
    robot: SingleArmPyBulletRobot,
    q1: JointPositions,
    q2: JointPositions,
) -> float:
    # NOTE: only using positions to calculate distance. Should use
    # orientations as well in the near future.
    from_ee = robot.forward_kinematics(q1).position
    to_ee = robot.forward_kinematics(q2).position
    return sum(np.subtract(from_ee, to_ee) ** 2)


def _get_weighted_joint_positions_distance(
    joint_infos: list[JointInfo],
    q1: JointPositions,
    q2: JointPositions,
    weights: list[float],
) -> float:
    assert len(joint_infos) == len(weights) == len(q1) == len(q2)
    diff = get_jointwise_difference(joint_infos, q2, q1)
    dist = np.abs(diff)
    return np.sum(weights * dist)
