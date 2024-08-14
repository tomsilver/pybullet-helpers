"""Motion Planning in PyBullet."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Collection, Iterable, Iterator, Optional

import numpy as np
from numpy.typing import NDArray
from tomsutils.motion_planning import BiRRT

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    check_collisions_with_held_object,
    sample_collision_free_inverse_kinematics,
)
from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    get_joint_infos,
    get_jointwise_difference,
)
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
    sampling_fn: Callable[[JointPositions], JointPositions] | None = None,
) -> Optional[list[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.

    If additional_state_constraint_fn is provided, the collision
    checking is augmented so that additional_state_constraint_fn() =
    False behaves as if a collision check failed. For example, if you
    want to make sure that a held object is not rotated beyond some
    threshold, you could use additional_state_constraint_fn to enforce
    that. The additional state constraint function can assume that the
    robot is already in the given joint positions because it will be
    called right after collision checking, which sets the robot state.

    If sampling_fn is not provided, defaults to uniform joint space.
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
        _initial_collision_fn = _collision_fn
        _collision_fn = lambda x: _initial_collision_fn(
            x
        ) or not additional_state_constraint_fn(x)

    _distance_fn = partial(
        get_joint_positions_distance, robot, joint_infos, metric="end_effector"
    )

    def _joint_space_sample_fn(pt: JointPositions) -> JointPositions:
        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        if isinstance(robot, SingleArmTwoFingerGripperPyBulletRobot):
            new_pt[robot.left_finger_joint_idx] = pt[robot.left_finger_joint_idx]
            new_pt[robot.right_finger_joint_idx] = pt[robot.right_finger_joint_idx]
        return new_pt

    if sampling_fn is None:
        sampling_fn = _joint_space_sample_fn

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
        sampling_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=hyperparameters.birrt_num_attempts,
        num_iters=hyperparameters.birrt_num_iters,
        smooth_amt=hyperparameters.birrt_smooth_amt,
    )

    return birrt.query(initial_positions, target_positions)


def get_motion_plan_distance(
    motion_plan: list[JointPositions],
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
) -> float:
    """Get the total distance for a motion plan under a given metric."""
    mp_dist = 0.0
    for t in range(len(motion_plan) - 1):
        q1, q2 = motion_plan[t], motion_plan[t + 1]
        dist = joint_distance_fn(q2, q1)
        mp_dist += dist
    return mp_dist


def select_shortest_motion_plan(
    motion_plans: Iterable[list[JointPositions]],
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
) -> list[JointPositions]:
    """Return the motion plan that has the least cumulative distance."""

    shortest_motion_plan: list[JointPositions] | None = None
    shortest_length = np.inf

    for motion_plan in motion_plans:
        mp_dist = get_motion_plan_distance(motion_plan, joint_distance_fn)
        if mp_dist < shortest_length:
            shortest_motion_plan = motion_plan
            shortest_length = mp_dist

    assert shortest_motion_plan is not None, "motion_plans was empty"
    return shortest_motion_plan


def run_smooth_motion_planning_to_pose(
    target_pose: Pose | Callable[[], Pose],
    robot: SingleArmPyBulletRobot,
    collision_ids: set[int],
    plan_frame_from_end_effector_frame: Pose,
    seed: int,
    held_object: int | None = None,
    base_link_to_held_obj: NDArray | None = None,
    max_time: float = 5.0,
    joint_geometric_scalar: float = 0.9,
    sampling_fn: Callable[[JointPositions], JointPositions] | None = None,
) -> Optional[list[JointPositions]]:
    """A naive smooth motion planner that reruns motion planning multiple times
    and then picks the "smoothest" result according to a geometric weighting of
    the joints (so the lowest joint should move the least)."""

    # Target poses can be sampled or singletons.
    if isinstance(target_pose, Pose):
        target_pose_sampler: Callable[[], Pose] = lambda: target_pose  # type: ignore
    else:
        target_pose_sampler = target_pose

    # Set up the geometrically weighted score function.
    def _score_motion_plan(plan: list[JointPositions]) -> float:
        weights = [1.0]
        num_joints = len(robot.arm_joints)
        for _ in range(num_joints - 1):
            weights.append(weights[-1] * joint_geometric_scalar)
        joint_infos = get_joint_infos(
            robot.robot_id, robot.arm_joints, robot.physics_client_id
        )
        dist_fn = partial(
            get_joint_positions_distance,
            robot,
            joint_infos,
            metric="weighted_joints",
            weights=weights,
        )
        return get_motion_plan_distance(plan, dist_fn)

    robot_initial_joints = robot.get_joint_positions()

    start_time = time.perf_counter()
    best_motion_plan: list[JointPositions] | None = None
    best_motion_plan_score: float = np.inf  # lower is better

    while time.perf_counter() - start_time < max_time:
        # Sample a target pose.
        target_pose = target_pose_sampler()
        # Transform to end effector space.
        end_effector_pose = multiply_poses(
            target_pose, plan_frame_from_end_effector_frame
        )
        # Sample a collision-free joint target. If none exist, we'll just
        # go back to sampling a different target pose.
        for target_joint_positions in sample_collision_free_inverse_kinematics(
            robot,
            end_effector_pose,
            collision_ids,
            max_candidates=1,
        ):
            # Try motion planning to the target.
            robot.set_joints(robot_initial_joints)
            motion_plan = run_motion_planning(
                robot,
                robot_initial_joints,
                target_joint_positions,
                collision_ids,
                seed,
                robot.physics_client_id,
                held_object=held_object,
                base_link_to_held_obj=base_link_to_held_obj,
                sampling_fn=sampling_fn,
            )
            # Score the motion plan.
            if motion_plan is not None:
                score = _score_motion_plan(motion_plan)
                if score < best_motion_plan_score:
                    best_motion_plan = motion_plan
                    best_motion_plan_score = score

    return best_motion_plan


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
