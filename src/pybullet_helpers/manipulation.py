"""Utilities for object manipulation."""

from typing import Iterator

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, iter_between_poses, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.states import KinematicState


def get_kinematic_plan_to_pick_object(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    surface_id: int,
    collision_ids: set[int],
    grasp_generator: Iterator[Pose],
    pregrasp_pad_scale: float = 1.1,
    postgrasp_translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    seed: int = 0,
) -> list[KinematicState] | None:
    """Make a plan to pick up the object from a surface.

    The grasp pose is in the object frame.

    The surface is used to determine the direction that the robot should move
    directly after picking (to remove contact between the object and surface).

    Users should make grasp_generator finite to prevent infinite loops, unless
    they are very confident that some feasible grasp plan exists.

    NOTE: this function updates pybullet directly and arbitrarily. Users should
    reset the pybullet state as appropriate after calling this function.
    """
    # Reset the simulator to the initial state to restart the planning.
    initial_state.set_pybullet(robot)
    state = initial_state
    all_object_ids = set(state.object_poses)
    joint_distance_fn = create_joint_distance_fn(robot)

    # Calculate pregrasp poses by translating the grasp away from the object.
    # The translation amount is determined based on the size of the axis aligned
    # bounding box for the object and the robot end effector.
    pregrasp_distance = _get_approach_distance_from_aabbs(
        robot, object_id, pad_scale=pregrasp_pad_scale
    )

    # Calculate once the direction to move after grasping succeeds. Using the
    # contact normal with the surface.
    postgrasp_translation = _get_approach_pose_from_contact_normals(
        object_id, surface_id, robot.physics_client_id, postgrasp_translation_magnitude
    )

    for relative_grasp in grasp_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # Calculate the grasp in the world frame.
        object_pose = state.object_poses[object_id]
        grasp = multiply_poses(object_pose, relative_grasp)

        # Calculate the pregrasp pose.
        pregrasp_translation_direction = np.array([0.0, 0.0, -1.0])
        pregrasp_tf = Pose(tuple(pregrasp_translation_direction * pregrasp_distance))
        pregrasp_pose = multiply_poses(grasp, pregrasp_tf)

        # Motion plan to the pregrasp pose.
        plan_to_pregrasp = run_smooth_motion_planning_to_pose(
            pregrasp_pose,
            robot,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
        )
        # If motion planning failed, try a different grasp.
        if plan_to_pregrasp is None:
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in plan_to_pregrasp:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Move to grasp.
        end_effector_pose = robot.get_end_effector_pose()
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                grasp,
                include_start=False,
            )
        )
        try:
            pregrasp_to_grasp_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids,
                joint_distance_fn,
                max_time=max_motion_planning_time,
                include_start=False,
            )
        except InverseKinematicsError:
            pregrasp_to_grasp_plan = None
        # If motion planning failed, try a different grasp.
        if pregrasp_to_grasp_plan is None:
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in pregrasp_to_grasp_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Update the state to include a grasp attachment.
        state = KinematicState.from_pybullet(
            robot, all_object_ids, attached_object_ids={object_id}
        )
        plan.append(state)

        # Move off the surface.
        end_effector_pose = robot.get_end_effector_pose()
        post_grasp_pose = multiply_poses(postgrasp_translation, end_effector_pose)
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                post_grasp_pose,
                include_start=False,
            )
        )
        try:
            grasp_to_postgrasp_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids - {object_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                include_start=False,
                held_object=object_id,
                base_link_to_held_obj=relative_grasp.invert(),
            )
        except InverseKinematicsError:
            grasp_to_postgrasp_plan = None
        # If motion planning failed, try a different grasp.
        if grasp_to_postgrasp_plan is None:
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in grasp_to_postgrasp_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Planning succeeded.
        return plan

    # No grasp worked.
    return None


def get_kinematic_plan_to_place_object(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    surface_id: int,
    collision_ids: set[int],
    placement_generator: Iterator[Pose],
    preplace_translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    seed: int = 0,
) -> list[KinematicState] | None:
    """Make a plan to place the held object onto the surface.

    The placement pose is in the surface frame.

    Users should make placement_grasp finite to prevent infinite loops, unless
    they are very confident that some feasible plan exists.

    NOTE: this function updates pybullet directly and arbitrarily. Users should
    reset the pybullet state as appropriate after calling this function.
    """
    assert object_id in initial_state.attachments

    # Reset the simulator to the initial state to restart the planning.
    initial_state.set_pybullet(robot)
    state = initial_state
    all_object_ids = set(state.object_poses)
    joint_distance_fn = create_joint_distance_fn(robot)

    for relative_placement in placement_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # Calculate the placement.
        surface_pose = state.object_poses[surface_id]
        world_to_object_placement = multiply_poses(surface_pose, relative_placement)
        end_effector_to_object = state.attachments[object_id]
        object_to_end_effector = end_effector_to_object.invert()
        placement = multiply_poses(world_to_object_placement, object_to_end_effector)

        # Temporary set the placement so that we can calculate contact normals
        # to determine the preplace pose.
        set_pose(object_id, world_to_object_placement, robot.physics_client_id)
        preplace_translation = _get_approach_pose_from_contact_normals(
            object_id,
            surface_id,
            robot.physics_client_id,
            translation_magnitude=preplace_translation_magnitude,
        )
        preplace_pose = multiply_poses(preplace_translation, placement)

        # Set the state back to continue planning.
        state.set_pybullet(robot)

        # Motion plan to the preplace pose.
        plan_to_preplace = run_smooth_motion_planning_to_pose(
            preplace_pose,
            robot,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
            held_object=object_id,
            base_link_to_held_obj=initial_state.attachments[object_id],
        )
        # If motion planning failed, try a different placement.
        if plan_to_preplace is None:
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in plan_to_preplace:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Move to place.
        end_effector_pose = robot.get_end_effector_pose()
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                placement,
                include_start=False,
            )
        )
        try:
            preplace_to_place_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids - {surface_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                include_start=False,
                held_object=object_id,
                base_link_to_held_obj=end_effector_to_object,
            )
        except InverseKinematicsError:
            preplace_to_place_plan = None
        # If motion planning failed, try a different placement.
        if preplace_to_place_plan is None:
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in preplace_to_place_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Update the state to remove the grasp attachment.
        new_attached_object_ids = set(state.attachments) - {object_id}
        state = KinematicState.from_pybullet(
            robot,
            all_object_ids,
            attached_object_ids=new_attached_object_ids,
        )
        plan.append(state)

        # Move back to the preplace pose.
        end_effector_pose = robot.get_end_effector_pose()
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                preplace_pose,
                include_start=False,
            )
        )
        try:
            place_to_postplace_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids,
                joint_distance_fn,
                max_time=max_motion_planning_time,
                include_start=False,
            )
        except InverseKinematicsError:
            place_to_postplace_plan = None
        # If motion planning failed, try a different placement.
        if place_to_postplace_plan is None:
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in place_to_postplace_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Planning succeeded.
        return plan

    return None


def _get_approach_distance_from_aabbs(
    robot: FingeredSingleArmPyBulletRobot, object_id: int, pad_scale: float = 1.1
) -> float:
    object_aabb = p.getAABB(object_id, -1, robot.physics_client_id)
    object_extent = max(
        object_aabb[1][0] - object_aabb[0][0],
        object_aabb[1][1] - object_aabb[0][1],
        object_aabb[1][2] - object_aabb[0][2],
    )
    object_radius = object_extent / 2
    robot_end_effector_radius = 0.0  # find max value over fingers
    for finger_id in robot.finger_ids:
        robot_end_effector_aabb = p.getAABB(
            robot.robot_id, finger_id, robot.physics_client_id
        )
        robot_end_effector_extent = max(
            robot_end_effector_aabb[1][0] - robot_end_effector_aabb[0][0],
            robot_end_effector_aabb[1][1] - robot_end_effector_aabb[0][1],
            robot_end_effector_aabb[1][2] - robot_end_effector_aabb[0][2],
        )
        robot_end_effector_radius = max(
            robot_end_effector_radius, robot_end_effector_extent / 2
        )

    return (object_radius + robot_end_effector_radius) * pad_scale


def _get_approach_pose_from_contact_normals(
    object_id: int,
    surface_id: int,
    physics_client_id: int,
    translation_magnitude: float = 0.05,
    contact_distance_threshold: float = 1e-3,
):
    contact_points = p.getClosestPoints(
        object_id,
        surface_id,
        distance=contact_distance_threshold,
        physicsClientId=physics_client_id,
    )
    assert len(contact_points) > 0
    contact_normals = []
    for contact_point in contact_points:
        contact_normal = contact_point[7]
        contact_normals.append(contact_normal)
    vec = np.mean(contact_normals, axis=0)
    translation_direction = vec / np.linalg.norm(vec)
    translation = translation_direction * translation_magnitude
    return Pose(tuple(translation))
