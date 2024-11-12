"""Utilities for object manipulation."""

from typing import Iterator

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import (
    Pose,
    get_half_extents_from_aabb,
    iter_between_poses,
    matrix_from_quat,
    multiply_poses,
    set_pose,
)
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.link import get_relative_link_pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.states import KinematicState
from pybullet_helpers.utils import get_closest_points_with_optional_links


def get_kinematic_plan_to_pick_object(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    surface_id: int,
    collision_ids: set[int],
    grasp_generator: Iterator[Pose],
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    pregrasp_pad_scale: float = 1.1,
    postgrasp_translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    max_motion_planning_candidates: int | None = None,
    max_smoothing_iters_per_step: int = 1,
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
        robot, object_id, object_link_id=object_link_id, pad_scale=pregrasp_pad_scale
    )

    # Calculate once the direction to move after grasping succeeds. Using the
    # contact normal with the surface.
    postgrasp_translation = _get_approach_pose_from_contact_normals(
        object_id,
        surface_id,
        robot.physics_client_id,
        surface_link_id=surface_link_id,
        translation_magnitude=postgrasp_translation_magnitude,
    )

    # Prepare to transform grasps relative to the link into the object frame.
    if object_link_id is None:
        object_to_link = Pose.identity()
    else:
        object_to_link = get_relative_link_pose(
            object_id, object_link_id, -1, robot.physics_client_id
        )

    for relative_grasp in grasp_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # Calculate the grasp in the world frame.
        object_pose = state.object_poses[object_id]
        grasp = multiply_poses(object_pose, object_to_link, relative_grasp)

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
            max_candidate_plans=max_motion_planning_candidates,
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
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
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
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
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
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    preplace_translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    max_motion_planning_candidates: int | None = None,
    max_smoothing_iters_per_step: int = 1,
    seed: int = 0,
    retract_after: bool = True,
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

    # Prepare to transform placements relative to parent frames.
    if object_link_id is None:
        object_to_link = Pose.identity()
    else:
        object_to_link = get_relative_link_pose(
            object_id, object_link_id, -1, robot.physics_client_id
        )
    if surface_link_id is None:
        surface_to_link = Pose.identity()
    else:
        surface_to_link = get_relative_link_pose(
            surface_id, surface_link_id, -1, robot.physics_client_id
        )

    for relative_placement in placement_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # Calculate the placement.
        surface_pose = state.object_poses[surface_id]
        object_to_surface_placement = multiply_poses(
            object_to_link.invert(), relative_placement, surface_to_link
        )
        world_to_object_placement = multiply_poses(
            surface_pose, object_to_surface_placement
        )
        end_effector_to_object = state.attachments[object_id]
        object_to_end_effector = end_effector_to_object.invert()
        placement = multiply_poses(world_to_object_placement, object_to_end_effector)

        # Temporarily set the placement so that we can calculate contact normals
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
            collision_ids=collision_ids - {object_id},
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
            max_candidate_plans=max_motion_planning_candidates,
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
                collision_ids - {object_id, surface_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
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

        if retract_after:

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
                    collision_ids - {object_id},
                    joint_distance_fn,
                    max_time=max_motion_planning_time,
                    max_smoothing_iters_per_step=max_smoothing_iters_per_step,
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

        # Planning succeeded.
        return plan

    return None


def get_kinematic_plan_to_retract(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    collision_ids: set[int],
    translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    max_smoothing_iters_per_step: int = 1,
) -> list[KinematicState] | None:
    """Make a plan to retract the robot in opposite direction of the fingers.

    It can be good practice to call this after picking or placing to make any
    subsequent calls to motion planning easier.

    NOTE: this function updates pybullet directly and arbitrarily. Users should
    reset the pybullet state as appropriate after calling this function.
    """
    # Reset the simulator to the initial state to restart the planning.
    initial_state.set_pybullet(robot)
    state = initial_state
    joint_distance_fn = create_joint_distance_fn(robot)

    end_effector_pose = robot.get_end_effector_pose()
    rot_mat = matrix_from_quat(end_effector_pose.orientation)
    retract_direction = -1 * rot_mat[:, 2]
    translation = Pose((tuple(translation_magnitude * retract_direction)))
    retract_pose = multiply_poses(translation, end_effector_pose)

    end_effector_path = list(
        iter_between_poses(
            end_effector_pose,
            retract_pose,
            include_start=True,
        )
    )

    if state.attachments:
        assert len(state.attachments) == 1
        held_object = next(iter(state.attachments))
        base_link_to_held_obj = state.attachments[held_object]
    else:
        held_object = None
        base_link_to_held_obj = None
    try:
        retract_plan = smoothly_follow_end_effector_path(
            robot,
            end_effector_path,
            state.robot_joints,
            collision_ids,
            joint_distance_fn,
            max_time=max_motion_planning_time,
            max_smoothing_iters_per_step=max_smoothing_iters_per_step,
            include_start=True,
            held_object=held_object,
            base_link_to_held_obj=base_link_to_held_obj,
        )
    except InverseKinematicsError:
        return None

    # Motion planning succeeded, so update the plan.
    kinematic_plan: list[KinematicState] = []
    for robot_joints in retract_plan:
        state = state.copy_with(robot_joints=robot_joints)
        kinematic_plan.append(state)

    return kinematic_plan


def generate_surface_placements(
    object_id: int,
    surface_id: int,
    rng: np.random.Generator,
    physics_client_id: int,
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
) -> Iterator[Pose]:
    """Generate placement poses relative to the object (link)."""
    while True:
        object_half_extents = get_half_extents_from_aabb(
            object_id, physics_client_id, link_id=object_link_id
        )
        surface_half_extents = get_half_extents_from_aabb(
            surface_id, physics_client_id, link_id=surface_link_id
        )
        relative_placement_position = (
            rng.uniform(
                -surface_half_extents[0] + object_half_extents[0],
                surface_half_extents[0] - object_half_extents[0],
            ),
            rng.uniform(
                -surface_half_extents[1] + object_half_extents[1],
                surface_half_extents[1] - object_half_extents[1],
            ),
            object_half_extents[2] + surface_half_extents[2],
        )
        relative_placement_yaw = rng.uniform(-np.pi, np.pi)
        relative_placement_orn = tuple(
            p.getQuaternionFromEuler([0, 0, relative_placement_yaw])
        )
        yield Pose(relative_placement_position, relative_placement_orn)


def _get_approach_distance_from_aabbs(
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    object_link_id: int | None = None,
    pad_scale: float = 1.1,
) -> float:
    object_half_extents = get_half_extents_from_aabb(
        object_id, physics_client_id=robot.physics_client_id, link_id=object_link_id
    )
    object_radius = max(object_half_extents)
    robot_end_effector_radius = 0.0  # find max value over fingers
    for finger_id in robot.finger_ids:
        robot_end_effector_half_extents = get_half_extents_from_aabb(
            robot.robot_id, physics_client_id=robot.physics_client_id, link_id=finger_id
        )
        robot_end_effector_radius = max(
            robot_end_effector_radius,
            *robot_end_effector_half_extents,
        )

    return (object_radius + robot_end_effector_radius) * pad_scale


def _get_approach_pose_from_contact_normals(
    object_id: int,
    surface_id: int,
    physics_client_id: int,
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    translation_magnitude: float = 0.05,
    contact_distance_threshold: float = 1e-3,
):
    contact_points = get_closest_points_with_optional_links(
        object_id,
        surface_id,
        physics_client_id=physics_client_id,
        link1=object_link_id,
        link2=surface_link_id,
        distance_threshold=contact_distance_threshold,
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
