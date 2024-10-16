"""Tests for manipulation.py."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.manipulation import (
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
)
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.states import KinematicState
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder


def test_kinematic_pick_place():
    """Tests for get_kinematic_plan_to_pick_object() and
    get_kinematic_plan_to_place_object."""

    rng = np.random.default_rng(123)

    # Set up a scene to test manipuation.
    physics_client_id = p.connect(p.DIRECT)

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_yaw=0)

    # Create robot.
    robot = create_pybullet_robot("panda", physics_client_id)

    # Create table.
    table_pose = Pose(position=(-0.5, 0.0, -0.2))
    table_rgba = (0.5, 0.5, 0.5, 1.0)
    table_half_extents = (0.1, 0.3, 0.2)
    table_id = create_pybullet_block(
        table_rgba,
        half_extents=table_half_extents,
        physics_client_id=physics_client_id,
    )
    set_pose(table_id, table_pose, physics_client_id)

    # Create object.
    object_pose = Pose(position=(-0.5, 0.0, 0.05))
    object_rgba = (0.9, 0.6, 0.3, 1.0)
    object_radius = 0.025
    object_length = 0.1
    object_id = create_pybullet_cylinder(
        object_rgba,
        object_radius,
        object_length,
        physics_client_id=physics_client_id,
    )
    set_pose(object_id, object_pose, physics_client_id)

    # Extract the initial state.
    initial_state = KinematicState.from_pybullet(robot, {object_id, table_id})

    # Set up a grasp generator.
    def _grasp_generator():
        while True:
            angle_offset = rng.uniform(-np.pi, np.pi)
            relative_pose = get_poses_facing_line(
                axis=(0.0, 0.0, 1.0),
                point_on_line=(0.0, 0.0, 0.0),
                radius=0.0125,
                num_points=1,
                angle_offset=angle_offset,
            )[0]
            rot = Pose.from_rpy((0, 0, 0), (0.0, 0.0, np.pi / 2))
            yield multiply_poses(relative_pose, rot)

    # Get a plan.
    plan = get_kinematic_plan_to_pick_object(
        initial_state,
        robot,
        object_id,
        table_id,
        collision_ids={table_id},
        grasp_generator=_grasp_generator(),
    )

    assert plan is not None

    # Advance to the end of the plan.
    initial_state = plan[-1]
    initial_state.set_pybullet(robot)

    # Set up a placement generator.
    placement_lb = (
        -table_half_extents[0] + object_radius,
        -table_half_extents[1] + object_radius,
        table_half_extents[2] + object_length / 2,
    )
    placement_ub = (
        table_half_extents[0] - object_radius,
        table_half_extents[1] - object_radius,
        table_half_extents[2] + object_length / 2,
    )

    def _placement_generator():
        # Sample on the surface of the table.
        while True:
            yield Pose(tuple(rng.uniform(placement_lb, placement_ub)))

    # Get a plan.
    plan = get_kinematic_plan_to_place_object(
        initial_state,
        robot,
        object_id,
        table_id,
        collision_ids={table_id},
        placement_generator=_placement_generator(),
    )

    assert plan is not None

    p.disconnect(physics_client_id)
