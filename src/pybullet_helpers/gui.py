"""Utilities for GUIs."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, matrix_from_quat
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


def run_interactive_joint_gui(robot: SingleArmPyBulletRobot) -> None:
    """Visualize a robot's joint space."""
    initial_joints = robot.get_joint_positions()

    slider_ids: list[int] = []
    for i, joint_name in enumerate(robot.arm_joint_names):
        lower, upper = robot.get_joint_limits_from_name(joint_name)
        # Handle circular joints.
        if np.isinf(lower):
            lower = -10
        if np.isinf(upper):
            upper = 10
        current = initial_joints[i]
        slider_id = p.addUserDebugParameter(
            paramName=joint_name,
            rangeMin=lower,
            rangeMax=upper,
            startValue=current,
            physicsClientId=robot.physics_client_id,
        )
        slider_ids.append(slider_id)
    show_end_effector_button_id = p.addUserDebugParameter(
        "Show end effector", 0, -1, 0, physicsClientId=robot.physics_client_id
    )

    p.setRealTimeSimulation(True, physicsClientId=robot.physics_client_id)
    frame_ids: set[int] = set()
    current_button_value = p.readUserDebugParameter(
        show_end_effector_button_id, physicsClientId=robot.physics_client_id
    )
    while True:
        joint_positions = []
        for slider_id in slider_ids:
            try:
                v = p.readUserDebugParameter(
                    slider_id, physicsClientId=robot.physics_client_id
                )
            except p.error:
                print("WARNING: failed to read parameter, skipping")
            joint_positions.append(v)
        robot.set_joints(joint_positions)
        try:
            button_value = p.readUserDebugParameter(
                show_end_effector_button_id, physicsClientId=robot.physics_client_id
            )
            if button_value != current_button_value:
                # Visualize the end effector pose.
                for frame_id in frame_ids:
                    p.removeUserDebugItem(
                        frame_id, physicsClientId=robot.physics_client_id
                    )
                frame_ids = visualize_pose(
                    robot.get_end_effector_pose(),
                    physics_client_id=robot.physics_client_id,
                )
                current_button_value = button_value
        except p.error:
            print("WARNING: failed to read button value")


def visualize_pose(
    pose: Pose,
    physics_client_id: int,
    axis_length: float = 0.2,
    x_axis_rgb=(1.0, 0.0, 0.0),
    y_axis_rgb=(0.0, 1.0, 0.0),
    z_axis_rgb=(0.0, 0.0, 1.0),
) -> set[int]:
    """Visualize a pose as a colored frame in the GUI.

    Returns the IDs of the debug lines.
    """

    # Define the axis unit vectors.
    x_axis_unit = np.array([axis_length, 0, 0])
    y_axis_unit = np.array([0, axis_length, 0])
    z_axis_unit = np.array([0, 0, axis_length])

    # Rotate the axis unit vectors.
    rotation_matrix = matrix_from_quat(pose.orientation)
    x_axis_end_position = pose.position + rotation_matrix.dot(x_axis_unit)
    y_axis_end_position = pose.position + rotation_matrix.dot(y_axis_unit)
    z_axis_end_position = pose.position + rotation_matrix.dot(z_axis_unit)

    # Draw x axis.
    x_id = p.addUserDebugLine(
        lineFromXYZ=pose.position,
        lineToXYZ=x_axis_end_position,
        lineColorRGB=x_axis_rgb,
        lifeTime=0,
        physicsClientId=physics_client_id,
    )

    # Draw y axis.
    y_id = p.addUserDebugLine(
        lineFromXYZ=pose.position,
        lineToXYZ=y_axis_end_position,
        lineColorRGB=y_axis_rgb,
        lifeTime=0,
        physicsClientId=physics_client_id,
    )

    # Draw z axis.
    z_id = p.addUserDebugLine(
        lineFromXYZ=pose.position,
        lineToXYZ=z_axis_end_position,
        lineColorRGB=z_axis_rgb,
        lifeTime=0,
        physicsClientId=physics_client_id,
    )

    return {x_id, y_id, z_id}
