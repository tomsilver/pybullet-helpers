"""Utilities for GUIs."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, Pose3D, matrix_from_quat
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


def create_gui_connection(
    camera_distance: float = 1.5,
    camera_yaw: float = 0,
    camera_pitch: float = -15,
    camera_target: Pose3D = (0, 0, 0.5),
    background_rgb: tuple[float, float, float] = (0, 0, 0),
    disable_preview_windows: bool = True,
) -> int:  # pragma: no cover
    """Creates a PyBullet GUI connection and initializes the camera.

    Returns the physics client ID for the connection.

    Not covered by unit tests because unit tests need to be headless.
    """
    physics_client_id = p.connect(
        p.GUI,
        options=(
            f"--background_color_red={background_rgb[0]} "
            f"--background_color_green={background_rgb[1]} "
            f"--background_color_blue={background_rgb[2]}"
        ),
    )
    # Disable the PyBullet GUI preview windows for faster rendering.
    if disable_preview_windows:
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, False, physicsClientId=physics_client_id
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, False, physicsClientId=physics_client_id
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False, physicsClientId=physics_client_id
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            False,
            physicsClientId=physics_client_id,
        )
    p.resetDebugVisualizerCamera(
        camera_distance,
        camera_yaw,
        camera_pitch,
        camera_target,
        physicsClientId=physics_client_id,
    )
    return physics_client_id


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
                ee_pose = robot.get_end_effector_pose()
                frame_ids = visualize_pose(
                    ee_pose,
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
