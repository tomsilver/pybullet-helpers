"""Utilities for GUIs."""

import numpy as np
import pybullet as p

from pybullet_helpers.robots.single_arm import SingleArmTwoFingerGripperPyBulletRobot


def run_interactive_joint_gui(robot: SingleArmTwoFingerGripperPyBulletRobot) -> None:
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

    p.setRealTimeSimulation(True, physicsClientId=robot.physics_client_id)
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
