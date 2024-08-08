"""Interactively visualize the joint space of a robot in pybullet."""

import pybullet as p

from pybullet_helpers.camera import create_gui_connection
from pybullet_helpers.robots.panda import PandaPyBulletRobot


def _main() -> None:
    physics_client_id = create_gui_connection()
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI, True, physicsClientId=physics_client_id
    )
    robot = PandaPyBulletRobot(physics_client_id, control_mode="reset")
    initial_joints = robot.get_joint_positions()

    slider_ids: list[int] = []
    for i, joint_name in enumerate(robot.arm_joint_names):
        lower, upper = robot.get_joint_limits_from_name(joint_name)
        current = initial_joints[i]
        slider_id = p.addUserDebugParameter(
            paramName=joint_name,
            rangeMin=lower,
            rangeMax=upper,
            startValue=current,
            physicsClientId=physics_client_id,
        )
        slider_ids.append(slider_id)

    p.setRealTimeSimulation(True, physicsClientId=physics_client_id)
    while True:
        joint_positions = []
        for slider_id in slider_ids:
            try:
                v = p.readUserDebugParameter(
                    slider_id, physicsClientId=physics_client_id
                )
            except p.error:
                print("WARNING: failed to read parameter, skipping")
            joint_positions.append(v)
        robot.set_joints(joint_positions)


if __name__ == "__main__":
    _main()
