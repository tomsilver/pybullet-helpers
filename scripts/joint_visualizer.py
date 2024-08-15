"""Interactively visualize the joint space of a robot in pybullet."""

import pybullet as p

from pybullet_helpers.gui import create_gui_connection, run_interactive_joint_gui
from pybullet_helpers.robots import create_pybullet_robot


def _main(robot_name: str) -> None:
    physics_client_id = create_gui_connection()
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI, True, physicsClientId=physics_client_id
    )
    # TODO remove
    initial_joints = [2.5, -1.5, -2.1, 1.9, 2.9, 1.5, -2.6, 0.0, 0.0]
    robot = create_pybullet_robot(
        robot_name,
        physics_client_id,
        control_mode="reset",
        home_joint_positions=initial_joints,
    )
    run_interactive_joint_gui(robot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", type=str)
    args = parser.parse_args()
    _main(args.robot)
