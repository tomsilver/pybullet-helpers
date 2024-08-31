"""Experimental script for one two-link robot moving another."""

import numpy as np
import pybullet as p
from numpy.typing import NDArray

from pybullet_helpers.geometry import Pose, matrix_from_quat
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


def _calculate_jacobian(robot: SingleArmPyBulletRobot) -> NDArray:
    joint_positions = robot.get_joint_positions()
    jac_t, jac_r = p.calculateJacobian(
        robot.robot_id,
        robot.link_from_name("end_effector_link"),
        [0, 0, 0],
        joint_positions,
        [0.0] * len(joint_positions),
        [0.0] * len(joint_positions),
        physicsClientId=robot.physics_client_id,
    )
    return np.concatenate((np.array(jac_t), np.array(jac_r)), axis=0)


def _calculate_mass_matrix(robot: SingleArmPyBulletRobot) -> NDArray:
    mass_matrix = p.calculateMassMatrix(
        robot.robot_id,
        robot.get_joint_positions(),
        physicsClientId=robot.physics_client_id,
    )
    return np.array(mass_matrix)


def _calculate_N_vector(robot: SingleArmPyBulletRobot) -> NDArray:
    joint_positions = robot.get_joint_positions()
    joint_velocities = robot.get_joint_velocities()
    joint_accel = [0.0] * len(joint_positions)
    n_vector = p.calculateInverseDynamics(
        robot.robot_id,
        joint_positions,
        joint_velocities,
        joint_accel,
        physicsClientId=robot.physics_client_id,
    )
    return np.array(n_vector)


def _get_active_to_passive_ee_twist(
    active_arm: SingleArmPyBulletRobot, passive_arm: SingleArmPyBulletRobot
) -> NDArray:
    active_ee_orn = active_arm.get_end_effector_pose().orientation
    passive_ee_orn = passive_arm.get_end_effector_pose().orientation
    active_to_passive_ee = matrix_from_quat(passive_ee_orn).T @ matrix_from_quat(
        active_ee_orn
    )
    active_to_passive_ee_twist = np.eye(6)
    active_to_passive_ee_twist[:3, :3] = active_to_passive_ee
    active_to_passive_ee_twist[3:, 3:] = active_to_passive_ee
    return active_to_passive_ee_twist


def _custom_step(
    active_arm: SingleArmPyBulletRobot,
    passive_arm: SingleArmPyBulletRobot,
    active_torque: list[float],
    active_to_passive_ee_twist: NDArray,
    dt: float = 1e-3,
) -> None:

    pos_r = np.array(active_arm.get_joint_positions())
    pos_h = np.array(passive_arm.get_joint_positions())
    vel_r = np.array(active_arm.get_joint_velocities())
    vel_h = np.array(passive_arm.get_joint_velocities())
    R = active_to_passive_ee_twist

    Jr = _calculate_jacobian(active_arm)
    Jh = _calculate_jacobian(passive_arm)
    Jhinv = np.linalg.pinv(Jh)

    Mr = _calculate_mass_matrix(active_arm)
    Mh = _calculate_mass_matrix(passive_arm)

    Nr = _calculate_N_vector(active_arm)
    Nh = _calculate_N_vector(passive_arm)

    acc_r = np.linalg.inv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
        (Jhinv @ R @ Jr).T
        @ (Mh * (1 / dt) @ (Jhinv @ R @ Jr) @ vel_r - Mh * (1 / dt) @ vel_h + Nh)
        + Nr
        - np.array(active_torque)
    )

    new_vel_r = vel_r + acc_r * dt
    robot_lin_vel = Jr @ new_vel_r
    human_lin_vel = R @ robot_lin_vel
    new_vel_h = Jhinv @ human_lin_vel

    acc_h = (new_vel_h - vel_h) / dt

    vel_r = vel_r + acc_r * dt
    vel_h = vel_h + acc_h * dt

    pos_r = pos_r + vel_r * dt
    pos_h = pos_h + vel_h * dt

    active_arm.set_joints(list(pos_r), joint_velocities=list(vel_r))
    passive_arm.set_joints(list(pos_h), joint_velocities=list(vel_h))


def _create_scenario(
    scenario: str,
) -> tuple[SingleArmPyBulletRobot, SingleArmPyBulletRobot, list[float]]:

    if scenario == "two-link":
        physics_client_id = create_gui_connection(camera_distance=2.0, camera_pitch=-40)

        active_arm_base_pose = Pose((-np.sqrt(2), 0.0, 0.0))
        active_arm_home_joint_positions = [-np.pi / 4, np.pi / 2]
        active_arm = create_pybullet_robot(
            "two-link",
            physics_client_id,
            base_pose=active_arm_base_pose,
            home_joint_positions=active_arm_home_joint_positions,
        )

        passive_arm_base_pose = Pose((np.sqrt(2), 0.0, 0.0))
        passive_arm_home_joint_positions = [np.pi / 2 + np.pi / 4, np.pi / 2]
        passive_arm = create_pybullet_robot(
            "two-link",
            physics_client_id,
            base_pose=passive_arm_base_pose,
            home_joint_positions=passive_arm_home_joint_positions,
        )

        torque = [0.01, 0.0]

        return active_arm, passive_arm, torque

    raise NotImplementedError


def _main(scenario: str) -> None:
    active_arm, passive_arm, torque = _create_scenario(scenario)
    active_to_passive_ee_twist = _get_active_to_passive_ee_twist(
        active_arm, passive_arm
    )
    while True:
        _custom_step(
            active_arm,
            passive_arm,
            active_torque=torque,
            active_to_passive_ee_twist=active_to_passive_ee_twist,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="two-link")
    args = parser.parse_args()

    _main(args.scenario)
