"""Experimental script for one two-link robot moving another."""

import time
from typing import Callable

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from pybullet_helpers.geometry import Pose, matrix_from_quat
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


def _calculate_jacobian(robot: SingleArmPyBulletRobot) -> NDArray:
    joint_positions = robot.get_joint_positions()
    jac_t, jac_r = p.calculateJacobian(
        robot.robot_id,
        robot.tool_link_id,
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
) -> tuple[
    SingleArmPyBulletRobot, SingleArmPyBulletRobot, Callable[[float], list[float]]
]:

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

        def _torque_fn(t: float) -> list[float]:
            if t < 0.01:
                return [1, 0.0]
            return [0.0] * 2

        return active_arm, passive_arm, _torque_fn

    if scenario == "panda-human":
        robot_init_pos = (0.8, -0.1, 0.5)
        human_init_pos = (0.15, 0.1, 1.4)

        physics_client_id = create_gui_connection(
            camera_target=robot_init_pos, camera_distance=1.75, camera_pitch=-50
        )

        robot_init_orn_obj = Rotation.from_euler("xyz", [0, 0, np.pi])
        robot_base_pose = Pose(robot_init_pos, robot_init_orn_obj.as_quat())
        human_init_orn_obj = Rotation.from_euler("xyz", [np.pi, 0, 0])
        human_base_pose = Pose(human_init_pos, human_init_orn_obj.as_quat())
        robot = create_pybullet_robot(
            "panda-limb-repo", physics_client_id, base_pose=robot_base_pose
        )
        human = create_pybullet_robot(
            "human-arm-6dof", physics_client_id, base_pose=human_base_pose
        )
        robot_init_joints = [
            0.94578431,
            -0.89487842,
            -1.67534487,
            -0.34826698,
            1.73607292,
            0.14233887,
        ]
        human_init_joints = [
            1.43252278,
            -0.81111486,
            -0.42373363,
            0.49931369,
            -1.17420521,
            0.37122887,
        ]
        robot.set_joints(robot_init_joints)
        human.set_joints(human_init_joints)

        def _torque_fn(t: float) -> list[float]:
            if t < 0.1:
                return [1, 0.0, 0.0, 0.0, 0.0, 0.0]
            return [0.0] * 6

        return robot, human, _torque_fn

    raise NotImplementedError


def _main(scenario: str) -> None:
    active_arm, passive_arm, torque_fn = _create_scenario(scenario)
    active_to_passive_ee_twist = _get_active_to_passive_ee_twist(
        active_arm, passive_arm
    )
    dt = 1e-3
    T = 10.0
    t = 0.0

    while t < T:
        _custom_step(
            active_arm,
            passive_arm,
            active_torque=torque_fn(t),
            active_to_passive_ee_twist=active_to_passive_ee_twist,
            dt=dt,
        )

        time.sleep(dt)
        t += dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="two-link")
    args = parser.parse_args()

    _main(args.scenario)
