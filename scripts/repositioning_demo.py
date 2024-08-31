"""Experimental script for one two-link robot moving another."""

import abc
import time
from functools import partial
from typing import Callable

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from pybullet_helpers.geometry import Pose, matrix_from_quat, multiply_poses
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


class RepositioningDynamicsModel(abc.ABC):
    """A model of forward dynamics."""

    def __init__(
        self,
        active_arm: SingleArmPyBulletRobot,
        passive_arm: SingleArmPyBulletRobot,
        dt: float,
    ) -> None:
        self._active_arm = active_arm
        self._passive_arm = passive_arm
        self._dt = dt

    @abc.abstractmethod
    def step(self, torque: list[float]) -> None:
        """Apply torque to the active arm and update in-place."""


class MathRepositioningDynamicsModel(RepositioningDynamicsModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_to_passive_ee_twist = self._get_active_to_passive_ee_twist(
            self._active_arm, self._passive_arm
        )

    def step(self, torque: list[float]) -> None:

        pos_r = np.array(self._active_arm.get_joint_positions())
        pos_h = np.array(self._passive_arm.get_joint_positions())
        vel_r = np.array(self._active_arm.get_joint_velocities())
        vel_h = np.array(self._passive_arm.get_joint_velocities())
        R = self._active_to_passive_ee_twist

        Jr = self._calculate_jacobian(self._active_arm)
        Jh = self._calculate_jacobian(self._passive_arm)
        Jhinv = np.linalg.pinv(Jh)

        Mr = self._calculate_mass_matrix(self._active_arm)
        Mh = self._calculate_mass_matrix(self._passive_arm)

        Nr = self._calculate_N_vector(self._active_arm)
        Nh = self._calculate_N_vector(self._passive_arm)

        acc_r = np.linalg.inv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ (
            (Jhinv @ R @ Jr).T
            @ (
                Mh * (1 / self._dt) @ (Jhinv @ R @ Jr) @ vel_r
                - Mh * (1 / self._dt) @ vel_h
                + Nh
            )
            + Nr
            - np.array(torque)
        )

        new_vel_r = vel_r + acc_r * self._dt
        r_lin_vel = Jr @ new_vel_r
        h_lin_vel = R @ r_lin_vel
        new_vel_h = Jhinv @ h_lin_vel

        acc_h = (new_vel_h - vel_h) / self._dt

        vel_r = vel_r + acc_r * self._dt
        vel_h = vel_h + acc_h * self._dt

        pos_r = pos_r + vel_r * self._dt
        pos_h = pos_h + vel_h * self._dt

        self._active_arm.set_joints(list(pos_r), joint_velocities=list(vel_r))
        self._passive_arm.set_joints(list(pos_h), joint_velocities=list(vel_h))

    @staticmethod
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

    @staticmethod
    def _calculate_mass_matrix(robot: SingleArmPyBulletRobot) -> NDArray:
        mass_matrix = p.calculateMassMatrix(
            robot.robot_id,
            robot.get_joint_positions(),
            physicsClientId=robot.physics_client_id,
        )
        return np.array(mass_matrix)

    @staticmethod
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

    @staticmethod
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


class PybulletConstraintRepositioningDynamicsModel(RepositioningDynamicsModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        tf = multiply_poses(
            self._active_arm.get_end_effector_pose(),
            self._passive_arm.get_end_effector_pose().invert(),
        )
        p.createConstraint(
            self._active_arm.robot_id,
            self._active_arm.tool_link_id,
            self._passive_arm.robot_id,
            self._passive_arm.tool_link_id,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=tf.position,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=tf.orientation,
            physicsClientId=self._active_arm.physics_client_id,
        )

    def step(self, torque: list[float]) -> None:
        import ipdb

        ipdb.set_trace()


def _create_dynamics_model(
    name: str,
    active_arm: SingleArmPyBulletRobot,
    passive_arm: SingleArmPyBulletRobot,
    dt: float,
) -> RepositioningDynamicsModel:
    if name == "math":
        return MathRepositioningDynamicsModel(active_arm, passive_arm, dt)

    if name == "pybullet-constraint":
        return PybulletConstraintRepositioningDynamicsModel(active_arm, passive_arm, dt)

    raise NotImplementedError


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
            if t < 0.05:
                return [0.0, 1.0]
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


def _main(scenario: str, dynamics: str) -> None:
    dt = 1e-3
    T = 10.0
    t = 0.0

    active_arm, passive_arm, torque_fn = _create_scenario(scenario)
    dynamics_model = _create_dynamics_model(dynamics, active_arm, passive_arm, dt)

    while t < T:
        torque = torque_fn(t)
        dynamics_model.step(torque)
        time.sleep(dt)
        t += dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="two-link")
    parser.add_argument("--dynamics", type=str, default="math")
    args = parser.parse_args()

    _main(args.scenario, args.dynamics)
