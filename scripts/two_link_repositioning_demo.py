"""Experimental script for one two-link robot moving another."""

from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.gui import create_gui_connection
import numpy as np
from numpy.typing import NDArray
import pybullet as p


def _calculate_jacobian(robot: SingleArmPyBulletRobot) -> NDArray:
    joint_positions = robot.get_joint_positions()
    jac_t, jac_r = p.calculateJacobian(robot.robot_id,
                             robot.link_from_name("end_effector_link"),
                             [0, 0, 0],
                             joint_positions,
                             [0.0] * len(joint_positions),
                             [0.0] * len(joint_positions),
                             physicsClientId=robot.physics_client_id,
                             )
    return np.concatenate((np.array(jac_t), np.array(jac_r)), axis=0)


def _calculate_mass_matrix(robot: SingleArmPyBulletRobot) -> NDArray:
    mass_matrix = p.calculateMassMatrix(robot.robot_id, robot.get_joint_positions(),
                                        physicsClientId=robot.physics_client_id)
    return np.array(mass_matrix)


def _calculate_N_vector(robot: SingleArmPyBulletRobot) -> NDArray:
    joint_positions = robot.get_joint_positions()
    joint_velocities = robot.get_joint_velocities()
    joint_accel = [0.0] * len(joint_positions)
    gravity_vector = p.calculateInverseDynamics(robot.robot_id, joint_positions, joint_velocities, joint_accel,
                                                physicsClientId=robot.physics_client_id)
    return np.array(gravity_vector)


def _custom_step(active_arm: SingleArmPyBulletRobot, passive_arm: SingleArmPyBulletRobot,
                 active_torque: list[float], dt: float = 1e-3) -> None:

    Jr = _calculate_jacobian(active_arm)
    Jh = _calculate_jacobian(passive_arm)
    Jhinv = np.linalg.pinv(Jh)

    Mr = _calculate_mass_matrix(active_arm)
    Mh = _calculate_mass_matrix(passive_arm)

    Nr = _calculate_N_vector(active_arm)
    Nh = _calculate_N_vector(passive_arm)

    import ipdb; ipdb.set_trace()
    u = np.linalg.inv((Jhinv @ R @ -Jr).T @ Mh @ (Jhinv @ R @ Jr) - Mr) @ ((Jhinv @ R @ Jr).T @ \
            (Mh * (1/dt) @ (Jhinv @ R @ Jr) @ vel_r_previous - Mh * (1/dt) @ vel_h_previous + Nh) + Nr - tau_robot)
        



def _main() -> None:
    physics_client_id = create_gui_connection()

    active_arm_base_pose = Pose.identity()
    active_arm_home_joint_positions = [-0.6426735129190922, 1.9461292850346736]
    active_arm = create_pybullet_robot("two-link", physics_client_id,
                                       base_pose=active_arm_base_pose,
                                       home_joint_positions=active_arm_home_joint_positions)

    passive_arm_base_pose = Pose((0.0, 1.0, 0.0))
    passive_arm_home_joint_positions = [-0.5542717476205666, 0.29770395550097417]
    passive_arm = create_pybullet_robot("two-link", physics_client_id,
                                       base_pose=passive_arm_base_pose,
                                       home_joint_positions=passive_arm_home_joint_positions)
    
    # get transformation from robot base to human base
    active_ee_to_passive_ee = multiply_poses(active_arm.get_end_effector_pose(),
                        passive_arm.get_end_effector_pose().invert())

    self.robot_to_human_ee_twist = np.eye(6) # rotation to apply to twist from robot linear velocities (J_r @ qdot_r)
    self.robot_to_human_ee_twist[:3, :3] = self.robot_to_human_ee
    self.robot_to_human_ee_twist[3:, 3:] = self.robot_to_human_ee

    active_ee_to_passive_ee = multiply_poses(active_arm.get_end_effector_pose(),
                        passive_arm.get_end_effector_pose().invert())
    
    # p.createConstraint(active_arm.robot_id,
    #                    active_arm.link_from_name("link2"),
    #                    passive_arm.robot_id,
    #                    passive_arm.link_from_name("link2"),
    #                    jointType=p.JOINT_FIXED,
    #                    jointAxis=[0, 0, 0],
    #                    parentFramePosition=[0, 0, 0],
    #                    childFramePosition=active_ee_to_passive_ee.position,
    #                    parentFrameOrientation=[0, 0, 0, 1],
    #                    childFrameOrientation=active_ee_to_passive_ee.orientation,
    #                     physicsClientId=physics_client_id
    #                    )

    while True:
        _custom_step(active_arm, passive_arm, active_torque=[0.0, 0.0])

        # world_to_active_ee = active_arm.get_end_effector_pose()
        # world_to_passive_ee = multiply_poses(world_to_active_ee, active_ee_to_passive_ee)
        # passive_joints = inverse_kinematics(passive_arm, world_to_passive_ee, validate=True)
        # passive_arm.set_joints(passive_joints)


if __name__ == "__main__":
    _main()