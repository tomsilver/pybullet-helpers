"""Vanilla PyBullet Inverse Kinematics.

The IKFast solver is preferred over PyBullet IK, if available for the
given robot.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics
from pybullet_helpers.joint import JointPositions, get_joint_infos, get_joints
from pybullet_helpers.link import get_link_pose, get_link_state
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
    SingleArmTwoFingerGripperPyBulletRobot,
)


class InverseKinematicsError(ValueError):
    """Thrown when inverse kinematics fails to find a solution."""


@dataclass(frozen=True)
class InverseKinematicsHyperparameters:
    """Hyperparameters for inverse kinematics."""

    convergence_atol: float = 1e-3
    max_iters: int = 100


def inverse_kinematics(
    robot: SingleArmPyBulletRobot,
    end_effector_pose: Pose,
    validate: bool = True,
    set_joints: bool = True,
    validation_atol: float = 1e-3,
) -> JointPositions:
    """Compute joint positions from a target end effector position, based on
    the robot's current joint positions. Uses IKFast if the robot has IKFast
    info specified.

    The robot's joint state is updated to the IK result. For convenience,
    the new joint positions are also returned.

    If set_joints is True, set the joints after running IK. This is
    recommended when using IKFast because IK is meant to be run
    sequentially from nearby states, since it is very sensitive to
    initialization.

    If validate is True, we guarantee that the returned joint positions
    would result in end_effector_pose if run through
    forward_kinematics.

    WARNING: if validate is True, physics may be overridden, and so it
    should not be used within simulation.
    """
    if robot.ikfast_info():
        joint_positions = _ikfast_inverse_kinematics(robot, end_effector_pose)
        if validate:
            try:
                _validate_joints_state(
                    robot, joint_positions, end_effector_pose, validation_atol
                )
            except ValueError as e:
                raise InverseKinematicsError(e)

    else:
        joint_positions = pybullet_inverse_kinematics(
            robot.robot_id,
            robot.end_effector_id,
            end_effector_pose.position,
            end_effector_pose.orientation,
            robot.arm_joints,
            physics_client_id=robot.physics_client_id,
            validate=validate,
        )

    if set_joints:
        robot.set_joints(joint_positions)

    return joint_positions


def pybullet_inverse_kinematics(
    robot: int,
    end_effector: int,
    target_position: Pose3D,
    target_orientation: Quaternion,
    joints: Sequence[int],
    physics_client_id: int,
    validate: bool = True,
    hyperparameters: InverseKinematicsHyperparameters | None = None,
) -> JointPositions:
    """Runs IK and returns joint positions for the given (free) joints.

    If validate is True, the PyBullet IK solver is called multiple
    times, resetting the robot state each time, until the target
    position is reached. If the target position is not reached after a
    maximum number of iters, an exception is raised.
    """
    if hyperparameters is None:
        hyperparameters = InverseKinematicsHyperparameters()

    # Figure out which joint each dimension of the return of IK corresponds to.
    all_joints = get_joints(robot, physics_client_id=physics_client_id)
    joint_infos = get_joint_infos(
        robot, all_joints, physics_client_id=physics_client_id
    )
    free_joints = [
        joint_info.jointIndex for joint_info in joint_infos if joint_info.qIndex > -1
    ]
    assert set(joints).issubset(set(free_joints))

    # Record the initial state of the joints (positions and velocities) so
    # that we can reset them after.
    initial_joints_states = p.getJointStates(
        robot, free_joints, physicsClientId=physics_client_id
    )
    if validate:
        assert len(initial_joints_states) == len(free_joints)

    # Running IK once is often insufficient, so we run it multiple times until
    # convergence. If it does not converge, an error is raised.
    for _ in range(hyperparameters.max_iters):
        free_joint_vals = p.calculateInverseKinematics(
            robot,
            end_effector,
            target_position,
            targetOrientation=target_orientation,
            physicsClientId=physics_client_id,
        )
        assert len(free_joints) == len(free_joint_vals)
        if not validate:
            break
        # Update the robot state and check if the desired position and
        # orientation are reached.
        for joint, joint_val in zip(free_joints, free_joint_vals):
            p.resetJointState(
                robot, joint, targetValue=joint_val, physicsClientId=physics_client_id
            )

        # Note: we are checking end-effector positions only for convergence.
        ee_link_pose = get_link_pose(robot, end_effector, physics_client_id)
        if np.allclose(
            ee_link_pose.position,
            target_position,
            atol=hyperparameters.convergence_atol,
        ):
            break
    else:
        raise InverseKinematicsError("Inverse kinematics failed to converge.")

    # Reset the joint state (positions and velocities) to their initial values
    # to avoid modifying the PyBullet internal state.
    if validate:
        for joint, (pos, vel, _, _) in zip(free_joints, initial_joints_states):
            p.resetJointState(
                robot,
                joint,
                targetValue=pos,
                targetVelocity=vel,
                physicsClientId=physics_client_id,
            )
    # Order the found free_joint_vals based on the requested joints.
    joint_vals = []
    for joint in joints:
        free_joint_idx = free_joints.index(joint)
        joint_val = free_joint_vals[free_joint_idx]
        joint_vals.append(joint_val)

    return joint_vals


def _ikfast_inverse_kinematics(
    robot: SingleArmPyBulletRobot, end_effector_pose: Pose
) -> JointPositions:
    """IK using IKFast.

    Returns the joint positions.
    """
    ik_solutions = ikfast_closest_inverse_kinematics(
        robot,
        world_from_target=end_effector_pose,
    )
    if not ik_solutions:
        raise InverseKinematicsError(
            f"No IK solution found for target pose {end_effector_pose} " "using IKFast"
        )

    # Use first solution as it is closest to current joint state.
    joint_positions = list(ik_solutions[0])

    # IKFast doesn't handle fingers, so we add them afterwards.
    if isinstance(robot, SingleArmTwoFingerGripperPyBulletRobot):
        # Add fingers to state.
        first_finger_idx, second_finger_idx = sorted(
            [robot.left_finger_joint_idx, robot.right_finger_joint_idx]
        )
        joint_positions.insert(first_finger_idx, robot.open_fingers)
        joint_positions.insert(second_finger_idx, robot.open_fingers)

    return joint_positions


def _validate_joints_state(
    robot: SingleArmPyBulletRobot,
    joint_positions: JointPositions,
    target_pose: Pose,
    validation_atol: float,
) -> None:
    """Validate that the given joint positions matches the target pose.

    This method should NOT be used during simulation mode as it resets
    the joint states.
    """
    # Store current joint positions so we can reset
    initial_joint_states = robot.get_joint_positions()

    # Set joint states, forward kinematics to determine EE position
    robot.set_joints(joint_positions)
    ee_pos = get_link_state(
        robot.robot_id,
        robot.end_effector_id,
        physics_client_id=robot.physics_client_id,
    ).worldLinkFramePosition
    target_pos = target_pose.position
    pos_is_close = np.allclose(ee_pos, target_pos, atol=validation_atol)

    # Reset joint positions before returning/raising error
    robot.set_joints(initial_joint_states)

    if not pos_is_close:
        raise ValueError(
            f"Joint states do not match target pose {target_pos} " f"from {ee_pos}"
        )
