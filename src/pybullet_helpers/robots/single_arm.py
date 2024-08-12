"""Abstract class for single armed manipulators with PyBullet helper
functions."""

import abc
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
from gymnasium.spaces import Box

from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast import IKFastInfo
from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    get_joint_infos,
    get_joint_lower_limits,
    get_joint_positions,
    get_joint_upper_limits,
    get_joints,
    get_kinematic_chain,
)
from pybullet_helpers.link import BASE_LINK, get_link_state


class SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot."""

    def __init__(
        self,
        physics_client_id: int,
        base_pose: Pose = Pose.identity(),
        control_mode: str = "position",
        home_joint_positions: JointPositions | None = None,
    ) -> None:
        self.physics_client_id = physics_client_id

        # Pose of base of robot.
        self._base_pose = base_pose

        # Control mode for the robot.
        self._control_mode = control_mode

        # Home joint positions.
        self._home_joint_positions = (
            home_joint_positions or self.default_home_joint_positions
        )

        # Load the robot and set base position and orientation.
        self.robot_id = p.loadURDF(
            str(self.urdf_path()),
            basePosition=self._base_pose.position,
            baseOrientation=self._base_pose.orientation,
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )

        # Robot initially at home pose.
        self.set_joints(self.home_joint_positions)

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the robot."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def urdf_path(cls) -> Path:
        """Get the path to the URDF file for the robot."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def default_home_joint_positions(self) -> JointPositions:
        """The default joint values for the robot."""
        raise NotImplementedError("Override me!")

    @property
    def home_joint_positions(self) -> JointPositions:
        """The home joint positions for this robot."""
        return self._home_joint_positions

    @property
    def action_space(self) -> Box:
        """The action space for the robot.

        Represents position control of the arm and finger joints.
        """
        return Box(
            np.array(self.joint_lower_limits, dtype=np.float32),
            np.array(self.joint_upper_limits, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    @abc.abstractmethod
    def end_effector_name(self) -> str:
        """The name of the end effector."""
        raise NotImplementedError("Override me!")

    @property
    def end_effector_id(self) -> int:
        """The PyBullet joint ID for the end effector."""
        return self.joint_from_name(self.end_effector_name)

    @property
    @abc.abstractmethod
    def tool_link_name(self) -> str:
        """The name of the end effector link (i.e., the tool link)."""
        raise NotImplementedError("Override me!")

    @cached_property
    def tool_link_id(self) -> int:
        """The PyBullet link ID for the tool link."""
        return self.link_from_name(self.tool_link_name)

    @cached_property
    def base_link_name(self) -> str:
        """Name of the base link for the robot."""
        base_info = p.getBodyInfo(self.robot_id, physicsClientId=self.physics_client_id)
        base_name = base_info[0].decode(encoding="UTF-8")
        return base_name

    @cached_property
    def arm_joints(self) -> list[int]:
        """The PyBullet joint IDs of the joints of the robot arm as determined
        by the kinematic chain.

        Note these are joint indices not body IDs, and that the arm
        joints may be a subset of all the robot joints.
        """
        joint_ids = get_kinematic_chain(
            self.robot_id, self.end_effector_id, self.physics_client_id
        )
        # NOTE: pybullet tools assumes sorted arm joints.
        joint_ids = sorted(joint_ids)
        return joint_ids

    @cached_property
    def arm_joint_names(self) -> list[str]:
        """The names of the arm joints."""
        return [
            info.jointName
            for info in get_joint_infos(
                self.robot_id, self.arm_joints, self.physics_client_id
            )
        ]

    @cached_property
    def joint_infos(self) -> list[JointInfo]:
        """Get the joint info for each joint of the robot.

        This may be a superset of the arm joints.
        """
        all_joint_ids = get_joints(self.robot_id, self.physics_client_id)
        return get_joint_infos(self.robot_id, all_joint_ids, self.physics_client_id)

    @cached_property
    def joint_names(self) -> list[str]:
        """Get the names of all the joints in the robot."""
        joint_names = [info.jointName for info in self.joint_infos]
        return joint_names

    def joint_from_name(self, joint_name: str) -> int:
        """Get the joint index for a joint name."""
        return self.joint_names.index(joint_name)

    def joint_info_from_name(self, joint_name: str) -> JointInfo:
        """Get the joint info for a joint name."""
        return self.joint_infos[self.joint_from_name(joint_name)]

    def link_from_name(self, link_name: str) -> int:
        """Get the link index for a given link name."""
        if link_name == self.base_link_name:
            return BASE_LINK

        # In PyBullet, each joint has an associated link.
        for joint_info in self.joint_infos:
            if joint_info.linkName == link_name:
                return joint_info.jointIndex
        raise ValueError(f"Could not find link {link_name}")

    @cached_property
    def joint_lower_limits(self) -> JointPositions:
        """Lower bound on the arm joint limits."""
        return get_joint_lower_limits(
            self.robot_id, self.arm_joints, self.physics_client_id
        )

    @cached_property
    def joint_upper_limits(self) -> JointPositions:
        """Upper bound on the arm joint limits."""
        return get_joint_upper_limits(
            self.robot_id, self.arm_joints, self.physics_client_id
        )

    def get_joint_limits_from_name(self, joint_name: str) -> tuple[float, float]:
        """Get the lower and upper limits for a given joint."""
        assert (
            joint_name in self.arm_joint_names
        ), f"Unrecognized joint name {joint_name}"
        idx = self.arm_joint_names.index(joint_name)
        lower = self.joint_lower_limits[idx]
        upper = self.joint_upper_limits[idx]
        return (lower, upper)

    def get_joint_positions(self) -> JointPositions:
        """Get the joint positions from the current PyBullet state."""
        return get_joint_positions(
            self.robot_id, self.arm_joints, self.physics_client_id
        )

    def set_joints(self, joint_positions: JointPositions) -> None:
        """Directly set the joint positions.

        Outside of resetting to an initial state, this should not be
        used with the robot that uses stepSimulation(); it should only
        be used for motion planning, collision checks, etc., in a robot
        that does not maintain state.
        """
        assert len(joint_positions) == len(self.arm_joints)
        for joint_id, joint_val in zip(self.arm_joints, joint_positions):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=joint_val,
                targetVelocity=0,
                physicsClientId=self.physics_client_id,
            )

    def get_end_effector_pose(self) -> Pose:
        """Get the robot end-effector pose based on the current PyBullet
        state."""
        ee_link_state = get_link_state(
            self.robot_id,
            self.end_effector_id,
            physics_client_id=self.physics_client_id,
        )
        return Pose(
            ee_link_state.worldLinkFramePosition,
            ee_link_state.worldLinkFrameOrientation,
        )

    def set_motors(self, joint_positions: JointPositions) -> None:
        """Update the motors to move toward the given joint positions."""
        assert len(joint_positions) == len(self.arm_joints)

        # Set arm joint motors.
        if self._control_mode == "position":
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.arm_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_positions,
                physicsClientId=self.physics_client_id,
            )
        elif self._control_mode == "reset":
            self.set_joints(joint_positions)
        else:
            raise NotImplementedError(
                "Unrecognized pybullet_control_mode: " f"{self._control_mode }"
            )

    def go_home(self) -> None:
        """Move the robot to its home end-effector pose."""
        self.set_motors(self.default_home_joint_positions)

    def forward_kinematics(self, joint_positions: JointPositions) -> Pose:
        """Compute the end effector pose that would result if the robot arm
        joint positions was equal to the input joint_positions.

        WARNING: This method will make use of resetJointState(), and so it
        should NOT be used during simulation.
        """
        self.set_joints(joint_positions)
        ee_link_state = get_link_state(
            self.robot_id,
            self.end_effector_id,
            physics_client_id=self.physics_client_id,
        )
        position = ee_link_state.worldLinkFramePosition
        orientation = ee_link_state.worldLinkFrameOrientation
        return Pose(position, orientation)

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        """IKFastInfo for this robot.

        If this is specified, then IK will use IKFast.
        """
        return None


class SingleArmTwoFingerGripperPyBulletRobot(SingleArmPyBulletRobot):
    """A single-arm fixed-base PyBullet robot with a two-finger gripper."""

    @cached_property
    def arm_joints(self) -> list[int]:
        """Add the fingers to the arm joints."""
        joint_ids = super().arm_joints
        joint_ids.extend([self.left_finger_id, self.right_finger_id])
        return joint_ids

    @property
    @abc.abstractmethod
    def left_finger_joint_name(self) -> str:
        """The name of the left finger joint."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def right_finger_joint_name(self) -> str:
        """The name of the right finger joint."""
        raise NotImplementedError("Override me!")

    @cached_property
    def left_finger_id(self) -> int:
        """The PyBullet joint ID for the left finger."""
        return self.joint_from_name(self.left_finger_joint_name)

    @cached_property
    def right_finger_id(self) -> int:
        """The PyBullet joint ID for the right finger."""
        return self.joint_from_name(self.right_finger_joint_name)

    @cached_property
    def left_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the left finger.

        Note this is not the joint ID, but the index of the joint within
        the list of arm joints.
        """
        return self.arm_joints.index(self.left_finger_id)

    @cached_property
    def right_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the right finger.

        Note this is not the joint ID, but the index of the joint within
        the list of arm joints.
        """
        return self.arm_joints.index(self.right_finger_id)

    @property
    @abc.abstractmethod
    def open_fingers_joint_value(self) -> float:
        """The value at which the finger joints should be open."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def closed_fingers_joint_value(self) -> float:
        """The value at which the finger joints should be closed."""
        raise NotImplementedError("Override me!")

    def open_fingers(self) -> None:
        """Execute opening the fingers."""
        self._change_fingers(self.open_fingers_joint_value)

    def close_fingers(self) -> None:
        """Execute closing the fingers."""
        self._change_fingers(self.closed_fingers_joint_value)

    def _change_fingers(self, new_value: float) -> None:
        current_joints = self.get_joint_positions()
        current_joints[self.left_finger_joint_idx] = new_value
        current_joints[self.right_finger_joint_idx] = new_value
        self.set_motors(current_joints)

    def get_finger_state(self) -> float:
        """Get the state of the gripper fingers."""
        return p.getJointState(
            self.robot_id,
            self.left_finger_id,
            physicsClientId=self.physics_client_id,
        )[0]
