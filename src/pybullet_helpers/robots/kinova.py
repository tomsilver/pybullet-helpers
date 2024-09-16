"""Kinova Gen3 robots."""

from pathlib import Path
from typing import Optional

import numpy as np

from pybullet_helpers.ikfast import IKFastInfo
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    FingeredSingleArmPyBulletRobot,
    SingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class KinovaGen3NoGripperPyBulletRobot(SingleArmPyBulletRobot):
    """A Kinova Gen3 robot arm with no gripper."""

    @classmethod
    def get_name(cls) -> str:
        return "kinova-gen3-no-gripper"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "kinova_no_gripper" / "GEN3_URDF_V12.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6]

    @property
    def end_effector_name(self) -> str:
        return "EndEffector"

    @property
    def tool_link_name(self) -> str:
        return "EndEffector_Link"


class KinovaGen3RobotiqGripperPyBulletRobot(FingeredSingleArmPyBulletRobot[float]):
    """A Kinova Gen3 robot arm with a robotiq gripper.

    The finger states are all determined by one value, but there are
    multiple mimic joints.
    """

    @classmethod
    def get_name(cls) -> str:
        return "kinova-gen3"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "kortex_description" / "gen3_7dof.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "tool_frame_joint"

    @property
    def tool_link_name(self) -> str:
        return "tool_frame"

    @property
    def finger_joint_names(self) -> list[str]:
        # The "real" joint, then 3 positive mimics, then 2 negative mimics.
        return [
            "finger_joint",
            "right_outer_knuckle_joint",
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]

    @property
    def open_fingers_state(self) -> float:
        return 0.0

    @property
    def closed_fingers_state(self) -> float:
        return 0.8

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state, state, state, state, -state, -state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 6
        finger_state = joint_positions[0]
        assert np.allclose(joint_positions[:4], finger_state)
        assert np.allclose(joint_positions[4:], -finger_state)
        return finger_state

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        return IKFastInfo(
            module_dir="kortex",
            module_name="ikfast_kortex",
            base_link="base_link",
            ee_link="end_effector_link",
            free_joints=["joint_7"],
        )


class KinovaGen36DOFRobotiqGripperPyBulletRobot(FingeredSingleArmPyBulletRobot[float]):
    """A 6dof Kinova Gen3 robot arm with a robotiq gripper."""

    @classmethod
    def get_name(cls) -> str:
        return "kinova-gen3-6dof"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "kortex_description" / "gen3_6dof.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "tool_frame_joint"

    @property
    def tool_link_name(self) -> str:
        return "tool_frame"

    @property
    def finger_joint_names(self) -> list[str]:
        # The "real" joint, then 3 positive mimics, then 2 negative mimics.
        return [
            "finger_joint",
            "right_outer_knuckle_joint",
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]

    @property
    def open_fingers_state(self) -> float:
        return 0.0

    @property
    def closed_fingers_state(self) -> float:
        return 0.8

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state, state, state, state, -state, -state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 6
        finger_state = joint_positions[0]
        assert np.allclose(joint_positions[:4], finger_state)
        assert np.allclose(joint_positions[4:], -finger_state)
        return finger_state

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        return IKFastInfo(
            module_dir="kortex",
            module_name="ikfast_kortex",
            base_link="base_link",
            ee_link="end_effector_link",
        )
