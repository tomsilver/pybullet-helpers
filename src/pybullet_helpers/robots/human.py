"""Humans."""

from pathlib import Path

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class RightArmHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with right arm animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-right-arm"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "right_arm_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class LeftArmHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with left arm animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-left-arm"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "left_arm_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class RightLegHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with right leg animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-right-leg"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "right_leg_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, -0.14448669, -0.26559232]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class LeftLegHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human with left leg animated."""

    @classmethod
    def get_name(cls) -> str:
        return "human-left-leg"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "left_leg_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, -0.14448669, -0.26559232]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"
