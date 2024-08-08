"""Kinova Gen3 robots."""

from pathlib import Path

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
    SingleArmTwoFingerGripperPyBulletRobot,
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


class KinovaGen3RobotiqGripperPyBulletRobot(SingleArmTwoFingerGripperPyBulletRobot):
    """A Kinova Gen3 robot arm with a robotiq gripper."""

    @classmethod
    def get_name(cls) -> str:
        return "kinova-gen3"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "kortex_description" / "gen3_robotiq_2f_85.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "end_effector"

    @property
    def tool_link_name(self) -> str:
        return "end_effector_link"

    @property
    def left_finger_joint_name(self) -> str:
        return "left_inner_finger_joint"

    @property
    def right_finger_joint_name(self) -> str:
        return "right_inner_finger_joint"

    @property
    def open_fingers(self) -> float:
        return 0.5

    @property
    def closed_fingers(self) -> float:
        return -0.5
