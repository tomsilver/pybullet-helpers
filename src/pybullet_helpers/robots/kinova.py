"""Kinova Gen3 robots."""

from pathlib import Path

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.utils import get_assets_path


class KinovaGen3NoGripperPyBulletRobot(SingleArmPyBulletRobot):
    """A Kinova Gen3 robot arm with no gripper."""

    @classmethod
    def get_name(cls) -> str:
        return "kinova-gen3-no-gripper"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "kinova_description" / "GEN3_URDF_V12.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6]

    @property
    def end_effector_name(self) -> str:
        return "EndEffector"

    @property
    def tool_link_name(self) -> str:
        return "EndEffector_Link"
