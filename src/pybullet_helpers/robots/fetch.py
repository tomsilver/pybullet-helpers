"""Fetch Robotics Mobile Manipulator (Fetch)."""

from pathlib import Path

from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.utils import get_assets_path
from pybullet_helpers.joint import JointPositions


class FetchPyBulletRobot(SingleArmPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    @classmethod
    def get_name(cls) -> str:
        return "fetch"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "fetch_description" / "robots" / "fetch.urdf"
    
    def home_joint_positions(self) -> JointPositions:
        import ipdb; ipdb.set_trace()

    @property
    def end_effector_name(self) -> str:
        return "gripper_axis"

    @property
    def tool_link_name(self) -> str:
        return "gripper_link"

    @property
    def left_finger_joint_name(self) -> str:
        return "l_gripper_finger_joint"

    @property
    def right_finger_joint_name(self) -> str:
        return "r_gripper_finger_joint"

    @property
    def open_fingers(self) -> float:
        return 0.04

    @property
    def closed_fingers(self) -> float:
        return 0.01
