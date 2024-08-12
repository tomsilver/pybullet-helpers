"""Fetch Robotics Mobile Manipulator (Fetch)."""

from pathlib import Path

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import SingleArmTwoFingerGripperPyBulletRobot
from pybullet_helpers.utils import get_assets_path


class FetchPyBulletRobot(SingleArmTwoFingerGripperPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    @classmethod
    def get_name(cls) -> str:
        return "fetch"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "fetch_description" / "robots" / "fetch.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [
            -0.5591804653688366,
            -0.5948112040931831,
            1.6380556206810288,
            1.2637851140067282,
            2.1300614898498007,
            -1.794984465148684,
            0.7899035789605409,
            0.0,
            0.0,
        ]

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
    def open_fingers_joint_value(self) -> float:
        return 0.04

    @property
    def closed_fingers_joint_value(self) -> float:
        return 0.01
