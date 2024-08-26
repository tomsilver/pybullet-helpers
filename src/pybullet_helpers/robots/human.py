"""Human arm."""

from pathlib import Path
from typing import Optional

from pybullet_helpers.ikfast import IKFastInfo
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class HumanArm6DoF(SingleArmPyBulletRobot):
    """Franka Emika Panda with the limb repo end effector block."""

    @classmethod
    def get_name(cls) -> str:
        return "human-arm-6dof"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human" / "arm_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, -1.08786023, -0.14448669, -0.26559232]

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        return IKFastInfo(
            module_dir="human",
            module_name="ikfast_human",
            base_link="base_link",
            ee_link="ee_link",
            free_joints=[],
        )
