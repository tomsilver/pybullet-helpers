"""Human from assistive gym with right arm animated."""

import itertools
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


class AssistiveHumanPyBulletRobot(SingleArmPyBulletRobot):
    """Human from assistive gym with right arm animated."""

    @classmethod
    def get_name(cls) -> str:
        return "assistive-human"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "human_description" / "human.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [
            0.0,
            0.0,
            0.0,
            np.pi / 2,
            0.0,
            0.0,
            -np.pi / 2
        ]

    @property
    def end_effector_name(self) -> str:
        return "end_effector"

    @property
    def tool_link_name(self) -> str:
        return "end_effector_link"
    
