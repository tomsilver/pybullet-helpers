"""Boston Dynamic Spot robot."""

from pathlib import Path

import numpy as np

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    FingeredSingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class SpotPyBulletRobot(FingeredSingleArmPyBulletRobot):
    """Boston Dynamic Spot robot."""

    @classmethod
    def get_name(cls) -> str:
        return "spot"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "spot" / "spot.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        # Magic joint values read off spot in standard standing position with
        # arm tucked.
        return [
            0.00010371208190917969,
            -3.115184783935547,
            3.132749557495117,
            1.5715421438217163,
            -0.01901412010192871,
            -1.5716896057128906,
            -0.008634686470031738,
        ]

    @property
    def end_effector_name(self) -> str:
        return "arm_wr1"

    @property
    def tool_link_name(self) -> str:
        return "arm_link_wr1"

    @property
    def finger_joint_names(self) -> list[str]:
        return ["arm_f1x"]

    @property
    def open_fingers_state(self) -> float:
        return -np.pi / 2

    @property
    def closed_fingers_state(self) -> float:
        return 0.0

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 1
        finger_state = joint_positions[0]
        return finger_state
