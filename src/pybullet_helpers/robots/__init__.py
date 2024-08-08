"""Create PyBullet robots."""

from typing import Type

from pybullet_helpers.robots.fetch import FetchPyBulletRobot
from pybullet_helpers.robots.kinova import KinovaGen3NoGripperPyBulletRobot
from pybullet_helpers.robots.panda import PandaPyBulletRobot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

_BUILT_IN_ROBOT_CLASSES: list[Type[SingleArmPyBulletRobot]] = [
    FetchPyBulletRobot,
    PandaPyBulletRobot,
    KinovaGen3NoGripperPyBulletRobot,
]


def create_pybullet_robot(
    robot_name: str, physics_client_id: int, *args, **kwargs
) -> SingleArmPyBulletRobot:
    """Create a known PyBullet robot from its name."""
    for cls in _BUILT_IN_ROBOT_CLASSES:
        if robot_name == cls.get_name():
            return cls(physics_client_id, *args, **kwargs)
    raise NotImplementedError(f"Unknown robot {robot_name}")
