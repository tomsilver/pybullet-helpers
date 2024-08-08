"""Create PyBullet robots."""

from pybullet_helpers.robots.fetch import FetchPyBulletRobot
from pybullet_helpers.robots.panda import PandaPyBulletRobot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


def create_pybullet_robot(
    robot_name: str, physics_client_id: int, *args, **kwargs
) -> SingleArmPyBulletRobot:
    """Create a known PyBullet robot from its name."""
    if robot_name == "panda":
        return PandaPyBulletRobot(physics_client_id, *args, **kwargs)
    if robot_name == "fetch":
        return FetchPyBulletRobot(physics_client_id, *args, **kwargs)
    raise NotImplementedError(f"Unknown robot {robot_name}")
