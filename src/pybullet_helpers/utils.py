"""Utility functions."""

from pathlib import Path
from typing import Sequence
import pybullet as p


def get_root_path() -> Path:
    """Get the path to the root directory of this package."""
    return Path(__file__).parent


def get_assets_path() -> Path:
    """Return the absolute path to the assets directory."""
    return get_root_path() / "assets"


def get_third_party_path() -> Path:
    """Return the absolute path to the third party directory."""
    return get_root_path() / "third_party"


def create_pybullet_block(color: tuple[float, float, float, float],
                          half_extents: tuple[float, float,
                                              float], mass: float,
                          friction: float, orientation: Sequence[float],
                          physics_client_id: int) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    pose = (0, 0, 0)

    # Create the collision shape.
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=half_extents,
                                          physicsClientId=physics_client_id)

    # Create the visual_shape.
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half_extents,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)

    # Create the body.
    block_id = p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=pose,
                                 baseOrientation=orientation,
                                 physicsClientId=physics_client_id)
    p.changeDynamics(
        block_id,
        linkIndex=-1,  # -1 for the base
        lateralFriction=friction,
        physicsClientId=physics_client_id)

    return block_id
