"""PyBullet helpers for cameras and rendering."""

import numpy as np
import pybullet as p
from tomsutils.structs import Image

from pybullet_helpers.geometry import Pose3D


def capture_image(
    physics_client_id: int,
    camera_distance: float = 1.5,
    camera_yaw: float = 0,
    camera_pitch: float = -15,
    camera_target: Pose3D = (0, 0, 0.5),
    image_width: int = 1674,
    image_height: int = 900,
) -> Image:
    """Capture an image."""
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client_id,
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(image_width / image_height),
        nearVal=0.1,
        farVal=100.0,
        physicsClientId=physics_client_id,
    )

    (_, _, px, _, _) = p.getCameraImage(
        width=image_width,
        height=image_height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=physics_client_id,
    )

    rgb_array = np.array(px).reshape((image_height, image_width, 4))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array
