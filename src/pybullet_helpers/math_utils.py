"""Math utilities."""

import numpy as np
from pybullet_utils.transformations import quaternion_from_matrix

from pybullet_helpers.geometry import Pose


def get_poses_facing_line(
    axis: tuple[float, float, float],
    point_on_line: tuple[float, float, float],
    radius: float,
    num_points: int,
) -> list[Pose]:
    """Generate poses that are rotated around a given line at a given radius,
    facing towards the line.

    "Facing" means that the z dim of the pose is pointing toward the
    line. The x dim is pointing right and the y dim is pointing down.

    A typical use case is generating multiple candidate grasps of an
    object.
    """
    assert np.isclose(np.linalg.norm(axis), 1.0), "axis should have unit norm"

    # Create a vector orthogonal to the axis to define a circle plane.
    if np.allclose(axis, [0, 0, 1]):
        ortho_vector = np.array([1, 0, 0])
    else:
        ortho_vector = np.cross(axis, [0, 0, 1])
    assert np.isclose(np.linalg.norm(ortho_vector), 1.0)

    # Compute points on the circle.
    poses = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        position_offset = np.cos(angle) * ortho_vector + np.sin(angle) * np.cross(
            axis, ortho_vector
        )
        position = point_on_line + radius * position_offset

        # Compute the orientation (rotation) quaternion
        z_axis = point_on_line - position  # vector from position to point_on_line
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, axis)
        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T
        M = np.identity(4)
        M[:3, :3] = rotation_matrix
        orientation = quaternion_from_matrix(M)

        poses.append(Pose(position=tuple(position), orientation=tuple(orientation)))

    return poses