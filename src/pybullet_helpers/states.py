"""Utilities for representing states."""

from __future__ import annotations

from dataclasses import dataclass, field

from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_state
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


@dataclass(frozen=True)
class KinematicState:
    """Robot positions, object poses, and attachments (grasps)."""

    robot_joints: JointPositions
    object_poses: dict[int, Pose]  # maps pybullet ID to pose
    # Maps held object IDs to end-effector transforms.
    attachments: dict[int, Pose] = field(default_factory=dict)

    @classmethod
    def from_pybullet(
        cls,
        robot: SingleArmPyBulletRobot,
        object_ids: set[int],
        attached_object_ids: set[int] | None = None,
    ) -> KinematicState:
        """Create a KinematicStatic from pybullet."""
        robot_joints = robot.get_joint_positions()
        object_poses = {o: get_pose(o, robot.physics_client_id) for o in object_ids}
        attachments: dict[int, Pose] = {}
        if attached_object_ids is not None:
            world_to_ee_link = get_link_state(
                robot.robot_id,
                robot.end_effector_id,
                physics_client_id=robot.physics_client_id,
            ).com_pose
            for object_id in attached_object_ids:
                world_to_object = get_pose(object_id, robot.physics_client_id)
                ee_link_to_object = multiply_poses(
                    world_to_ee_link.invert(), world_to_object
                )
                attachments[object_id] = ee_link_to_object
        return KinematicState(robot_joints, object_poses, attachments)

    def set_pybullet(self, robot: SingleArmPyBulletRobot) -> None:
        """Reset the pybullet simulator to this kinematic state."""
        robot.set_joints(self.robot_joints)
        for object_id, pose in self.object_poses.items():
            set_pose(object_id, pose, robot.physics_client_id)
        world_to_ee_link = get_link_state(
            robot.robot_id,
            robot.end_effector_id,
            physics_client_id=robot.physics_client_id,
        ).com_pose
        for object_id, ee_link_to_object in self.attachments.items():
            world_to_object = multiply_poses(world_to_ee_link, ee_link_to_object)
            set_pose(object_id, world_to_object, robot.physics_client_id)