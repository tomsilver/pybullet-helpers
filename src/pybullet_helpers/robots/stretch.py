"""Stretch SE3 robot with SG3 end effector."""

import importlib.resources as importlib_resources
from pathlib import Path

import numpy as np

from pybullet_helpers.geometry import Pose, multiply_poses, matrix_from_quat
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_relative_link_pose, get_link_pose
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot


class StretchPyBulletRobot(FingeredSingleArmPyBulletRobot[float]):
    """Stretch SE3 robot with SG3 end effector."""

    def __init__(
        self, physics_client_id: int, fixed_base: bool = False, **kwargs
    ) -> None:
        """By default, stretch can move its based."""
        super().__init__(physics_client_id, fixed_base=fixed_base, **kwargs)

        # The wrist is used for IK 'calibration'.
        self._wrist_id = self.link_from_name("link_arm_l0")

        # Set up values for custom IK.
        
        # Determine reachability bounds for the arm extending outward.
        joints = list(self.joint_lower_limits)
        self.set_joints(joints)
        min_pose = self._get_relative_wrist_pose()
        self._arm_horizontal_bounds = [min_pose.position[2]]
        for arm_joint_name in ["joint_arm_l3", "joint_arm_l2", "joint_arm_l1", "joint_arm_l0"]:
            joint_idx = self.arm_joints.index(self.joint_from_name(arm_joint_name))
            joints[joint_idx] = self.joint_upper_limits[joint_idx]
            self.set_joints(joints)
            joint_max_pose = self._get_relative_wrist_pose()
            # Only the z dimension should be changing.
            assert np.allclose(min_pose.position[:2], joint_max_pose.position[:2])
            assert np.allclose(min_pose.orientation, joint_max_pose.orientation)
            self._arm_horizontal_bounds.append(joint_max_pose.position[2])

        # Determine reachability bounds for the arm moving up and down.
        joints = list(self.joint_lower_limits)
        self.set_joints(joints)
        self._arm_vertical_bounds = [min_pose.position[1]]
        arm_joint_name = "joint_lift"
        joint_idx = self.arm_joints.index(self.joint_from_name(arm_joint_name))
        joints[joint_idx] = self.joint_upper_limits[joint_idx]
        self.set_joints(joints)
        joint_max_pose = self._get_relative_wrist_pose()
        # Only the y dimension should be changing, but there is small error.
        assert np.isclose(min_pose.position[0], joint_max_pose.position[0], atol=1e-3)
        assert np.isclose(min_pose.position[2], joint_max_pose.position[2], atol=1e-3)
        assert np.allclose(min_pose.orientation, joint_max_pose.orientation, atol=1e-3)
        self._arm_vertical_bounds.append(joint_max_pose.position[1])

        # Determine approximate "radius" from wrist to end effector and the
        # offset between the end effector and the wrist plane in the nominal.
        # TODO simplify and remove a lot of this
        self.set_joints(self.joint_lower_limits)
        ee_pose = self.get_end_effector_pose()
        wrist_pose = get_link_pose(self.robot_id, self._wrist_id, self.physics_client_id)
        self._wrist_radius = np.sqrt(np.sum(np.square(np.subtract(ee_pose.position, wrist_pose.position))))
        self._wrist_ee_offset = Pose(tuple(np.subtract(wrist_pose.position, ee_pose.position)))

        # Reset the joints after 'calibration'.
        self.set_joints(self._home_joint_positions)

    @classmethod
    def get_name(cls) -> str:
        return "stretch"

    @classmethod
    def urdf_path(cls) -> Path:
        pkg = Path(str(importlib_resources.files("stretch_urdf")))
        filepath = pkg / "SE3" / "stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf"
        with open(filepath, "r", encoding="utf-8") as f:
            urdf_str = f.read()
        # The original URDF file has lower == upper == 0 for the fingers, which
        # breaks some assumptions and prevents moving the fingers.
        urdf_str = urdf_str.replace(
            '<limit effort="0" lower="0" upper="0" velocity="0"/>',
            '<limit effort="0" lower="0" upper="0.5" velocity="0"/>',
        )
        # NOTE: it is unfortunately necessary to write in the same directory
        # as the original URDF file because otherwise the STL files won't be
        # found (because they are defined relatively in the URDF).
        new_filepath = filepath.parent / (filepath.stem + "-PYBULLET-HELPERS.urdf")
        with open(new_filepath, mode="w", encoding="utf-8") as new_f:
            new_f.write(urdf_str)
        return new_filepath

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.65, 0.025, 0.025, 0.04, 0.0157, 0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "joint_gripper_s3_body"

    @property
    def tool_link_name(self) -> str:
        return "link_gripper_s3_body"

    @property
    def finger_joint_names(self) -> list[str]:
        return ["joint_gripper_finger_right", "joint_gripper_finger_left"]

    @property
    def open_fingers_state(self) -> float:
        return 0.5

    @property
    def closed_fingers_state(self) -> float:
        return 0.0

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state, state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 2
        assert np.isclose(joint_positions[0], joint_positions[1])
        return joint_positions[0]

    @property
    def default_inverse_kinematics_method(self) -> str:
        return "custom"

    def custom_inverse_kinematics(
        self,
        end_effector_pose: Pose,
        validate: bool = True,
        validation_atol: float = 1e-3,
    ) -> JointPositions:
        
        # Find a good initialization for differential IK by approximating the
        # wrist->end effector as a sphere of constant radius.

        # Start by transforming the end effector pose into the robot frame.
        initial_joints = self.get_joint_positions()
        self.set_joints(self.joint_lower_limits)
        world_to_robot = get_link_pose(self.robot_id, self._wrist_id, self.physics_client_id)
        world_to_target = end_effector_pose
        robot_to_target = multiply_poses(world_to_robot.invert(), world_to_target)

        # Use the sphere radius to determine a point near the reachable plane.
        z_axis_start = np.array([0, 0, -self._wrist_radius])
        rotation_matrix = matrix_from_quat(robot_to_target.orientation)
        z_axis_end = robot_to_target.position + rotation_matrix.dot(z_axis_start)

        world_wrist_target = multiply_poses(self._wrist_ee_offset, end_effector_pose)

        # Determine the reachable plane for the wrist relative to the robot origin.
        reachable_plane_width = max(self._arm_horizontal_bounds) - min(self._arm_horizontal_bounds)
        reachable_plane_height = max(self._arm_vertical_bounds) - min(self._arm_vertical_bounds)

        # Project the point onto the plane.


        self.set_joints(initial_joints)

        import pybullet as p
        from pybullet_helpers.utils import create_pybullet_block
        from pybullet_helpers.gui import visualize_pose
        
        # visualize the reachable plane
        plane_id = create_pybullet_block((0.9, 0.1, 0.9, 0.25), [1e-2, reachable_plane_height / 2, reachable_plane_width / 2], self.physics_client_id)
        p.setCollisionFilterGroupMask(plane_id, -1, 0, 0, self.physics_client_id)
        plane_origin_to_plane = Pose((0, -reachable_plane_height / 2, -reachable_plane_width / 2))
        world_to_plane = world_to_robot
        world_to_plane_origin = multiply_poses(world_to_plane, plane_origin_to_plane.invert())
        p.resetBasePositionAndOrientation(plane_id,
                                          world_to_plane_origin.position,
                                      world_to_plane_origin.orientation,
                                      self.physics_client_id)
        for joint in range(-1, len(self.joint_infos)):
            p.changeVisualShape(
                self.robot_id,
                joint,
                rgbaColor=[0.0, 1.0, 0.0, 0.5],
                physicsClientId=self.physics_client_id,
            )

        # visualize the key points
        visualize_pose(get_link_pose(self.robot_id, self._wrist_id, self.physics_client_id),
                       physics_client_id=self.physics_client_id)

        p.addUserDebugLine(
            lineFromXYZ=multiply_poses(world_to_robot, robot_to_target).position,
            lineToXYZ=multiply_poses(world_to_robot, Pose(z_axis_end)).position,
            lineColorRGB=(1, 0, 0),
            lifeTime=0,
            physicsClientId=self.physics_client_id,
        )
        
        p.addUserDebugPoints([world_wrist_target.position],
                             [(1, 0, 0)],
                             pointSize=10,
                             lifeTime=0,
                             physicsClientId=self.physics_client_id)
        
        # p.addUserDebugLine(
        #     lineFromXYZ=self.get_end_effector_pose().position ,
        #     lineToXYZ=np.subtract(self.get_end_effector_pose().position, self._wrist_ee_offset),
        #     lineColorRGB=(0, 0, 1),
        #     lifeTime=0,
        #     physicsClientId=self.physics_client_id,
        # )

        # p.addUserDebugPoints([multiply_poses(world_to_robot, Pose(relative_wrist_target_position)).position],
        #                      [(0, 0, 1)],
        #                      pointSize=10,
        #                      lifeTime=0,
        #                      physicsClientId=self.physics_client_id)


        # p.addUserDebugPoints([multiply_poses(world_to_robot, Pose(relative_wrist_target_position)).position],
        #                      [(0, 0, 1)],
        #                      pointSize=10,
        #                      lifeTime=0,
        #                      physicsClientId=self.physics_client_id)

        while True:
            p.stepSimulation()

        

    def _get_relative_wrist_pose(self) -> Pose:
        """Get the pose of the wrist."""
        return get_relative_link_pose(
            self.robot_id,
            -1,
            self._wrist_id,
            physics_client_id=self.physics_client_id,
        )
