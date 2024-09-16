"""Stretch SE3 robot with SG3 end effector."""

import importlib.resources as importlib_resources
from pathlib import Path

import numpy as np

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_relative_link_pose
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

        # Set up custom IK, which works by transforming the target pose into
        # the wrist frame and then handles the target wrist position (which has
        # two degrees of freedom).
        
        # Determine reachability bounds for the arm extending outward.
        joints = list(self.joint_lower_limits)
        self.set_joints(joints)
        arm_l3_min_pose = self._get_relative_wrist_pose()
        self._arm_horizontal_bounds = [arm_l3_min_pose.position[2]]
        for arm_joint_name in ["joint_arm_l3", "joint_arm_l2", "joint_arm_l1", "joint_arm_l0"]:
            joint_idx = self.arm_joints.index(self.joint_from_name(arm_joint_name))
            joints[joint_idx] = self.joint_upper_limits[joint_idx]
            self.set_joints(joints)
            joint_max_pose = self._get_relative_wrist_pose()
            # Only the z dimension should be changing.
            assert np.allclose(arm_l3_min_pose.position[:2], joint_max_pose.position[:2])
            assert np.allclose(arm_l3_min_pose.orientation, joint_max_pose.orientation)
            self._arm_horizontal_bounds.append(joint_max_pose.position[2])

        # Determine reachability bounds for the arm moving up and down.
        import ipdb; ipdb.set_trace()

        # Reset the joints after 'calibration'.
        # self.set_joints(self._home_joint_positions)

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
    ):
        import ipdb

        ipdb.set_trace()

    def _get_relative_wrist_pose(self) -> Pose:
        """Get the pose of the wrist."""
        return get_relative_link_pose(
            self.robot_id,
            -1,
            self._wrist_id,
            physics_client_id=self.physics_client_id,
        )
