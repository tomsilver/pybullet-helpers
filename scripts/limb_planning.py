"""Experimenting with motion planning for limb repositioning."""

from typing import Iterator

import imageio.v2 as iio
import numpy as np
from scipy.spatial.transform import Rotation as R
from tomsutils.motion_planning import BiRRT

from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, interpolate_poses, multiply_poses
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, get_joint_infos
from pybullet_helpers.motion_planning import MotionPlanningHyperparameters
from pybullet_helpers.robots import create_pybullet_robot


def _main():
    # Initialize hyperparameters for planning.
    seed = 0
    fps = 5
    hyperparameters = MotionPlanningHyperparameters(birrt_extend_num_interp=25)
    num_interp = hyperparameters.birrt_extend_num_interp

    # Create physics client.
    physics_client_id = create_gui_connection()

    # Set up the robots.
    robot_init_pos = [0.8, -0.1, 0.5]
    robot_init_orn_obj = R.from_euler("xyz", [0, 0, np.pi])
    robot_base_pose = Pose(robot_init_pos, robot_init_orn_obj.as_quat())
    human_init_pos = [0.15, 0.1, 1.4]
    human_init_orn_obj = R.from_euler("xyz", [np.pi, 0, 0])
    human_base_pose = Pose(human_init_pos, human_init_orn_obj.as_quat())
    robot = create_pybullet_robot(
        "panda-limb-repo", physics_client_id, base_pose=robot_base_pose
    )
    human = create_pybullet_robot(
        "human-arm-6dof", physics_client_id, base_pose=human_base_pose
    )
    robot_init_joints = [
        0.94578431,
        -0.89487842,
        -1.67534487,
        -0.34826698,
        1.73607292,
        0.14233887,
    ]
    human_init_joints = [
        1.43252278,
        -0.81111486,
        -0.42373363,
        0.49931369,
        -1.17420521,
        0.37122887,
    ]
    robot.set_joints(robot_init_joints)
    human.set_joints(human_init_joints)
    init_joints = robot_init_joints + human_init_joints
    robot_joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, physics_client_id
    )
    num_robot_dof = len(robot_joint_infos)

    # Get transform between human and robot end effectors.
    robot_init_ee = robot.get_end_effector_pose()
    human_init_ee = human.get_end_effector_pose()
    robot_ee_to_human_ee = multiply_poses(robot_init_ee.invert(), human_init_ee)

    # Create a target state for the human.
    human_ee_relative_target = Pose(
        (-0.3, 0.0, 0.2), R.from_euler("xyz", [0, 0, -np.pi / 4]).as_quat()
    )
    human_target_ee = multiply_poses(human_init_ee, human_ee_relative_target)
    robot_target_ee = multiply_poses(human_target_ee, robot_ee_to_human_ee.invert())
    human_target_joints = inverse_kinematics(human, human_target_ee)
    robot_target_joints = inverse_kinematics(robot, robot_target_ee)
    target_joints = robot_target_joints + human_target_joints

    # The state for motion planning is the robot's joints concatenated with
    # the human's joints. But we will sample in the robot's joint space and
    # then run IK for the human.
    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)

    def _sampling_fn(_: JointPositions) -> JointPositions:
        # Retry sampling until we find a feasible state that works for both the
        # robot and the human.
        while True:
            new_robot_joints: JointPositions = list(joint_space.sample())
            # Run FK to get new robot joint position.
            new_robot_ee = robot.forward_kinematics(new_robot_joints)
            # Run IK to get a new human joint position.
            new_human_ee = multiply_poses(new_robot_ee, robot_ee_to_human_ee)
            try:
                new_human_joints = inverse_kinematics(human, new_human_ee)
            except InverseKinematicsError:
                continue
            # Succeeded.
            return new_robot_joints + new_human_joints

    # Interpolate in end effector space.
    def _extend_fn(
        pt1: JointPositions, pt2: JointPositions
    ) -> Iterator[JointPositions]:
        # Interpolate in end effector space.
        robot_joints1 = pt1[:num_robot_dof]
        robot_joints2 = pt2[:num_robot_dof]
        robot_ee_pose1 = robot.forward_kinematics(robot_joints1)
        robot_ee_pose2 = robot.forward_kinematics(robot_joints2)
        for robot_ee_pose in interpolate_poses(
            robot_ee_pose1, robot_ee_pose2, num_interp=num_interp, include_start=False
        ):
            # Run inverse kinematics for both robot and human.
            robot_joints = inverse_kinematics(robot, robot_ee_pose)
            human_ee_pose = multiply_poses(robot_ee_pose, robot_ee_to_human_ee)
            human_joints = inverse_kinematics(human, human_ee_pose)
            yield robot_joints + human_joints

    def _distance_fn(pt1: JointPositions, pt2: JointPositions) -> float:
        robot_joints1 = pt1[:num_robot_dof]
        robot_joints2 = pt2[:num_robot_dof]
        from_ee = robot.forward_kinematics(robot_joints1).position
        to_ee = robot.forward_kinematics(robot_joints2).position
        return sum(np.subtract(from_ee, to_ee) ** 2)

    # Collision function doesn't do anything.
    _collision_fn = lambda _: False

    birrt = BiRRT(
        _sampling_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=hyperparameters.birrt_num_attempts,
        num_iters=hyperparameters.birrt_num_iters,
        smooth_amt=hyperparameters.birrt_smooth_amt,
    )

    plan = birrt.query(init_joints, target_joints)

    imgs = []
    for s in plan:
        robot_joints, human_joints = s[:num_robot_dof], s[num_robot_dof:]
        robot.set_joints(robot_joints)
        human.set_joints(human_joints)
        img = capture_image(physics_client_id, camera_target=human_init_ee.position)
        imgs.append(img.astype(np.uint8))

    outfile = "limb_planning.mp4"
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
