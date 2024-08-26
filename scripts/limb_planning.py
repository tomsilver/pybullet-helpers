"""Experimenting with motion planning for limb repositioning."""

import numpy as np
import pybullet as p

from tomsutils.motion_planning import BiRRT
from pybullet_helpers.robots import create_pybullet_robot
from scipy.spatial.transform import Rotation as R
from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import inverse_kinematics

def _main():
    physics_client_id = create_gui_connection()

    # Set up the robots.
    robot_init_pos = [0.8, -0.1, 0.5]
    robot_init_orn_obj = R.from_euler('xyz', [0,0,np.pi])
    robot_base_pose = Pose(robot_init_pos, robot_init_orn_obj.as_quat())
    human_init_pos = [0.15, 0.1, 1.4]
    human_init_orn_obj = R.from_euler('xyz', [np.pi,0,0])
    human_base_pose = Pose(human_init_pos, human_init_orn_obj.as_quat())
    robot = create_pybullet_robot("panda-limb-repo", physics_client_id, base_pose=robot_base_pose)
    human = create_pybullet_robot("human-arm-6dof", physics_client_id, base_pose=human_base_pose)
    robot_init_joints = [0.94578431,-0.89487842,-1.67534487,-0.34826698,1.73607292,0.14233887]
    human_init_joints = [1.43252278,-0.81111486,-0.42373363,0.49931369,-1.17420521,0.37122887]
    robot.set_joints(robot_init_joints)
    human.set_joints(human_init_joints)

    # Create a target state for the human.
    human_init_ee = human.get_end_effector_pose()
    human_target_ee = Pose(tuple(np.add((0.0, 0.0, 0.0), human_init_ee.position)),
                           human_init_ee.orientation)

    inverse_kinematics(human, human_init_ee)

    while True:
        p.stepSimulation(physics_client_id)


    # Sample in the robot's joint space and then run IK for the human.


if __name__ == "__main__":
    _main()
