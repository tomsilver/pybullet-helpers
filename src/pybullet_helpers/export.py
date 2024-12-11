"""Code for exporting to URDF."""

import os

import pybullet as p

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import get_joint_info, get_num_joints
from pybullet_helpers.link import get_link_state


def _get_urdf_pose_str(pos: list[float], orn: list[float]) -> str:
    euler = p.getEulerFromQuaternion(orn)
    return f'<origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{euler[0]} {euler[1]} {euler[2]}"/>'


def _get_urdf_geometry_str(geom_type: int, dims: list[int], filename: str) -> str:
    if geom_type == p.GEOM_SPHERE:
        geometry_str = f'<sphere radius="{dims[0]}"/>'
    elif geom_type == p.GEOM_BOX:
        # dims are half-extents.
        geometry_str = f'<box size="{dims[0]} {dims[1]} {dims[2]}"/>'
    elif geom_type == p.GEOM_CYLINDER:
        # dims = [length, radius].
        length = dims[0]
        radius = dims[1]
        geometry_str = f'<cylinder radius="{radius}" length="{length}"/>'
    elif geom_type == p.GEOM_MESH:
        # dims for mesh are scaling factors.
        if filename == "unknown_file":
            raise ValueError("Tried to create mesh with unknown file.")
        filepath = os.path.relpath(filename)
        geometry_str = (
            f'<mesh filename="{filepath}" scale="{dims[0]} {dims[1]} {dims[2]}"/>'
        )
    else:
        raise ValueError(f"Unsupported geom type: {geom_type}")
    return geometry_str


def _get_urdf_joint_type(joint_type: int) -> str:
    if joint_type == p.JOINT_REVOLUTE:
        jtype_str = "revolute"
    elif joint_type == p.JOINT_PRISMATIC:
        jtype_str = "prismatic"
    elif joint_type == p.JOINT_FIXED:
        jtype_str = "fixed"
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")
    return jtype_str


def create_urdf_from_body_id(
    body_id: int, physics_client_id: int, name: str = "pybullet-extracted"
) -> str:
    """Create a URDF string for a body that's loaded into pybullet."""

    # Start building the URDF content.
    urdf_content = '<?xml version="1.0"?>\n'
    urdf_content += f'<robot name="{name}">\n'

    num_joints = get_num_joints(body_id, physics_client_id)

    # Gather visual shape data once, so we don't repeat calls.
    visual_data = p.getVisualShapeData(body_id, physicsClientId=physics_client_id)

    # Handle each link, including the base link (which is index -1 in PyBullet).
    for link_idx in range(-1, num_joints):
        if link_idx == -1:
            link_name = "base_link"
        else:
            joint_info = get_joint_info(body_id, link_idx, physics_client_id)
            link_name = joint_info.linkName

        # Visual shapes for this link.
        link_visuals = [v for v in visual_data if v[1] == link_idx]

        # Collision shapes for this link.
        collision_data = p.getCollisionShapeData(
            body_id, link_idx, physicsClientId=physics_client_id
        )

        urdf_content += f'  <link name="{link_name}">\n'

        # Add visual elements.
        for v in link_visuals:
            # Visual shape data format:
            # v = (bodyUniqueId, linkIndex, visualGeometryType, dimensions, filename,
            #      localVisualFramePos, localVisualFrameOrn, color, specular).
            geom_type = v[2]
            dims = v[3]
            filename = v[4].decode("UTF-8")
            pos = v[5]
            orn = v[6]
            color = v[7]

            urdf_content += "    <visual>\n"

            # Add pose.
            pose_str = _get_urdf_pose_str(pos, orn)
            urdf_content += f"      {pose_str}\n"

            # Add geometry.
            geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
            urdf_content += "      <geometry>\n"
            urdf_content += f"        {geometry_str}\n"
            urdf_content += "      </geometry>\n"

            # Add color.
            r, g, b, a = color
            urdf_content += f'      <material name="color_{link_idx+1}">\n'
            urdf_content += f'        <color rgba="{r} {g} {b} {a}"/>\n'
            urdf_content += "     </material>\n"

            urdf_content += "    </visual>\n"

        # Add collision elements.
        for c in collision_data:
            # Collision shape data format:
            # c = (object id, linkIndex, geometryType, dimensions, filename,
            #      localCollisionFramePos, localCollisionFrameOrn).
            geom_type = c[2]
            dims = c[3]
            filename = c[4].decode("UTF-8")
            com_to_pose = Pose(c[5], c[6])
            if c[1] == -1:
                origin_to_pose = com_to_pose
            else:
                link_state = get_link_state(c[0], c[1], physics_client_id)
                tf = Pose(
                    link_state.localInertialFramePosition,
                    link_state.localInertialFrameOrientation,
                )
                origin_to_pose = multiply_poses(com_to_pose, tf)
            pos = origin_to_pose.position
            orn = origin_to_pose.orientation

            urdf_content += "    <collision>\n"

            # Add pose.
            pose_str = _get_urdf_pose_str(pos, orn)
            urdf_content += f"      {pose_str}\n"

            # Add geometry.
            try:
                geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
            except ValueError:
                # PyBullet seems to internally replace cylinders with meshes
                # when loading from URDF. In this case the best we can do is
                # steal the geometry from visual shape data.
                matching_link_visuals = [v for v in link_visuals if v[:2] == c[:2]]
                if len(matching_link_visuals) == 1:
                    print("WARNING: using visual geometry for collisions.")
                    v = matching_link_visuals[0]
                    geom_type = v[2]
                    dims = v[3]
                    geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
            urdf_content += "      <geometry>\n"
            urdf_content += f"        {geometry_str}\n"
            urdf_content += "      </geometry>\n"

            urdf_content += "    </collision>\n"

        # Add inertial properties.
        dynamics_info = p.getDynamicsInfo(
            body_id, link_idx, physicsClientId=physics_client_id
        )
        mass = dynamics_info[0]
        ixx, iyy, izz = dynamics_info[2]
        inertial_pos = dynamics_info[3]
        inertial_orn = dynamics_info[4]
        inertial_pose_str = _get_urdf_pose_str(inertial_pos, inertial_orn)

        urdf_content += "    <inertial>\n"
        urdf_content += f"      {inertial_pose_str}\n"
        urdf_content += f'      <mass value="{mass}"/>\n'
        urdf_content += f'      <inertia ixx="{ixx}" ixy="0.0" ixz="0.0" iyy="{iyy}" iyz="0.0" izz="{izz}"/>\n'
        urdf_content += "    </inertial>\n"

        urdf_content += "  </link>\n"

    # Add joints.
    for joint_idx in range(num_joints):
        joint_info = get_joint_info(body_id, joint_idx, physics_client_id)
        joint_name = joint_info.jointName
        joint_type = joint_info.jointType
        jtype_str = _get_urdf_joint_type(joint_type)

        parent_idx = joint_info.parentIndex
        child_name = joint_info.linkName
        parent_name = (
            "base_link"
            if parent_idx == -1
            else get_joint_info(body_id, parent_idx, physics_client_id).linkName
        )

        parent_frame_pos = joint_info.parentFramePos
        parent_frame_orn = joint_info.parentFrameOrn

        if parent_idx != -1:
            parent_link_state = get_link_state(body_id, parent_idx, physics_client_id)
            tf = Pose(
                parent_link_state.localInertialFramePosition,
                link_state.localInertialFrameOrientation,
            )
            local_frame = multiply_poses(Pose(parent_frame_pos, parent_frame_orn), tf)
            parent_frame_pos = local_frame.position
            parent_frame_orn = local_frame.orientation

        peuler = p.getEulerFromQuaternion(parent_frame_orn)

        urdf_content += f'  <joint name="{joint_name}" type="{jtype_str}">\n'
        urdf_content += f'    <parent link="{parent_name}"/>\n'
        urdf_content += f'    <child link="{child_name}"/>\n'
        urdf_content += (
            f'    <origin xyz="{parent_frame_pos[0]} {parent_frame_pos[1]} '
            f'{parent_frame_pos[2]}" rpy="{peuler[0]} {peuler[1]} {peuler[2]}"/>\n'
        )

        # If not fixed, add the axis and limits.
        if jtype_str in ["revolute", "prismatic"]:
            joint_axis = joint_info.jointAxis
            urdf_content += (
                f'    <axis xyz="{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"/>\n'
            )
            lower = joint_info.jointLowerLimit
            upper = joint_info.jointUpperLimit
            max_velocity = joint_info.jointMaxVelocity
            max_effort = joint_info.jointMaxForce
            urdf_content += f'    <limit effort="{max_effort}" velocity="{max_velocity}" lower="{lower}" upper="{upper}"/>\n'

        urdf_content += "  </joint>\n"

    urdf_content += "</robot>\n"

    return urdf_content
