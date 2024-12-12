"""Code for exporting to URDF."""

import os

import pybullet as p

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import get_joint_info, get_num_joints
from pybullet_helpers.link import get_link_state


def _get_visual_shape_data_for_link(
    body_id: int, link_idx: int, physics_client_id: int
) -> list:
    return [
        v
        for v in p.getVisualShapeData(body_id, physicsClientId=physics_client_id)
        if v[1] == link_idx
    ]


def _get_urdf_pose_str(
    pos: tuple[float, float, float], orn: tuple[float, float, float, float]
) -> str:
    euler = p.getEulerFromQuaternion(orn)
    return (
        f'<origin xyz="{pos[0]} {pos[1]} {pos[2]}" '
        f'rpy="{euler[0]} {euler[1]} {euler[2]}"/>'
    )


def _get_urdf_geometry_str(
    geom_type: int, dims: tuple[float, float, float], filename: str
) -> str:
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


def _get_urdf_lines_for_link(
    body_id: int, link_idx: int, physics_client_id: int
) -> list[tuple[str, int]]:
    # Get the link name.
    if link_idx == -1:
        link_name = "base_link"
    else:
        joint_info = get_joint_info(body_id, link_idx, physics_client_id)
        link_name = joint_info.linkName
        assert link_name != "base_link", "Cannot have multiple base links"

    # Start the URDF for the link.
    urdf_lines = [(f'<link name="{link_name}">', 2)]

    # Add visual tags.
    visual_urdf_lines = _get_visual_urdf_lines_for_link(
        body_id, link_idx, physics_client_id
    )
    urdf_lines.extend(visual_urdf_lines)

    # Add collision tags.
    collision_urdf_lines = _get_collision_urdf_lines_for_link(
        body_id, link_idx, physics_client_id
    )
    urdf_lines.extend(collision_urdf_lines)

    # Add inertial tag.
    inertial_urdf_lines = _get_inertial_urdf_lines_for_link(
        body_id, link_idx, physics_client_id
    )
    urdf_lines.extend(inertial_urdf_lines)

    # Finish the URDF for the link.
    urdf_lines.append(("</link>", 2))

    return urdf_lines


def _get_visual_urdf_lines_for_link(
    body_id: int, link_idx: int, physics_client_id: int
) -> list[tuple[str, int]]:
    urdf_lines = []
    link_visuals = _get_visual_shape_data_for_link(body_id, link_idx, physics_client_id)
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

        urdf_lines.extend(
            _get_single_visual_urdf_lines_for_link(
                geom_type, dims, filename, pos, orn, color
            )
        )
    return urdf_lines


def _get_single_visual_urdf_lines_for_link(
    geom_type: int,
    dims: tuple[float, float, float],
    filename: str,
    pos: tuple[float, float, float],
    orn: tuple[float, float, float, float],
    color: tuple[float, float, float, float],
) -> list[tuple[str, int]]:
    urdf_lines = [("<visual>", 4)]

    # Add pose.
    pose_str = _get_urdf_pose_str(pos, orn)
    urdf_lines.append((pose_str, 6))

    # Add geometry.
    geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
    urdf_lines.append(("<geometry>", 6))
    urdf_lines.append((geometry_str, 8))
    urdf_lines.append(("</geometry>", 6))

    # Add color.
    r, g, b, a = color
    color_name = f"color-{int(255*r)}-{int(255*g)}-{int(255*b)}-{int(255*a)}"
    color_tag = f'<color rgba="{r} {g} {b} {a}"/>'
    urdf_lines.append((f'<material name="{color_name}">', 6))
    urdf_lines.append((color_tag, 8))
    urdf_lines.append(("</material>", 6))

    urdf_lines.append(("</visual>", 4))

    return urdf_lines


def _get_collision_urdf_lines_for_link(
    body_id: int, link_idx: int, physics_client_id: int
) -> list[tuple[str, int]]:
    urdf_lines = []

    collision_data = p.getCollisionShapeData(
        body_id, link_idx, physicsClientId=physics_client_id
    )

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
            origin_to_pose = multiply_poses(tf, com_to_pose)
        pos = origin_to_pose.position
        orn = origin_to_pose.orientation

        urdf_lines.extend(
            _get_single_collision_urdf_lines_for_link(
                body_id,
                link_idx,
                physics_client_id,
                geom_type,
                dims,
                filename,
                pos,
                orn,
            )
        )

    return urdf_lines


def _get_single_collision_urdf_lines_for_link(
    body_id: int,
    link_idx: int,
    physics_client_id: int,
    geom_type: int,
    dims: tuple[float, float, float],
    filename: str,
    pos: tuple[float, float, float],
    orn: tuple[float, float, float, float],
) -> list[tuple[str, int]]:
    urdf_lines = [("<collision>", 4)]

    # Add pose.
    pose_str = _get_urdf_pose_str(pos, orn)
    urdf_lines.append((pose_str, 6))

    # Add geometry.
    try:
        geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
    except ValueError:
        # PyBullet seems to internally replace cylinders with meshes
        # when loading from URDF. In this case the best we can do is
        # steal the geometry from visual shape data.
        matching_link_visuals = _get_visual_shape_data_for_link(
            body_id, link_idx, physics_client_id
        )
        if len(matching_link_visuals) == 1:
            print("WARNING: using visual geometry for collisions.")
            v = matching_link_visuals[0]
            geom_type = v[2]
            dims = v[3]
            geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
        else:
            import ipdb

            ipdb.set_trace()
            raise NotImplementedError

    urdf_lines.append(("<geometry>", 6))
    urdf_lines.append((geometry_str, 8))
    urdf_lines.append(("</geometry>", 6))

    urdf_lines.append(("</collision>", 4))

    return urdf_lines


def _get_inertial_urdf_lines_for_link(
    body_id: int, link_idx: int, physics_client_id: int
) -> list[tuple[str, int]]:
    dynamics_info = p.getDynamicsInfo(
        body_id, link_idx, physicsClientId=physics_client_id
    )
    mass = dynamics_info[0]
    ixx, iyy, izz = dynamics_info[2]
    inertial_pos = dynamics_info[3]
    inertial_orn = dynamics_info[4]

    urdf_lines = [("<inertial>", 4)]
    inertial_pose_str = _get_urdf_pose_str(inertial_pos, inertial_orn)
    mass_str = f'<mass value="{mass}"/>'
    inertia_str = (
        f'<inertia ixx="{ixx}" ixy="0.0" ixz="0.0" '
        f'iyy="{iyy}" iyz="0.0" izz="{izz}"/>'
    )
    urdf_lines.append((inertial_pose_str, 6))
    urdf_lines.append((mass_str, 6))
    urdf_lines.append((inertia_str, 6))
    urdf_lines.append(("</inertial>", 4))

    return urdf_lines


def _get_urdf_lines_for_joint(
    body_id: int, joint_idx: int, physics_client_id: int
) -> list[tuple[str, int]]:

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
            parent_link_state.localInertialFrameOrientation,
        )
        local_frame = multiply_poses(tf, Pose(parent_frame_pos, parent_frame_orn))
        parent_frame_pos = local_frame.position
        parent_frame_orn = local_frame.orientation

    peuler = p.getEulerFromQuaternion(parent_frame_orn)

    pose_str = (
        f'<origin xyz="{parent_frame_pos[0]} {parent_frame_pos[1]} '
        f'{parent_frame_pos[2]}" rpy="{peuler[0]} {peuler[1]} {peuler[2]}"/>'
    )

    urdf_lines = [(f'<joint name="{joint_name}" type="{jtype_str}">\n', 2)]
    urdf_lines.append((f'<parent link="{parent_name}"/>', 4))
    urdf_lines.append((f'<child link="{child_name}"/>', 4))
    urdf_lines.append((pose_str, 4))

    # If not fixed, add the axis and limits.
    if jtype_str in ["revolute", "prismatic"]:
        joint_axis = joint_info.jointAxis
        axis_str = f'<axis xyz="{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"/>'
        urdf_lines.append((axis_str, 4))

        lower = joint_info.jointLowerLimit
        upper = joint_info.jointUpperLimit
        max_velocity = joint_info.jointMaxVelocity
        max_effort = joint_info.jointMaxForce
        limit_str = (
            f'    <limit effort="{max_effort}" '
            f'velocity="{max_velocity}" lower="{lower}" upper="{upper}"/>\n'
        )
        urdf_lines.append((limit_str, 4))

    urdf_lines.append(("</joint>", 2))

    return urdf_lines


def create_urdf_from_body_id(
    body_id: int, physics_client_id: int, name: str = "pybullet-extracted"
) -> str:
    """Create a URDF string for a body that's loaded into pybullet."""

    # Start building the URDF content.
    urdf_lines = [('<?xml version="1.0"?>', 0)]
    urdf_lines.append((f'<robot name="{name}">', 0))

    num_joints = get_num_joints(body_id, physics_client_id)

    # Handle each link, including the base link (which is index -1 in PyBullet).
    for link_idx in range(-1, num_joints):
        link_urdf_lines = _get_urdf_lines_for_link(body_id, link_idx, physics_client_id)
        urdf_lines.extend(link_urdf_lines)

    # Handle joints.
    for joint_idx in range(num_joints):
        joint_urdf_lines = _get_urdf_lines_for_joint(
            body_id, joint_idx, physics_client_id
        )
        urdf_lines.extend(joint_urdf_lines)

    urdf_lines.append(("</robot>", 0))

    urdf_content = ""
    for urdf_str, indent in urdf_lines:
        urdf_content += " " * indent + urdf_str + "\n"

    return urdf_content
