"""Code for exporting to URDF."""

import os
from dataclasses import dataclass

import pybullet as p

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import get_joint_info, get_num_joints
from pybullet_helpers.link import get_link_state


@dataclass
class URDFStringContainer:
    """Light container around URDF strings."""

    link_strs: list[tuple[str, int]]
    joint_strs: list[tuple[str, int]]

    def to_string(self, urdf_name: str = "pybullet-extracted") -> str:
        """Get the full URDF string."""
        urdf_lines = [('<?xml version="1.0"?>', 0)]
        urdf_lines.append((f'<robot name="{urdf_name}">', 0))
        urdf_lines.extend(self.link_strs)
        urdf_lines.extend(self.joint_strs)
        urdf_lines.append(("</robot>", 0))
        urdf_content = ""
        for urdf_str, indent in urdf_lines:
            urdf_content += " " * indent + urdf_str + "\n"
        return urdf_content


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


def _add_urdf_lines_for_link(
    body_id: int,
    link_idx: int,
    physics_client_id: int,
    container: URDFStringContainer,
) -> None:
    # Get the link name.
    if link_idx == -1:
        link_name = "base_link"
    else:
        joint_info = get_joint_info(body_id, link_idx, physics_client_id)
        link_name = joint_info.linkName
        assert link_name != "base_link", "Cannot have multiple base links"

    # Get visual data.
    all_visual_data = _get_visual_shape_data_for_link(
        body_id, link_idx, physics_client_id
    )
    assert len(all_visual_data) <= 1
    visual_data = all_visual_data[0] if all_visual_data else None

    # Get collision data.
    all_collision_data = p.getCollisionShapeData(
        body_id, link_idx, physicsClientId=physics_client_id
    )
    assert len(all_collision_data) <= 1
    collision_data = all_collision_data[0] if all_collision_data else None

    # Get inertial data.
    inertial_data = p.getDynamicsInfo(
        body_id, link_idx, physicsClientId=physics_client_id
    )

    # Start the URDF for the link.
    urdf_lines = [(f'<link name="{link_name}">', 2)]

    # Add visual tags.
    if visual_data:
        visual_urdf_lines = _get_visual_urdf_lines_for_link(visual_data)
        urdf_lines.extend(visual_urdf_lines)

    # Add collision tags.
    if collision_data:
        collision_urdf_lines = _get_collision_urdf_lines_for_link(
            collision_data, physics_client_id
        )
        urdf_lines.extend(collision_urdf_lines)

    # Add inertial tag.
    inertial_urdf_lines = _get_inertial_urdf_lines_for_link(inertial_data)
    urdf_lines.extend(inertial_urdf_lines)

    # Finish the URDF for the link.
    urdf_lines.append(("</link>", 2))

    container.link_strs.extend(urdf_lines)


def _get_visual_urdf_lines_for_link(
    visual_data: tuple,
) -> list[tuple[str, int]]:
    # Visual shape data format:
    # v = (bodyUniqueId, linkIndex, visualGeometryType, dimensions, filename,
    #      localVisualFramePos, localVisualFrameOrn, color, specular).
    geom_type = visual_data[2]
    dims = visual_data[3]
    filename = visual_data[4].decode("UTF-8")
    pos = visual_data[5]
    orn = visual_data[6]
    color = visual_data[7]

    return _get_single_visual_urdf_lines_for_link(
        geom_type, dims, filename, pos, orn, color
    )


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
    collision_data: tuple,
    physics_client_id: int,
) -> list[tuple[str, int]]:
    # Collision shape data format:
    # c = (object id, linkIndex, geometryType, dimensions, filename,
    #      localCollisionFramePos, localCollisionFrameOrn).
    geom_type = collision_data[2]
    dims = collision_data[3]
    filename = collision_data[4].decode("UTF-8")
    com_to_pose = Pose(collision_data[5], collision_data[6])
    if collision_data[1] == -1:
        origin_to_pose = com_to_pose
    else:
        link_state = get_link_state(
            collision_data[0], collision_data[1], physics_client_id
        )
        tf = Pose(
            link_state.localInertialFramePosition,
            link_state.localInertialFrameOrientation,
        )
        origin_to_pose = multiply_poses(tf, com_to_pose)
    pos = origin_to_pose.position
    orn = origin_to_pose.orientation

    return _get_single_collision_urdf_lines_for_link(
        collision_data[0],
        collision_data[1],
        physics_client_id,
        geom_type,
        dims,
        filename,
        pos,
        orn,
    )


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
            raise NotImplementedError

    urdf_lines.append(("<geometry>", 6))
    urdf_lines.append((geometry_str, 8))
    urdf_lines.append(("</geometry>", 6))

    urdf_lines.append(("</collision>", 4))

    return urdf_lines


def _get_inertial_urdf_lines_for_link(
    inertial_data: tuple,
) -> list[tuple[str, int]]:
    mass = inertial_data[0]
    ixx, iyy, izz = inertial_data[2]
    inertial_pos = inertial_data[3]
    inertial_orn = inertial_data[4]

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


def _add_urdf_lines_for_joint(
    body_id: int,
    joint_idx: int,
    physics_client_id: int,
    container: URDFStringContainer,
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

    container.joint_strs.extend(urdf_lines)


def create_urdf_from_body_id(
    body_id: int, physics_client_id: int, name: str = "pybullet-extracted"
) -> str:
    """Create a URDF string for a body that's loaded into pybullet."""

    # Start building the URDF content.
    container = URDFStringContainer([], [])
    num_joints = get_num_joints(body_id, physics_client_id)

    # Handle each link, including the base link (which is index -1 in PyBullet).
    for link_idx in range(-1, num_joints):
        _add_urdf_lines_for_link(body_id, link_idx, physics_client_id, container)

    # Handle joints.
    for joint_idx in range(num_joints):
        _add_urdf_lines_for_joint(body_id, joint_idx, physics_client_id, container)

    return container.to_string(name)
