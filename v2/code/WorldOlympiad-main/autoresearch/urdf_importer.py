"""Import an articulated URDF into WorldIR while preserving its asset bundle."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation

from .materials import MATERIALS
from .models import BodyIR, Geometry, Inertial, JointIR, LinkIR, Pose
from .math3d import compose


def _model_frames(links: list[LinkIR], joints: list[JointIR]) -> None:
    """Convert URDF parent-local origins to WorldIR model-frame zero poses.

    Geometry and inertia remain link-local; axes remain joint-local. Never
    flatten a movable joint, or bake the body's world placement into a link.
    """
    names = {link.name for link in links}
    if len(names) != len(links):
        raise ValueError('URDF has duplicate link names')
    parents = {}
    for joint in joints:
        if (joint.parent not in names or joint.child not in names
                or joint.child in parents or joint.parent == joint.child):
            raise ValueError('URDF has invalid or ambiguous joint topology')
        parents[joint.child] = joint
    roots = names - set(parents)
    if len(roots) != 1:
        raise ValueError('URDF requires exactly one topological root')
    frames = {next(iter(roots)): Pose()}
    visiting = set()

    def frame(name):
        if name in frames:
            return frames[name]
        if name in visiting or name not in parents:
            raise ValueError('URDF has cyclic or disconnected topology')
        visiting.add(name)
        joint = parents[name]
        frames[name] = compose(frame(joint.parent), joint.pose)
        visiting.remove(name)
        return frames[name]

    for link in links:
        link.pose = frame(link.name)
    for joint in joints:
        joint.pose = frames[joint.child]



def _floats(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split()) if value else default


def _pose(node: ET.Element | None) -> Pose:
    if node is None:
        return Pose()
    xyz = _floats(node.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _floats(node.get("rpy"), (0.0, 0.0, 0.0))
    quat = tuple(float(item) for item in Rotation.from_euler("xyz", rpy).as_quat())
    return Pose(xyz=xyz, quat_xyzw=quat)  # type: ignore[arg-type]


def _mesh_path(urdf: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        return urdf.parent / filename.removeprefix("package://")
    path = Path(filename)
    return path if path.is_absolute() else urdf.parent / path


def _geometry(urdf: Path, node: ET.Element, name: str, rgba) -> Geometry:
    geometry = node.find("geometry")
    if geometry is None:
        raise ValueError(f"URDF geometry is missing for {name}")
    pose = _pose(node.find("origin"))
    mesh = geometry.find("mesh")
    if mesh is not None:
        path = _mesh_path(urdf, str(mesh.get("filename"))).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        scale = _floats(mesh.get("scale"), (1.0, 1.0, 1.0))
        return Geometry(name=name, kind="mesh", path=str(path), scale=scale, pose=pose, rgba=rgba)  # type: ignore[arg-type]
    box = geometry.find("box")
    if box is not None:
        return Geometry(name=name, kind="box", size=_floats(box.get("size"), ()), pose=pose, rgba=rgba)  # type: ignore[arg-type]
    sphere = geometry.find("sphere")
    if sphere is not None:
        return Geometry(name=name, kind="sphere", radius=float(sphere.get("radius")), pose=pose, rgba=rgba)
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return Geometry(name=name, kind="cylinder", radius=float(cylinder.get("radius")), length=float(cylinder.get("length")), pose=pose, rgba=rgba)
    raise ValueError(f"unsupported URDF geometry for {name}")


def import_urdf_body(
    urdf_path: str | Path,
    *,
    body_name: str,
    pose: Pose,
    static: bool = True,
    semantic_category: str = "robot_arm",
    material_name: str = "robot",
) -> BodyIR:
    urdf = Path(urdf_path).resolve()
    root = ET.parse(urdf).getroot()
    material_colors: dict[str, tuple[float, float, float, float]] = {}
    for material in root.findall("material"):
        color = material.find("color")
        if material.get("name") and color is not None and color.get("rgba"):
            material_colors[str(material.get("name"))] = _floats(color.get("rgba"), ())  # type: ignore[assignment]
    links: list[LinkIR] = []
    zero_mass_links: list[str] = []
    for link_node in root.findall("link"):
        link_name = str(link_node.get("name"))
        inertial_node = link_node.find("inertial")
        inertial = None
        if inertial_node is not None:
            mass_node = inertial_node.find("mass")
            matrix = inertial_node.find("inertia")
            if mass_node is not None and matrix is not None:
                mass = float(mass_node.get("value"))
                if mass <= 0:
                    zero_mass_links.append(link_name)
                    mass = 1e-6
                # URDF tensor is in the inertial-origin frame; WorldIR stores
                # the tensor in the link frame (COM translation stays local).
                from .worldir_fixed_joint_collapse import _inertia_matrix, _inertia_values
                inertial_origin = _pose(inertial_node.find("origin"))
                tensor = _inertia_matrix(tuple(float(matrix.get(key, "0"))
                    for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
                rotation = Rotation.from_quat(inertial_origin.quat_xyzw).as_matrix()
                tensor = rotation @ tensor @ rotation.T
                if not np.isfinite(tensor).all():
                    raise ValueError('URDF inertia must be finite')
                inertial = Inertial(
                    mass=mass,
                    com=_pose(inertial_node.find("origin")).xyz,
                    inertia=_inertia_values(tensor),
                )
        visuals: list[Geometry] = []
        for index, visual in enumerate(link_node.findall("visual")):
            rgba = (0.92, 0.92, 0.92, 1.0)
            material = visual.find("material")
            if material is not None:
                color = material.find("color")
                if color is not None and color.get("rgba"):
                    rgba = _floats(color.get("rgba"), ())  # type: ignore[assignment]
                elif material.get("name") in material_colors:
                    rgba = material_colors[str(material.get("name"))]
            visuals.append(_geometry(urdf, visual, f"{link_name}_visual_{index}", rgba))
        collisions = [
            _geometry(urdf, node, f"{link_name}_collision_{index}", None)
            for index, node in enumerate(link_node.findall("collision"))
        ]
        links.append(LinkIR(name=link_name, inertial=inertial, visuals=visuals, collisions=collisions))
    joints: list[JointIR] = []
    for joint_node in root.findall("joint"):
        joint_type = str(joint_node.get("type"))
        if joint_type not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError(f"unsupported URDF joint type: {joint_type}")
        parent = joint_node.find("parent")
        child = joint_node.find("child")
        if parent is None or child is None:
            raise ValueError("URDF joint lacks parent or child")
        limit = joint_node.find("limit")
        axis = joint_node.find("axis")
        joints.append(
            JointIR(
                name=str(joint_node.get("name")),
                type=joint_type,  # type: ignore[arg-type]
                parent=str(parent.get("link")),
                child=str(child.get("link")),
                pose=_pose(joint_node.find("origin")),
                axis=_floats(axis.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),  # type: ignore[arg-type]
                lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
                upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
                effort=float(limit.get("effort", "100")) if limit is not None else 100.0,
                velocity=float(limit.get("velocity", "10")) if limit is not None else 10.0,
            )
        )
    _model_frames(links, joints)
    if not static and (zero_mass_links or any(link.inertial is None for link in links)):
        raise ValueError('dynamic URDF requires measured positive mass on every link')
    return BodyIR(
        name=body_name,
        semantic_category=semantic_category,
        pose=pose,
        links=links,
        joints=joints,
        material=MATERIALS[material_name],
        static=static,
        affordances=(["articulated", "manipulator", "fixed_base"]
                     if static and semantic_category == "robot_arm"
                     else ["articulated", "fixed_base" if static else "floating_base"]),
        source_path=str(urdf),
        metadata={
            "precompiled_urdf_source": str(urdf),
            "precompiled_asset_policy": "copy_complete_urdf_directory",
            "fixed_base": static,
            "urdf_frame_contract": "model_frame_zero_pose; joint_local_axes; link_local_geometry",
            **({"end_effector_link_name": "panda_grasptarget"}
               if semantic_category == "robot_arm" else {}),
            "zero_mass_links_represented_with_epsilon_in_worldir": zero_mass_links,
        },
    )
