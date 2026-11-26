"""Collapse fixed links while preserving articulated WorldIR geometry and inertia."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .articulated_link_frame_rebase import _relative
from .math3d import compose
from .models import BodyIR, Inertial, Pose


def _inertia_matrix(values: tuple[float, ...]) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = values
    return np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def _inertia_values(matrix: np.ndarray) -> tuple[float, ...]:
    return (float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]),
            float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[1, 2]))


def collapse_fixed_joint_tree(body: BodyIR) -> dict:
    """Keep movable joints; merge every fixed subtree into its moving root.

    Link poses and joint poses are model-frame transforms. The derivative is
    admitted only if all visual/collision world poses and total mass are
    invariant. The original SDF and mesh files are never rewritten.
    """

    links = {link.name: link for link in body.links}
    parents = {joint.child: joint for joint in body.joints}
    roots = set(links) - set(parents)
    if len(roots) != 1 or len(parents) != len(body.joints):
        raise ValueError("fixed-joint collapse requires one unambiguous tree")
    root = next(iter(roots))
    visiting: set[str] = set()
    representatives: dict[str, str] = {}

    def representative(name: str) -> str:
        if name in representatives:
            return representatives[name]
        if name in visiting:
            raise ValueError("fixed-joint collapse found a cycle")
        visiting.add(name)
        joint = parents.get(name)
        selected = (
            representative(joint.parent)
            if joint is not None and joint.type == "fixed" else name
        )
        visiting.remove(name)
        representatives[name] = selected
        return selected

    for name in links:
        representative(name)
    if representative(root) != root:
        raise ValueError("fixed-joint collapse lost its root")
    groups: dict[str, list[str]] = {}
    for name in links:
        groups.setdefault(representatives[name], []).append(name)
    before = {
        (link.name, role, geometry.name): compose(
            body.pose, compose(link.pose, geometry.pose)
        )
        for link in body.links
        for role, geometries in (("visual", link.visuals), ("collision", link.collisions))
        for geometry in geometries
    }
    total_mass_before = sum(
        link.inertial.mass for link in body.links if link.inertial is not None
    )
    collapsed = []
    for rep_name, names in groups.items():
        rep = links[rep_name]
        visual, collision = [], []
        inertial_parts = []
        for name in names:
            link = links[name]
            transform = _relative(rep.pose, link.pose)
            for role, destination in (("visual", visual), ("collision", collision)):
                for geometry in getattr(link, role + "s"):
                    destination.append(geometry.model_copy(update={
                        "name": f"{name}__{geometry.name}",
                        "pose": compose(transform, geometry.pose),
                    }))
            if link.inertial is not None:
                mass = float(link.inertial.mass)
                com_pose = compose(transform, Pose(xyz=link.inertial.com))
                rotation = Rotation.from_quat(transform.quat_xyzw).as_matrix()
                inertial_parts.append((mass, np.asarray(com_pose.xyz),
                                       rotation @ _inertia_matrix(link.inertial.inertia)
                                       @ rotation.T))
        if not inertial_parts:
            raise ValueError("fixed-joint component has no measured inertia")
        mass_total = sum(part[0] for part in inertial_parts)
        com = sum((mass * xyz for mass, xyz, _ in inertial_parts),
                  np.zeros(3)) / mass_total
        tensor = np.zeros((3, 3))
        for mass, xyz, local_tensor in inertial_parts:
            offset = xyz - com
            tensor += local_tensor + mass * (
                float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
            )
        if np.linalg.eigvalsh(tensor).min() <= 0:
            raise ValueError("collapsed inertial tensor is not positive definite")
        collapsed.append(rep.model_copy(update={
            "visuals": visual, "collisions": collision,
            "inertial": Inertial(mass=mass_total, com=tuple(com),
                                 inertia=_inertia_values(tensor)),
        }))
    moving_joints = [joint.model_copy(update={
        "parent": representatives[joint.parent],
        "child": representatives[joint.child],
    }) for joint in body.joints if joint.type != "fixed"]
    if any(joint.parent == joint.child for joint in moving_joints):
        raise ValueError("movable joint vanished inside a fixed component")
    body.links = collapsed
    body.joints = moving_joints
    after = {
        (source, role, name): compose(body.pose, compose(link.pose, geometry.pose))
        for link in body.links
        for role, geometries in (("visual", link.visuals), ("collision", link.collisions))
        for geometry in geometries
        for source, separator, name in [geometry.name.partition("__")]
        if separator
    }
    if set(before) != set(after):
        raise ValueError("fixed-joint collapse changed geometry inventory")
    max_position = max((float(np.linalg.norm(
        np.asarray(before[key].xyz) - np.asarray(after[key].xyz)))
        for key in before), default=0.0)
    max_rotation = max((float((
        Rotation.from_quat(before[key].quat_xyzw).inv()
        * Rotation.from_quat(after[key].quat_xyzw)).magnitude())
        for key in before), default=0.0)
    total_mass_after = sum(link.inertial.mass for link in collapsed)
    if (max_position > 1e-6 or max_rotation > 1e-6
            or abs(total_mass_after - total_mass_before) > 1e-6):
        raise ValueError("fixed-joint collapse changed geometry or mass")
    return {
        "schema_version": "worldir_fixed_joint_collapse.v1",
        "body": body.name,
        "source_link_count": len(links),
        "collapsed_link_count": len(collapsed),
        "source_fixed_joint_count": sum(joint.type == "fixed" for joint in parents.values()),
        "preserved_movable_joint_count": len(moving_joints),
        "representative_by_source_link": representatives,
        "visual_count": sum(len(link.visuals) for link in collapsed),
        "collision_count": sum(len(link.collisions) for link in collapsed),
        "total_mass_kg": total_mass_after,
        "maximum_world_geometry_position_error_m": max_position,
        "maximum_world_geometry_orientation_error_rad": max_rotation,
        "source_asset_geometry_unchanged": True,
        "passed": True,
    }
