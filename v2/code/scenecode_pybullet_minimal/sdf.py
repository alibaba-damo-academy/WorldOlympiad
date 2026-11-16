"""Create a PyBullet-readable physics view of an official SceneCode SDF."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import xml.etree.ElementTree as ET

from dataclasses import asdict, dataclass
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass(frozen=True)
class BridgeArtifact:
    source_sdf: str
    physics_sdf: str
    source_sha256: str
    physics_sha256: str
    removed_visual_count: int
    collision_mesh_count: int
    collision_geometry_count: int
    visual_bounds_fallback: bool = False
    visual_bounds_source: str | None = None
    visual_bounds_box_size_xyz_m: tuple[float, float, float] | None = None
    inertial_repair_count: int = 0
    inertial_repairs: tuple[dict[str, object], ...] = ()
    collision_scale_by_link: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in parent if _local_name(node.tag) == name]


def _child(parent: ET.Element, name: str) -> ET.Element | None:
    matches = _children(parent, name)
    if len(matches) > 1:
        raise ValueError(f"{_local_name(parent.tag)} contains duplicate {name} elements")
    return matches[0] if matches else None


def _qualified_tag(parent: ET.Element, name: str) -> str:
    if parent.tag.startswith("{"):
        return f"{parent.tag.split('}', 1)[0]}}}{name}"
    return name


def _numbers(text: str | None, count: int, *, label: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in (text or "").split())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain {count} finite numbers")
    return values


def _mesh_vertices(path: Path) -> list[tuple[float, float, float]]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        vertices: list[tuple[float, float, float]] = []
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                if raw.startswith("v "):
                    values = raw.split()
                    if len(values) >= 4:
                        vertices.append(tuple(float(value) for value in values[1:4]))
        if not vertices:
            raise ValueError(f"collision OBJ has no vertices: {path}")
        return vertices
    if suffix != ".stl":
        raise ValueError(f"unsupported collision mesh for inertia bounds: {path}")
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            vertices = []
            offset = 84
            for _ in range(triangle_count):
                offset += 12
                for _ in range(3):
                    vertices.append(struct.unpack_from("<fff", data, offset))
                    offset += 12
                offset += 2
            if vertices:
                return vertices
    vertices = []
    for raw in data.decode("utf-8", errors="replace").splitlines():
        values = raw.strip().split()
        if len(values) == 4 and values[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in values[1:4]))
    if not vertices:
        raise ValueError(f"collision STL has no vertices: {path}")
    return vertices


def _rotation_matrix(rpy: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _transform_point(
    point: tuple[float, float, float],
    xyz: tuple[float, float, float],
    rotation: tuple[tuple[float, ...], ...],
) -> tuple[float, float, float]:
    return tuple(
        xyz[row] + sum(rotation[row][column] * point[column] for column in range(3))
        for row in range(3)
    )


def _collision_points(collision: ET.Element) -> list[tuple[float, float, float]]:
    pose = _child(collision, "pose")
    pose_values = (
        _numbers(pose.text, 6, label="collision pose")
        if pose is not None
        else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    xyz = pose_values[:3]
    rotation = _rotation_matrix(pose_values[3:])
    geometry = _child(collision, "geometry")
    if geometry is None:
        raise ValueError("collision geometry is missing")
    shapes = [
        node
        for node in geometry
        if _local_name(node.tag) in {"mesh", "box", "sphere", "cylinder", "capsule"}
    ]
    if len(shapes) != 1:
        raise ValueError("collision geometry is missing or unsupported")
    shape = shapes[0]
    kind = _local_name(shape.tag)
    if kind == "mesh":
        uri = _child(shape, "uri")
        if uri is None or not uri.text:
            raise ValueError("collision mesh is missing a URI")
        scale_node = _child(shape, "scale")
        scale = (
            _numbers(scale_node.text, 3, label="collision mesh scale")
            if scale_node is not None
            else (1.0, 1.0, 1.0)
        )
        local_points = [
            tuple(vertex[axis] * scale[axis] for axis in range(3))
            for vertex in _mesh_vertices(Path(uri.text).resolve())
        ]
    else:
        if kind == "box":
            size = _child(shape, "size")
            if size is None:
                raise ValueError("collision box is missing size")
            extent = tuple(value / 2.0 for value in _numbers(size.text, 3, label="box size"))
        elif kind == "sphere":
            radius = _child(shape, "radius")
            if radius is None:
                raise ValueError("collision sphere is missing radius")
            value = _numbers(radius.text, 1, label="sphere radius")[0]
            extent = (value, value, value)
        else:
            radius = _child(shape, "radius")
            length = _child(shape, "length")
            if radius is None or length is None:
                raise ValueError(f"collision {kind} is missing radius or length")
            r = _numbers(radius.text, 1, label=f"{kind} radius")[0]
            half_z = _numbers(length.text, 1, label=f"{kind} length")[0] / 2.0
            if kind == "capsule":
                half_z += r
            extent = (r, r, half_z)
        if any(value <= 0.0 for value in extent):
            raise ValueError(f"collision {kind} has non-positive extent")
        local_points = [
            (sx * extent[0], sy * extent[1], sz * extent[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    return [_transform_point(point, xyz, rotation) for point in local_points]


def _repair_incomplete_inertials(root: ET.Element) -> tuple[dict[str, object], ...]:
    """Complete malformed generated inertials from collision AABB evidence.

    SceneCode assets sometimes contain ``mass`` and ``pose`` but omit the
    required inertia tensor. The source remains immutable; the PyBullet-only
    projection receives a deterministic enclosing-box tensor.
    """

    repairs: list[dict[str, object]] = []
    required = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    for link in (node for node in root.iter() if _local_name(node.tag) == "link"):
        inertial = _child(link, "inertial")
        if inertial is None:
            continue
        mass_node = _child(inertial, "mass")
        if mass_node is None:
            raise ValueError("inertial element is missing mass")
        mass = _numbers(mass_node.text, 1, label="inertial mass")[0]
        if mass < 0.0:
            raise ValueError("inertial mass must not be negative")
        inertia = _child(inertial, "inertia")
        complete = inertia is not None and all(_child(inertia, name) is not None for name in required)
        if complete:
            diagonal = [
                _numbers(_child(inertia, name).text, 1, label=f"inertia {name}")[0]
                for name in required[:3]
            ]
            # SceneCode uses a complete zero-mass root link for fixed-base
            # articulated assets. PyBullet accepts that inertial as-is; only
            # incomplete zero-mass data is invalid for a physics projection.
            if all(value > 0.0 for value in diagonal):
                continue
        if mass <= 0.0:
            raise ValueError("zero-mass inertial must have a complete positive tensor")
        collisions = _children(link, "collision")
        points = [point for collision in collisions for point in _collision_points(collision)]
        if not points:
            raise ValueError("cannot derive missing inertia without collision geometry")
        lower = tuple(min(point[axis] for point in points) for axis in range(3))
        upper = tuple(max(point[axis] for point in points) for axis in range(3))
        size = tuple(upper[axis] - lower[axis] for axis in range(3))
        if any(not math.isfinite(value) or value <= 0.0 for value in size):
            raise ValueError("collision bounds are degenerate; cannot derive inertia")
        ixx = mass * (size[1] ** 2 + size[2] ** 2) / 12.0
        iyy = mass * (size[0] ** 2 + size[2] ** 2) / 12.0
        izz = mass * (size[0] ** 2 + size[1] ** 2) / 12.0
        if inertia is not None:
            inertial.remove(inertia)
        inertia = ET.SubElement(inertial, _qualified_tag(inertial, "inertia"))
        values = {"ixx": ixx, "iyy": iyy, "izz": izz, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}
        for name in required:
            node = ET.SubElement(inertia, _qualified_tag(inertia, name))
            node.text = f"{values[name]:.12g}"
        repairs.append(
            {
                "link": link.get("name") or "",
                "mass_kg": mass,
                "collision_aabb_size_xyz_m": size,
                "policy": "solid_enclosing_collision_aabb",
                "inertia_diagonal_kg_m2": (ixx, iyy, izz),
            }
        )
    return tuple(repairs)


def _visual_gltf_box(
    root: ET.Element, source: Path
) -> tuple[ET.Element, tuple[float, float, float], tuple[float, float, float], Path]:
    """Return a conservative SDF box for one transform-free glTF visual.

    SceneCode's deterministic region-marker fallback emits a single glTF box
    without collision geometry. glTF is Y-up, while SDF is Z-up, so its X/Z/Y
    extents map to SDF X/Y/Z. Complex or transformed visuals fail closed.
    """

    candidates: list[tuple[ET.Element, Path]] = []
    for link in (node for node in root.iter() if _local_name(node.tag) == "link"):
        for visual in (node for node in link if _local_name(node.tag) == "visual"):
            if any(_local_name(node.tag) == "pose" for node in visual):
                raise ValueError("visual-bounds fallback does not accept SDF visual poses")
            uri = next(
                (
                    node
                    for node in visual.iter()
                    if _local_name(node.tag) == "uri"
                ),
                None,
            )
            if uri is None or not uri.text or not uri.text.strip():
                continue
            raw_uri = uri.text.strip()
            if "://" in raw_uri:
                raise ValueError(f"unsupported visual mesh URI: {raw_uri}")
            path = Path(raw_uri)
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            if path.suffix.lower() == ".gltf":
                candidates.append((link, path))
    if len(candidates) != 1:
        raise ValueError(
            "visual-bounds fallback requires exactly one local glTF visual"
        )
    link, gltf_path = candidates[0]
    if not gltf_path.is_file():
        raise FileNotFoundError(gltf_path)
    payload = json.loads(gltf_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes") or []
    if any(
        any(key in node for key in ("matrix", "rotation", "scale", "translation"))
        for node in nodes
    ):
        raise ValueError("visual-bounds fallback does not accept glTF node transforms")
    accessors = payload.get("accessors") or []
    position_indices: set[int] = set()
    for mesh in payload.get("meshes") or []:
        for primitive in mesh.get("primitives") or []:
            index = (primitive.get("attributes") or {}).get("POSITION")
            if isinstance(index, int):
                position_indices.add(index)
    if not position_indices:
        raise ValueError("glTF visual has no POSITION accessor")
    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for index in sorted(position_indices):
        if index < 0 or index >= len(accessors):
            raise ValueError("glTF POSITION accessor index is out of range")
        accessor = accessors[index]
        low = accessor.get("min")
        high = accessor.get("max")
        if not (
            isinstance(low, list)
            and isinstance(high, list)
            and len(low) == 3
            and len(high) == 3
        ):
            raise ValueError("glTF POSITION accessor lacks 3D min/max bounds")
        low_values = [float(value) for value in low]
        high_values = [float(value) for value in high]
        if not all(math.isfinite(value) for value in low_values + high_values):
            raise ValueError("glTF POSITION bounds must be finite")
        lows.append(low_values)
        highs.append(high_values)
    gltf_low = [min(values[axis] for values in lows) for axis in range(3)]
    gltf_high = [max(values[axis] for values in highs) for axis in range(3)]
    gltf_size = [gltf_high[axis] - gltf_low[axis] for axis in range(3)]
    if any(value <= 0.0 for value in gltf_size):
        raise ValueError("glTF visual bounds are degenerate")
    gltf_center = [
        (gltf_low[axis] + gltf_high[axis]) / 2.0 for axis in range(3)
    ]
    size_xyz = (gltf_size[0], gltf_size[2], gltf_size[1])
    center_xyz = (gltf_center[0], -gltf_center[2], gltf_center[1])
    return link, size_xyz, center_xyz, gltf_path


def prepare_physics_sdf(
    source_sdf: str | Path,
    destination_sdf: str | Path,
    *,
    allow_visual_bounds_fallback: bool = False,
    collision_scales_by_link: dict[str, float] | None = None,
) -> BridgeArtifact:
    """Project SceneCode SDF data into the subset accepted by PyBullet.

    Official SceneCode uses glTF for high-quality visuals and OBJ convex meshes
    for collision. PyBullet rejects an otherwise valid SDF as soon as it sees a
    glTF visual URI. This function therefore removes visual elements only; the
    source file and every collision mesh remain immutable.
    """

    source = Path(source_sdf).resolve()
    destination = Path(destination_sdf).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        raise ValueError("physics SDF must not overwrite the official source SDF")

    tree = ET.parse(source)
    root = tree.getroot()
    if _local_name(root.tag) != "sdf":
        raise ValueError(f"expected an SDF root, got {root.tag!r}")

    requested_scales = collision_scales_by_link or {}
    for link_name, scale in requested_scales.items():
        if not isinstance(link_name, str) or not link_name:
            raise ValueError("collision scale link names must be non-empty strings")
        if not math.isfinite(float(scale)) or float(scale) < 1.0:
            raise ValueError("collision link scales must be finite and >= 1")
    link_names = {
        link.get("name")
        for link in root.iter()
        if _local_name(link.tag) == "link" and link.get("name")
    }
    missing_scale_links = sorted(set(requested_scales) - link_names)
    if missing_scale_links:
        raise ValueError(f"collision scale links are absent from SDF: {missing_scale_links}")

    fallback: tuple[
        ET.Element,
        tuple[float, float, float],
        tuple[float, float, float],
        Path,
    ] | None = None
    if allow_visual_bounds_fallback:
        has_collision = any(
            _local_name(node.tag) == "collision" for node in root.iter()
        )
        if not has_collision:
            fallback = _visual_gltf_box(root, source)

    removed_visuals = 0
    for link in (node for node in root.iter() if _local_name(node.tag) == "link"):
        for child in list(link):
            if _local_name(child.tag) == "visual":
                link.remove(child)
                removed_visuals += 1

    if fallback is not None:
        link, size_xyz, center_xyz, _ = fallback
        collision = ET.SubElement(
            link, "collision", {"name": "visual_bounds_fallback"}
        )
        pose = ET.SubElement(collision, "pose")
        pose.text = " ".join(f"{value:.12g}" for value in (*center_xyz, 0.0, 0.0, 0.0))
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = " ".join(f"{value:.12g}" for value in size_xyz)

    collision_meshes = 0
    collision_geometries = 0
    for collision in (
        node for node in root.iter() if _local_name(node.tag) == "collision"
    ):
        geometries = [
            node for node in collision if _local_name(node.tag) == "geometry"
        ]
        if len(geometries) != 1:
            raise ValueError("collision must contain exactly one geometry")
        supported = [
            node
            for node in geometries[0]
            if _local_name(node.tag) in {"mesh", "box", "sphere", "cylinder", "capsule"}
        ]
        if len(supported) != 1:
            raise ValueError("collision geometry is missing or unsupported")
        collision_geometries += 1
        for mesh in (node for node in collision.iter() if _local_name(node.tag) == "mesh"):
            uri = next(
                (node for node in mesh if _local_name(node.tag) == "uri"), None
            )
            if uri is None or not uri.text or not uri.text.strip():
                raise ValueError("collision mesh is missing a URI")
            raw_uri = uri.text.strip()
            if "://" in raw_uri:
                raise ValueError(f"unsupported collision mesh URI: {raw_uri}")
            mesh_path = Path(raw_uri)
            if not mesh_path.is_absolute():
                mesh_path = (source.parent / mesh_path).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            if mesh_path.suffix.lower() not in {".obj", ".stl"}:
                raise ValueError(
                    f"PyBullet collision mesh must be OBJ/STL, got {mesh_path.suffix}"
                )
            uri.text = str(mesh_path)
            for child in list(mesh):
                if _local_name(child.tag) == "declare_convex":
                    mesh.remove(child)
            collision_meshes += 1

    # A collision-only scale is an explicit, deterministic PyBullet variant.
    # It never changes the official SDF or its visual geometry.  The caller
    # records the link/scale pair in the artifact and rollout manifest.
    applied_scales: list[tuple[str, float]] = []
    for link in (node for node in root.iter() if _local_name(node.tag) == "link"):
        link_name = link.get("name")
        scale = float(requested_scales.get(link_name, 1.0))
        if scale == 1.0:
            continue
        for collision in (node for node in link if _local_name(node.tag) == "collision"):
            geometry = _child(collision, "geometry")
            assert geometry is not None
            shape = next(iter(geometry), None)
            if shape is None:
                raise ValueError("collision geometry is missing or unsupported")
            kind = _local_name(shape.tag)
            if kind == "mesh":
                scale_node = _child(shape, "scale")
                if scale_node is None:
                    scale_node = ET.SubElement(shape, _qualified_tag(shape, "scale"))
                    existing = (1.0, 1.0, 1.0)
                else:
                    existing = _numbers(scale_node.text, 3, label="collision mesh scale")
                scale_node.text = " ".join(f"{value * scale:.12g}" for value in existing)
            elif kind == "box":
                size = _child(shape, "size")
                if size is None:
                    raise ValueError("collision box is missing size")
                values = _numbers(size.text, 3, label="box size")
                size.text = " ".join(f"{value * scale:.12g}" for value in values)
            elif kind == "sphere":
                radius = _child(shape, "radius")
                if radius is None:
                    raise ValueError("collision sphere is missing radius")
                radius.text = f"{_numbers(radius.text, 1, label='sphere radius')[0] * scale:.12g}"
            elif kind in {"cylinder", "capsule"}:
                radius = _child(shape, "radius")
                length = _child(shape, "length")
                if radius is None or length is None:
                    raise ValueError(f"collision {kind} is missing radius or length")
                radius.text = f"{_numbers(radius.text, 1, label=f'{kind} radius')[0] * scale:.12g}"
                length.text = f"{_numbers(length.text, 1, label=f'{kind} length')[0] * scale:.12g}"
            else:
                raise ValueError(f"unsupported collision geometry for scale: {kind}")
        applied_scales.append((str(link_name), scale))

    if collision_geometries == 0:
        raise ValueError("SceneCode SDF contains no collision geometry")

    inertial_repairs = _repair_incomplete_inertials(root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return BridgeArtifact(
        source_sdf=str(source),
        physics_sdf=str(destination),
        source_sha256=_sha256(source),
        physics_sha256=_sha256(destination),
        removed_visual_count=removed_visuals,
        collision_mesh_count=collision_meshes,
        collision_geometry_count=collision_geometries,
        visual_bounds_fallback=fallback is not None,
        visual_bounds_source=str(fallback[3]) if fallback is not None else None,
        visual_bounds_box_size_xyz_m=fallback[1] if fallback is not None else None,
        inertial_repair_count=len(inertial_repairs),
        inertial_repairs=inertial_repairs,
        collision_scale_by_link=tuple(sorted(applied_scales)),
    )
