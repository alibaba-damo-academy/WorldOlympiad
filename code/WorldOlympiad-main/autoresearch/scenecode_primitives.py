"""Deterministic fallbacks for explicitly requested solid task primitives.

These are deliberately narrow: they activate only after SceneCode exhausts its
normal Code_Object retries and only when the description unambiguously asks for
a closed, single-piece cylinder, cuboid, or sphere.  Complex objects remain
owned by SceneCode's generative asset routes.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


PRIMITIVE_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (190, 45, 45),
    "green": (45, 165, 75),
    "blue": (45, 90, 190),
    "yellow": (220, 185, 45),
    "orange": (220, 115, 35),
    "purple": (135, 70, 175),
    "teal": (25, 155, 150),
    "turquoise": (40, 180, 170),
    "cyan": (35, 175, 200),
    "magenta": (190, 45, 145),
    "pink": (220, 105, 150),
    "lime": (125, 190, 45),
    "navy": (35, 55, 130),
    "beige": (190, 170, 125),
    "maroon": (125, 40, 50),
    "brown": (125, 80, 45),
    "gray": (125, 125, 125),
    "grey": (125, 125, 125),
    "black": (30, 30, 30),
    "white": (225, 225, 225),
}


def solid_primitive_contract(description: str) -> str | None:
    """Return a safe primitive kind for an unambiguous closed solid request."""

    text = description.lower()
    if not re.search(r"\b(solid|single-piece|single piece)\b", text):
        return None
    if re.search(r"\b(hollow|tube|cup|bowl|container|open-top|open top|through-hole|through hole)\b", text):
        return None
    if re.search(r"\b(cylinder|cylindrical)\b", text):
        return "cylinder"
    if re.search(r"\b(cube|cuboid|rectangular block|box-shaped block)\b", text):
        return "cuboid"
    if re.search(r"\b(sphere|spherical|ball)\b", text):
        return "sphere"
    return None


def primitive_color(description: str) -> tuple[str, tuple[int, int, int]]:
    text = description.lower()
    matches: list[tuple[int, str, tuple[int, int, int]]] = []
    for name, rgb in PRIMITIVE_COLORS.items():
        match = re.search(rf"\b{re.escape(name)}\b", text)
        if match:
            matches.append((match.start(), name, rgb))
    if not matches:
        return "gray", PRIMITIVE_COLORS["gray"]
    _, name, rgb = min(matches)
    return name, rgb


def _append_face(
    vertices: list[list[float]],
    normals: list[list[float]],
    uvs: list[list[float]],
    indices: list[int],
    points: list[list[float]],
    normal: list[float],
) -> None:
    start = len(vertices)
    vertices.extend(points)
    normals.extend([normal] * 4)
    uvs.extend([[0, 0], [1, 0], [1, 1], [0, 1]])
    indices.extend([start, start + 1, start + 2, start, start + 2, start + 3])


def primitive_mesh(
    kind: str, dimensions: tuple[float, float, float] | list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a closed Z-up primitive with its bottom on z=0."""

    x, y, z = (max(float(value), 1e-3) for value in dimensions)
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    indices: list[int] = []

    if kind == "cuboid":
        hx, hy = x / 2.0, y / 2.0
        _append_face(vertices, normals, uvs, indices, [[hx, -hy, 0], [hx, hy, 0], [hx, hy, z], [hx, -hy, z]], [1, 0, 0])
        _append_face(vertices, normals, uvs, indices, [[-hx, hy, 0], [-hx, -hy, 0], [-hx, -hy, z], [-hx, hy, z]], [-1, 0, 0])
        _append_face(vertices, normals, uvs, indices, [[hx, hy, 0], [-hx, hy, 0], [-hx, hy, z], [hx, hy, z]], [0, 1, 0])
        _append_face(vertices, normals, uvs, indices, [[-hx, -hy, 0], [hx, -hy, 0], [hx, -hy, z], [-hx, -hy, z]], [0, -1, 0])
        _append_face(vertices, normals, uvs, indices, [[-hx, -hy, z], [hx, -hy, z], [hx, hy, z], [-hx, hy, z]], [0, 0, 1])
        _append_face(vertices, normals, uvs, indices, [[-hx, hy, 0], [hx, hy, 0], [hx, -hy, 0], [-hx, -hy, 0]], [0, 0, -1])
    elif kind == "cylinder":
        segments = 48
        radius_x, radius_y = x / 2.0, y / 2.0
        for index in range(segments):
            a0 = 2.0 * math.pi * index / segments
            a1 = 2.0 * math.pi * (index + 1) / segments
            p0 = [radius_x * math.cos(a0), radius_y * math.sin(a0), 0.0]
            p1 = [radius_x * math.cos(a1), radius_y * math.sin(a1), 0.0]
            p2 = [p1[0], p1[1], z]
            p3 = [p0[0], p0[1], z]
            mid = (a0 + a1) / 2.0
            normal = [math.cos(mid), math.sin(mid), 0.0]
            _append_face(vertices, normals, uvs, indices, [p0, p1, p2, p3], normal)
            top = len(vertices)
            vertices.extend([[0, 0, z], p3, p2])
            normals.extend([[0, 0, 1]] * 3)
            uvs.extend([[0.5, 0.5], [0, 0], [1, 0]])
            indices.extend([top, top + 1, top + 2])
            bottom = len(vertices)
            vertices.extend([[0, 0, 0], p1, p0])
            normals.extend([[0, 0, -1]] * 3)
            uvs.extend([[0.5, 0.5], [1, 0], [0, 0]])
            indices.extend([bottom, bottom + 1, bottom + 2])
    elif kind == "sphere":
        latitudes, longitudes = 24, 48
        rx, ry, rz = x / 2.0, y / 2.0, z / 2.0
        for lat in range(latitudes):
            phi0 = -math.pi / 2.0 + math.pi * lat / latitudes
            phi1 = -math.pi / 2.0 + math.pi * (lat + 1) / latitudes
            for lon in range(longitudes):
                theta0 = 2.0 * math.pi * lon / longitudes
                theta1 = 2.0 * math.pi * (lon + 1) / longitudes
                points: list[list[float]] = []
                face_normals: list[list[float]] = []
                for phi, theta in ((phi0, theta0), (phi0, theta1), (phi1, theta1), (phi1, theta0)):
                    nx = math.cos(phi) * math.cos(theta)
                    ny = math.cos(phi) * math.sin(theta)
                    nz = math.sin(phi)
                    points.append([rx * nx, ry * ny, rz * nz + rz])
                    face_normals.append([nx, ny, nz])
                start = len(vertices)
                vertices.extend(points)
                normals.extend(face_normals)
                uvs.extend([[0, 0], [1, 0], [1, 1], [0, 1]])
                indices.extend([start, start + 1, start + 2, start, start + 2, start + 3])
    else:
        raise ValueError(f"unsupported primitive kind: {kind}")

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def create_solid_primitive_glb(
    *, description: str, dimensions: list[float], output_dir: Path
) -> tuple[Path, str] | None:
    """Materialize a self-contained GLB for a safe primitive contract."""

    kind = solid_primitive_contract(description)
    if kind is None:
        return None

    from PIL import Image
    from scenecode.utils.gltf_generation import (
        create_glb_from_mesh_data,
        zup_to_yup_transform,
    )

    color_name, rgb = primitive_color(description)
    output_dir.mkdir(parents=True, exist_ok=True)
    color = output_dir / "base_color.png"
    normal = output_dir / "normal.png"
    roughness = output_dir / "roughness.png"
    Image.new("RGB", (64, 64), rgb).save(color)
    Image.new("RGB", (64, 64), (128, 128, 255)).save(normal)
    Image.new("L", (64, 64), 205).save(roughness)

    vertices, normals, uvs, indices = primitive_mesh(kind, dimensions)
    glb = output_dir / f"solid_{color_name}_{kind}.glb"
    create_glb_from_mesh_data(
        vertices=zup_to_yup_transform(vertices),
        normals=zup_to_yup_transform(normals),
        uvs=uvs,
        indices=indices,
        color_texture_path=color,
        normal_texture_path=normal,
        roughness_texture_path=roughness,
        output_path=glb,
    )
    return glb, kind
