"""Small, exporter-faithful closed-box mesh boundary used by source checks.

The source builder uses X/Y footprint and Z height.  Visual GLB export is
Y-up, while collision OBJ export is Z-up; the conversion is intentionally
explicit so an axis-swapped collision package cannot pass a size audit.
"""
from __future__ import annotations

from math import isfinite


def box_inertial(mass: float, dimensions: tuple[float, float, float],
                 *, center: tuple[float, float, float] | None = None) -> dict:
    """Return the physical inertial block for a uniform closed box.

    ``dimensions`` are native-link X/Y/Z dimensions and ``center`` is the
    mesh COM in that link frame.  Keeping this calculation at the exporter
    boundary prevents a translated mesh from silently receiving the old
    origin/placeholder inertia.  It is deliberately geometry-derived, not a
    task-specific correction.
    """
    if not isfinite(mass) or mass <= 0 or any(not isfinite(v) or v <= 0 for v in dimensions):
        raise ValueError("mass and dimensions must be finite and positive")
    com = center if center is not None else tuple(v / 2.0 for v in dimensions)
    if len(com) != 3 or any(not isfinite(v) for v in com):
        raise ValueError("center must contain three finite coordinates")
    x, y, z = dimensions
    return {
        "mass": float(mass),
        "center": tuple(float(v) for v in com),
        "ixx": float(mass * (y * y + z * z) / 12.0),
        "iyy": float(mass * (x * x + z * z) / 12.0),
        "izz": float(mass * (x * x + y * y) / 12.0),
    }


def closed_box_mesh(ring_count: int = 32):
    if ring_count < 3:
        raise ValueError("ring_count must be at least three")
    bottom = range(ring_count)
    top = range(ring_count, 2 * ring_count)
    faces = []
    for i in range(ring_count):
        j = (i + 1) % ring_count
        faces.append((i, j, ring_count + j, ring_count + i))
    bc, tc = 2 * ring_count, 2 * ring_count + 1
    for i in range(ring_count):
        j = (i + 1) % ring_count
        faces.extend(((bc, j, i), (tc, ring_count + i, ring_count + j)))
    return faces


def visual_to_collision_axes(vertices):
    """Convert GLB visual coordinates to the native collision Z-up frame.

    The exporter writes visual vertices in ``(x, y_up, z_forward)`` while the
    physics OBJ is ``(x, y_depth, z_up)``.  The signed conversion is
    ``(x, -z_forward, y_up)``; keeping the sign and coordinate origin is
    essential for translated meshes, not just for extent comparisons.
    """
    return [(x, -z, y) for x, y, z in vertices]
