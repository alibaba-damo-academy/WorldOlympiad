"""Deterministic mesh-structure gates for SceneCode task assets.

The SceneCode VLM may accept a plan or render whose noun is correct while the
actual mesh is structurally wrong (for example a capped ``open-top`` box or a
container with one missing side).  This module checks the generated triangles
directly.  It derives contracts from identity-bearing words in the selected
description and never chooses a case, object name, color, or task recipe.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np


STRUCTURE_SCHEMA_VERSION = "scenecode_structure_gate.v1"


def structure_contract(visual_description: str) -> str | None:
    """Return the generic geometry contract implied by one description."""

    lowered = " ".join(visual_description.lower().split())
    vessel_words = (
        "box",
        "bin",
        "bowl",
        "container",
        "cup",
        "pot",
        "tray",
    )
    if (
        re.search(r"\b(?:open[- ]top|open top|hollow)\b", lowered)
        and any(word in lowered for word in vessel_words)
    ):
        return "open_vessel"
    if re.search(r"\b(?:through[- ]hole|vertical hole)\b", lowered):
        return "through_hole"
    return None


def _unique_distances(values: np.ndarray, tolerance: float) -> list[float]:
    if values.size == 0:
        return []
    ordered = np.sort(values.astype(np.float64))
    result = [float(ordered[0])]
    for value in ordered[1:]:
        if abs(float(value) - result[-1]) > tolerance:
            result.append(float(value))
    return result


def ray_triangle_distances(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    origin: np.ndarray,
    direction: np.ndarray,
) -> list[float]:
    """Return de-duplicated positive ray/triangle intersection distances.

    This is a vectorized Moller--Trumbore test.  It deliberately avoids spatial
    index packages so the same gate runs in the minimal SceneCode runtime.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    repeated_direction = np.broadcast_to(direction, edge2.shape)
    h = np.cross(repeated_direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    epsilon = 1e-10
    candidate = np.abs(determinant) > epsilon
    if not np.any(candidate):
        return []
    inverse = np.zeros_like(determinant)
    inverse[candidate] = 1.0 / determinant[candidate]
    s = origin - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("ij,ij->i", repeated_direction, q)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    valid = (
        candidate
        & (u >= -epsilon)
        & (u <= 1.0 + epsilon)
        & (v >= -epsilon)
        & (u + v <= 1.0 + epsilon)
        & (distance > epsilon)
    )
    extent = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    return _unique_distances(distance[valid], max(extent * 1e-6, 1e-9))


def _axis_hit_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    axis: int,
    fixed_point: np.ndarray,
    positive: bool = True,
) -> list[float]:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extent = high - low
    direction = np.zeros(3, dtype=np.float64)
    direction[axis] = 1.0 if positive else -1.0
    origin = np.asarray(fixed_point, dtype=np.float64).copy()
    origin[axis] = (
        low[axis] - 0.1 * extent[axis]
        if positive
        else high[axis] + 0.1 * extent[axis]
    )
    distances = ray_triangle_distances(
        vertices, faces, origin=origin, direction=direction
    )
    return [float(origin[axis] + direction[axis] * value) for value in distances]


def _normalized_hits(values: list[float], low: float, high: float) -> list[float]:
    extent = high - low
    if extent <= 0:
        return []
    return [round((value - low) / extent, 6) for value in values]


def _evaluate_open_axis(
    vertices: np.ndarray, faces: np.ndarray, *, up_axis: int
) -> dict[str, Any]:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extent = high - low
    center = (low + high) / 2.0
    side_axes = [axis for axis in range(3) if axis != up_axis]

    vertical_point = center.copy()
    vertical_hits = _axis_hit_positions(
        vertices,
        faces,
        axis=up_axis,
        fixed_point=vertical_point,
        positive=False,
    )
    vertical_normalized = _normalized_hits(
        vertical_hits, float(low[up_axis]), float(high[up_axis])
    )
    top_clear = not any(value >= 0.78 for value in vertical_normalized)
    bottom_present = any(value <= 0.22 for value in vertical_normalized)
    interior_clear = not any(0.28 < value < 0.72 for value in vertical_normalized)

    side_results: list[dict[str, Any]] = []
    for axis in side_axes:
        point = center.copy()
        point[up_axis] = low[up_axis] + 0.58 * extent[up_axis]
        hits = _axis_hit_positions(
            vertices, faces, axis=axis, fixed_point=point, positive=True
        )
        normalized = _normalized_hits(hits, float(low[axis]), float(high[axis]))
        negative_wall = any(value <= 0.22 for value in normalized)
        positive_wall = any(value >= 0.78 for value in normalized)
        side_results.append(
            {
                "axis": axis,
                "normalized_hits": normalized,
                "negative_wall_present": negative_wall,
                "positive_wall_present": positive_wall,
                "passed": negative_wall and positive_wall,
            }
        )

    checks = {
        "top_open": top_clear,
        "bottom_present": bottom_present,
        "interior_clear": interior_clear,
        "two_sided_enclosure_axis_0": side_results[0]["passed"],
        "two_sided_enclosure_axis_1": side_results[1]["passed"],
    }
    return {
        "axis": up_axis,
        "passed": all(checks.values()),
        "checks": checks,
        "vertical_normalized_hits": vertical_normalized,
        "side_probes": side_results,
        "score": sum(bool(value) for value in checks.values()),
    }


def evaluate_open_vessel(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Check one open end, a bottom, clear interior, and four side directions."""

    candidates = [
        _evaluate_open_axis(vertices, faces, up_axis=axis) for axis in range(3)
    ]
    best = max(candidates, key=lambda item: (int(item["passed"]), item["score"]))
    return {
        "passed": bool(best["passed"]),
        "selected_open_axis": best["axis"],
        "checks": best["checks"],
        "measurements": {
            "vertical_normalized_hits": best["vertical_normalized_hits"],
            "side_probes": best["side_probes"],
        },
    }


def _evaluate_through_axis(
    vertices: np.ndarray, faces: np.ndarray, *, axis: int
) -> dict[str, Any]:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extent = high - low
    center = (low + high) / 2.0
    center_hits = _axis_hit_positions(
        vertices, faces, axis=axis, fixed_point=center, positive=True
    )
    side_axes = [value for value in range(3) if value != axis]
    solid_probe_counts: list[int] = []
    for side_axis in side_axes:
        for sign in (-1.0, 1.0):
            point = center.copy()
            point[side_axis] += sign * 0.32 * extent[side_axis]
            hits = _axis_hit_positions(
                vertices, faces, axis=axis, fixed_point=point, positive=True
            )
            solid_probe_counts.append(len(hits))
    checks = {
        "center_channel_clear": len(center_hits) == 0,
        "surrounding_body_present": any(count >= 2 for count in solid_probe_counts),
    }
    return {
        "axis": axis,
        "passed": all(checks.values()),
        "checks": checks,
        "center_hit_count": len(center_hits),
        "solid_probe_hit_counts": solid_probe_counts,
        "score": sum(bool(value) for value in checks.values()),
    }


def evaluate_through_hole(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    candidates = [
        _evaluate_through_axis(vertices, faces, axis=axis) for axis in range(3)
    ]
    best = max(candidates, key=lambda item: (int(item["passed"]), item["score"]))
    return {
        "passed": bool(best["passed"]),
        "selected_hole_axis": best["axis"],
        "checks": best["checks"],
        "measurements": {
            "center_hit_count": best["center_hit_count"],
            "solid_probe_hit_counts": best["solid_probe_hit_counts"],
        },
    }


def _load_mesh_arrays(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(Path(mesh_path), process=False, force="scene")
    meshes = []
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            meshes.append(geometry)
    if not meshes:
        raise ValueError(f"mesh contains no triangle geometry: {mesh_path}")
    combined = trimesh.util.concatenate(meshes)
    vertices = np.asarray(combined.vertices, dtype=np.float64)
    faces = np.asarray(combined.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
        raise ValueError(f"invalid triangle arrays in mesh: {mesh_path}")
    return vertices, faces


def validate_mesh_structure(
    mesh_path: str | Path, visual_description: str
) -> dict[str, Any]:
    """Validate all deterministic structure implied by ``visual_description``."""

    contract = structure_contract(visual_description)
    result: dict[str, Any] = {
        "schema_version": STRUCTURE_SCHEMA_VERSION,
        "contract": contract,
        "applicable": contract is not None,
        "passed": True,
        "visual_description": visual_description,
    }
    if contract is None:
        return result
    try:
        vertices, faces = _load_mesh_arrays(mesh_path)
        if contract == "open_vessel":
            evaluated = evaluate_open_vessel(vertices, faces)
        elif contract == "through_hole":
            evaluated = evaluate_through_hole(vertices, faces)
        else:  # pragma: no cover - guarded by structure_contract
            raise ValueError(f"unsupported structure contract: {contract}")
        result.update(evaluated)
        result["vertex_count"] = int(len(vertices))
        result["triangle_count"] = int(len(faces))
    except Exception as exc:
        result.update(
            {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def failed_structure_reason(result: dict[str, Any]) -> str:
    if result.get("passed") is True:
        return ""
    failed = [
        name for name, passed in (result.get("checks") or {}).items() if not passed
    ]
    details = ", ".join(failed) if failed else str(result.get("error") or "unknown")
    return f"{result.get('contract') or 'structure'} failed: {details}"
