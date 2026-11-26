"""Deterministic canonical-surface object flow from exact simulator poses."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from .math3d import compose
from .models import BodyIR, Geometry, Pose, WorldIR


def _transform(points: np.ndarray, pose: Pose) -> np.ndarray:
    quaternion = np.asarray(pose.quat_xyzw, dtype=np.float64)
    xyz = quaternion[:3]
    scalar = quaternion[3]
    # q * p * q^-1, vectorized without constructing matrices.
    first = 2.0 * np.cross(xyz[None, :], points)
    rotated = points + scalar * first + np.cross(xyz[None, :], first)
    return rotated + np.asarray(pose.xyz, dtype=np.float64)


def _surface_area(geometry: Geometry) -> float:
    if geometry.kind == "box":
        x, y, z = geometry.size or (0.0, 0.0, 0.0)
        return 2.0 * (x * y + x * z + y * z)
    if geometry.kind == "sphere":
        return 4.0 * np.pi * float(geometry.radius) ** 2
    if geometry.kind == "cylinder":
        radius, length = float(geometry.radius), float(geometry.length)
        return 2.0 * np.pi * radius * (radius + length)
    if geometry.kind == "mesh":
        import trimesh

        loaded = trimesh.load(Path(str(geometry.path)), force="mesh", process=False)
        vertices = np.asarray(loaded.vertices, dtype=np.float64) * np.asarray(geometry.scale)
        triangles = vertices[np.asarray(loaded.faces, dtype=np.int64)]
        return float(
            (np.linalg.norm(np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ), axis=1) / 2.0).sum()
        )
    raise ValueError(geometry.kind)


def _sample_box(size: tuple[float, float, float], count: int, rng: np.random.Generator) -> np.ndarray:
    half = np.asarray(size, dtype=np.float64) / 2.0
    face_areas = np.asarray([half[1] * half[2], half[1] * half[2], half[0] * half[2], half[0] * half[2], half[0] * half[1], half[0] * half[1]])
    faces = rng.choice(6, size=count, p=face_areas / face_areas.sum())
    points = rng.uniform(-1.0, 1.0, size=(count, 3)) * half
    for axis in range(3):
        negative = faces == axis * 2
        positive = faces == axis * 2 + 1
        points[negative, axis] = -half[axis]
        points[positive, axis] = half[axis]
    return points


def _sample_sphere(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * radius


def _sample_cylinder(radius: float, length: float, count: int, rng: np.random.Generator) -> np.ndarray:
    lateral_area = 2.0 * np.pi * radius * length
    cap_area = 2.0 * np.pi * radius * radius
    lateral = rng.random(count) < lateral_area / (lateral_area + cap_area)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = rng.uniform(-length / 2.0, length / 2.0, size=count)
    cap_indices = np.flatnonzero(~lateral)
    cap_radius = radius * np.sqrt(rng.random(len(cap_indices)))
    points[cap_indices, 0] = cap_radius * np.cos(theta[cap_indices])
    points[cap_indices, 1] = cap_radius * np.sin(theta[cap_indices])
    points[cap_indices, 2] = rng.choice((-length / 2.0, length / 2.0), size=len(cap_indices))
    return points


def _sample_mesh(geometry: Geometry, count: int, rng: np.random.Generator) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(Path(str(geometry.path)), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(geometry.scale)
    triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2.0
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0:
        raise ValueError(f"mesh has no sampleable surface: {geometry.path}")
    chosen = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    selected = triangles[chosen]
    return selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])


def _allocate_counts(areas: Iterable[float], total: int) -> np.ndarray:
    values = np.asarray(list(areas), dtype=np.float64)
    raw = values / values.sum() * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def sample_body_surface(body: BodyIR, *, count: int = 1024, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample a rigid target's collision surface in its PyBullet base frame."""

    if body.joints or len(body.links) != 1:
        raise NotImplementedError("canonical surface v1 supports one-link rigid targets")
    geometries = body.links[0].collisions
    if not geometries:
        raise ValueError(f"target has no collision geometry: {body.name}")
    rng = np.random.default_rng(seed)
    counts = _allocate_counts((_surface_area(item) for item in geometries), count)
    chunks: list[np.ndarray] = []
    geometry_ids: list[np.ndarray] = []
    for index, (geometry, geometry_count) in enumerate(zip(geometries, counts)):
        if not geometry_count:
            continue
        if geometry.kind == "box":
            points = _sample_box(geometry.size, int(geometry_count), rng)
        elif geometry.kind == "sphere":
            points = _sample_sphere(float(geometry.radius), int(geometry_count), rng)
        elif geometry.kind == "cylinder":
            points = _sample_cylinder(float(geometry.radius), float(geometry.length), int(geometry_count), rng)
        else:
            points = _sample_mesh(geometry, int(geometry_count), rng)
        chunks.append(_transform(points, geometry.pose))
        geometry_ids.append(np.full(int(geometry_count), index, dtype=np.int32))
    return np.concatenate(chunks), np.concatenate(geometry_ids)


def write_simulator_object_flow(
    output: str | Path,
    result: object,
    spec: object,
    world: WorldIR,
    *,
    point_count: int = 1024,
) -> Path:
    """Write canonical points and their exact per-observation-frame rigid flow."""

    target_name = str(getattr(result, "target_name"))
    target = next(body for body in world.bodies if body.name == target_name)
    canonical, geometry_ids = sample_body_surface(
        target, count=point_count, seed=int(getattr(spec, "seed"))
    )
    stride = round(1.0 / (float(getattr(spec, "fps")) * float(getattr(spec, "physics_dt_s"))))
    timestamps = np.asarray(getattr(result, "timestamps"), dtype=np.float64)
    indices = np.arange(0, len(timestamps), stride, dtype=np.int64)
    if indices[-1] != len(timestamps) - 1:
        indices = np.append(indices, len(timestamps) - 1)
    target_index = int(getattr(result, "target_index"))
    positions = np.asarray(getattr(result, "positions"), dtype=np.float64)[indices, target_index]
    orientations = np.asarray(getattr(result, "orientations"), dtype=np.float64)[indices, target_index]
    world_points = np.stack(
        [_transform(canonical, Pose(xyz=tuple(position), quat_xyzw=tuple(quaternion))) for position, quaternion in zip(positions, orientations)]
    )
    flow = world_points - world_points[0:1]
    aggregate = np.median(flow, axis=1)
    visible_surface_world: np.ndarray | None = None
    depth_frames = list(getattr(result, "depth_frames", []))
    segmentation_frames = list(getattr(result, "segmentation_frames", []))
    camera = getattr(result, "camera", None)
    target_instance_id = getattr(result, "target_instance_id", None)
    if depth_frames and segmentation_frames and camera is not None and target_instance_id is not None:
        if len(depth_frames) != len(indices) or len(segmentation_frames) != len(indices):
            raise ValueError("rendered surface labels must match observation-frame count")
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        camera_to_world = np.asarray(
            camera["camera_to_world_opencv"], dtype=np.float64
        )
        visible_chunks: list[np.ndarray] = []
        for depth, segmentation in zip(depth_frames, segmentation_frames):
            rows, columns = np.nonzero(segmentation == int(target_instance_id))
            if not len(rows):
                raise ValueError("target has no exact visible surface pixels")
            z = np.asarray(depth[rows, columns], dtype=np.float64)
            camera_points = np.stack(
                (
                    (columns - intrinsics[0, 2]) / intrinsics[0, 0] * z,
                    (rows - intrinsics[1, 2]) / intrinsics[1, 1] * z,
                    z,
                    np.ones_like(z),
                ),
                axis=-1,
            )
            world_surface = camera_points @ camera_to_world.T
            visible_chunks.append(np.median(world_surface[:, :3], axis=0))
        visible_surface_world = np.stack(visible_chunks)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "simulator_object_flow.v1",
        "surface_source": "collision_geometry",
        "sampling": "area_weighted_deterministic",
        "seed": int(getattr(spec, "seed")),
        "point_count": int(point_count),
        "target": target_name,
        "rigidity": "single_link_rigid_body",
        "coordinate_frame": "pybullet_world_z_up",
        "units": "m",
        "visible_surface_source": (
            "exact_metric_depth_and_instance_segmentation"
            if visible_surface_world is not None
            else "unavailable_legacy_or_nonrendered_result"
        ),
    }
    np.savez_compressed(
        destination,
        canonical_points_body=canonical.astype(np.float32),
        geometry_ids=geometry_ids,
        world_points=world_points.astype(np.float32),
        flow=flow.astype(np.float32),
        aggregate_median_displacement=aggregate.astype(np.float32),
        visible_surface_median_world=(
            visible_surface_world.astype(np.float32)
            if visible_surface_world is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        visible_surface_median_displacement=(
            (visible_surface_world - visible_surface_world[0]).astype(np.float32)
            if visible_surface_world is not None
            else np.empty((0, 3), dtype=np.float32)
        ),
        base_positions=positions.astype(np.float64),
        base_orientations_xyzw=orientations.astype(np.float64),
        timestamps=timestamps[indices],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return destination


def rebuild_simulator_object_flow(episode_dir: str | Path) -> Path:
    """Rebuild surface GT from an already recorded, exact-state episode."""

    root = Path(episode_dir).resolve()
    world = WorldIR.model_validate_json((root / "world_ir.json").read_text(encoding="utf-8"))
    from .models import EpisodeSpec

    spec = EpisodeSpec.model_validate_json(
        (root / "episode_spec.json").read_text(encoding="utf-8")
    )
    with np.load(root / "state_trajectory.npz", allow_pickle=False) as payload:
        names = [str(item) for item in payload["body_names"]]
        result = SimpleNamespace(
            target_name=spec.target,
            target_index=names.index(spec.target),
            timestamps=np.asarray(payload["timestamps"], dtype=np.float64),
            positions=np.asarray(payload["positions"], dtype=np.float64),
            orientations=np.asarray(payload["orientations_xyzw"], dtype=np.float64),
            depth_frames=[
                np.load(path) for path in sorted((root / "depth").glob("*.npy"))
            ],
            segmentation_frames=[
                np.load(path)
                for path in sorted((root / "segmentation").glob("*.npy"))
            ],
            camera=json.loads((root / "calibration.json").read_text(encoding="utf-8")),
            target_instance_id=json.loads(
                (root / "segmentation_encoding.json").read_text(encoding="utf-8")
            )["target_instance_id"],
        )
    return write_simulator_object_flow(
        root / "simulator_object_flow.npz", result, spec, world
    )
