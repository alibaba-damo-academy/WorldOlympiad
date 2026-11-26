"""Deterministic grasp-band cross-section profile of one SceneCode SDF asset.

A parallel-jaw gripper does not have to span an object's whole horizontal
footprint; it only has to span the material that lies between its pads at the
height where it closes.  Sizing a manipuland from its overall AABB (a kettle's
spout-to-body span, a jug's handle-to-spout span) therefore shrinks objects
far below their authored, realistic size even though a perfectly good grasp
feature exists.

This module measures the real collision geometry through SceneCode's own SDF
projection (``prepare_physics_sdf``) and returns the object's collision
cross-section per vertical band, in the object's own frame with the origin at
the root link.  The result is a pure function of the SDF, so the scale and
grasp plan derived from it are reproducible and can be recorded as evidence.

Usage:
    python -m scenecode_pybullet_bridge.grasp_profile --sdf PATH [--out JSON]
"""

from __future__ import annotations

import argparse
import hashlib
import json

from . import json_evidence
import math
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p

from .sdf import prepare_physics_sdf

PROFILE_SCHEMA = "scenecode_grasp_band_profile.v1"
DEFAULT_PROFILE_STEP_M = 0.005
DEFAULT_RAY_STEP_M = 0.003
DEFAULT_BAND_HEIGHT_M = 0.06


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _body_links(body: int) -> list[int]:
    return [-1, *range(p.getNumJoints(body))]


def sdf_grasp_band_profile(
    sdf_path: str | Path,
    *,
    profile_step_m: float = DEFAULT_PROFILE_STEP_M,
    ray_step_m: float = DEFAULT_RAY_STEP_M,
    band_height_m: float = DEFAULT_BAND_HEIGHT_M,
    base_orientation_xyzw: tuple[float, float, float, float] | None = None,
    work_dir: str | Path | None = None,
) -> dict:
    """Return the object's grasp-band cross-section profile.

    The profile is expressed in the placed object root frame: ``base_offset_m``
    is the height above the lowest collision point and the horizontal centres
    are relative to the root link origin.  When ``base_orientation_xyzw`` is
    given it is applied exactly as the reference pipeline places the subject,
    so the reported horizontal axes are the world axes the gripper moves in
    (jaw yaw 0 closes along +X, jaw yaw 90 closes along +Y).
    """

    sdf_path = Path(sdf_path).resolve()
    if not sdf_path.is_file():
        raise FileNotFoundError(sdf_path)
    for label, value in (
        ("profile step", profile_step_m),
        ("ray step", ray_step_m),
        ("band height", band_height_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be positive and finite")

    owns_work = work_dir is None
    work = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp())
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    physics_sdf = work / f"{sdf_path.stem}.physics.sdf"
    prepare_physics_sdf(sdf_path, physics_sdf)

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("unable to start a PyBullet DIRECT client")
    try:
        bodies = p.loadSDF(str(physics_sdf))
        if len(bodies) != 1:
            raise ValueError(f"SDF must contain exactly one model, got {len(bodies)}")
        body = bodies[0]
        if base_orientation_xyzw is not None:
            values = tuple(float(value) for value in base_orientation_xyzw)
            if len(values) != 4 or not all(math.isfinite(value) for value in values):
                raise ValueError("base orientation must be four finite quaternion values")
            norm = math.sqrt(sum(value * value for value in values))
            if norm <= 0.0:
                raise ValueError("base orientation quaternion must be non-zero")
            p.resetBasePositionAndOrientation(
                body, [0.0, 0.0, 0.0], [value / norm for value in values]
            )
        origin, orientation = p.getBasePositionAndOrientation(body)
        origin = np.asarray(origin, dtype=np.float64)
        lows = []
        highs = []
        for link in _body_links(body):
            low, high = p.getAABB(body, link)
            lows.append(np.asarray(low, dtype=np.float64))
            highs.append(np.asarray(high, dtype=np.float64))
        low = np.min(np.stack(lows), axis=0)
        high = np.max(np.stack(highs), axis=0)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("object collision bounds are not finite")
        size = high - low
        if float(np.max(size)) <= 0.0:
            raise ValueError("object has no measurable collision geometry")

        margin = 0.02
        x_start, x_end = float(low[0] - margin), float(high[0] + margin)
        y_start, y_end = float(low[1] - margin), float(high[1] + margin)
        z_values = np.arange(low[2] + 0.5 * profile_step_m, high[2], profile_step_m)
        x_rays = np.arange(x_start, x_end + ray_step_m, ray_step_m)
        y_rays = np.arange(y_start, y_end + ray_step_m, ray_step_m)

        links = set(_body_links(body))
        hits: list[tuple[float, float, float]] = []

        def cast(start, end):
            hit = p.rayTest(list(start), list(end))[0]
            if int(hit[0]) != body or int(hit[1]) not in links:
                return None
            return np.asarray(hit[3], dtype=np.float64)

        for z in z_values:
            for y in y_rays:
                for start, end in (
                    ((x_start, y, z), (x_end, y, z)),
                    ((x_end, y, z), (x_start, y, z)),
                ):
                    point = cast(start, end)
                    if point is not None:
                        hits.append((float(point[0]), float(point[1]), float(point[2])))
            for x in x_rays:
                for start, end in (
                    ((x, y_start, z), (x, y_end, z)),
                    ((x, y_end, z), (x, y_start, z)),
                ):
                    point = cast(start, end)
                    if point is not None:
                        hits.append((float(point[0]), float(point[1]), float(point[2])))
    finally:
        p.disconnect(client)
        if owns_work:
            for path in sorted(work.glob("*")):
                path.unlink(missing_ok=True)
            try:
                work.rmdir()
            except OSError:
                pass

    if not hits:
        raise ValueError("no collision geometry was sampled for the grasp profile")
    samples = np.asarray(hits, dtype=np.float64) - origin
    base_z = float(samples[:, 2].min())
    top_z = float(samples[:, 2].max())
    if top_z - base_z <= 1.0e-6:
        raise ValueError("object collision geometry has no vertical extent")

    band_edges = np.arange(base_z, top_z + profile_step_m, profile_step_m)
    bands: list[dict] = []
    for index in range(len(band_edges) - 1):
        z0 = float(band_edges[index])
        z1 = float(band_edges[index + 1])
        selected = samples[(samples[:, 2] >= z0) & (samples[:, 2] < z1)]
        if selected.shape[0] < 3:
            continue
        bands.append(
            {
                "band_index": index,
                "base_offset_low_m": z0 - base_z,
                "base_offset_high_m": z1 - base_z,
                "base_offset_center_m": 0.5 * (z0 + z1) - base_z,
                "x_low_m": float(selected[:, 0].min()),
                "x_high_m": float(selected[:, 0].max()),
                "y_low_m": float(selected[:, 1].min()),
                "y_high_m": float(selected[:, 1].max()),
                "x_center_m": float(selected[:, 0].mean()),
                "y_center_m": float(selected[:, 1].mean()),
                "x_extent_m": float(selected[:, 0].max() - selected[:, 0].min()),
                "y_extent_m": float(selected[:, 1].max() - selected[:, 1].min()),
                "sample_count": int(selected.shape[0]),
            }
        )
    if not bands:
        raise ValueError("grasp-band profile produced no usable band")
    return {
        "schema_version": PROFILE_SCHEMA,
        "sdf_path": str(sdf_path),
        "sdf_sha256": _sha256(sdf_path),
        "aabb_low_m": [float(value) for value in (low - origin)],
        "aabb_high_m": [float(value) for value in (high - origin)],
        "base_offset_to_root_origin_m": float(-origin[2] + low[2]),
        "object_height_m": float(top_z - base_z),
        "profile_step_m": float(profile_step_m),
        "ray_step_m": float(ray_step_m),
        "band_height_m": float(band_height_m),
        "bands": bands,
    }


def plan_grasp_band(
    profile: dict,
    *,
    usable_aperture_m: float,
    minimum_aperture_extent_m: float = 0.0,
    maximum_lateral_offset_m: float | None = None,
    minimum_straddle_margin_m: float = 0.004,
    band_center_step_m: float = 0.005,
) -> dict:
    """Choose the vertical band the jaws can actually span.

    Candidate bands are scored by how much uniform scale they permit, then by
    how close the band centre is to the object's vertical mid-height (stability).
    A band is only feasible when the material inside it fits the gripper
    aperture, is thick enough to be pinched, and its lateral centre lies close
    enough to the object root axis that the two opposing finger contacts sit on
    opposite sides of that axis - which is exactly what the reference
    certificates verify before a carry is admitted.
    """

    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("grasp-band profile has an unsupported schema")
    if not math.isfinite(usable_aperture_m) or usable_aperture_m <= 0.0:
        raise ValueError("usable aperture must be positive and finite")
    if (
        not math.isfinite(minimum_straddle_margin_m)
        or minimum_straddle_margin_m < 0.0
    ):
        raise ValueError("minimum straddle margin must be finite and non-negative")
    band_height = float(profile["band_height_m"])
    bands = profile["bands"]
    base_z = float(profile["bands"][0]["base_offset_low_m"])
    top_z = float(profile["bands"][-1]["base_offset_high_m"])
    height = top_z - base_z
    candidates = []
    center = base_z + 0.5 * band_height
    while center <= top_z:
        z_low = center - 0.5 * band_height
        z_high = center + 0.5 * band_height
        selected = [
            band
            for band in bands
            if band["base_offset_high_m"] > z_low
            and band["base_offset_low_m"] < z_high
        ]
        if selected:
            x_extent = max(band["x_high_m"] for band in selected) - min(
                band["x_low_m"] for band in selected
            )
            y_extent = max(band["y_high_m"] for band in selected) - min(
                band["y_low_m"] for band in selected
            )
            axis = 0 if x_extent <= y_extent else 1
            extent = min(x_extent, y_extent)
            x_low = min(band["x_low_m"] for band in selected)
            x_high = max(band["x_high_m"] for band in selected)
            y_low = min(band["y_low_m"] for band in selected)
            y_high = max(band["y_high_m"] for band in selected)
            lateral_center = 0.5 * (x_low + x_high) if axis == 0 else 0.5 * (y_low + y_high)
            candidates.append(
                {
                    "band_center_offset_m": center,
                    "band_low_offset_m": z_low,
                    "band_high_offset_m": z_high,
                    "grip_axis": axis,
                    "grip_extent_m": float(extent),
                    "grip_axis_lateral_center_m": float(lateral_center),
                    "lateral_center_xy_m": [
                        float(0.5 * (x_low + x_high)),
                        float(0.5 * (y_low + y_high)),
                    ],
                    "x_extent_m": float(x_extent),
                    "y_extent_m": float(y_extent),
                    "band_count": len(selected),
                }
            )
        center += band_center_step_m
    if not candidates:
        raise ValueError("no grasp band could be sampled from the profile")
    feasible = [
        candidate
        for candidate in candidates
        if candidate["grip_extent_m"]
        <= usable_aperture_m + 1.0e-9
        and candidate["grip_extent_m"] >= minimum_aperture_extent_m - 1.0e-9
        and (
            maximum_lateral_offset_m is None
            or abs(candidate["grip_axis_lateral_center_m"])
            <= maximum_lateral_offset_m + 1.0e-9
        )
        and (
            0.5 * candidate["grip_extent_m"]
            - abs(candidate["grip_axis_lateral_center_m"])
            >= minimum_straddle_margin_m - 1.0e-9
        )
    ]
    pool = feasible if feasible else candidates

    def admissible_scale(candidate: dict) -> float:
        raw = min(1.0, usable_aperture_m / candidate["grip_extent_m"])
        return math.floor(raw / 0.05 + 1.0e-9) * 0.05

    best = min(
        pool,
        key=lambda candidate: (
            0 if candidate["grip_extent_m"] <= usable_aperture_m else 1,
            -admissible_scale(candidate),
            abs(candidate["band_center_offset_m"] - 0.5 * height),
            round(candidate["grip_extent_m"], 9),
            candidate["band_center_offset_m"],
        ),
    )
    return {
        "schema_version": "scenecode_grasp_band_plan.v1",
        "profile_schema_version": PROFILE_SCHEMA,
        "object_height_m": height,
        "usable_aperture_m": float(usable_aperture_m),
        "minimum_aperture_extent_m": float(minimum_aperture_extent_m),
        "minimum_straddle_margin_m": float(minimum_straddle_margin_m),
        "maximum_lateral_offset_m": (
            None if maximum_lateral_offset_m is None else float(maximum_lateral_offset_m)
        ),
        "band_center_offset_m": float(best["band_center_offset_m"]),
        "band_low_offset_m": float(best["band_low_offset_m"]),
        "band_high_offset_m": float(best["band_high_offset_m"]),
        "grip_axis": int(best["grip_axis"]),
        "grip_extent_m": float(best["grip_extent_m"]),
        "grip_axis_lateral_center_m": float(best["grip_axis_lateral_center_m"]),
        "predicted_straddle_margin_m": float(
            0.5 * best["grip_extent_m"] - abs(best["grip_axis_lateral_center_m"])
        ),
        "lateral_center_xy_m": [
            float(best["lateral_center_xy_m"][0]),
            float(best["lateral_center_xy_m"][1]),
        ],
        "admissible_uniform_scale": float(admissible_scale(best)),
        "x_extent_m": float(best["x_extent_m"]),
        "y_extent_m": float(best["y_extent_m"]),
        "feature_grasp_feasible": bool(feasible),
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "policy": "widest_admissible_scale_at_narrowest_root_aligned_grasp_band",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--out")
    parser.add_argument("--profile-step", type=float, default=DEFAULT_PROFILE_STEP_M)
    parser.add_argument("--ray-step", type=float, default=DEFAULT_RAY_STEP_M)
    parser.add_argument("--band-height", type=float, default=DEFAULT_BAND_HEIGHT_M)
    parser.add_argument("--usable-aperture", type=float)
    parser.add_argument("--minimum-extent", type=float, default=0.0)
    parser.add_argument("--maximum-lateral-offset", type=float)
    parser.add_argument("--minimum-straddle-margin", type=float, default=0.004)
    parser.add_argument("--orientation-xyzw", help="comma separated x,y,z,w")
    args = parser.parse_args()
    profile = sdf_grasp_band_profile(
        args.sdf,
        profile_step_m=args.profile_step,
        ray_step_m=args.ray_step,
        band_height_m=args.band_height,
        base_orientation_xyzw=(
            tuple(float(v) for v in args.orientation_xyzw.split(","))
            if args.orientation_xyzw
            else None
        ),
    )
    payload: dict = {"profile": profile}
    if args.usable_aperture is not None:
        payload["plan"] = plan_grasp_band(
            profile,
            usable_aperture_m=args.usable_aperture,
            minimum_aperture_extent_m=args.minimum_extent,
            maximum_lateral_offset_m=args.maximum_lateral_offset,
            minimum_straddle_margin_m=args.minimum_straddle_margin,
        )
    text = json_evidence.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
