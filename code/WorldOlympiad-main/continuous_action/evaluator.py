"""Manifest-driven full-rollout Continuous Action Evaluation.

WorldOlympiad semantic chunks are post-hoc diagnostic intervals. They never
partition model inference, reset model state, or inject future GT observations.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .adapters import load_dream2flow_video_plan
from .chunks import ChunkSpec, validate_worldolympiad_chunks
from .schema import ActionTrajectory


def _resolve_path(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def _load_trajectory(
    spec: dict[str, Any], *, manifest_dir: Path, duration: float
) -> ActionTrajectory:
    kind = str(spec.get("type", "canonical"))
    if "path" not in spec:
        raise ValueError(f"trajectory spec of type {kind!r} is missing 'path'")
    path = _resolve_path(manifest_dir, str(spec["path"]))
    timing: dict[str, float] = {}
    if spec.get("sample_rate") is not None:
        timing["sample_rate"] = float(spec["sample_rate"])
    else:
        timing["duration"] = float(spec.get("duration", duration))

    trajectory: ActionTrajectory
    if kind == "canonical":
        trajectory = ActionTrajectory.load(path)
    elif kind == "dream2flow_video_plan":
        if spec.get("video_path") is None:
            raise ValueError("dream2flow_video_plan specs must set video_path")
        if spec.get("object_name") is None:
            raise ValueError("dream2flow_video_plan specs must set object_name")
        trajectory = load_dream2flow_video_plan(
            path,
            **timing,
            video_path=_resolve_path(manifest_dir, str(spec["video_path"])),
            video_role=str(spec.get("video_role", "")),
            object_name=str(spec["object_name"]),
            source_model=(
                str(spec["source_model"])
                if spec.get("source_model") is not None
                else None
            ),

        )
    else:
        raise ValueError(
            f"unsupported trajectory type: {kind!r}; active CAE accepts only "
            "dream2flow_video_plan and exact-state validation accepts canonical"
        )

    metadata = dict(trajectory.metadata)
    for key in ("benchmark_status", "checkpoint_revision", "calibration_id", "sample_id"):
        if spec.get(key) is not None:
            metadata[key] = spec[key]
    return ActionTrajectory(
        positions=trajectory.positions,
        timestamps=trajectory.timestamps,
        coordinate_frame=trajectory.coordinate_frame,
        units=trajectory.units,
        metadata=metadata,
    )


def slice_trajectory(
    trajectory: ActionTrajectory,
    start_sec: float,
    end_sec: float,
) -> ActionTrajectory:
    """Slice a completed rollout for post-hoc diagnostics.

    Exact interval boundaries are inserted by linear interpolation. The model
    has already finished its rollout before this function is called.
    """

    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError(f"invalid diagnostic interval [{start_sec}, {end_sec})")
    if end_sec > trajectory.duration + 1e-6:
        raise ValueError(
            "diagnostic interval exceeds completed rollout: "
            f"end={end_sec:.6f}s duration={trajectory.duration:.6f}s"
        )
    interior = trajectory.timestamps[
        (trajectory.timestamps > start_sec) & (trajectory.timestamps < end_sec)
    ]
    timestamps = np.unique(np.concatenate(([start_sec], interior, [end_sec]))).astype(
        np.float64
    )
    positions = np.stack(
        [
            np.interp(timestamps, trajectory.timestamps, trajectory.positions[:, axis])
            for axis in range(3)
        ],
        axis=-1,
    )
    metadata = dict(trajectory.metadata)
    metadata.update(
        {
            "posthoc_slice": True,
            "parent_interval_sec": [float(start_sec), float(end_sec)],
        }
    )
    return ActionTrajectory(
        positions=positions,
        timestamps=timestamps,
        coordinate_frame=trajectory.coordinate_frame,
        units=trajectory.units,
        metadata=metadata,
    )


def evaluate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Evaluate the single current complete-trajectory manifest."""
    from .current_evaluator import evaluate_current_manifest
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return evaluate_current_manifest(path, manifest, _load_trajectory)


def write_evaluation(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"refusing to overwrite malformed CAE result: {path}") from exc
        if existing.get("metric_version") != result.get("metric_version"):
            raise FileExistsError(
                "refusing to overwrite a CAE artifact from another metric version; "
                "write a versioned output path"
            )
        if existing != result:
            raise FileExistsError(
                "refusing to mutate an existing CAE result with different inputs or "
                "provenance; write a new versioned output path"
            )
        return path

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                "refusing to race-overwrite an immutable CAE result"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path
