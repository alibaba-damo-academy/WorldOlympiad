"""Canonical action-trajectory artifact used by the CAE evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ActionTrajectory:
    """A Cartesian action path sampled at physical timestamps.

    ``positions`` contains the released Dream2Flow trajectory optimizer's
    recovered Cartesian EE positions in the active video protocol. Both video
    roles use the same pinned extraction and robot registration. Raw object
    flow and policy action heads are not formal CAE inputs.

    Compared trajectories share registered axes; scoring may subtract each
    initial translation according to the explicitly selected metric version.
        """

    positions: np.ndarray
    timestamps: np.ndarray
    coordinate_frame: str = "initial_ee"
    units: str = "m"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (T, 3), got {positions.shape}")
        if len(positions) < 2:
            raise ValueError("an action trajectory needs at least two samples")
        if timestamps.ndim != 1 or timestamps.shape[0] != positions.shape[0]:
            raise ValueError(
                "timestamps must have shape (T,) and match positions; "
                f"got {timestamps.shape} for {positions.shape}"
            )
        if not np.isfinite(positions).all() or not np.isfinite(timestamps).all():
            raise ValueError("trajectory contains NaN or infinite values")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame must be non-empty")
        if self.units != "m":
            raise ValueError(f"CAE currently requires metre positions, got units={self.units!r}")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "timestamps", timestamps - timestamps[0])
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1])

    @property
    def relative_positions(self) -> np.ndarray:
        """Return a translated copy whose first sample is the origin."""

        return self.positions - self.positions[0]

    def save(self, path: str | Path) -> Path:
        """Save a portable compressed ``.npz`` trajectory artifact."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            positions=self.positions,
            timestamps=self.timestamps,
            coordinate_frame=np.asarray(self.coordinate_frame),
            units=np.asarray(self.units),
            metadata_json=np.asarray(json.dumps(dict(self.metadata), ensure_ascii=False)),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ActionTrajectory":
        """Load an artifact produced by :meth:`save`."""

        artifact_path = Path(path)
        with np.load(artifact_path, allow_pickle=False) as data:
            metadata_raw = str(data["metadata_json"].item()) if "metadata_json" in data else "{}"
            return cls(
                positions=data["positions"],
                timestamps=data["timestamps"],
                coordinate_frame=(
                    str(data["coordinate_frame"].item()) if "coordinate_frame" in data else "initial_ee"
                ),
                units=str(data["units"].item()) if "units" in data else "m",
                metadata=json.loads(metadata_raw),
            )


def timestamps_for_samples(
    num_samples: int,
    *,
    duration: float | None = None,
    sample_rate: float | None = None,
) -> np.ndarray:
    """Build timestamps without silently inventing a time scale."""

    if num_samples < 2:
        raise ValueError("num_samples must be at least two")
    if duration is not None and sample_rate is not None:
        raise ValueError("provide duration or sample_rate, not both")
    if duration is not None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        return np.linspace(0.0, float(duration), num_samples, dtype=np.float64)
    if sample_rate is not None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return np.arange(num_samples, dtype=np.float64) / float(sample_rate)
    raise ValueError("a physical duration or sample_rate is required")
