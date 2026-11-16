"""Quality gates for video-derived action-carrier trajectories.

The scorer must fail closed: a visually plausible curve is not evidence that
the end effector was tracked for the complete rollout.  These checks run
before any trajectory metric is computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingQualityConfig:
    min_points_per_frame: int = 8
    min_frame_coverage: float = 0.90
    max_untracked_gap_frames: int = 2
    min_anchor_groups: int = 2


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def assess_tracking_quality(
    valid_point_counts: np.ndarray,
    *,
    anchor_groups_used: int,
    config: TrackingQualityConfig | None = None,
) -> dict[str, object]:
    """Return a JSON-safe pass/fail report for one complete video timeline."""

    cfg = config or TrackingQualityConfig()
    counts = np.asarray(valid_point_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("valid_point_counts must be a 1-D full-video array")
    if np.any(counts < 0):
        raise ValueError("valid point counts cannot be negative")

    tracked = counts >= cfg.min_points_per_frame
    coverage = float(np.mean(tracked))
    longest_gap = _longest_false_run(tracked)
    failures: list[str] = []
    if coverage < cfg.min_frame_coverage:
        failures.append(
            f"frame coverage {coverage:.3f} < {cfg.min_frame_coverage:.3f}"
        )
    if longest_gap > cfg.max_untracked_gap_frames:
        failures.append(
            f"untracked gap {longest_gap} > {cfg.max_untracked_gap_frames} frames"
        )
    if anchor_groups_used < cfg.min_anchor_groups:
        failures.append(
            f"anchor groups {anchor_groups_used} < {cfg.min_anchor_groups}"
        )

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "frame_count": int(counts.size),
        "valid_point_counts": counts.tolist(),
        "tracked_frame_count": int(np.count_nonzero(tracked)),
        "frame_coverage": coverage,
        "longest_untracked_gap_frames": int(longest_gap),
        "anchor_groups_used": int(anchor_groups_used),
        "thresholds": asdict(cfg),
    }


def require_tracking_quality(metadata: dict[str, object]) -> dict[str, object]:
    """Validate the embedded extraction report and reject unscoreable tracks."""

    quality = metadata.get("tracking_quality")
    if not isinstance(quality, dict):
        raise ValueError("video action trajectory is missing tracking_quality metadata")
    if quality.get("status") != "pass":
        failures = quality.get("failures", ["unspecified tracking failure"])
        raise ValueError(
            "video action trajectory failed tracking quality gates: "
            + "; ".join(str(item) for item in failures)
        )
    anchor_quality = metadata.get("anchor_selection_quality")
    if not isinstance(anchor_quality, dict):
        raise ValueError("video action trajectory is missing anchor_selection_quality metadata")
    if anchor_quality.get("status") != "pass":
        raise ValueError(
            "video action trajectory failed action-carrier anchor gates: "
            + "; ".join(
                str(item)
                for item in anchor_quality.get("failures", ["unspecified anchor failure"])
            )
        )
    return quality
