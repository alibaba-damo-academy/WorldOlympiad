"""Compare the entire supplied action trajectory using two scores only.

Positions are measured in shared registered axes. Path comparison removes only
starting translation, retaining shape, direction and physical extent. Speed
comes from the original timestamps. Observation masks are diagnostics only.
"""
from dataclasses import dataclass
import numpy as np
from .schema import ActionTrajectory
from .current_metric import (
    CURRENT_METRIC_VERSION, CURRENT_METRIC_PROFILE, CURRENT_SCHEMA_VERSION,
    current_metric_config, current_metric_config_sha256,
)

# Stable public names for batch consumers, all referring to this ONE metric.
FORMAL_INDEPENDENT_METRIC_VERSION = CURRENT_METRIC_VERSION
FORMAL_METRIC_PROFILE_ID = CURRENT_METRIC_PROFILE
FORMAL_METRIC_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION
formal_metric_config = current_metric_config
formal_metric_config_sha256 = current_metric_config_sha256
METRIC_SCHEMA_BY_VERSION = {CURRENT_METRIC_VERSION: CURRENT_SCHEMA_VERSION}


@dataclass(frozen=True)
class ContinuousActionScore:
    details: dict
    @property
    def path_score(self): return self.details["path_score"]
    @property
    def velocity_score(self): return self.details["velocity_score"]
    @property
    def continuous_score(self): return self.details["continuous_score"]
    @property
    def metric_version(self): return CURRENT_METRIC_VERSION
    @property
    def metric_profile(self): return CURRENT_METRIC_PROFILE
    def to_dict(self): return dict(self.details)


def _dot_integral(a, b, steps):
    """Exact dot-product integral of two piecewise linear curves."""
    return float(np.sum(steps * (2*np.sum(a[:-1]*b[:-1], axis=1)
        + np.sum(a[:-1]*b[1:], axis=1) + np.sum(a[1:]*b[:-1], axis=1)
        + 2*np.sum(a[1:]*b[1:], axis=1))/6))


def _observation(trajectory):
    raw = trajectory.metadata.get("observed_frame_mask", trajectory.metadata.get("observed"))
    if raw is None:
        return {"fraction": None, "observed_seconds": None, "frame_ranges": None}
    mask = np.asarray(raw)
    if mask.shape != (len(trajectory.positions),) or not np.isin(mask, [0, 1]).all():
        raise ValueError("observed_frame_mask must contain one boolean per trajectory sample")
    mask = mask.astype(bool)
    seconds = float(np.diff(trajectory.timestamps)[mask[:-1] & mask[1:]].sum())
    if seconds <= 0:
        raise ValueError("trajectory has no observed adjacent samples to support its reconstruction")
    ranges, start = [], None
    for i, observed in enumerate(mask):
        if observed and start is None:
            start = i
        if start is not None and (not observed or i == len(mask)-1):
            end = i if observed else i-1
            if end > start:
                ranges.append([start, end])
            start = None
    return {"fraction": seconds/trajectory.duration, "observed_seconds": seconds, "frame_ranges": ranges}


def score_action_trajectories(
    reference: ActionTrajectory, predicted: ActionTrajectory, *,
    metric_version: str = CURRENT_METRIC_VERSION,
    robot_workspace_diameter_m: float | None = None,
) -> ContinuousActionScore:
    """Score all source samples; workspace size is provenance, not a divisor."""
    if metric_version != CURRENT_METRIC_VERSION:
        raise ValueError("retired CAE metric; only full_trajectory_shape_speed is supported")
    if reference.coordinate_frame != predicted.coordinate_frame:
        raise ValueError("CAE requires the same registered coordinate frame")
    if robot_workspace_diameter_m is not None and (
        not np.isfinite(robot_workspace_diameter_m) or robot_workspace_diameter_m <= 0
    ):
        raise ValueError("robot workspace diameter must be finite and positive")
    ro, po = _observation(reference), _observation(predicted)
    r, p = reference.relative_positions, predicted.relative_positions
    rl = np.linalg.norm(np.diff(r, axis=0), axis=1)
    pl = np.linalg.norm(np.diff(p, axis=0), axis=1)
    lr, lp = float(rl.sum()), float(pl.sum())
    path = speed = float(lr == 0 and lp == 0)
    route_rms, path_knots, time_knots = None, 0, 0
    if lr > 0 and lp > 0:
        def curve(points, lengths):
            s = np.r_[0.0, np.cumsum(lengths)] / lengths.sum()
            keep = np.r_[True, np.diff(s) > 0]
            return s[keep], points[keep]
        rs, rp = curve(r, rl)
        ps, pp = curve(p, pl)
        s = np.union1d(rs, ps)
        ds = np.diff(s)
        ar = np.column_stack([np.interp(s, rs, rp[:, i]) for i in range(3)])
        ap = np.column_stack([np.interp(s, ps, pp[:, i]) for i in range(3)])
        def rms(x): return float(np.sqrt(max(0.0, _dot_integral(x, x, ds))))
        route_rms = rms(ar-ap)
        path = float(np.clip(1-route_rms/(rms(ar)+rms(ap)), 0, 1))
        # Only the horizontal time axis is normalized; speed remains m/s.
        ur = reference.timestamps/reference.duration
        up = predicted.timestamps/predicted.duration
        u = np.union1d(ur, up)
        du, mid = np.diff(u), (u[:-1]+u[1:])/2
        vr, vp = rl/np.diff(reference.timestamps), pl/np.diff(predicted.timestamps)
        a = vr[np.clip(np.searchsorted(ur, mid, side="right")-1, 0, len(vr)-1)]
        b = vp[np.clip(np.searchsorted(up, mid, side="right")-1, 0, len(vp)-1)]
        speed = float(np.clip(2*np.sum(du*a*b)/np.sum(du*(a*a+b*b)), 0, 1))
        path_knots, time_knots = len(s), len(u)
    fractions = [ro["fraction"], po["fraction"]]
    partial = any(x is not None and x < 1-1e-12 for x in fractions)
    status = "partial" if partial else "unknown" if None in fractions else "complete"
    return ContinuousActionScore({
        "metric_version": CURRENT_METRIC_VERSION, "metric_profile": CURRENT_METRIC_PROFILE,
        "metric_config_sha256": current_metric_config_sha256(),
        "path_score": path, "velocity_score": speed,
        "continuous_score": float(np.sqrt(path*speed)),
        "shape_score": path, "speed_curve_score": speed,
        "scored_components": ["trajectory_shape", "speed_curve"],
        "aggregation": "sqrt(path_score * velocity_score)",
        "reference_path_length": lr, "predicted_path_length": lp,
        "reference_duration": reference.duration, "predicted_duration": predicted.duration,
        "reference_characteristic_speed": lr/reference.duration,
        "predicted_characteristic_speed": lp/predicted.duration,
        "reference_samples": len(r), "predicted_samples": len(p),
        "reference_source_node_range": [0, len(r)-1], "predicted_source_node_range": [0, len(p)-1],
        "path_integration_knots": path_knots, "speed_integration_knots": time_knots,
        "route_rms_distance_m": route_rms,
        "prediction_has_motion": lp > 0, "orientation_evaluated": False,
        "partial_observation": partial, "observation_coverage": status,
        "reference_observation_coverage": ro["fraction"], "prediction_observation_coverage": po["fraction"],
        "reference_observed_seconds": ro["observed_seconds"], "prediction_observed_seconds": po["observed_seconds"],
        "reference_observed_frame_ranges": ro["frame_ranges"], "prediction_observed_frame_ranges": po["frame_ranges"],
        "measurement_scope": "complete supplied Cartesian trajectories; source metadata distinguishes observed support from reconstruction",
        "visibility_affects_score": False,
    })
