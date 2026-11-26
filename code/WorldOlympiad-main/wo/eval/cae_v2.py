"""CAE v2: continuous-action fidelity, a submetric of the Interaction track.

Compares the motion of task entities (robot end effector + task objects) in a
model episode against the reference.  For WAMs the trajectories are the ones
their actions produced in their own simulator; for pure world models they are
reconstructed from video (objects only, with a calibrated noise floor).

Per entity:
  1. resample on a 15 Hz grid, 0.2 s Gaussian low-pass, subtract the start;
  2. cut each side to its active segment: from the first 2 cm departure to task
     completion (if known) or to when it settles at its final pose;
  3. align both segments by progress (pace and an initial wait are free and only
     reported as diagnostics);
  4. track = 1 - rms(r-p)/(rms(r)+rms(p)) (noise floor removed);
     speed = similarity of the normalised speed profiles; entity = sqrt(track*speed).
CAE = 0.5 * end_effector + 0.5 * mean(task objects); if one group is absent the
other carries the full weight.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

METRIC_VERSION = "cae_v2_active_segment_progress"
GRID_HZ = 15.0
SMOOTH_S = 0.2
EE_WEIGHT = 0.5

CONFIG = {
    "metric_version": METRIC_VERSION, "grid_hz": GRID_HZ, "smooth_sigma_s": SMOOTH_S,
    "alignment": "each side's active segment (onset 2 cm -> completion or settle), progress-normalised",
    "end_effector_weight": EE_WEIGHT,
    "track": "1 - rms(r(u)-p(u)) / (rms(r)+rms(p)) over progress u, noise floor removed in quadrature",
    "speed": "similarity of mean-normalised speed profiles over progress (pace-free)",
    "diagnostics": "onset delay, pace ratio, path lengths (not scored)",
    "entity": "sqrt(track*speed)", "aggregate": "0.5*EE + 0.5*mean(objects)",
}
CONFIG_SHA256 = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Track:
    positions: np.ndarray  # (N, 3) metres
    timestamps: np.ndarray  # (N,) seconds, strictly increasing
    kind: str = "object"  # end_effector | object
    noise_floor_m: float = 0.0  # reconstruction RMS error of this pipeline (calibrated)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        p, t = np.asarray(self.positions, float), np.asarray(self.timestamps, float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(t) or len(p) < 2:
            raise ValueError("track needs (N,3) positions and N timestamps, N >= 2")
        if not (np.isfinite(p).all() and np.isfinite(t).all()) or np.any(np.diff(t) <= 0):
            raise ValueError("track must be finite with increasing timestamps")


def _prepare(tr: Track) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter1d

    t = np.asarray(tr.timestamps, float) - tr.timestamps[0]
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / GRID_HZ)
    pos = np.column_stack([np.interp(grid, t, tr.positions[:, i]) for i in range(3)])
    pos = gaussian_filter1d(pos, SMOOTH_S * GRID_HZ, axis=0, mode="nearest")
    return grid, pos - pos[0]


def _speed(pos: np.ndarray) -> np.ndarray:
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) * GRID_HZ
    return np.r_[v, v[-1:]]


ONSET_M = 0.02  # motion starts when the entity leaves its start by this much
SETTLE_M = 0.02  # motion ends when the entity stays within this of its final pose
N_PROGRESS = 200


def active_segment(t: np.ndarray, pos: np.ndarray, end_time: float | None = None) -> tuple[int, int] | None:
    """[onset, end] indices of the entity's active motion (None if it never moves)."""
    moved = np.linalg.norm(pos - pos[0], axis=1) > ONSET_M
    if not moved.any():
        return None
    onset = max(0, int(np.argmax(moved)) - 1)
    if end_time is not None:
        end = int(np.clip(np.searchsorted(t, end_time), onset + 1, len(t) - 1))
    else:
        away = np.linalg.norm(pos - pos[-1], axis=1) > SETTLE_M
        end = int(np.nonzero(away)[0][-1]) + 1 if away.any() else len(t) - 1
        end = min(max(end, onset + 1), len(t) - 1)
    return onset, end


def _progress(t, pos, seg):
    a, b = seg
    u = np.linspace(t[a], t[b], N_PROGRESS)
    q = np.column_stack([np.interp(u, t, pos[:, i]) for i in range(3)])
    return q - q[0], float(t[b] - t[a])


def score_entity(ref: Track, pred: Track, *, ref_end_s: float | None = None,
                 pred_end_s: float | None = None) -> dict:
    """Motion fidelity over each side's active segment, aligned by progress.

    Waiting before moving and overall pace are not penalised (reported as
    diagnostics); shape, direction, extent and the speed profile within the
    motion are.  An unfinished motion ends at its last movement (or the window
    end), so stopping early or drifting still lowers the score."""
    tr, r = _prepare(ref)
    tp, p = _prepare(pred)
    floor2 = ref.noise_floor_m ** 2 + pred.noise_floor_m ** 2
    sr, sp = active_segment(tr, r, ref_end_s), active_segment(tp, p, pred_end_s)
    diag = {"reference_path_m": float(np.sum(np.linalg.norm(np.diff(r, axis=0), axis=1))),
            "prediction_path_m": float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))),
            "noise_floor_m": float(np.sqrt(floor2))}
    if sr is None and sp is None:
        return {"score": 1.0, "track": 1.0, "speed": 1.0, "both_static": True, **diag}
    if sr is None or sp is None:
        return {"score": 0.0, "track": 0.0, "speed": 0.0,
                "static_side": "reference" if sr is None else "prediction", **diag}
    a, dur_r = _progress(tr, r, sr)
    b, dur_p = _progress(tp, p, sp)
    err2 = float(np.mean(np.sum((a - b) ** 2, 1)))
    err = float(np.sqrt(max(0.0, err2 - floor2)))
    scale = float(np.sqrt(np.mean(np.sum(a * a, 1)))) + float(np.sqrt(np.mean(np.sum(b * b, 1))))
    track = 1.0 if scale <= 1e-9 else float(np.clip(1 - err / scale, 0, 1))
    # speed profile shape over progress (pace-free): each normalised by its mean
    va = np.linalg.norm(np.diff(a, axis=0), axis=1)
    vb = np.linalg.norm(np.diff(b, axis=0), axis=1)
    va, vb = va / max(va.mean(), 1e-12), vb / max(vb.mean(), 1e-12)
    speed = float(np.clip(2 * np.mean(va * vb) / np.mean(va * va + vb * vb), 0, 1))
    return {"score": float(np.sqrt(track * speed)), "track": track, "speed": speed,
            "rms_error_m": float(np.sqrt(err2)), "rms_error_after_floor_m": err,
            "reference_active_s": dur_r, "prediction_active_s": dur_p,
            "prediction_onset_delay_s": float(tp[sp[0]] - tr[sr[0]]),
            "pace_ratio": dur_r / max(dur_p, 1e-9), **diag}


def score(reference: dict[str, Track], prediction: dict[str, Track], *,
          ref_end_s: float | None = None, pred_end_s: float | None = None) -> dict:
    """Entities are matched by name; every reference entity must be predicted.
    ``*_end_s``: task-completion time of each episode, when known."""
    missing = set(reference) - set(prediction)
    if missing:
        return {"status": "pending", "score": None, "reason": f"no predicted track for {sorted(missing)}",
                "metric_version": METRIC_VERSION}
    entities = {name: {"kind": tr.kind, **score_entity(tr, prediction[name], ref_end_s=ref_end_s,
                                                        pred_end_s=pred_end_s)}
                for name, tr in reference.items()}
    ee = [e["score"] for e in entities.values() if e["kind"] == "end_effector"]
    obj = [e["score"] for e in entities.values() if e["kind"] != "end_effector"]
    if not ee and not obj:
        return {"status": "not_applicable", "score": None, "reason": "no tracked entity",
                "metric_version": METRIC_VERSION}
    if ee and obj:
        value = EE_WEIGHT * float(np.mean(ee)) + (1 - EE_WEIGHT) * float(np.mean(obj))
    else:
        value = float(np.mean(ee or obj))
    return {"status": "scored", "score": value, "metric_version": METRIC_VERSION,
            "config_sha256": CONFIG_SHA256,
            "end_effector_score": float(np.mean(ee)) if ee else None,
            "object_score": float(np.mean(obj)) if obj else None,
            "entities": entities}


def tracks_from_trajectory(path, objects: list[str], *, noise_floor_m: float = 0.0) -> dict[str, Track]:
    """EE + named objects from a wo / Isaac trajectory.json (frames at 15 Hz)."""
    frames = json.loads(open(path).read())["frames"]
    t = np.array([f.get("time_s", i / 15.0) for i, f in enumerate(frames)], float)
    if np.any(np.diff(t) <= 0):
        t = np.arange(len(frames)) / 15.0
    out = {"end_effector": Track(np.array([f["robot"]["end_effector_position"] for f in frames]), t,
                                 "end_effector", noise_floor_m)}
    for oid in objects:
        if all(oid in f.get("objects", {}) for f in frames):
            pos = np.array([f["objects"][oid]["link_position"] for f in frames])
            out[oid] = Track(pos, t, "object", noise_floor_m)
    return out
