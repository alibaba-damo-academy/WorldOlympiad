"""Three equal macro tracks; CAE v2 is half of Interaction.

Measurement rows: {"status": scored|pending|not_applicable, "score": float|None}.
The CAE row must declare its trajectory source.  Formal sources (decision
2026-10-05): "simulator_execution" for WAMs (the trajectory their own actions
produced in their own simulator) and "video_reconstruction" for pure world
models (objects only, with a calibrated noise floor).
"""

from __future__ import annotations

from statistics import mean

SCORE_POLICY = "three_track_equal.interaction_half_cae_v2.v1"
FORMAL_CAE_SOURCES = {"simulator_execution", "video_reconstruction"}


def _row(row: dict | None) -> dict:
    if row is None:
        return {"status": "pending", "score": None, "reason": "not measured"}
    if row.get("status") == "scored" and not (0 <= float(row["score"]) <= 1):
        raise ValueError("scored measurement must lie in [0, 1]")
    return dict(row)


def compose(physical: dict | None, geometry: dict | None, semantic_interaction: dict | None,
            cae: dict | None) -> dict:
    cae = _row(cae)
    if cae["status"] == "scored" and cae.get("source") not in FORMAL_CAE_SOURCES:
        cae = {"status": "pending", "score": None, "diagnostic": cae,
               "reason": f"CAE source {cae.get('source')!r} is diagnostic, not formal"}
    sub = {"semantic_interaction": _row(semantic_interaction), "continuous_action": cae}
    live = [r for r in sub.values() if r["status"] != "not_applicable"]
    if not live:
        inter = {"status": "not_applicable", "score": None}
    elif any(r["status"] == "pending" for r in live):
        inter = {"status": "pending", "score": None}
    else:
        inter = {"status": "scored", "score": mean(r["score"] for r in live)}
    inter.update(submetrics=sub, weights={"semantic_interaction": 0.5, "continuous_action": 0.5})
    tracks = {"physical": _row(physical), "geometry": _row(geometry), "interaction": inter}
    pending = [k for k, r in tracks.items() if r["status"] == "pending"]
    scored = [r["score"] for r in tracks.values() if r["status"] == "scored"]
    return {"score_policy": SCORE_POLICY, "tracks": tracks,
            "status": "pending" if pending or not scored else "complete",
            "pending_tracks": pending,
            "worldolympiad2_score": None if pending or not scored else mean(scored)}
