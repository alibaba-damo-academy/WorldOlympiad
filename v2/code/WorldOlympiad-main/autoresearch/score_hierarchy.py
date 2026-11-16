"""Three equal macro tracks; motion fidelity is an Interaction submetric."""
from __future__ import annotations

from copy import deepcopy
import math
from statistics import mean
from typing import Any

SCORE_POLICY = "three_track_equal_interaction_cae_half.v1"
MACRO_TRACKS = ("physical", "geometry", "interaction")


def _measurement(row: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(row)
    status = result.get("status")
    if status == "scored":
        value = result.get("score")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("scored submetric requires a finite score in [0, 1]")
    elif status not in {"pending", "not_applicable"}:
        raise ValueError(f"unsupported measurement status: {status}")
    elif result.get("score") is not None:
        raise ValueError("pending or not-applicable measurements cannot carry scores")
    return result


def compose_three_tracks(measurements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(measurements) != {"physical", "geometry", "interaction", "continuous_action"}:
        raise ValueError("three-track aggregation requires the four underlying measurements")
    parts = {name: _measurement(row) for name, row in measurements.items()}
    submetrics = {"semantic_interaction": parts["interaction"],
                  "continuous_action": parts["continuous_action"]}
    applicable = [row for row in submetrics.values() if row["status"] != "not_applicable"]
    interaction = {"submetrics": submetrics,
                   "weights": {"semantic_interaction": 0.5, "continuous_action": 0.5}}
    if not applicable:
        interaction.update(status="not_applicable", score=None,
                           reason="no applicable interaction submetric")
    elif any(row["status"] == "pending" for row in applicable):
        interaction.update(status="pending", score=None,
                           reason="required interaction submetric evidence is missing")
    else:
        interaction.update(status="scored", score=mean(row["score"] for row in applicable))
    tracks = {"physical": parts["physical"], "geometry": parts["geometry"],
              "interaction": interaction}
    pending = any(row["status"] == "pending" for row in tracks.values())
    scored = [row["score"] for row in tracks.values() if row["status"] == "scored"]
    return {"score_policy": SCORE_POLICY, "track_results": tracks,
            "status": "pending" if pending or not scored else "complete",
            "worldolympiad2_score": None if pending or not scored else mean(scored)}


def from_judge_summaries(physical: dict | None, interaction: dict | None,
                         geometry: dict | None, action: dict) -> dict:
    def core(summary: dict | None, key: str) -> dict:
        value = (summary or {}).get(key)
        return {"status": "pending" if value is None else "scored", "score": value}

    value = action.get("score")
    if action.get("status") == "applicable" and value is not None:
        cae = {"status": "scored", "score": value,
               "components": {"trajectory_shape": action.get("path_score"),
                              "speed_curve": action.get("velocity_score")},
               "metric_version": action.get("metric_version")}
    elif action.get("status") == "not_applicable" and action.get("reason") != "no_cae_manifest":
        cae = {"status": "not_applicable", "score": None, "reason": action.get("reason")}
    else:
        cae = {"status": "pending", "score": None,
               "reason": action.get("reason") or "required motion evidence is missing"}
    return compose_three_tracks({"physical": core(physical, "score"),
        "geometry": core(geometry, "final_score_normalized"),
        "interaction": core(interaction, "score"), "continuous_action": cae})
