"""Compact WAM report: action shape in Geometry, speed profile in Interaction.

Uses existing native measurements. Historical v2 episode results remain intact.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean

VERSION = "geometry_action_shape_interaction_speed.v1"
DISPLAY = {"cosmos3_nano_droid": "Cosmos3 Nano-DROID", "gr00t_n1.6_droid": "GR00T N1.6-DROID",
           "fastwam": "FastWAM", "lingbot_va": "LingBot-VA", "longwam": "Long-WAM RoboTwin2.0-IDM"}


def measured(value, *, source, scale=(0., 1.)):
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
    if not valid or not scale[0] <= value <= scale[1]:
        return {"status": "pending", "score": None, "reason": "required finite measurement missing", "source": source}
    return {"status": "scored", "score": float(value), "source": source}


def combine(parts):
    live = [p for p in parts.values() if p.get("status") != "not_applicable"]
    if any(p.get("status") != "scored" for p in live):
        state, value = "pending", None
    elif not live:
        state, value = "not_applicable", None
    else:
        state, value = "scored", mean(p["score"] for p in live)
    return {"status": state, "score": value, "submetrics": parts,
            "declared_weights": {k: 1 / len(parts) for k in parts},
            "weights": {k: 0. if p.get("status") == "not_applicable" else 1 / max(1, len(live))
                        for k, p in parts.items()}}


def action_component(cae, key, source):
    if cae.get("status") == "not_applicable" and cae.get("reason"):
        return {"status": "not_applicable", "score": None, "reason": cae["reason"], "source": source}
    if cae.get("status") != "scored":
        return measured(None, source=source)
    entities = list((cae.get("entities") or {}).values())
    groups = {}
    for kind in ("end_effector", "object"):
        chosen = [e for e in entities if e.get("kind") == kind]
        observations = [measured(e.get(key), source=source) for e in chosen]
        if not observations or any(x["status"] != "scored" for x in observations):
            return measured(None, source=source)
        groups[kind] = measured(mean(x["score"] for x in observations), source=source)
    result = combine(groups)
    result.update(source=source, metric_version=cae.get("metric_version"), config_sha256=cae.get("config_sha256"))
    return result


def semantic_component(row, source):
    details = row.get("interaction_details") or {}
    summary = details.get("summary") or {}
    scores = [measured(summary.get(k), source=source, scale=(0., 5.))
              for k in ("chunk_mean", "transition_mean", "global_score")]
    clip = measured(((row.get("clip_details") or {}).get("summary") or {}).get("semantic_adherence"),
                    source=source, scale=(-1., 1.))
    if any(x["status"] != "scored" for x in scores) or clip["status"] != "scored":
        return measured(None, source=source)
    mllm = mean(x["score"] / 5. for x in scores)
    aux = max(0., min(1., (clip["score"] - .20) / .20))
    return {"status": "scored", "score": .9 * mllm + .1 * aux, "mllm": mllm,
            "clip_raw": clip["score"], "clip_aux": aux, "source": source,
            "metric_version": "WorldOlympiad_paper_Eq9_11"}


def project(row):
    result = deepcopy(row)
    if row.get("status") == "not_applicable" and row.get("reason"):
        return result
    source = str(Path(row.get("episode", "")) / "case/score.json")
    native = str(Path(row.get("episode", "")) / "native_run/simulation/trajectory.json")
    geometry = row.get("geometry_details") or {}
    action_track = action_component(row.get("cae_details") or {}, "track", native)
    action_speed = action_component(row.get("cae_details") or {}, "speed", native)
    spatial = combine({"reconstruction": measured(geometry.get("gs_score"), source=source),
                       "meta_view": measured(geometry.get("meta_score"), source=source),
                       "camera_trajectory": measured(geometry.get("camera_motion_score"), source=source),
                       "action_trajectory": action_track})
    semantic = semantic_component(row, source)
    summary = (row.get("interaction_details") or {}).get("summary") or {}
    interaction_parts = {}
    for name, key in (("chunk", "chunk_mean"), ("transition", "transition_mean"), ("global", "global_score")):
        observation = measured(summary.get(key), source=source, scale=(0., 5.))
        if observation["status"] == "scored":
            observation["score"] /= 5.
        interaction_parts[name] = observation
    interaction_parts["clip_aux"] = measured(semantic.get("clip_aux"), source=source)
    interaction_parts["action_speed"] = action_speed
    interaction = combine(interaction_parts)
    physical = deepcopy((row.get("tracks") or {}).get("physical") or measured(row.get("physical"), source=source))
    overall = combine({"physical": physical, "geometry": spatial, "interaction": interaction})
    result.update(historical_v2_scores={k: row.get(k) for k in
                  ("physical", "geometry", "semantic_interaction", "cae_v2", "interaction", "worldolympiad2_score")},
                  metric_version=VERSION, geometry=spatial["score"], interaction=interaction["score"],
                  semantic_interaction=semantic["score"], action_trajectory=action_track["score"],
                  action_speed=action_speed["score"], worldolympiad2_score=overall["score"],
                  tracks=overall["submetrics"], status="complete" if overall["status"] == "scored" else overall["status"],
                  pending_tracks=[k for k, p in overall["submetrics"].items() if p["status"] == "pending"])
    return result


def availability(observations):
    valid = [p["score"] for p in observations if p["status"] == "scored"]
    pending = sum(p["status"] == "pending" for p in observations)
    return {"mean": mean(valid) if valid and not pending else None,
            "measured_mean": mean(valid) if valid else None, "N": len(valid), "pending": pending,
            "NA": sum(p["status"] == "not_applicable" for p in observations)}


def diagnostics(row):
    g, i = row.get("geometry_details") or {}, row.get("interaction_details") or {}
    summary, global_scores = i.get("summary") or {}, i.get("global") or {}
    source = row.get("episode", "")
    metrics = {k: measured(row.get(k), source=source) for k in
               ("physical", "geometry", "interaction", "worldolympiad2_score", "action_trajectory", "action_speed")}
    for key in ("gs_score", "meta_score", "camera_motion_score"):
        metrics[key] = measured(g.get(key), source=source)
    for key in ("chunk_mean", "transition_mean", "global_score"):
        observation = measured(summary.get(key), source=source, scale=(0., 5.))
        if observation["status"] == "scored":
            observation["score"] /= 5.
        metrics[key] = observation
    for key in ("long_range_consistency", "global_text_alignment"):
        observation = measured(global_scores.get(key), source=source, scale=(0., 5.))
        if observation["status"] == "scored":
            observation["score"] /= 5.
        metrics[key] = observation
    semantic = semantic_component(row, source)
    metrics["semantic_base"] = semantic
    metrics["clip_raw"] = measured(semantic.get("clip_raw"), source=source, scale=(-1., 1.))
    metrics["clip_aux"] = measured(semantic.get("clip_aux"), source=source)
    physical = row.get("physical_details") or {}
    for dimension in ("mechanics", "thermotics", "material"):
        value = (physical.get("dimension_scores") or {}).get(dimension)
        questions = [q for q in physical.get("results", []) if q.get("dimension") == dimension]
        if value is None and questions and all(q.get("related") is False for q in questions):
            metrics[dimension] = {"status": "not_applicable", "score": None, "reason": "all reference rules unrelated", "source": source}
        else:
            metrics[dimension] = measured(value, source=source)
    return metrics


def build(document, source=None):
    rows = [project(r) for r in document["episodes"]]
    models = {}
    for model in sorted({r["model"] for r in rows}):
        selected = [r for r in rows if r["model"] == model]
        diagnostics_rows = [diagnostics(r) for r in selected]
        metrics = {key: availability([r[key] for r in diagnostics_rows]) for key in diagnostics_rows[0]}
        models[model] = {"cases": len(selected), "metrics": metrics}
    ranked = sorted((m for m in models if models[m]["metrics"]["worldolympiad2_score"]["mean"] is not None),
                    key=lambda m: models[m]["metrics"]["worldolympiad2_score"]["mean"], reverse=True)
    for rank, model in enumerate(ranked, 1):
        models[model]["rank"] = rank
    return {"schema_version": "wo_results.v3", "metric_version": VERSION, "episodes": rows,
            "models": models, "coverage": deepcopy(document.get("coverage", {})), "source": source,
            "historical_source_audit": deepcopy(document.get("historical_source_audit", {})),
            "protocol": {**document.get("protocol", {}),
                         "geometry": "mean(reconstruction, meta, camera_trajectory, action_trajectory)",
                         "interaction": "mean(chunk/5, transition/5, global/5, clip_aux, action_speed); five independent submetrics, each weight 0.2",
                         "semantic_interaction": "chunk, transition, global and clip_aux enter independently; no half-weight semantic block",
                         "clip_aux_calibration": {"tau_min": .20, "tau_max": .40, "interaction_weight": .2},
                         "paper_semantic_diagnostic": "0.9 * mean(chunk/5, transition/5, global/5) + 0.1 * clip_aux; diagnostic only",
                         "action_entity_weights": "0.5 end-effector + 0.5 mean(target objects)",
                         "action_speed_definition": "existing CAE v2 normalized speed-profile similarity; not absolute execution pace",
                         "physical": "existing mean(generic video physics, task probes), unchanged",
                         "overall": "mean(physical, geometry, interaction)"}}


TABLES = [
    ("主结果", [("N", None), ("Physical", "physical"), ("3D Cons.", "geometry"),
                ("Interact.", "interaction"), ("All", "worldolympiad2_score"), ("Rank", None)]),
    ("Physical 子指标", [("Overall", "physical"), ("Mechanics", "mechanics"),
                         ("Thermodynamics", "thermotics"), ("Material", "material")]),
    ("Interaction 子指标", [("Chunk", "chunk_mean"), ("Trans.", "transition_mean"),
                            ("Global", "global_score"), ("Long Range", "long_range_consistency"),
                            ("Global Text", "global_text_alignment"), ("CLIP Raw", "clip_raw"),
                            ("CLIP Aux.", "clip_aux"), ("Action Speed", "action_speed"), ("Interact.", "interaction")]),
    ("Geometry 子指标", [("S_recon", "gs_score"), ("S_meta", "meta_score"),
                         ("S_traj (camera)", "camera_motion_score"),
                         ("S_action", "action_trajectory"), ("3D Cons.", "geometry")]),
]


def markdown(document):
    models = document["models"]
    ordered = sorted(models, key=lambda m: (models[m].get("rank", 999), m))
    lines = ["Robotics / bench_v1：24 个 case × 4 模型 × seed 0。粗体为最高分。", ""]
    for title, columns in TABLES:
        lines += ["**" + title + "**", "", "| Model | " + " | ".join(k for k, _ in columns) + " |",
                  "|---|" + "---:|" * len(columns)]
        for model in ordered:
            values = []
            for label, key in columns:
                if key is None:
                    values.append(str(models[model]["cases"] if label == "N" else models[model].get("rank", "—")))
                    continue
                metric = models[model]["metrics"][key]
                value = metric["mean"]
                if value is None:
                    values.append("pending" if metric["pending"] else f"NA (N=0)")
                    continue
                number = f"{value:.3f}"
                contenders = [r["metrics"][key]["mean"] for r in models.values() if r["metrics"][key]["mean"] is not None]
                if contenders and value == max(contenders):
                    number = "**" + number + "**"
                if key in ("mechanics", "thermotics", "material"):
                    number += f" (N={metric['N']})"
                values.append(number)
            lines.append("| " + DISPLAY.get(model, model) + " | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["Geometry 四项等权；Interaction 的 Chunk、Transition、Global、CLIP Aux、Action Speed 五项等权；All 为 P/G/I 均值。Physical 沿用视频物理与任务探针各半的定义。",
              "动作轨迹/速度沿用 CAE v2 进度对齐：Action Speed 是归一化速度曲线相似度。各子指标有效 N/NA/pending 以实际测量和 JSON 审计为准，行 N 不是所有子指标的共同分母；NA 不记零。GR00T 为 N1.6-DROID。"]
    return "\n".join(lines) + "\n"


def write_report(document, out, source=None):
    out = Path(out)
    result = build(document, source)
    out.write_text(json.dumps(result, indent=1))
    out.with_suffix(".md").write_text(markdown(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.source.resolve() != args.out.resolve(), "keep historical source separate"
    source = {"path": str(args.source.resolve()), "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest()}
    result = write_report(json.loads(args.source.read_text()), args.out, source)
    print(markdown(result))


if __name__ == "__main__":
    main()
