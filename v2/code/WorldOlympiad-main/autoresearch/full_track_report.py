"""Build a numeric, coverage-aware four-track report from WorldOlympiad judges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _score_file(case_dir: Path) -> Path:
    matches = [
        path
        for path in case_dir.glob("*_judge_*.json")
        if "continuous_action" not in path.name
    ]
    direct = case_dir / "score.json"
    if direct.is_file():
        matches.append(direct)
    if len(matches) != 1:
        raise ValueError(f"expected one WorldOlympiad judge in {case_dir}, got {matches}")
    return matches[0]


def _required_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric {label}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} is outside [0, 1]: {result}")
    return result


def case_record(
    case_dir: str | Path, *, action_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    directory = Path(case_dir).resolve()
    case = json.loads((directory / "worldolympiad_case.json").read_text())
    judge_path = _score_file(directory)
    judge = json.loads(judge_path.read_text())
    physical = _required_number(judge.get("physical", {}).get("score"), label="physical")
    geometry = _required_number(
        judge.get("three_d", {}).get("final_score_normalized"), label="geometry"
    )
    interaction = _required_number(
        judge.get("interaction", {}).get("score"), label="interaction"
    )
    continuous = judge.get("continuous_action") or {}
    if action_override is not None:
        action = (0.0 if action_override.get("evaluator_coverage_failure") else
                  _required_number(action_override.get("continuous_action"), label="continuous_action"))
        action_status = str(action_override.get("status") or "missing")
        action_semantics = str(action_override.get("score_semantics") or "missing")
        coverage_failure = bool(action_override.get("evaluator_coverage_failure"))
        action_source = "official_dream2flow_action_report"
    else:
        action = 0.0
        declared = case.get("tracks", {}).get("continuous_action")
        action_status = str(declared or continuous.get("status") or "missing")
        action_semantics = "fail_closed_missing_official_optimizer_report"
        coverage_failure = True
        action_source = "missing_official_dream2flow_action_report"
    tracks = {
        "physical": physical,
        "geometry": geometry,
        "interaction": interaction,
        "continuous_action": action,
    }
    return {
        "case_id": str(case["case_id"]),
        "model_id": case.get("model_id"),
        "benchmark_status": case.get("benchmark_status"),
        "track_scores": tracks,
        "continuous_action_status": action_status,
        "continuous_action_score_semantics": action_semantics,
        "continuous_action_source": action_source,
        "evaluator_coverage_failure": coverage_failure,
        "coverage_enforced_four_track_score": mean(tracks.values()),
        "applicable_track_score": judge.get(
            "applicable_track_score_before_coverage_enforcement",
            judge.get("worldolympiad2_score"),
        ),
        "judge": str(judge_path),
    }


def build_report(
    roots: list[str | Path], *, action_report: dict[str, Any] | None = None
) -> dict[str, Any]:
    action_by_case = {}
    for row in (action_report or {}).get("cases", []):
        case_id = str(row["case_id"])
        if case_id in action_by_case:
            raise ValueError(f"duplicate action case id: {case_id}")
        action_by_case[case_id] = row
    records = []
    seen_cases = {}
    for root_value in roots:
        root = Path(root_value).resolve()
        for case_manifest in sorted(root.glob("*/worldolympiad_case.json")):
            case_id = str(json.loads(case_manifest.read_text())["case_id"])
            resolved = case_manifest.resolve()
            if case_id in seen_cases:
                if seen_cases[case_id] == resolved:
                    continue
                raise ValueError(f"duplicate case id across roots: {case_id}")
            seen_cases[case_id] = resolved
            records.append(
                case_record(case_manifest.parent, action_override=action_by_case.get(case_id))
            )
    if not records:
        raise ValueError("no scored cases found")
    return {
        "schema_version": "worldolympiad_full_track_report.v1",
        "policy": {
            "all_tracks_numeric": True,
            "missing_or_unsupported_continuous_action_score": 0.0,
            "zero_semantics": "evaluation-pipeline coverage failure, not inferred model motion quality",
            "aggregate": "unweighted mean of physical, geometry, interaction, continuous_action",
            "continuous_action_precedence": "official Dream2Flow optimizer action report only; no historical judge fallback",
        },
        "case_count": len(records),
        "coverage_failure_count": sum(
            bool(record["evaluator_coverage_failure"]) for record in records
        ),
        "mean_coverage_enforced_four_track_score": mean(
            float(record["coverage_enforced_four_track_score"]) for record in records
        ),
        "cases": records,
    }


def build_report_v2(
    roots: list[str | Path], *, applicability_plan: dict[str, Any],
    action_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Four-track report with explicit semantic N/A and pending evidence.

    The historical all-numeric v1 report above remains unchanged. Applicability
    is a frozen task-side declaration, never inferred from the model's score.
    """
    if applicability_plan.get("schema_version") != "worldolympiad_track_applicability.v1":
        raise ValueError("unsupported applicability plan")
    entries = applicability_plan.get("cases") or []
    plans = {str(row["case_id"]): row for row in entries}
    if len(plans) != len(entries):
        raise ValueError("duplicate applicability case")
    actions = (action_report or {}).get("cases") or []
    action_by_case = {str(row["case_id"]): row for row in actions}
    if len(action_by_case) != len(actions):
        raise ValueError("duplicate action case id")
    records = []
    seen = set()
    for root_value in roots:
        root = Path(root_value).resolve()
        direct = root / "worldolympiad_case.json"
        manifests = [direct] if direct.is_file() else sorted(root.glob("*/worldolympiad_case.json"))
        for manifest in manifests:
            case = json.loads(manifest.read_text())
            case_id = str(case["case_id"])
            if case_id in seen:
                raise ValueError("duplicate case id across roots")
            seen.add(case_id)
            if case_id not in plans:
                raise ValueError("case has no frozen applicability declaration")
            declaration = plans[case_id]
            if not declaration.get("task_sha256"):
                raise ValueError("applicability needs a task digest")
            declared = declaration.get("tracks") or {}
            required = {"physical", "geometry", "interaction", "continuous_action"}
            if set(declared) != required:
                raise ValueError("applicability must declare all four tracks")
            judge_path = _score_file(manifest.parent)
            judge = json.loads(judge_path.read_text())
            measured = {
                "physical": judge.get("physical", {}).get("score"),
                "geometry": judge.get("three_d", {}).get("final_score_normalized"),
                "interaction": judge.get("interaction", {}).get("score"),
            }
            action = action_by_case.get(case_id)
            if action and action.get("evaluator_coverage_failure") is not True:
                measured["continuous_action"] = action.get("continuous_action",
                                                            action.get("continuous_score"))
            tracks = {}
            for name in ("physical", "geometry", "interaction", "continuous_action"):
                policy = declared[name]
                applicable = policy.get("applicable") if isinstance(policy, dict) else policy
                if not isinstance(applicable, bool):
                    raise ValueError("track applicability must be boolean")
                if not applicable:
                    reason = policy.get("reason") if isinstance(policy, dict) else None
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError("N/A requires a task-side reason")
                    tracks[name] = {"status": "not_applicable", "score": None,
                                    "reason": reason}
                    continue
                value = measured.get(name)
                if value is None or (name == "continuous_action" and not action):
                    tracks[name] = {"status": "pending", "score": None,
                                    "reason": "required evaluator evidence is missing"}
                    continue
                tracks[name] = {"status": "scored",
                                "score": _required_number(value, label=name),
                                "source": str(judge_path if name != "continuous_action"
                                              else action.get("source") or "action_report")}
            pending = any(row["status"] == "pending" for row in tracks.values())
            scores = [row["score"] for row in tracks.values() if row["status"] == "scored"]
            records.append({"case_id": case_id, "model_id": case.get("model_id"),
                            "status": "pending" if pending else "complete",
                            "track_results": tracks,
                            "worldolympiad2_score": None if pending or not scores else mean(scores),
                            "judge": str(judge_path)})
    if not records or set(plans) != seen:
        raise ValueError("applicability and scored cases differ")
    if set(action_by_case) - seen:
        raise ValueError("action report contains an unknown case")
    if any(not any(track["status"] == "scored" for track in row["track_results"].values())
           and row["status"] == "complete" for row in records):
        raise ValueError("completed case has no applicable tracks")
    identity = hashlib.sha256(json.dumps(applicability_plan, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    completed = [row["worldolympiad2_score"] for row in records
                 if row["status"] == "complete"]
    return {"schema_version": "worldolympiad_full_track_report.v2",
            "applicability_plan_sha256": identity,
            "case_count": len(records), "complete_count": len(completed),
            "mean_score": mean(completed) if len(completed) == len(records) else None,
            "cases": records}


def build_report_v3(roots: list[str | Path], *, applicability_plan: dict[str, Any],
                    action_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reuse measured evidence, nesting CAE once inside Interaction."""
    from .score_hierarchy import SCORE_POLICY, compose_three_tracks
    source = build_report_v2(roots, applicability_plan=applicability_plan,
                             action_report=action_report)
    actions = {str(row["case_id"]): row for row in (action_report or {}).get("cases", [])}
    cases = []
    for row in source["cases"]:
        measurements = row["track_results"]
        action = actions.get(row["case_id"], {})
        measurements["continuous_action"]["components"] = {
            "trajectory_shape": action.get("path_score"),
            "speed_curve": action.get("velocity_score"),
        }
        measurements["continuous_action"]["metric_version"] = action.get("metric_version")
        composed = compose_three_tracks(measurements)
        cases.append({**row, **composed})
    complete = [row["worldolympiad2_score"] for row in cases if row["status"] == "complete"]
    return {"schema_version": "worldolympiad_three_track_report.v1",
            "score_policy": SCORE_POLICY,
            "weights": {"physical": 1/3, "geometry": 1/3, "interaction": 1/3},
            "interaction_weights": {"semantic_interaction": 0.5, "continuous_action": 0.5},
            "applicability_plan_sha256": source["applicability_plan_sha256"],
            "case_count": len(cases), "complete_count": len(complete),
            "mean_score": mean(complete) if len(complete) == len(cases) else None,
            "cases": cases}


def write_report(report: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "case_id",
        "model_id",
        "benchmark_status",
        "physical",
        "geometry",
        "interaction",
        "continuous_action",
        "continuous_action_status",
        "continuous_action_score_semantics",
        "continuous_action_source",
        "evaluator_coverage_failure",
        "coverage_enforced_four_track_score",
        "applicable_track_score",
        "judge",
    ]
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in report["cases"]:
            writer.writerow(
                {
                    **{key: record.get(key) for key in fields},
                    **record["track_scores"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--action-report", type=Path)
    parser.add_argument("--applicability-plan", type=Path)
    args = parser.parse_args()
    action_report = (
        json.loads(args.action_report.read_text(encoding="utf-8"))
        if args.action_report is not None
        else None
    )
    if args.applicability_plan is None:
        parser.error("current three-track reporting requires --applicability-plan")
    plan = json.loads(args.applicability_plan.read_text())
    report = build_report_v3(args.root, applicability_plan=plan, action_report=action_report)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "model_id", "status", "physical", "geometry",
                         "interaction", "interaction_semantic", "interaction_cae", "worldolympiad2_score"])
        for row in report["cases"]:
            submetrics = row["track_results"]["interaction"]["submetrics"]
            writer.writerow([row["case_id"], row["model_id"], row["status"],
                *(row["track_results"][name]["score"] for name in ("physical", "geometry", "interaction")),
                submetrics["semantic_interaction"]["score"], submetrics["continuous_action"]["score"],
                row["worldolympiad2_score"]])
    print(args.output_json)


if __name__ == "__main__":
    main()
