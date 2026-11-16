"""Score complete official Dream2Flow plans; observation support is diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from continuous_action.adapters import load_dream2flow_video_plan
from continuous_action.metrics import score_action_trajectories
from continuous_action.workspaces import validate_robot_workspace_contract
from continuous_action.current_metric import CURRENT_METRIC_VERSION



SCHEMA_VERSION = "worldolympiad-official-dream2flow-action-report.v1"
PROTOCOL = "video_imagination_dream2flow_official_action_trajectory_v2"
EVIDENCE_GLOB = "cae_pair/{role}/dream2flow/trajectory_optimization_plan.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_dirs(roots: list[str | Path]) -> list[Path]:
    result: dict[str, Path] = {}
    for value in roots:
        root = Path(value).resolve()
        for manifest in sorted(root.glob("*/worldolympiad_case.json")):
            case = json.loads(manifest.read_text(encoding="utf-8"))
            case_id = str(case["case_id"])
            if case_id in result and result[case_id] != manifest.parent:
                raise ValueError(f"duplicate case id across roots: {case_id}")
            result[case_id] = manifest.parent
    return [result[key] for key in sorted(result)]


def _unmeasured(case_id: str, case_dir: Path, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "continuous_action": None,
        "status": "evaluator_coverage_failure",
        "score_semantics": "unmeasured_evaluator_coverage_failure",
        "evaluator_coverage_failure": True,
        "reason": reason,
        "evaluation_protocol": PROTOCOL,
        "trajectory_source": "official_dream2flow_trajectory_optimizer_ee_trajectory",
    }


def _not_applicable(case_id: str, case_dir: Path, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "continuous_action": None,
        "status": "not_applicable",
        "score_semantics": "task_semantics_outside_single_rigid_grasp_optimizer",
        "evaluator_coverage_failure": False,
        "reason": reason,
        "evaluation_protocol": PROTOCOL,
        "trajectory_source": None,
    }


def score_case(case_dir: str | Path, *, metric_version: str = CURRENT_METRIC_VERSION) -> dict[str, Any]:
    if metric_version != CURRENT_METRIC_VERSION:
        raise ValueError("retired CAE metric")
    root = Path(case_dir).resolve()
    case = json.loads((root / "worldolympiad_case.json").read_text(encoding="utf-8"))
    case_id = str(case["case_id"])
    declared_not_applicable = (
        (case.get("tracks") or {}).get("continuous_action")
        == "not_applicable_to_official_rigid_grasp_optimizer"
        and case.get("rigid_grasp_cae_applicable") is False
    )
    spec_path = root / "official_cae_pair_spec.json"
    if not spec_path.is_file():
        spec_path = root / "cae_pair_spec.json"
    if not spec_path.is_file():
        if declared_not_applicable:
            return _not_applicable(case_id, root, str(case.get("cae_mode")))
        return _unmeasured(case_id, root, "missing_cae_pair_spec")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    pair_root = root / str(spec.get("output_dir", "cae_pair"))
    reference_plan = pair_root / "reference/dream2flow/trajectory_optimization_plan.npz"
    imagination_plan = pair_root / "imagination/dream2flow/trajectory_optimization_plan.npz"
    reference_video = pair_root / "reference/rgb.mp4"
    imagination_video = pair_root / "imagination/rgb.mp4"
    required = [reference_plan, imagination_plan, reference_video, imagination_video]
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        if declared_not_applicable:
            return _not_applicable(case_id, root, str(case.get("cae_mode")))
        return _unmeasured(case_id, root, "missing_official_optimizer_evidence:" + ",".join(missing))
    try:
        workspace = validate_robot_workspace_contract(
            spec.get("robot_workspace_contract"), source_model=str(spec["source_model"])
        )
        reference = load_dream2flow_video_plan(
            reference_plan,
            video_path=reference_video,
            video_role="reference",
        )
        prediction = load_dream2flow_video_plan(
            imagination_plan,
            video_path=imagination_video,
            video_role="imagination",
            source_model=str(spec["source_model"]),
        )
        tracked = {
            str(reference.metadata.get("tracked_object", "")).strip(),
            str(prediction.metadata.get("tracked_object", "")).strip(),
        }
        component_id = str(spec.get("tracked_component_id", "")).strip()
        if component_id:
            if not all(tracked) or len(tracked) != 1:
                raise ValueError(f"reference/imagination tracked names disagree: {sorted(tracked)}")
        else:
            expected_object = str(spec.get("object_name", "")).strip()
            if tracked != {expected_object}:
                raise ValueError(f"tracked object mismatch: expected {expected_object!r}, got {tracked!r}")
        for key in ["registration_sha256","registered_urdf_sha256"]:
            if reference.metadata.get(key) != prediction.metadata.get(key):
                raise ValueError("reference/imagination provenance mismatch: "+key)
        score = score_action_trajectories(
            reference,
            prediction,
            metric_version=metric_version,
            robot_workspace_diameter_m=float(workspace["diameter_m"]),
        )
    except Exception as exc:
        return _unmeasured(case_id, root, f"invalid_official_optimizer_evidence:{type(exc).__name__}:{exc}")
    details = score.to_dict()
    partial = bool(details.get("partial_observation", False))
    return {
        "case_id": case_id,
        "case_dir": str(root),
        "continuous_action": float(score.continuous_score),
        "path_score": float(score.path_score),
        "velocity_score": float(score.velocity_score),
        "status": "scored_partial_coverage" if partial else "scored",
        "score_semantics": (
            "official_dream2flow_optimizer_action_trajectory_partial_coverage"
            if partial else "official_dream2flow_optimizer_action_trajectory"),
        "evaluator_coverage_failure": False,
        "reason": None,
        "evaluation_protocol": PROTOCOL,
        "trajectory_source": "official_dream2flow_trajectory_optimizer_ee_trajectory",
        "metric_version": score.metric_version,
        "metric_profile": score.metric_profile,
        "observation_coverage": details.get("observation_coverage", "complete"),
        "reference_observation_coverage": details.get("reference_observation_coverage"),
        "prediction_observation_coverage": details.get("prediction_observation_coverage"),
        "visibility_affects_score": False,
        "robot_workspace_contract": workspace,
        "reference_plan": reference_plan.relative_to(root).as_posix(),
        "reference_plan_sha256": _sha256(reference_plan),
        "imagination_plan": imagination_plan.relative_to(root).as_posix(),
        "imagination_plan_sha256": _sha256(imagination_plan),
        "measurement_resolution": {role: {key: trajectory.metadata.get(key) for key in
            ["source_frame_step_max","trajectory_sample_interval_max_sec","source_video_fps"]}
            for role,trajectory in [("reference",reference),("imagination",prediction)]},
        "measurement_limits": "CAE measures supplied optimized EE trajectories; finer video-frame motion, local rigidity, orientation and contact are not directly scored",
        "score": score.to_dict(),
    }


def build_report(roots: list[str | Path], *, metric_version: str = CURRENT_METRIC_VERSION) -> dict[str, Any]:
    cases = [score_case(path, metric_version=metric_version) for path in _case_dirs(roots)]
    if not cases:
        raise ValueError("no scored cases found")
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_protocol": PROTOCOL,
        "metric_version": metric_version,
        "policy": {
            "real_score_requirement": "released Dream2Flow trajectory optimizer output for both roles",
            "missing_or_invalid_score": None,
            "partial_observation": "score the complete supplied optimizer plan; observation support is diagnostic only",
            "unmeasured_semantics": "missing or invalid evidence is null, never a model-quality zero",
            "historical_object_flow_median_sidecars_used": False,
        },
        "case_count": len(cases),
        "scored_count": sum(row["continuous_action"] is not None for row in cases),
        "not_applicable_count": sum(row["status"] == "not_applicable" for row in cases),
        "coverage_failure_count": sum(bool(row["evaluator_coverage_failure"]) for row in cases),
        "mean_continuous_action": (
            None if any(r["evaluator_coverage_failure"] for r in cases)
            or not any(r["continuous_action"] is not None for r in cases)
            else mean(float(r["continuous_action"]) for r in cases
                      if r["continuous_action"] is not None)
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--metric-version',choices=[CURRENT_METRIC_VERSION],default=CURRENT_METRIC_VERSION)
    args = parser.parse_args()
    import os
    workspace = Path(os.environ.get("WORLDOLYMPIAD_WORKSPACE", Path(__file__).resolve().parents[3]))
    selected_scope = None
    if any(Path(root).resolve().is_relative_to(workspace / "outputs") for root in args.root):
        selected_manifest = os.environ.get("WORLDEVAL_SELECTED_MODEL_CASE_SET_MANIFEST")
        if selected_manifest:
            from scripts.build_selected_model_case_set import validate_case_set
            selected_path = Path(selected_manifest).resolve()
            if len(args.root) != 1 or Path(args.root[0]).resolve() != selected_path.parent:
                raise ValueError("selected-model score roots differ from their frozen case set")
            selected_scope = validate_case_set(
                output_root=workspace / "outputs", manifest_path=selected_path,
                expected_model="lingbot_va_robotwin",
            )
    report = build_report(args.root, metric_version=args.metric_version)
    if selected_scope is not None:
        if report["case_count"] != selected_scope["case_count"]:
            raise ValueError("selected-model CAE report omits a frozen cell")
        report["selected_model_scope"] = selected_scope
    if args.output.exists():
        if json.loads(args.output.read_text()) != report:
            raise FileExistsError("write a new versioned report path; existing CAE evidence is immutable")
        print(args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
