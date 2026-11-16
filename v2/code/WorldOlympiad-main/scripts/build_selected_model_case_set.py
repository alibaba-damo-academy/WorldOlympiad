#!/usr/bin/env python3
"""Assemble one frozen model's scored cases without changing source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from autoresearch.io import sha256_file, write_json


def build_case_set(
    *, output_root: Path, plan_path: Path, model: str, case_set_id: str
) -> Path:
    output_root = output_root.resolve()
    plan_path = plan_path.resolve()
    if model != "lingbot_va_robotwin":
        raise ValueError("this case-set adapter requires native LingBot evaluation bundles")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", case_set_id):
        raise ValueError("unsafe case-set id")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = [cell for cell in plan.get("cells", []) if cell.get("model") == model]
    if not selected or len({cell["task_id"] for cell in selected}) != len(selected):
        raise ValueError("frozen plan has no unique selected model cells")
    candidates = []
    for bundle_path in (output_root / "evaluation_cases").glob(
        "lingbot_robotwin_*/evaluation_bundle.json"
    ):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        case_path = Path(str(bundle.get("case_manifest", "")))
        model_path = Path(str(bundle.get("source_run_manifest", "")))
        pair_path = Path(str(bundle.get("evaluation_pair", "")))
        if not all(path.is_file() for path in (case_path, model_path, pair_path)):
            continue
        if (sha256_file(case_path) != bundle.get("case_manifest_sha256")
                or sha256_file(model_path) != bundle.get("source_run_manifest_sha256")
                or sha256_file(pair_path) != bundle.get("evaluation_pair_sha256")):
            continue
        candidates.append((bundle_path, bundle,
                           json.loads(model_path.read_text(encoding="utf-8")),
                           json.loads(pair_path.read_text(encoding="utf-8"))))
    resolved = []
    for cell in selected:
        reference = cell["reference"]
        matches = [row for row in candidates
                   if row[2].get("task") == reference["native_task"]
                   and row[2].get("seed") == reference["seed"]
                   and row[3].get("instruction") == cell["instruction"]]
        if len(matches) != 1:
            raise ValueError(
                f"selected {cell['task_id']} needs one digest-valid evaluation bundle; got {len(matches)}"
            )
        resolved.append((cell, matches[0]))
    final = output_root / "evaluation_cases" / case_set_id
    if final.exists():
        raise FileExistsError(final)
    final.mkdir(parents=True)
    records = []
    try:
        for cell, (bundle_path, bundle, run, pair) in resolved:
            source = Path(bundle["case_manifest"]).resolve().parent
            if any(item.is_symlink() for item in source.rglob("*")):
                raise ValueError("selected case inputs contain symlinks")
            destination = final / source.name
            shutil.copytree(
                source, destination,
                ignore=shutil.ignore_patterns(
                    "cae_pair", "*_judge_*.json", "*_judge_*.md"
                ),
            )
            copied = destination / "worldolympiad_case.json"
            if sha256_file(copied) != bundle["case_manifest_sha256"]:
                raise ValueError("selected case changed during copy")
            records.append({
                "task_class": cell["task_id"],
                "native_task": cell["reference"]["native_task"],
                "seed": cell["reference"]["seed"],
                "run_id": bundle["run_id"],
                "source_evaluation_bundle": str(bundle_path),
                "source_evaluation_bundle_sha256": sha256_file(bundle_path),
                "source_model_run_manifest_sha256": bundle["source_run_manifest_sha256"],
                "case_manifest": f"{source.name}/worldolympiad_case.json",
                "case_manifest_sha256": bundle["case_manifest_sha256"],
            })
        write_json(final / "selection_manifest.json", {
            "schema_version": "worldolympiad_selected_model_case_set.v1",
            "model": model,
            "scope": "selected_model_only_full_suite_pending",
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "cases": records,
        })
    except Exception:
        shutil.rmtree(final, ignore_errors=True)
        raise
    return final / "selection_manifest.json"


def validate_case_set(
    *, output_root: Path, manifest_path: Path, expected_model: str,
    require_cae: bool = False,
) -> dict:
    """Recheck every selected cell before model-scoped scoring."""
    output_root = output_root.resolve()
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    root.relative_to(output_root / "evaluation_cases")
    if manifest_path.name != "selection_manifest.json":
        raise ValueError("selected case set lacks its fixed manifest name")
    selected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (selected.get("schema_version") != "worldolympiad_selected_model_case_set.v1"
            or selected.get("model") != expected_model
            or selected.get("scope") != "selected_model_only_full_suite_pending"):
        raise ValueError("case set does not declare a model-scoped score")
    plan_path = Path(str(selected["plan"])).resolve()
    plan_path.relative_to(output_root)
    if sha256_file(plan_path) != selected.get("plan_sha256"):
        raise ValueError("selected case set plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_rows = {cell["task_id"]: cell for cell in plan.get("cells", [])
                 if cell.get("model") == expected_model}
    rows = selected.get("cases") or []
    if len(rows) != len(plan_rows) or {row["task_class"] for row in rows} != set(plan_rows):
        raise ValueError("selected case set omits or duplicates a frozen model cell")
    for row in rows:
        cell = plan_rows[row["task_class"]]
        reference = cell["reference"]
        if (row.get("native_task") != reference["native_task"]
                or row.get("seed") != reference["seed"]):
            raise ValueError("selected native task differs from frozen plan")
        copied = (root / str(row["case_manifest"])).resolve()
        copied.relative_to(root)
        if copied.name != "worldolympiad_case.json":
            raise ValueError("selected case path changed")
        copied_sha = sha256_file(copied)
        source_bundle = Path(str(row["source_evaluation_bundle"])).resolve()
        source_bundle.relative_to(output_root / "evaluation_cases")
        if sha256_file(source_bundle) != row["source_evaluation_bundle_sha256"]:
            raise ValueError("selected evaluation source changed")
        bundle = json.loads(source_bundle.read_text(encoding="utf-8"))
        model_manifest = Path(str(bundle["source_run_manifest"])).resolve()
        model_manifest.relative_to(output_root / "robotics_wam" / expected_model)
        pair_path = Path(str(bundle["evaluation_pair"])).resolve()
        pair_path.relative_to(output_root / "robotics_evaluation_pairs")
        if (bundle.get("case_manifest_sha256") != row["case_manifest_sha256"]
                or sha256_file(model_manifest) != row["source_model_run_manifest_sha256"]):
            raise ValueError("selected model or case provenance changed")
        model = json.loads(model_manifest.read_text(encoding="utf-8"))
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        if (model.get("task") != reference["native_task"]
                or model.get("seed") != reference["seed"]
                or pair.get("instruction") != cell["instruction"]):
            raise ValueError("selected model output belongs to another frozen task")
        receipt_path = root / f"cae_preparation_{row['run_id']}.v2.json"
        receipt = (
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.is_file() else None
        )
        if copied_sha != row["case_manifest_sha256"]:
            if (not isinstance(receipt, dict)
                    or receipt.get("status") != "prepared"
                    or receipt.get("selected_input_case_manifest_sha256")
                    != row["case_manifest_sha256"]
                    or receipt.get("derived_case_manifest_sha256") != copied_sha):
                raise ValueError("selected case input changed without CAE derivation")
            original = json.loads(Path(bundle["case_manifest"]).read_text(encoding="utf-8"))
            derived = json.loads(copied.read_text(encoding="utf-8"))
            if (derived.pop("continuous_action_manifest", None)
                    != "continuous_action_manifest.json"
                    or derived.pop("continuous_action_manifests", None)
                    != {"full_trajectory_shape_speed": "continuous_action_manifest.json"}):
                raise ValueError("selected CAE case declares an unexpected manifest")
            tracks = dict(derived.get("tracks") or {})
            if (tracks.get("continuous_action") != "required"
                    or (original.get("tracks") or {}).get("continuous_action")
                    != "pending_symmetric_dream2flow_manifest"):
                raise ValueError("selected CAE case changed track applicability")
            tracks["continuous_action"] = "pending_symmetric_dream2flow_manifest"
            derived["tracks"] = tracks
            if derived != original:
                raise ValueError("selected CAE case changed frozen task fields")
        if require_cae:
            if bundle.get("rigid_grasp_cae_applicable") is False:
                case_record = json.loads(copied.read_text(encoding="utf-8"))
                if ((case_record.get("tracks") or {}).get("continuous_action")
                        != "not_applicable_to_official_rigid_grasp_optimizer"):
                    raise ValueError("semantic CAE N/A is not frozen in the case")
                continue
            if bundle.get("rigid_grasp_cae_applicable") is not True:
                raise ValueError("selected CAE applicability is unresolved")
            if receipt is None:
                raise ValueError(f"selected {row['task_class']} has no official CAE receipt")
            if (receipt.get("schema_version") != "lingbot_robotwin_cae_preparation.v2"
                    or receipt.get("status") != "prepared"
                    or receipt.get("selected_input_case_manifest_sha256")
                    != row["case_manifest_sha256"]
                    or receipt.get("derived_case_manifest_sha256") != copied_sha
                    or receipt.get("source_run_manifest_sha256")
                    != row["source_model_run_manifest_sha256"]
                    or Path(str(receipt.get("case_dir", ""))).resolve() != copied.parent
                    or (receipt.get("selected_model_scope") or {}).get(
                        "case_set_manifest_sha256") != sha256_file(manifest_path)):
                raise ValueError("selected official CAE receipt belongs to another run")
            plans = receipt.get("optimizer_plans") or {}
            if set(plans) != {"reference", "imagination"}:
                raise ValueError("selected official CAE lacks both optimizer roles")
            for role, record in plans.items():
                path = Path(str(record.get("path", ""))).resolve()
                path.relative_to(copied.parent)
                if sha256_file(path) != record.get("sha256"):
                    raise ValueError(f"selected {role} optimizer evidence changed")
            manifest_record = receipt.get("continuous_action_manifest") or {}
            cae_manifest = Path(str(manifest_record.get("path", ""))).resolve()
            cae_manifest.relative_to(copied.parent)
            if sha256_file(cae_manifest) != manifest_record.get("sha256"):
                raise ValueError("selected continuous-action manifest changed")
    return {
        "schema_version": "worldolympiad_selected_model_score_admission.v1",
        "passed": True,
        "model": expected_model,
        "case_count": len(rows),
        "case_set_manifest_sha256": sha256_file(manifest_path),
        "full_suite_scoring_allowed": False,
    }


def validate_selected_case_root(
    *, output_root: Path, manifest_path: Path, case_root: Path,
    expected_model: str, require_cae: bool = False,
) -> dict:
    """Admit one CAE worker only for a copied case in the frozen selection."""
    admission = validate_case_set(
        output_root=output_root, manifest_path=manifest_path,
        expected_model=expected_model,
        require_cae=require_cae,
    )
    manifest_path = manifest_path.resolve()
    case_root = case_root.resolve()
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        row for row in selection["cases"]
        if (manifest_path.parent / row["case_manifest"]).resolve().parent == case_root
    ]
    if len(matches) != 1:
        raise ValueError("CAE case is absent or ambiguous in selected model set")
    return {**admission, "task_class": matches[0]["task_class"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--case-set-id")
    parser.add_argument("--validate-existing", type=Path)
    parser.add_argument("--require-cae", action="store_true")
    args = parser.parse_args()
    if args.validate_existing is not None:
        if args.model is None:
            parser.error("validation requires --model")
        print(json.dumps(validate_case_set(
            output_root=args.output_root,
            manifest_path=args.validate_existing,
            expected_model=args.model,
            require_cae=args.require_cae,
        )))
        return
    if args.plan is None or args.model is None or args.case_set_id is None:
        parser.error("building requires --plan, --model, and --case-set-id")
    print(build_case_set(
        output_root=args.output_root, plan_path=args.plan,
        model=args.model, case_set_id=args.case_set_id,
    ))


if __name__ == "__main__":
    main()
