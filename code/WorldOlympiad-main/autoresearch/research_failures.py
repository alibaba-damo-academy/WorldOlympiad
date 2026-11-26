"""Evidence intake for the shared research and repair loop."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from .io import sha256_file as sha, write_json as atomic_json
from .robotics_rsi_trace import append_trace, module_for_stage


def capture_failure(output_root: Path, failure_path: Path) -> Path:
    """Retain each failure once for diagnosis by the shared research specialist."""
    output_root, failure_path = Path(output_root).resolve(), Path(failure_path).resolve()
    failure_path.relative_to(output_root)
    failure = json.loads(failure_path.read_text())
    digest = sha(failure_path)
    stage = failure.get("failed_stage", "unknown")
    trace = append_trace(output_root / "robotics_rsi/traces",
        module=module_for_stage(stage), outcome="failure",
        task_id=failure.get("case_id"), run_id=failure.get("case_id"), stage=stage,
        evidence={"failure_path": str(failure_path), "failure_sha256": digest,
                  "classification": failure.get("classification")},
        error=failure.get("reason"))
    root = output_root / "olympiad_forge/research_cases" / digest
    evidence = root / "failure.json"
    item = root / "case.json"
    if not evidence.exists():
        atomic_json(evidence, failure)
    if not item.exists():
        atomic_json(item, {"schema_version": "olympiad_forge_research_case.v1",
            "status": "pending", "trace_id": trace.stem,
            "failure_evidence": str(evidence), "failure_evidence_sha256": sha(evidence),
            "case_id": failure.get("case_id"), "batch_id": failure.get("batch_id"),
            "classification": failure.get("classification"), "stage": stage,
            "created_at": datetime.now(timezone.utc).isoformat()})
    return item


def capture_compiled_gap(workspace: Path, batch_id: str, case_id: str) -> Path | None:
    """Bind a real compiled capability gap to an immutable RSI work item."""
    for value in (batch_id, case_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
            raise ValueError("unsafe capability gap identity")
    workspace = Path(workspace).resolve()
    output_root = workspace / "outputs"
    task = output_root / "robotics_task_batches" / batch_id / "tasks" / f"{case_id}.json"
    source_plan_path = output_root / "robotics_compilation_plans" / batch_id / f"{case_id}.json"
    source_plan = json.loads(source_plan_path.read_text())
    if source_plan.get("case_id") != case_id:
        raise ValueError("compiled capability plan belongs to another case")
    from .robotics_contracts import TaskInstructionSpec
    typed_task = TaskInstructionSpec.model_validate_json(task.read_text())
    if typed_task.case_id != case_id:
        raise ValueError("selected task belongs to another case")
    from .robotics_task_compiler import compile_task
    plan = compile_task(typed_task)
    def executable_contract(value: dict) -> tuple[list[dict], list[dict]]:
        return (
            [{k: v for k, v in row.items() if k != "capability_status"}
             for row in value.get("controller_program", [])],
            [{k: v for k, v in row.items() if k != "evaluator_status"}
             for row in value.get("success_program", [])],
        )
    if executable_contract(source_plan) != executable_contract(plan):
        raise ValueError("published plan and current compiler disagree on task execution")
    reports = plan.get("native_capabilities") or {}
    current = plan["execution_capability_preflight"]
    if (not any(row.get("status") == "blocked" for row in reports.values())
            and current["passed"]):
        return None
    plan_digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan_path = (output_root / "olympiad_forge/compiled_plans" / batch_id / case_id /
                 f"{plan_digest}.json")
    if plan_path.is_file():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("derived RSI plan changed")
    else:
        atomic_json(plan_path, plan)
    operators = sorted({op for row in reports.values()
                        for op in row.get("unsupported_operators", [])} |
                       {op for row in current["wams"].values()
                        for op in row.get("unsupported_operators", [])})
    predicates = sorted({kind for row in reports.values()
                         for kind in row.get("unsupported_terminal_predicates", [])})
    evidence = {
        "schema_version": "robotics_compiled_capability_gap.v1",
        "batch_id": batch_id, "case_id": case_id,
        "classification": "native_capability", "failed_stage": "capability_preflight",
        "reason": "compiled task needs an unverified execution capability",
        "task_sha256": sha(task),
        "compilation_plan": str(plan_path),
        "compilation_plan_sha256": sha(plan_path),
        "source_compilation_plan_sha256": sha(source_plan_path),
        "unsupported_operators": operators,
        "unsupported_terminal_predicates": predicates,
    }
    report = (output_root / "robotics_capability_gaps" / batch_id / case_id /
              f"{plan_digest}.json")
    if report.is_file():
        if json.loads(report.read_text()) != evidence:
            raise ValueError("published capability gap differs from current plan")
    else:
        atomic_json(report, evidence)
    return capture_failure(output_root, report)
