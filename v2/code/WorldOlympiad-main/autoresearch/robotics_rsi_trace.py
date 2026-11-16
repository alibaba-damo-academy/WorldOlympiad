"""Append-only task traces used by the shared research and repair specialist."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROBOTICS_RSI_STAGE_MODULES = {
    "reference": "robotics_reference",
    "preflight": "robotics_preflight",
    "wam": "robotics_wam",
    "audit": "robotics_audit",
    "pair": "robotics_pair",
    "case": "robotics_case",
    "evaluation": "robotics_evaluation",
}

RSI_MODULES = frozenset({
    "agent_loop", "tool_use", "observation_management",
    "context_management", "completion_detection",
}) | frozenset(ROBOTICS_RSI_STAGE_MODULES.values())
TRACE_MODULES = RSI_MODULES | {"unclassified"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def trace_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def module_for_stage(stage: str) -> str:
    return ROBOTICS_RSI_STAGE_MODULES.get(stage, "unclassified")


def append_trace(root: str | Path, *, module: str, outcome: str,
                 task_id: str | None = None, run_id: str | None = None,
                 stage: str | None = None, evidence: dict[str, Any] | None = None,
                 error: str | None = None) -> Path:
    """Write one immutable JSONL trace and return its content-addressed path."""

    if module not in TRACE_MODULES:
        raise ValueError(f"unknown RSI module: {module!r}")
    if outcome not in {"success", "failure", "warning"}:
        raise ValueError(f"unknown trace outcome: {outcome!r}")
    payload: dict[str, Any] = {
        "schema_version": "robotics_rsi_trace.v1",
        "module": module,
        "outcome": outcome,
        "task_id": task_id,
        "run_id": run_id,
        "stage": stage,
        "evidence": evidence or {},
        "error": error,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = trace_id({k: v for k, v in payload.items() if k != "recorded_at"})
    payload["trace_id"] = digest
    directory = Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("trace_id") != digest:
            raise ValueError(f"trace collision: {path}")
        return path
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")
    return path


def load_traces(root: str | Path, *, module: str | None = None) -> list[dict[str, Any]]:
    directory = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema_version") != "robotics_rsi_trace.v1":
            continue
        if module is None or row.get("module") == module:
            rows.append(row)
    return rows


def contrastive_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize outcomes by harness module and exact task, without taxonomy."""

    summary: dict[str, dict[str, int]] = {}
    task_outcomes: dict[str, set[str]] = {}
    task_modules: dict[str, set[str]] = {}
    module_task_outcomes: dict[tuple[str, str], dict[str, list[str]]] = {}
    recurring_failures: dict[tuple[str, str], set[str]] = {}
    failure_trace_ids: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        module = str(row.get("module"))
        outcome = str(row.get("outcome"))
        bucket = summary.setdefault(module, {"success": 0, "failure": 0, "warning": 0})
        if outcome in bucket:
            bucket[outcome] += 1
        task_id = row.get("task_id")
        if task_id:
            task_id = str(task_id)
            task_outcomes.setdefault(task_id, set()).add(outcome)
            task_modules.setdefault(task_id, set()).add(module)
            module_rows = module_task_outcomes.setdefault((task_id, module), {})
            module_rows.setdefault(outcome, []).append(str(row.get("trace_id") or ""))
            if outcome == "failure":
                evidence = row.get("evidence") or {}
                key = (module, str(evidence.get("finding_key") or row.get("error") or row.get("stage") or "unspecified"))
                recurring_failures.setdefault(key, set()).add(task_id)
                failure_trace_ids.setdefault(key, []).append(str(row.get("trace_id") or ""))
    task_groups = {
        task_id: {
            "group": (
                "contrastive" if {"success", "failure"} <= outcomes
                else "positive" if outcomes == {"success"}
                else "negative" if outcomes == {"failure"}
                else "mixed_or_incomplete"
            ),
            "outcomes": sorted(outcomes),
            "modules": sorted(task_modules[task_id]),
        }
        for task_id, outcomes in sorted(task_outcomes.items())
    }
    trajectory_pairs = [
        {
            "task_id": task_id,
            "module": module,
            "success_trace_ids": sorted(set(outcomes.get("success", []))),
            "failure_trace_ids": sorted(set(outcomes.get("failure", []))),
        }
        for (task_id, module), outcomes in sorted(module_task_outcomes.items())
        if outcomes.get("success") and outcomes.get("failure")
    ]
    cross_task_findings = [
        {
            "module": module,
            "finding_key": finding_key,
            "supporting_task_ids": sorted(task_ids),
            "supporting_trace_ids": sorted(set(failure_trace_ids[(module, finding_key)])),
            "distinct_task_count": len(task_ids),
        }
        for (module, finding_key), task_ids in sorted(recurring_failures.items())
        if len(task_ids) > 1
    ]
    return {
        "schema_version": "robotics_rsi_contrastive_summary.v2",
        "modules": summary,
        "tasks": task_groups,
        "trajectory_pairs": trajectory_pairs,
        "cross_task_findings": cross_task_findings,
        "contrastive_task_count": sum(
            row["group"] == "contrastive" for row in task_groups.values()
        ),
    }
