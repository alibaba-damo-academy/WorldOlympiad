"""Freeze a model-authored, semantically de-duplicated robotics eval suite.

An upstream structured task author proposes declarative data. This module
validates and freezes it; it
does not select from a fixed task catalog, bind a task-specific skill, or turn
layout changes into fake task diversity. Scene and simulator compilers later
decide whether each proposal is executable and certify it before model rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .robotics_contracts import (
    RoboticsTaskBatchCase,
    RoboticsTaskBatchManifest,
    TaskInstructionSpec,
    WAMName,
)
from .robotics_views import sha256_file


GENERATOR_ID = "structured_model_authored_suite_v2"
COMPILER_ID = "agentic_worldir_compiler_v1"


def _safe_id(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError(f"{label} must be a safe 1-64 character identifier")
    return value


def _identity_seed(batch_id: str, wam: str) -> int:
    payload = f"{GENERATOR_ID}\0{batch_id}\0{wam}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _case_seed(identity_seed: int, index: int) -> int:
    payload = f"{identity_seed}\0{index}".encode()
    # JSON number consumers use IEEE-754, so keep exact integer round-tripping.
    return int.from_bytes(hashlib.sha256(payload).digest()[:6], "big")


def _normalized_text(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _semantic_payload(task: TaskInstructionSpec) -> dict[str, Any]:
    """Return task meaning while excluding case id, seed, and sampled layout."""

    objects = sorted(
        (
            item.object_id,
            item.category,
            item.role,
            _normalized_text(item.visual_description),
            item.geometry_constraints,
        )
        for item in task.objects
    )
    articulation_components = sorted(
        (
            item.component_id,
            item.host_object_id,
            item.semantic_kind,
            item.joint_type,
            _normalized_text(item.visual_description),
            item.geometry_constraints,
        )
        for item in task.articulation_components
    )
    actions = [
        {
            "operation": step.operation,
            "subjects": sorted(step.subjects),
            "target": step.target,
            "constraints": step.constraints,
        }
        for step in task.action_program
    ]
    predicates = [
        (item.kind, item.subject, item.object, item.threshold)
        for item in task.success_predicates
    ]
    predicates.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {
        "instruction": _normalized_text(task.instruction),
        "scene_brief": _normalized_text(task.scene_brief),
        "objects": objects,
        "articulation_components": articulation_components,
        "action_program": actions,
        "success_predicates": predicates,
    }


def semantic_signature(task: TaskInstructionSpec) -> str:
    payload = json.dumps(
        _semantic_payload(task),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task_from_proposal(
    proposal: dict[str, Any],
    *,
    case_id: str,
    case_seed: int,
    requested_wam: WAMName,
) -> TaskInstructionSpec:
    forbidden = {
        "schema_version",
        "case_id",
        "case_seed",
        "requested_wams",
        "output_size_wh",
        "duration_sec",
        "fps",
    } & set(proposal)
    if forbidden:
        raise ValueError(
            "suite proposals cannot override harness/model-owned fields: "
            f"{sorted(forbidden)}"
        )
    if not proposal.get("action_program"):
        raise ValueError(f"declarative case {case_id} needs an action program")
    parameters = proposal.get("case_parameters")
    if parameters is None:
        parameters = {"layout_seed": case_seed}
    payload = {
        **proposal,
        "schema_version": "robotics_task_instruction.v3",
        "case_id": case_id,
        "case_seed": case_seed,
        "case_parameters": parameters,
        "robot": "franka_robotiq_2f85",
        "simulator": "pybullet",
        # Compatibility provenance for the current reference builder. Generic
        # compilation ends the expert on success and records measured timing;
        # these values never constrain the WAM rollout.
        "duration_sec": 12.0,
        "fps": 15.0,
        "output_size_wh": None,
        "requested_wams": [requested_wam],
    }
    return TaskInstructionSpec.model_validate(payload)


def _load_proposals(
    value: str | Path | Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") not in {
            "robotics_task_suite_proposal.v1",
            "robotics_task_suite_proposal.v2",
        }:
            raise ValueError("unsupported robotics task suite proposal")
        value = payload.get("cases", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("robotics suite proposals must be a list")
    proposals = [dict(item) for item in value]
    if not proposals:
        raise ValueError("robotics suite proposal contains no cases")
    return proposals


def validate_proposals(
    *,
    batch_id: str,
    case_count: int,
    requested_wam: WAMName,
    proposals: str | Path | Sequence[dict[str, Any]],
) -> list[TaskInstructionSpec]:
    """Validate and deterministically identify a proposal without publishing it."""

    batch_id = _safe_id(batch_id, label="batch_id")
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    proposed = _load_proposals(proposals)
    if len(proposed) != case_count:
        raise ValueError(
            f"case_count={case_count} but the model proposed {len(proposed)} cases"
        )
    identity_seed = _identity_seed(batch_id, requested_wam)
    tasks: list[TaskInstructionSpec] = []
    signatures: set[str] = set()
    for index, proposal in enumerate(proposed):
        case_id = f"{batch_id}_case_{index:04d}"
        task = _task_from_proposal(
            proposal,
            case_id=case_id,
            case_seed=_case_seed(identity_seed, index),
            requested_wam=requested_wam,
        )
        signature = semantic_signature(task)
        if signature in signatures:
            raise ValueError(
                "suite contains an exact semantic duplicate; change the task, "
                "not only its seed or layout"
            )
        signatures.add(signature)
        tasks.append(task)
    return tasks


def generate_batch(
    *,
    output_root: str | Path,
    batch_id: str,
    case_count: int,
    requested_wam: WAMName,
    proposals: str | Path | Sequence[dict[str, Any]],
) -> Path:
    batch_id = _safe_id(batch_id, label="batch_id")
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    proposed = _load_proposals(proposals)
    proposal_json = json.dumps(
        proposed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    proposal_sha256 = hashlib.sha256(proposal_json.encode("utf-8")).hexdigest()

    root = Path(output_root).resolve()
    batch_root = root / "robotics_task_batches" / batch_id
    manifest_path = batch_root / "batch_manifest.json"
    if batch_root.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"incomplete generated batch requires inspection: {batch_root}")
        existing = RoboticsTaskBatchManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            existing.requested_wam != requested_wam
            or len(existing.cases) != case_count
            or existing.generator_id != GENERATOR_ID
            or existing.proposal_sha256 != proposal_sha256
        ):
            raise ValueError("batch id is already frozen to different inputs")
        for case in existing.cases:
            task = root / case.task_path
            if not task.is_file() or sha256_file(task) != case.task_sha256:
                raise ValueError(f"published generated task changed: {case.case_id}")
        return manifest_path

    identity_seed = _identity_seed(batch_id, requested_wam)
    tasks = validate_proposals(
        batch_id=batch_id,
        case_count=case_count,
        requested_wam=requested_wam,
        proposals=proposed,
    )

    batch_root.mkdir(parents=True)
    tasks_root = batch_root / "tasks"
    tasks_root.mkdir()
    records: list[RoboticsTaskBatchCase] = []
    for task in tasks:
        task_path = tasks_root / f"{task.case_id}.json"
        task.write_json(task_path)
        records.append(
            RoboticsTaskBatchCase(
                case_id=task.case_id,
                case_seed=int(task.case_seed),
                task_path=task_path.relative_to(root).as_posix(),
                task_sha256=sha256_file(task_path),
                compiler_id=COMPILER_ID,
                semantic_signature=semantic_signature(task),
                reference_run_id=f"ref_{task.case_id}",
                wam_run_id=f"wam_{task.case_id}",
            )
        )
    manifest = RoboticsTaskBatchManifest(
        batch_id=batch_id,
        requested_wam=requested_wam,
        identity_seed=identity_seed,
        generator_id=GENERATOR_ID,
        proposal_sha256=proposal_sha256,
        generation_mode="model_authored_declarative",
        cases=records,
    )
    return manifest.write_json(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--case-count", type=int, required=True)
    parser.add_argument("--proposal-file", required=True)
    parser.add_argument(
        "--requested-wam",
        choices=["cosmos3_nano_droid", "lingbot_va_base", "lingbot_va_robotwin"],
        required=True,
    )
    args = parser.parse_args()
    print(
        generate_batch(
            output_root=args.output_root,
            batch_id=args.batch_id,
            case_count=args.case_count,
            requested_wam=args.requested_wam,
            proposals=args.proposal_file,
        )
    )


if __name__ == "__main__":
    main()
