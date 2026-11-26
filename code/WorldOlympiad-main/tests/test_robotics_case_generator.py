from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.robotics_case_generator import generate_batch
from autoresearch.robotics_contracts import (
    RoboticsTaskBatchManifest,
    TaskInstructionSpec,
)
from autoresearch.robotics_views import sha256_file


def _pick_inside(*, object_id: str = "apple", color: str = "red") -> dict:
    return {
        "instruction": f"Put the {color} {object_id} in the blue bowl.",
        "scene_brief": f"A {color} {object_id}, a blue bowl, and a worktable.",
        "initial_state": f"The {object_id} is on the table outside the bowl.",
        "objects": [
            {
                "object_id": object_id,
                "category": "fruit",
                "role": "manipuland",
                "visual_description": f"{color} {object_id}",
            },
            {
                "object_id": "bowl",
                "category": "bowl",
                "role": "destination",
                "visual_description": "blue ceramic bowl",
            },
            {
                "object_id": "table",
                "category": "worktable",
                "role": "support",
                "visual_description": "wood worktable",
            },
        ],
        "action_program": [
            {"operation": "pick", "subjects": [object_id]},
            {"operation": "place_inside", "subjects": [object_id], "target": "bowl"},
        ],
        "success_predicates": [
            {"kind": "inside", "subject": object_id, "object": "bowl"},
            {"kind": "released", "subject": object_id},
        ],
    }


def _push_to_zone() -> dict:
    return {
        "instruction": "Push the wooden block into the green target zone.",
        "scene_brief": "A wooden block and green target zone on a worktable.",
        "initial_state": "The block starts outside the target zone.",
        "objects": [
            {
                "object_id": "block",
                "category": "block",
                "role": "manipuland",
                "visual_description": "small wooden block",
            },
            {
                "object_id": "zone",
                "category": "target_zone",
                "role": "destination",
                "visual_description": "flat green target zone",
            },
            {
                "object_id": "table",
                "category": "worktable",
                "role": "support",
                "visual_description": "wood worktable",
            },
        ],
        "action_program": [
            {"operation": "push_to", "subjects": ["block"], "target": "zone"}
        ],
        "success_predicates": [
            {"kind": "inside", "subject": "block", "object": "zone"}
        ],
    }


def test_agent_authored_suite_freezes_distinct_tasks_deterministically(
    tmp_path: Path,
) -> None:
    proposals = [_pick_inside(), _push_to_zone()]
    manifest_path = generate_batch(
        output_root=tmp_path,
        batch_id="diverse_batch",
        case_count=2,
        requested_wam="cosmos3_nano_droid",
        proposals=proposals,
    )
    manifest = RoboticsTaskBatchManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.schema_version == "robotics_task_batch.v2"
    assert manifest.generation_mode == "model_authored_declarative"
    assert len(manifest.cases) == 2
    assert len({case.semantic_signature for case in manifest.cases}) == 2
    assert all(case.recipe_id is None for case in manifest.cases)
    assert all(case.case_seed < 2**53 for case in manifest.cases)
    for case in manifest.cases:
        task_path = tmp_path / case.task_path
        assert sha256_file(task_path) == case.task_sha256
        task = TaskInstructionSpec.model_validate_json(
            task_path.read_text(encoding="utf-8")
        )
        assert task.schema_version == "robotics_task_instruction.v3"
        assert task.action_program

    assert generate_batch(
        output_root=tmp_path,
        batch_id="diverse_batch",
        case_count=2,
        requested_wam="cosmos3_nano_droid",
        proposals=proposals,
    ) == manifest_path


def test_exact_semantic_duplicate_is_rejected_even_if_layout_changes(
    tmp_path: Path,
) -> None:
    first = _pick_inside()
    second = _pick_inside()
    first["case_parameters"] = {"manipuland_xy": [0.1, 0.2]}
    second["case_parameters"] = {"manipuland_xy": [-0.2, 0.1]}
    with pytest.raises(ValueError, match="exact semantic duplicate"):
        generate_batch(
            output_root=tmp_path,
            batch_id="duplicates",
            case_count=2,
            requested_wam="cosmos3_nano_droid",
            proposals=[first, second],
        )


def test_shared_action_family_is_allowed_when_task_semantics_differ(
    tmp_path: Path,
) -> None:
    manifest = generate_batch(
        output_root=tmp_path,
        batch_id="shared_controller",
        case_count=2,
        requested_wam="cosmos3_nano_droid",
        proposals=[_pick_inside(object_id="apple"), _pick_inside(object_id="pear", color="green")],
    )
    assert manifest.is_file()


def test_agent_cannot_override_harness_or_model_owned_fields(tmp_path: Path) -> None:
    proposal = _pick_inside()
    proposal["duration_sec"] = 500
    with pytest.raises(ValueError, match="cannot override"):
        generate_batch(
            output_root=tmp_path,
            batch_id="owned_fields",
            case_count=1,
            requested_wam="cosmos3_nano_droid",
            proposals=[proposal],
        )


def test_generated_batch_refuses_changed_identity(tmp_path: Path) -> None:
    generate_batch(
        output_root=tmp_path,
        batch_id="frozen_batch",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[_pick_inside()],
    )
    with pytest.raises(ValueError, match="different inputs"):
        generate_batch(
            output_root=tmp_path,
            batch_id="frozen_batch",
            case_count=1,
            requested_wam="lingbot_va_base",
            proposals=[_pick_inside()],
        )
