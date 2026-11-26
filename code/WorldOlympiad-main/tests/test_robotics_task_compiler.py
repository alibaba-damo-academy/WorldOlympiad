from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.robotics_case_generator import generate_batch
from autoresearch.robotics_contracts import ActionStep, TaskInstructionSpec
from autoresearch.robotics_task_compiler import (
    compile_batch,
    compile_task,
    declares_upright_terminal,
)


def _upright_place_task() -> TaskInstructionSpec:
    return TaskInstructionSpec.model_validate(
        {
            "case_id": "upright_place_case",
            "instruction": (
                "Pick up the red bottle from the table and place it upright "
                "inside the basket."
            ),
            "scene_brief": "A bottle and basket on a worktable.",
            "initial_state": "The bottle stands beside the basket.",
            "objects": [
                {
                    "object_id": "bottle",
                    "category": "bottle",
                    "role": "manipuland",
                    "visual_description": "red bottle",
                },
                {
                    "object_id": "basket",
                    "category": "basket",
                    "role": "destination",
                    "visual_description": "woven basket",
                },
                {
                    "object_id": "table",
                    "category": "table",
                    "role": "support",
                    "visual_description": "worktable",
                },
            ],
            "action_program": [
                {"operation": "pick", "subjects": ["bottle"]},
                {"operation": "place", "subjects": ["bottle"], "target": "basket"},
            ],
            "success_predicates": [
                {"kind": "in_container", "subject": "bottle", "object": "basket"}
            ],
            "requested_wams": ["cosmos3_nano_droid"],
        }
    )


def test_upright_instruction_compiles_explicit_terminal_orientation() -> None:
    """An instruction that asks for an upright placement must encode it.

    The shared task-intent gate rejects a bare pick/place pair for an upright
    instruction. The compiler records the requirement explicitly on the terminal
    controller step and its success predicate so the reference must certify the
    measured terminal alignment.
    """

    task = _upright_place_task()
    assert declares_upright_terminal(task) is True
    plan = compile_task(task)
    terminal = plan["controller_program"][-1]
    assert terminal["constraints"]["terminal_upright"] is True
    assert terminal["constraints"]["terminal_upright_axis_world"] == [0.0, 0.0, 1.0]
    assert terminal["constraints"]["terminal_upright_min_cosine"] == 0.99
    predicate = plan["success_program"][0]
    assert predicate["terminal_upright"] is True
    assert predicate["terminal_upright_min_cosine"] == 0.99


def test_non_upright_instruction_is_not_reinterpreted() -> None:
    task = _upright_place_task().model_copy(
        update={"instruction": "Pick up the red bottle and put it in the basket."}
    )
    assert declares_upright_terminal(task) is False
    plan = compile_task(task)
    assert "terminal_upright" not in plan["controller_program"][-1]["constraints"]
    assert "terminal_upright" not in plan["success_program"][0]


def _handle_pull_task() -> TaskInstructionSpec:
    return TaskInstructionSpec.model_validate(
        {
            "case_id": "handle_pull_case",
            "instruction": (
                "Grasp the cabinet handle and pull the door open until it is "
                "fully open."
            ),
            "scene_brief": "A single-door cabinet on a table.",
            "initial_state": "The cabinet door is fully closed.",
            "objects": [
                {
                    "object_id": "cabinet_host",
                    "category": "cabinet",
                    "role": "destination",
                    "visual_description": "white cabinet",
                },
                {
                    "object_id": "table",
                    "category": "table",
                    "role": "support",
                    "visual_description": "worktable",
                },
            ],
            "articulation_components": [
                {
                    "component_id": "cabinet_door",
                    "host_object_id": "cabinet_host",
                    "joint_type": "revolute",
                    "semantic_kind": "door",
                    "visual_description": "hinged door with a handle",
                }
            ],
            "action_program": [{"operation": "pull", "subjects": ["cabinet_door"]}],
            "success_predicates": [
                {
                    "kind": "door_open",
                    "subject": "cabinet_door",
                    "object": "cabinet_host",
                    "threshold": 1.0,
                }
            ],
            "requested_wams": ["cosmos3_nano_droid"],
        }
    )


def test_handle_instruction_compiles_required_handle_grasp() -> None:
    from autoresearch.robotics_task_compiler import declares_handle_articulation

    task = _handle_pull_task()
    assert declares_handle_articulation(task) is True
    plan = compile_task(task)
    constraints = plan["controller_program"][0]["constraints"]
    assert constraints["grasp_target"] == "handle"
    assert constraints["grasp_target_required"] is True
    assert constraints["articulation_component"] == "cabinet_door"
    assert constraints["terminal_joint_fraction"] == 1.0
    assert plan["success_program"][0]["terminal_joint_fraction"] == 1.0


def test_explicit_final_handle_release_has_action_and_measured_postcondition():
    from autoresearch.robotics_reference_materializer import compiled_execution_steps
    task = _handle_pull_task().model_copy(update={
        "instruction":"Grasp the handle and pull the door fully open, then release the handle."})
    plan = compile_task(task)
    constraints = plan["controller_program"][0]["constraints"]
    assert constraints["release_after_joint_motion"] is True
    release = [item for item in plan["success_program"] if item["predicate"] == "released"]
    assert len(release) == 1 and release[0]["subject"] == "cabinet_door"
    assert release[0]["evaluator"] == "gripper_detached"
    assert release[0]["finger_contact_required"] is False
    assert compiled_execution_steps(plan)[0]["terminal_joint_fraction"] == 1.
    retained = _handle_pull_task().model_copy(update={
        "instruction":"Open the cabinet door without releasing the handle."})
    assert not any(row["predicate"] == "released" for row in compile_task(retained)["success_program"])


def _hanging_task() -> TaskInstructionSpec:
    return TaskInstructionSpec.model_validate(
        {
            "case_id": "hanging_case",
            "case_seed": 1,
            "case_parameters": {"layout_seed": 1},
            "instruction": "Pick up the mug and hang it securely on the rack hook.",
            "scene_brief": "A mug and open hook on a table.",
            "initial_state": "The mug is beside the open hook and is not engaged.",
            "objects": [
                {"object_id": "mug", "category": "ceramic_mug", "role": "manipuland", "visual_description": "blue mug"},
                {"object_id": "rack_hook", "category": "rack_hook", "role": "destination", "visual_description": "open metal hook"},
                {"object_id": "table", "category": "table", "role": "support", "visual_description": "wood table"},
            ],
            "action_program": [
                {"operation": "pick", "subjects": ["mug"]},
                {"operation": "insert", "subjects": ["mug"], "target": "rack_hook"},
            ],
            "success_predicates": [
                {"kind": "inserted", "subject": "mug", "object": "rack_hook"}
            ],
            "requested_wams": ["cosmos3_nano_droid", "lingbot_va_robotwin"],
        }
    )


def test_open_hook_is_compiled_as_hang_and_capability_checked() -> None:
    plan = compile_task(_hanging_task())
    assert [step["operator"] for step in plan["controller_program"]] == ["pick", "hang"]
    assert plan["success_program"][0]["predicate"] == "hung"
    assert plan["native_capabilities"]["cosmos3_nano_droid"]["status"] == "native_supported"
    assert plan["native_capabilities"]["cosmos3_nano_droid"]["unsupported_operators"] == []
    assert plan["native_capabilities"]["lingbot_va_robotwin"]["status"] == "native_supported"


def test_unseen_operator_is_preserved_as_a_capability_gap() -> None:
    task = _upright_place_task().model_copy(update={"action_program": [
        ActionStep(operation="new_reusable_operator", subjects=["bottle"], target="basket")
    ]})
    plan = compile_task(task)
    assert plan["controller_program"][0]["operator"] == "new_reusable_operator"
    assert plan["controller_program"][0]["controller_primitive"] is None
    assert plan["controller_program"][0]["capability_status"] == "missing"
    assert plan["execution_capability_preflight"]["passed"] is False


def test_semantic_pour_mapping_does_not_claim_executable_capability() -> None:
    task = _upright_place_task().model_copy(update={"action_program": [
        ActionStep(operation="pick", subjects=["bottle"]),
        ActionStep(operation="pour", subjects=["bottle"], target="basket"),
    ]})
    plan = compile_task(task)
    assert plan["controller_program"][-1]["capability_status"] == "declared_unverified"
    assert plan["execution_capability_preflight"]["passed"] is False
    assert plan["native_capabilities"]["cosmos3_nano_droid"]["unsupported_operators"] == ["pour"]


def test_declared_contents_pour_uses_registered_independent_execution() -> None:
    task = TaskInstructionSpec.model_validate({
        "case_id": "unseen_container_pour",
        "instruction": "Pour the blocks from the small bin into the large bin.",
        "scene_brief": "Two bins and three loose blocks on a table.",
        "initial_state": "Blocks begin inside the small bin.",
        "objects": [
            {"object_id": "small_bin", "category": "bin", "role": "support",
             "visual_description": "small graspable bin"},
            {"object_id": "blocks", "category": "blocks", "role": "manipuland",
             "visual_description": "three loose blocks"},
            {"object_id": "large_bin", "category": "bin", "role": "destination",
             "visual_description": "large stable bin"},
        ],
        "action_program": [
            {"operation": "pick", "subjects": ["small_bin"]},
            {"operation": "pour", "subjects": ["small_bin"], "target": "large_bin",
             "constraints": {"source_container": "small_bin", "contents_object": "blocks",
                             "pour_required": True, "tilt_required": True}},
        ],
        "success_predicates": [
            {"kind": "inside", "subject": "blocks", "object": "large_bin"}],
        "requested_wams": ["cosmos3_nano_droid"],
    })
    plan = compile_task(task)
    assert plan["controller_program"][-1]["capability_status"] == "native_registered"
    assert plan["native_capabilities"]["cosmos3_nano_droid"]["status"] == "native_supported"
    assert plan["execution_capability_preflight"]["passed"] is True


def _proposal(operation: str = "push") -> dict:
    return {
        "instruction": "Push the block to the target.",
        "scene_brief": "A block and target zone on a worktable.",
        "initial_state": "The block starts outside the target.",
        "objects": [
            {
                "object_id": "block",
                "category": "block",
                "role": "manipuland",
                "visual_description": "wood block",
            },
            {
                "object_id": "target",
                "category": "target_zone",
                "role": "destination",
                "visual_description": "green target",
            },
            {
                "object_id": "table",
                "category": "worktable",
                "role": "support",
                "visual_description": "worktable",
            },
        ],
        "action_program": [
            {"operation": operation, "subjects": ["block"], "target": "target"}
        ],
        "success_predicates": [
            {"kind": "on_region", "subject": "block", "object": "target"}
        ],
    }


def test_batch_compiler_maps_reusable_operators_without_task_recipes(
    tmp_path: Path,
) -> None:
    batch = generate_batch(
        output_root=tmp_path,
        batch_id="compile_me",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[_proposal()],
    )
    output = compile_batch(batch_manifest=batch, output_root=tmp_path)
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "static_plan_validated"
    assert value["rollout_admission"] == "blocked_until_physical_certification"
    plan = json.loads((tmp_path / value["cases"][0]["plan"]).read_text())
    assert plan["controller_program"][0]["controller_primitive"] == "cartesian_push"
    assert plan["success_program"][0]["evaluator"] == "object_in_planar_region"
    assert plan["rollout_admission"] == "blocked_until_all_required_certificates_pass"


def test_unknown_action_remains_in_task_for_shared_rsi_loop() -> None:
    payload = _proposal(operation="new_reusable_operator")
    payload.update(
        {
            "schema_version": "robotics_task_instruction.v3",
            "case_id": "unknown_operator",
            "case_seed": 1,
            "case_parameters": {"layout_seed": 1},
        }
    )
    task = TaskInstructionSpec.model_validate(payload)
    plan = compile_task(task)
    assert plan["controller_program"][0]["operator"] == "new_reusable_operator"
    assert plan["controller_program"][0]["capability_status"] == "missing"
    assert plan["native_capabilities"]["cosmos3_nano_droid"]["status"] == "blocked"


def test_recompiling_unchanged_batch_reuses_published_plans(
    tmp_path: Path,
) -> None:
    """A re-dispatched compile stage must reuse its unchanged result.

    The scheduler re-dispatches a stage whenever its owner or the accepted
    engineering revision changed, and its contract is that a stage with
    unchanged inputs reuses the existing result. The compiler previously raised
    FileExistsError for any re-dispatch, which parked the case with no legal way
    to reach the already published plans.
    """

    batch = generate_batch(
        output_root=tmp_path,
        batch_id="recompile_me",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[_proposal(operation="push")],
    )
    first = compile_batch(batch_manifest=batch, output_root=tmp_path)
    manifest = json.loads(first.read_text(encoding="utf-8"))
    plan_path = tmp_path / manifest["cases"][0]["plan"]
    digest = plan_path.read_bytes()

    second = compile_batch(batch_manifest=batch, output_root=tmp_path)

    assert second == first
    assert plan_path.read_bytes() == digest


def test_recompiled_batch_still_fails_closed_on_a_changed_plan(
    tmp_path: Path,
) -> None:
    """Published plans stay frozen: a different plan must not overwrite them."""

    batch = generate_batch(
        output_root=tmp_path,
        batch_id="frozen_me",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[_proposal(operation="push")],
    )
    compiled = compile_batch(batch_manifest=batch, output_root=tmp_path)
    manifest = json.loads(compiled.read_text(encoding="utf-8"))
    plan_path = tmp_path / manifest["cases"][0]["plan"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["compiler_id"] = "different_compiler"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        compile_batch(batch_manifest=batch, output_root=tmp_path)
