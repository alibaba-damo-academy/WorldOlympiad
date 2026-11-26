"""Shared capability/schema extension for authorized articulated tasks.

These tests pin the frozen contracts that let the OpenAI task author describe a
real articulated control event ("press") and a real articulated joint motion
("door_open"/"drawer_open") without inventing a per-case skill.  They assert the
whole reusable chain: strict author schema -> proposal validation -> controller
compilation -> reference-materializer execution steps.
"""

from __future__ import annotations

from autoresearch.robotics_reference_materializer import compiled_execution_steps
from autoresearch.robotics_task_author import (
    SHARED_ACTION_OPERATORS,
    SHARED_PROGRAM_SHAPES,
    AuthoredCase,
    _strict_output_schema,
)
from autoresearch.robotics_task_compiler import compile_task
from autoresearch.robotics_case_generator import validate_proposals


def _articulated_door_proposal() -> dict:
    return {
        "instruction": "Grasp the cabinet handle and pull the door open until it is fully open.",
        "scene_brief": (
            "A closed two-door cabinet with a visible handle stands on a broad "
            "tabletop; the door is fully closed and the task is unsolved."
        ),
        "initial_state": (
            "The cabinet door is fully closed; the handle is not yet grasped."
        ),
        "objects": [
            {
                "object_id": "cabinet",
                "category": "cabinet",
                "role": "destination",
                "visual_description": "A light-gray rectangular cabinet with one hinged door.",
            },
            {
                "object_id": "table_surface",
                "category": "table",
                "role": "support",
                "visual_description": "A broad light-gray horizontal tabletop.",
            },
        ],
        "articulation_components": [
            {
                "component_id": "cabinet_door",
                "host_object_id": "cabinet",
                "semantic_kind": "door",
                "joint_type": "revolute",
                "visual_description": "A hinged light-gray door with a slim metal handle.",
            }
        ],
        "action_program": [
            {"operation": "pull", "subjects": ["cabinet_door"], "target": None}
        ],
        "success_predicates": [
            {
                "kind": "door_open",
                "subject": "cabinet_door",
                "object": "cabinet",
                "threshold": None,
            }
        ],
    }


def _press_proposal() -> dict:
    return {
        "instruction": "Move the gripper above the red button and press it down once.",
        "scene_brief": (
            "A raised red spring-loaded push button sits in a dark fixture on a "
            "clear light-gray table; the button is fully extended."
        ),
        "initial_state": (
            "The red button is fully extended and has not been pressed."
        ),
        "objects": [
            {
                "object_id": "red_button",
                "category": "push_button",
                "role": "manipuland",
                "visual_description": "A bright red cylindrical spring-loaded push button.",
            },
            {
                "object_id": "button_fixture",
                "category": "button_fixture",
                "role": "destination",
                "visual_description": "A compact matte-black button housing.",
            },
            {
                "object_id": "table_support",
                "category": "table",
                "role": "support",
                "visual_description": "A broad stationary light-gray tabletop.",
            },
        ],
        "articulation_components": [
            {
                "component_id": "button_cap",
                "host_object_id": "button_fixture",
                "semantic_kind": "button",
                "joint_type": "prismatic",
                "visual_description": "A bright red cylindrical cap with short visible travel.",
            }
        ],
        "action_program": [
            {"operation": "press", "subjects": ["button_cap"], "target": None}
        ],
        "success_predicates": [
            {
                "kind": "pressed",
                "subject": "button_cap",
                "object": "button_fixture",
                "threshold": None,
            }
        ],
    }


def test_strict_schema_offers_shared_operators_and_articulation_declaration() -> None:
    schema = _strict_output_schema(
        case_count=1, predicates=["pressed", "door_open", "stacked"]
    )
    case = schema["properties"]["cases"]["items"]
    operations = case["properties"]["action_program"]["items"]["properties"][
        "operation"
    ]["enum"]
    assert operations == list(SHARED_ACTION_OPERATORS)
    assert {"press", "pull", "close"}.issubset(set(operations))
    assert "articulation_components" in case["required"]
    component = case["properties"]["articulation_components"]["items"]
    assert set(component["required"]) == {
        "component_id",
        "host_object_id",
        "semantic_kind",
        "joint_type",
        "visual_description",
    }
    assert component["properties"]["joint_type"]["enum"] == ["prismatic", "revolute"]
    assert case["properties"]["success_predicates"]["items"]["properties"]["kind"][
        "enum"
    ] == ["pressed", "door_open", "stacked"]


def test_author_accepts_articulated_case_without_rigid_manipuland() -> None:
    case = AuthoredCase.model_validate(_articulated_door_proposal())
    assert case.articulation_components[0].component_id == "cabinet_door"


def test_pull_program_compiles_and_materializes_door_open() -> None:
    proposal = _articulated_door_proposal()
    tasks = validate_proposals(
        batch_id="capability_extension_test",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[proposal],
    )
    plan = compile_task(tasks[0])
    assert plan["native_capabilities"]["cosmos3_nano_droid"]["status"] == (
        "native_supported"
    )
    assert plan["native_capabilities"]["cosmos3_nano_droid"][
        "unsupported_terminal_predicates"
    ] == []
    steps = compiled_execution_steps(plan)
    assert steps == [
        {
            "subject_id": "cabinet_door",
            "destination_id": "cabinet",
            "terminal_operator": "pull",
            "terminal_predicate": "door_open",
            "terminal_joint_fraction": 1.0,
        }
    ]


def test_press_program_compiles_and_materializes_pressed() -> None:
    proposal = _press_proposal()
    tasks = validate_proposals(
        batch_id="capability_extension_press",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[proposal],
    )
    plan = compile_task(tasks[0])
    assert plan["native_capabilities"]["cosmos3_nano_droid"] == {
        "status": "native_supported",
        "unsupported_operators": [],
        "unsupported_terminal_predicates": [],
        "policy": "reference_may_publish_native_wam_must_fail_closed",
    }
    steps = compiled_execution_steps(plan)
    assert steps == [
        {
            "subject_id": "button_cap",
            "destination_id": "button_fixture",
            "terminal_operator": "press",
            "terminal_predicate": "pressed",
        }
    ]


def test_articulated_host_prompt_declares_a_task_target() -> None:
    from autoresearch.robotics_scene_prompt_compiler import compile_task_prompt
    from autoresearch.scene_prompting import assess_scene_prompt, compile_scene_prompt

    tasks = validate_proposals(
        batch_id="capability_extension_prompt",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[_articulated_door_proposal()],
    )
    spec = compile_task_prompt(tasks[0])
    roles = {item.task_role for item in spec.objects}
    assert "target" in roles
    assert {item.name for item in spec.objects} <= {
        item.visual_description.strip().rstrip(".") for item in tasks[0].objects
    }
    prompt = compile_scene_prompt(spec)
    report = assess_scene_prompt(spec, prompt)
    assert report.passed, report.problems
    # The table stays the primary support, so the prompt keeps a spatial anchor
    # and covers every declared object, including the freestanding cabinet.
    assert "center" in prompt.lower()
    for item in spec.objects:
        assert item.name.strip().lower() in prompt.lower()


def test_press_prompt_declares_a_task_target() -> None:
    from autoresearch.robotics_scene_prompt_compiler import compile_task_prompt

    tasks = validate_proposals(
        batch_id="capability_extension_press_prompt",
        case_count=1,
        requested_wam="cosmos3_nano_droid",
        proposals=[_press_proposal()],
    )
    spec = compile_task_prompt(tasks[0])
    roles = {item.task_role for item in spec.objects}
    assert roles & {"target", "support_target"}
