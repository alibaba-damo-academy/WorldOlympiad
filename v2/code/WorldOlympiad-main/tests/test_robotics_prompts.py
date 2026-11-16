from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from autoresearch.robotics_contracts import (
    SuccessPredicate,
    TaskInstructionSpec,
    TaskObject,
)
from autoresearch.robotics_prompts import (
    ROBOTICS_PROMPT_REGISTRY,
    RoboticsPromptManifest,
    render_robotics_prompt,
)


def _task(
    *,
    case_id: str = "arbitrary_case_name",
    duration_sec: float = 12.0,
    fps: float = 15.0,
    relation: str = "inside",
) -> TaskInstructionSpec:
    return TaskInstructionSpec(
        case_id=case_id,
        instruction="Put the red cup in the blue tray.",
        objects=[
            TaskObject(
                object_id="red_cup",
                category="cup",
                role="manipuland",
                visual_description="matte red ceramic cup",
            ),
            TaskObject(
                object_id="blue_tray",
                category="sorting_tray",
                role="destination",
                visual_description="shallow blue tray",
            ),
            TaskObject(
                object_id="wood_table",
                category="worktable",
                role="support",
                visual_description="wooden worktable",
            ),
        ],
        initial_state="The cup starts on the table outside the tray.",
        success_predicates=[
            SuccessPredicate(kind="pose_delta", subject="red_cup", threshold=0.12),
            SuccessPredicate(
                kind=relation,
                subject="red_cup",
                object="blue_tray",
            ),
            SuccessPredicate(
                kind="contact", subject="red_cup", object="blue_tray"
            ),
            SuccessPredicate(kind="released", subject="red_cup"),
        ],
        duration_sec=duration_sec,
        fps=fps,
    )


def test_registry_is_typed_and_model_specific() -> None:
    assert set(ROBOTICS_PROMPT_REGISTRY) == {
        ("cosmos3_nano_droid", "inside"),
        ("lingbot_va_base", "inside"),
    }
    cosmos = ROBOTICS_PROMPT_REGISTRY[("cosmos3_nano_droid", "inside")]
    lingbot = ROBOTICS_PROMPT_REGISTRY[("lingbot_va_base", "inside")]
    assert cosmos.template_id != lingbot.template_id
    assert cosmos.max_utf8_bytes == 160
    assert lingbot.max_utf8_bytes == 220


def test_cosmos_uses_verified_short_command_without_case_hardcoding() -> None:
    manifest = render_robotics_prompt(
        _task(case_id="not_cup_to_tray"), "cosmos3_nano_droid"
    )
    assert manifest.rendered_prompt == "Put the red cup in the blue tray."
    assert manifest.template_id == "cosmos3_droid.short_task_instruction"
    assert manifest.template_version == "2.0.0"
    assert manifest.case_id == "not_cup_to_tray"
    assert len(manifest.task_contract_sha256) == 64
    assert manifest.semantic_bindings is None
    assert {item.role for item in manifest.object_bindings} >= {
        "manipuland", "destination"
    }


def test_cosmos_short_template_renders_different_task_semantics() -> None:
    task = TaskInstructionSpec(
        case_id="fruit_sorting_trial",
        instruction="Move the fruit to its destination.",
        objects=[
            TaskObject(
                object_id="yellow_banana",
                category="banana",
                role="manipuland",
                visual_description="yellow banana",
            ),
            TaskObject(
                object_id="green_bowl",
                category="bowl",
                role="destination",
                visual_description="green bowl",
            ),
            TaskObject(
                object_id="counter",
                category="counter",
                role="support",
                visual_description="kitchen counter",
            ),
        ],
        initial_state="The banana starts beside the bowl.",
        success_predicates=[
            SuccessPredicate(
                kind="inside", subject="yellow_banana", object="green_bowl"
            ),
            SuccessPredicate(kind="released", subject="yellow_banana"),
        ],
    )
    manifest = render_robotics_prompt(task, "cosmos3_nano_droid")
    assert manifest.rendered_prompt == "Move the fruit to its destination."


def test_lingbot_uses_its_task_only_template() -> None:
    manifest = render_robotics_prompt(_task(), "lingbot_va_base")
    assert manifest.rendered_prompt == "Put the red cup in the blue tray."
    assert manifest.template_id == "lingbot_va_base.short_task_instruction"


def test_prompt_ignores_reference_timeline_fields() -> None:
    short = render_robotics_prompt(
        _task(duration_sec=12.0, fps=15.0), "cosmos3_nano_droid"
    )
    long = render_robotics_prompt(
        _task(duration_sec=99.0, fps=60.0), "cosmos3_nano_droid"
    )
    assert short.rendered_prompt == long.rendered_prompt
    assert short.prompt_sha256 == long.prompt_sha256
    assert short.object_bindings_sha256 == long.object_bindings_sha256
    assert short.action_program_sha256 == long.action_program_sha256
    assert "frame" not in short.rendered_prompt.lower()
    assert "second" not in short.rendered_prompt.lower()


def test_manifest_records_and_validates_content_hashes() -> None:
    manifest = render_robotics_prompt(_task(), "cosmos3_nano_droid")
    assert manifest.prompt_sha256 == hashlib.sha256(
        manifest.rendered_prompt.encode("utf-8")
    ).hexdigest()
    assert (
        manifest.object_bindings_sha256
        == hashlib.sha256(
            "|".join(item.content_hash() for item in manifest.object_bindings).encode()
        ).hexdigest()
    )
    payload = manifest.model_dump(mode="json")
    payload["prompt_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="prompt SHA-256 changed"):
        RoboticsPromptManifest.model_validate(payload)


def test_manifest_rejects_timeline_language_even_with_matching_hash() -> None:
    manifest = render_robotics_prompt(_task(), "cosmos3_nano_droid")
    payload = manifest.model_dump(mode="json")
    payload["rendered_prompt"] = "Put the red cup in the blue tray in 408 frames."
    payload["prompt_sha256"] = hashlib.sha256(
        payload["rendered_prompt"].encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValidationError, match="may not encode.*timing"):
        RoboticsPromptManifest.model_validate(payload)


def test_manifest_rejects_non_template_text_even_with_matching_hash() -> None:
    manifest = render_robotics_prompt(_task(), "lingbot_va_base")
    payload = manifest.model_dump(mode="json")
    payload["rendered_prompt"] = "Move the red cup into the blue tray."
    payload["prompt_sha256"] = hashlib.sha256(
        payload["rendered_prompt"].encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValidationError, match="preserve the selected short task instruction"):
        RoboticsPromptManifest.model_validate(payload)


def test_timing_language_in_selected_instruction_fails_closed() -> None:
    task = _task().model_copy(
        update={"instruction": "Finish this task within 27.2 seconds."}
    )
    with pytest.raises(ValueError, match="may not encode.*timing"):
        render_robotics_prompt(task, "cosmos3_nano_droid")


def test_unknown_wam_fails_and_prompt_manifest_preserves_declared_roles() -> None:
    with pytest.raises(ValueError, match="no registered robotics prompt style"):
        render_robotics_prompt(_task(), "unknown_wam")

    task = _task()
    objects = [
        item.model_copy(update={"role": "support"})
        if item.object_id == "red_cup"
        else item
        for item in task.objects
    ]
    mismatched = task.model_copy(update={"objects": objects})
    manifest = render_robotics_prompt(mismatched, "cosmos3_nano_droid")
    assert next(
        item for item in manifest.object_bindings if item.object_id == "red_cup"
    ).role == "support"
