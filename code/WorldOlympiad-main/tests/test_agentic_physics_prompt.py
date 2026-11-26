import hashlib
import json

import pytest

from autoresearch.agentic_contracts import AgentTask
from autoresearch.agentic_physics_prompt import (
    PROFILE, MINIMAL_PROFILE, compile_minimal_prompt, compile_motion_prompt,
    select_prompt, verify_prompt_receipt,
)


def _task():
    return AgentTask.model_validate({
        "task_id": "unseen_mug_drop", "instruction": "Release the mug onto the table.",
        "scene_brief": "A mug and table.", "initial_state": "Mug above table.",
        "objects": [
            {"object_id": "mug", "category": "ceramic_mug", "role": "manipuland",
             "visual_description": "A blue ceramic mug with one handle"},
            {"object_id": "table", "category": "wooden_table", "role": "support",
             "visual_description": "A light wooden table with a flat top"},
        ],
        "operation_program": [{"operation": "release", "subjects": ["mug"]}],
        "success_predicates": [{"kind": "contact", "subject": "mug", "object": "table"}],
        "requested_models": ["lingbot_world2_14b_causal_fast"],
        "model_prompts": {"lingbot_world2_14b_causal_fast":
                          "A blue ceramic mug hangs above a wooden table. "
                          "The mug falls under gravity and reaches the table."},
    })


def test_motion_prompt_compiles_declared_events_without_changing_task(tmp_path):
    task = _task()
    original_hash = task.content_hash()
    prompt, receipt = select_prompt(task, "lingbot_world2_14b_causal_fast", PROFILE)
    assert prompt == compile_motion_prompt(task)
    assert "A blue ceramic mug" in prompt
    assert "Mug above table" in prompt
    assert "At the start" in prompt
    assert prompt.index("released") < prompt.index("falls downward") < prompt.index("contacts")
    assert "bounces" not in prompt and "to the right" not in prompt
    assert task.content_hash() == original_hash
    assert receipt["task_sha256"] == original_hash
    assert receipt["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    path = tmp_path / "prompt_receipt.json"
    path.write_text(json.dumps(receipt))
    assert verify_prompt_receipt(path, task, "lingbot_world2_14b_causal_fast", PROFILE) == prompt
    path.write_text(json.dumps({**receipt, "prompt_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="differs"):
        verify_prompt_receipt(path, task, "lingbot_world2_14b_causal_fast", PROFILE)


def test_motion_prompt_requires_supported_evidence():
    task = _task()
    task.success_predicates = []
    with pytest.raises(ValueError, match="contact target"):
        compile_motion_prompt(task)


def test_minimal_prompt_front_loads_one_explicit_motion_chain():
    task = _task()
    task.initial_state = "The mug starts 0.35 m above the wooden table."
    prompt, receipt = select_prompt(
        task, "lingbot_world2_14b_causal_fast", MINIMAL_PROFILE)
    assert prompt == compile_minimal_prompt(task)
    assert prompt.startswith("A blue ceramic mug")
    assert "0.35 m above the wooden table" in prompt
    assert prompt.index("released") < prompt.index("falls") < prompt.index("contacts")
    assert "camera" not in prompt.casefold()
    assert receipt["profile"] == MINIMAL_PROFILE
