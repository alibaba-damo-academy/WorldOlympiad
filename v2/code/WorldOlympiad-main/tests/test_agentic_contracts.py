import pytest

from autoresearch.agentic_contracts import AgentTask, StageReceipt, from_robotics_task


OBJECTS = [
    {"object_id": "ball", "category": "sphere", "role": "manipuland",
     "visual_description": "red ball"},
    {"object_id": "table", "category": "table", "role": "support",
     "visual_description": "wooden table"},
]


def test_passive_physics_task_uses_the_shared_contract():
    task = AgentTask.model_validate({
        "task_id": "drop_case", "instruction": "Release the ball over the table.",
        "scene_brief": "A ball and a table.", "initial_state": "The ball is above the table.",
        "objects": OBJECTS,
        "operation_program": [{"operation": "release", "subjects": ["ball"]}],
        "success_predicates": [{"kind": "contact", "subject": "ball", "object": "table"}],
        "requested_models": ["lingbot_world2_14b_causal_fast"],
    })
    assert task.embodiment is None
    assert len(task.content_hash()) == 64


def test_legacy_robotics_task_enters_the_same_contract():
    old = {
        "case_id": "place_case", "instruction": "Place the ball on the table.",
        "initial_state": "The ball is beside the table.", "objects": OBJECTS,
        "action_program": [
            {"operation": "pick", "subjects": ["ball"]},
            {"operation": "place", "subjects": ["ball"], "target": "table"},
        ],
        "success_predicates": [{"kind": "on", "subject": "ball", "object": "table"}],
        "requested_wams": ["cosmos3_nano_droid"],
    }
    task = from_robotics_task(old)
    assert task.task_id == "place_case"
    assert task.embodiment == "franka_panda"
    assert [step.operation for step in task.operation_program] == ["pick", "place"]
    assert task.source["schema_version"].startswith("robotics_task_instruction")


def test_wam_requires_embodiment_but_never_a_family():
    with pytest.raises(ValueError, match="robot embodiment"):
        AgentTask.model_validate({
            "task_id": "case", "instruction": "Move the ball.",
            "scene_brief": "A ball and table.", "initial_state": "A ball is on the table.",
            "objects": OBJECTS,
            "operation_program": [{"operation": "push", "subjects": ["ball"], "target": "table"}],
            "success_predicates": [{"kind": "on", "subject": "ball", "object": "table"}],
            "requested_models": ["cosmos3_nano_droid"],
        })


def test_articulation_component_is_a_task_addressable_subject():
    task = AgentTask.model_validate({
        "task_id": "door", "instruction": "Open the cabinet door.",
        "scene_brief": "Cabinet on a table.", "initial_state": "Door closed.",
        "objects": [
            {"object_id": "cabinet", "category": "cabinet", "role": "destination",
             "visual_description": "white cabinet"},
            {"object_id": "table", "category": "table", "role": "support",
             "visual_description": "wood table"},
        ],
        "articulation_components": [{"component_id": "door_panel",
            "host_object_id": "cabinet", "semantic_kind": "door",
            "joint_type": "revolute", "visual_description": "cabinet door"}],
        "operation_program": [{"operation": "pull", "subjects": ["door_panel"]}],
        "success_predicates": [{"kind": "door_open", "subject": "door_panel",
                                "object": "cabinet"}],
        "embodiment": "franka_panda", "requested_models": ["cosmos3_nano_droid"],
    })
    assert task.articulation_components[0].component_id == "door_panel"


def test_stage_receipt_requires_external_artifact_evidence():
    digest = "a" * 64
    base = {"task_id": "case", "run_id": "run", "stage": "model",
            "model_profile": "lingbot_world2_14b_causal_fast",
            "task_sha256": digest, "input_sha256": digest, "code_sha256": digest}
    with pytest.raises(ValueError, match="artifact digests"):
        StageReceipt.model_validate({**base, "status": "complete"})
    receipt = StageReceipt.model_validate({**base, "status": "complete",
                                           "artifacts": {"run_manifest.json": digest}})
    assert receipt.status == "complete"
    with pytest.raises(ValueError, match="owner"):
        StageReceipt.model_validate({**base, "status": "failed"})
