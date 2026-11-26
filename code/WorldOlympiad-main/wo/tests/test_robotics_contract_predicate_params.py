from autoresearch.robotics_contracts import TaskInstructionSpec


def test_predicate_params_survive_scene_compiler_contract():
    payload = {
        "case_id": "axis-test",
        "instruction": "Stand the bottle upright.",
        "objects": [
            {"object_id": "bottle", "category": "bottle", "role": "manipuland", "visual_description": "bottle"},
            {"object_id": "tray", "category": "tray", "role": "destination", "visual_description": "tray"},
            {"object_id": "table", "category": "table", "role": "support", "visual_description": "table"},
        ],
        "initial_state": "The bottle lies on its side.",
        "success_predicates": [{
            "kind": "pose_delta", "subject": "bottle", "object": "table",
            "params": {"semantic_up_axis": [0, 1, 0]},
        }],
    }
    task = TaskInstructionSpec.model_validate(payload)
    assert task.success_predicates[0].params["semantic_up_axis"] == [0, 1, 0]
