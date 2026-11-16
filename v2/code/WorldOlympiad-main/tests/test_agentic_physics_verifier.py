import json

import numpy as np

from autoresearch.agentic_contracts import AgentTask
from autoresearch.agentic_physics_verifier import verify_passive_reference


def _task(predicates):
    return AgentTask.model_validate({
        "task_id": "physics", "instruction": "Release the ball.",
        "scene_brief": "Ball and table.", "initial_state": "Ball above table.",
        "objects": [
            {"object_id": "ball", "category": "ball", "role": "manipuland",
             "visual_description": "red ball"},
            {"object_id": "table", "category": "table", "role": "support",
             "visual_description": "wood table"},
        ],
        "operation_program": [{"operation": "release", "subjects": ["ball"]}],
        "success_predicates": predicates,
        "requested_models": ["lingbot_world2_14b_causal_fast"],
    })


def _evidence(tmp_path, *, with_contact=True):
    trajectory = tmp_path / "state.npz"
    contacts = tmp_path / "contacts.json"
    positions = np.array([[[0., 0., .4], [0., 0., 0.]],
                          [[0., 0., .2], [0., 0., 0.]],
                          [[0., 0., .05], [0., 0., 0.]]])
    velocities = np.array([[[0., 0., -.1], [0., 0., 0.]],
                           [[0., 0., -.5], [0., 0., 0.]],
                           [[0., 0., .2], [0., 0., 0.]]])
    orientations = np.tile(np.array([0., 0., 0., 1.]), (3, 2, 1))
    np.savez(trajectory, body_names=np.array(["subject", "table"]),
             positions=positions, orientations_xyzw=orientations,
             linear_velocities=velocities, timestamps=np.array([0., .1, .2]))
    contacts.write_text(json.dumps([{"body_a": 1, "body_b": 2, "time_s": .15}]
                                   if with_contact else []))
    return trajectory, contacts


def test_independent_contact_and_motion_certificate(tmp_path):
    trajectory, contacts = _evidence(tmp_path)
    result = verify_passive_reference(
        task=_task([{"kind": "contact", "subject": "ball", "object": "table"},
                    {"kind": "moved", "subject": "ball", "threshold": .2}]),
        trajectory=trajectory, contacts_path=contacts,
        body_ids={"ball": 1, "table": 2},
        aliases={"ball": "subject", "table": "table"},
    )
    assert result["passed"] is True
    assert result["checks"][1]["displacement_m"] > .3


def test_missing_contact_is_a_failed_task_not_a_candidate_verdict(tmp_path):
    trajectory, contacts = _evidence(tmp_path, with_contact=False)
    result = verify_passive_reference(
        task=_task([{"kind": "contact", "subject": "ball", "object": "table"}]),
        trajectory=trajectory, contacts_path=contacts,
        body_ids={"ball": 1, "table": 2}, aliases={"ball": "subject"},
    )
    assert result["passed"] is False
    assert result["checks"][0]["contact_count"] == 0
