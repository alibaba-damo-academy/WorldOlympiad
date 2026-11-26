import numpy as np

from wo import predicates
from wo.spec import Predicate, Step, TaskSpec
from wo.expert.run import audit_initial_goals, SETTLED_ENDPOINT_PRIMITIVES
from wo.expert.goals import candidate_orientations
from wo.expert.kin import Pose


class _World:
    def aabb(self, _):
        return np.zeros(3), np.ones(3)
    def object_pose(self, _):
        return np.zeros(3), (0.0, 0.0, 0.0, 1.0)
    def articulation_fraction(self, _):
        return 0.0


def test_upright_can_use_measured_semantic_asset_axis():
    w = _World()
    p = Predicate(kind="upright", subject="bottle", params={"semantic_up_axis": [0, 1, 0]})
    ok, evidence = predicates.evaluate(w, p)
    assert not ok
    assert evidence["semantic_up_axis"] == [0.0, 1.0, 0.0]


def test_initial_goal_audit_rejects_pre_solved_endpoint():
    w = _World()
    step = Step("reorient", "bottle", Predicate("upright", "bottle"))
    findings = audit_initial_goals(w, type("Spec", (), {"steps": (step,)})(), {})
    assert findings[0]["passed"] is True


def test_initial_goal_audit_ignores_contact_events():
    w = _World()
    step = Step("tool_contact", "brush", Predicate("contact", "brush", "crumb"))
    assert audit_initial_goals(w, type("Spec", (), {"steps": (step,)})(), {}) == []


def test_inverted_orientation_uses_declared_semantic_axis():
    step = Step("reorient", "bottle", Predicate(
        "inverted", "bottle", params={"semantic_up_axis": [0, 1, 0]}))
    rotations = candidate_orientations(step, Pose(np.zeros(3), np.eye(3)), n_yaw=1)
    assert np.allclose(rotations[0] @ np.array([0., 1., 0.]), [0., 0., -1.])


def test_settled_endpoint_includes_push():
    assert "push" in SETTLED_ENDPOINT_PRIMITIVES


def test_step_endpoint_applies_keep_upright_to_non_upright_goal():
    w = _World()
    # The mock is enough to exercise the shared terminal constraint path.
    step = Step("reorient", "bottle", Predicate(
        "upright", "bottle", params={"semantic_up_axis": [0, 1, 0]}),
        constraints={"keep_upright": True})
    ok, evidence = predicates.evaluate_step(w, step)
    assert not ok
    assert evidence["upright_cos"] == 0.0


def test_multi_step_audit_reports_later_preexisting_goal_without_classifying_task():
    w = _World()
    steps = (
        Step("articulate", "door", Predicate("joint_open", "door")),
        Step("articulate", "door", Predicate("joint_closed", "door")),
    )
    findings = audit_initial_goals(w, type("Spec", (), {"steps": steps})(), {})
    assert [item["passed"] for item in findings] == [False, True]
