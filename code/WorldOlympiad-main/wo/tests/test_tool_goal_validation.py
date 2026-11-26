from wo.expert import rollout
from wo.spec import Predicate, Step


def test_tool_goal_validation_keeps_declared_predicate_evidence(monkeypatch):
    step = Step("tool_contact", "brush", Predicate("on_region", "scrap", "pan"),
                target="scrap")
    monkeypatch.setattr(rollout.predicates, "evaluate",
                        lambda world, goal, **kwargs: (False, {"inside": False}))
    ok, evidence = rollout.verify_tool_goal(object(), step, {"peak_contact_force_n": 2.0})
    assert not ok
    assert evidence["peak_contact_force_n"] == 2.0
    assert evidence["predicate"] == {"inside": False}


def test_tool_contact_run_steps_uses_verified_event_evidence(monkeypatch):
    """Exercise the production tool branch, not only the helper."""
    step = Step("tool_contact", "brush", Predicate("contact", "brush", "scrap"),
                target="scrap")

    class Physics:
        def saveState(self, **kwargs): return 1
        def restoreState(self, *args, **kwargs): pass

    class WorldStub:
        bodies = {"brush": 1, "scrap": 2}
        frames = []
        cid = 0
        p = Physics()
        step_index = 0
        def object_pose(self, oid): return ((0, 0, 0), (0, 0, 0, 1))

    class KinStub:
        def __init__(self, world): pass

    monkeypatch.setattr(rollout, "Kinematics", KinStub)
    monkeypatch.setattr(rollout, "sync", lambda *args: None)
    import wo.expert.tool as tool_module
    monkeypatch.setattr(tool_module, "run_strike", lambda *args: {"peak_contact_force_n": 2.0})
    monkeypatch.setattr(rollout, "verify_tool_goal",
                        lambda *args, **kwargs: (True, {"predicate": {"contact": True}}))
    reports = rollout.run_steps(WorldStub(), WorldStub(), lambda _: None, (step,))
    assert reports[0].passed
    assert reports[0].evidence["predicate"] == {"contact": True}


def test_run_strike_does_not_evaluate_endpoint_at_transient_contact(monkeypatch):
    """The real strike loop must leave endpoint certification to run_steps."""
    import numpy as np
    import wo.expert.tool as tool_module

    step = Step("tool_contact", "brush", Predicate("pose_delta", "scrap"), target="scrap")

    class P:
        def __init__(self, p=(0.0, 0.0, 0.0), R=None):
            self.p = np.asarray(p, dtype=float)
            self.R = np.eye(3) if R is None else R
        @classmethod
        def from_pq(cls, p, q): return cls(p)
        def inv(self): return self
        def __matmul__(self, other): return P(self.p + other.p)
        def moved(self, delta): return P(self.p + np.asarray(delta))

    class Physics:
        def getBasePositionAndOrientation(self, *args, **kwargs):
            return ((0, 0, 0), (0, 0, 0, 1))

    class W:
        bodies = {"brush": 1, "scrap": 2}; cid = 0; p = Physics()
        def object_pose(self, oid): return ((0, 0, 0), (0, 0, 0, 1))
        def aabb(self, oid): return (np.zeros(3), np.ones(3))
        def arm_q(self): return np.zeros(2)
        def ee_pose(self): return ((0, 0, 0), (0, 0, 0, 1))
        def grasped(self, oid): return True
        def contacts(self, a, b): return [(None,) * 9 + (2.0,)]
        def act(self, q): pass

    class G:
        tcp_in_obj = P((0.0, 0.0, 0.0)); score = 0; kind = "handle"; width = 0.02
        def world(self, obj): return P()

    class K:
        p = Physics(); bodies = W.bodies; cid = 0; ignore = set()
        def object_pose(self, oid): return ((0, 0, 0), (0, 0, 0, 1))
        def aabb(self, oid): return (np.zeros(3), np.ones(3))
        def arm_q(self): return np.zeros(2)
        def ik(self, *args, **kwargs): return np.zeros(2)
        def in_collision(self, q): return None
        def cartesian_path(self, *args, **kwargs): return [np.zeros(2), np.zeros(2)]
        def plan(self, *args, **kwargs): return [np.zeros(2), np.zeros(2)]

    class E:
        def __init__(self, *args): pass
        def run(self, *args, **kwargs): pass
        def settle(self, *args, **kwargs): pass

    monkeypatch.setattr(tool_module, "Pose", P)
    monkeypatch.setattr(tool_module, "local_aabb", lambda *args: (np.zeros(3), np.ones(3)))
    monkeypatch.setattr(tool_module, "sample_grasps", lambda *args: [G()])
    monkeypatch.setattr(tool_module, "tcp_to_ee", lambda x: x)
    monkeypatch.setattr(tool_module, "Executor", E)
    monkeypatch.setattr(tool_module, "sync", lambda *args: None)
    monkeypatch.setattr(tool_module, "fine_actions", lambda qs, *args: qs)
    monkeypatch.setattr(tool_module.predicates, "evaluate",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("endpoint evaluated during contact")))

    ev = tool_module.run_strike(W(), K(), K(), step)
    assert ev["peak_contact_force_n"] == 2.0
    assert "contact_predicate" not in ev


def test_tool_pose_delta_passes_initial_state_to_predicate(monkeypatch):
    step = Step("tool_contact", "brush", Predicate("pose_delta", "scrap"), target="scrap")
    seen = {}
    monkeypatch.setattr(rollout.predicates, "evaluate",
                        lambda world, goal, **kwargs: (seen.setdefault("initial", kwargs["initial"]) is not None, {}))
    ok, _ = rollout.verify_tool_goal(object(), step, {}, initial={"scrap": ((0, 0, 0), (0, 0, 0, 1))})
    assert ok and seen["initial"]
