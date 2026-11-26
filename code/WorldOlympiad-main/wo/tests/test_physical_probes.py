"""Task-physics probes: measured quantities, not the judge's confidence."""

import numpy as np

from wo.eval import physical_probes as P
from wo.spec import Predicate, Step, TaskObject, TaskSpec


def _t(n, hz=15.0):
    return np.arange(n) / hz


def _spec(primitive, subject="cube", target="bowl", kind="inside"):
    return TaskSpec(
        case_id="unit", family="unit", instruction="x",
        objects=(TaskObject(object_id=subject, category="cube", role="manipuland", description="d"),
                 TaskObject(object_id=target, category="container", role="destination", description="d"),
                 TaskObject(object_id="table", category="table", role="support", description="d")),
        steps=(Step(primitive=primitive, subject=subject, target=target,
                    goal=Predicate(kind=kind, subject=subject, target=target)),),
        seed=0)


def _carried(n=80, rise=0.10):
    """A clean pick and place: rise, carry rigidly, lower, then rest."""
    k = n // 4
    z = np.concatenate([np.linspace(0, rise, k), np.full(k, rise),
                        np.linspace(rise, 0.0, k), np.zeros(n - 3 * k)])
    x = np.concatenate([np.zeros(k), np.linspace(0, 0.2, k), np.full(n - 2 * k, 0.2)])
    obj = np.stack([x, np.zeros(n), z], axis=1)
    ee = obj + np.array([0, 0, 0.02])
    return _t(n), obj, ee


def test_clean_transport_scores_near_one():
    t, obj, ee = _carried()
    probes = P.probes_for_spec(_spec("transport"), {"cube": obj, "end_effector": ee}, t)
    agg = P.aggregate(probes)
    assert agg["status"] == "scored"
    assert agg["per_probe"]["lift_rise"] == 1.0
    assert agg["per_probe"]["carry_attachment"] > 0.95
    assert agg["per_probe"]["release_settling"] > 0.9
    assert agg["score"] > 0.9


def test_object_that_never_rises_fails_the_lift_probe():
    t = _t(40)
    obj = np.zeros((40, 3))
    ee = obj + np.array([0, 0, 0.3])
    agg = P.aggregate(P.probes_for_spec(_spec("transport"), {"cube": obj, "end_effector": ee}, t))
    assert agg["per_probe"]["lift_rise"] == 0.0
    assert agg["subscores"]["task_events"] < 0.6


def test_slipping_grasp_loses_attachment_but_keeps_the_lift():
    t, obj, ee = _carried()
    rng = np.random.default_rng(0)
    slip = obj + np.concatenate([np.zeros((22, 3)), rng.normal(0, 0.05, (len(obj) - 22, 3))])
    agg = P.aggregate(P.probes_for_spec(_spec("transport"), {"cube": slip, "end_effector": ee}, t))
    assert agg["per_probe"]["lift_rise"] == 1.0
    assert agg["per_probe"]["carry_attachment"] < 0.5


def test_free_fall_is_not_a_teleport_but_a_jump_across_the_room_is():
    t = _t(30)
    drop = np.clip(0.8 - 4.9 * (t ** 2), 0.0, None)
    g = np.stack([np.zeros(30), np.zeros(30), drop], axis=1)
    assert P.no_teleport({"cube": g}, t)["score"] > 0.99
    blown = g.copy()
    blown[20:] += np.array([30.0, 0, 0])
    assert P.no_teleport({"cube": blown}, t)["score"] < 0.05


def test_untouched_object_may_fall_but_may_not_slide():
    t = _t(40)
    ee = np.tile(np.array([1.0, 1.0, 0.5]), (40, 1))
    falling = np.stack([np.zeros(40), np.zeros(40), np.linspace(0.3, 0.0, 40)], axis=1)
    assert P.no_spontaneous_motion({"cube": falling, "end_effector": ee}, t)["score"] > 0.9
    sliding = np.stack([np.linspace(0, 0.3, 40), np.zeros(40), np.zeros(40)], axis=1)
    assert P.no_spontaneous_motion({"cube": sliding, "end_effector": ee}, t)["score"] < 0.1


def test_articulated_parts_are_exempt_from_the_free_body_probes():
    t = _t(40)
    ee = np.tile(np.array([1.0, 1.0, 0.5]), (40, 1))
    swinging = np.stack([np.linspace(0, 0.4, 40), np.zeros(40), np.zeros(40)], axis=1)
    tracks = {"door": swinging, "end_effector": ee}
    assert P.no_spontaneous_motion(tracks, t, {"door"})["score"] is None
    assert P.no_teleport(tracks, t, {"door"})["score"] > 0.99


def test_joint_chatter_scores_below_a_single_sweep():
    t = _t(40)
    clean = np.linspace(0, 1.5, 40)
    assert P.joint_monotonic(clean, t)["score"] > 0.99
    chatter = clean + 0.3 * np.sin(np.arange(40))
    assert P.joint_monotonic(chatter, t)["score"] < 0.7
    assert P.joint_monotonic(np.zeros(40), t)["score"] is None


def test_push_requires_contact_first_and_the_goal_direction():
    t = _t(40)
    obj = np.stack([np.linspace(0, 0.2, 40), np.zeros(40), np.zeros(40)], axis=1)
    near = obj - np.array([0.05, 0, 0])
    far = np.tile(np.array([2.0, 2.0, 0.5]), (40, 1))
    assert P.contact_before_motion(obj, near, t, name="p")["score"] > 0.9
    assert P.contact_before_motion(obj, far, t, name="p")["score"] < 0.1
    assert P.push_direction(obj, np.array([0.5, 0, 0]))["score"] > 0.95
    assert P.push_direction(obj, np.array([-0.5, 0, 0]))["score"] < 0.05


def test_unmeasurable_probes_are_excluded_not_counted_as_zero():
    probes = [P._probe("a", 1.0, {}), P._probe("no_teleport", None, {})]
    agg = P.aggregate(probes)
    assert agg["score"] == 1.0
    assert agg["measured_probe_count"] == 1 and agg["probe_count"] == 2
    assert agg["subscores"]["plausibility"] is None
