"""Ranking must not turn untested certified candidates into infeasibility."""
from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import rollout
from wo.expert.kin import Pose


def setup(monkeypatch, reachable, *, ngrasps=3, ngoals=1):
    pose = Pose.from_pq([0, 0, 1], [0, 0, 0, 1])
    world = SimpleNamespace(
        object_pose=lambda _: (pose.p, pose.q),
        aabb=lambda _: (np.zeros(3), np.ones(3)), arm_q=lambda: np.zeros(7),
        bodies={'subject': 1, 'target': 2}, cid=0, robot=3,
        set_arm=lambda _: None, set_gripper_state=lambda _: None,
        set_link_pose=lambda *a: None,
        p=SimpleNamespace(getClosestPoints=lambda *a, **kw: []))
    grasps = [SimpleNamespace(world=lambda _: pose, kind=str(i),
                              tcp_in_obj=pose, score=1) for i in range(ngrasps)]
    goals = [SimpleNamespace(obj_pose=pose, score=1) for _ in range(ngoals)]
    monkeypatch.setattr(rollout, 'sample_grasps', lambda *a: grasps)
    monkeypatch.setattr(rollout, 'sample_goals', lambda *a, **kw: (goals, {}))
    calls = []

    def ik(*a, **kw):
        calls.append(1)
        return np.zeros(7) if len(calls) in reachable else None

    kin = SimpleNamespace(ik=ik, in_collision=lambda *a, **kw: None,
                          cartesian_path=lambda *a, **kw: [np.zeros(7)],
                          attached=None, ignore=set())
    step = SimpleNamespace(subject='subject', target='target', primitive='transport',
                           goal=SimpleNamespace(target='target'))
    return world, kin, step, calls


def test_collision_branch_fallback_retains_checks(monkeypatch):
    world, kin, step, calls = setup(monkeypatch, set(), ngrasps=1)
    options = []
    def ik(*args, **kwargs):
        options.append(kwargs)
        return np.zeros(7)
    kin.ik = ik
    hits = iter(['table', None, None])
    kin.in_collision = lambda *a, **kw: next(hits)
    plans, info = rollout.plan_transport(world, kin, world, step)
    assert plans
    assert options[1]['collision_free'] is True
    assert info['rejections'] == {'pregrasp_collision:0:table': 1}
    assert info['pregrasp_branch_recoveries'] == 1
    assert kin.ignore == set()


def test_exhausts_remaining_grasps_after_ranked_search_fails(monkeypatch):
    # Initial single-grasp search fails, then first expanded grasp fails;
    # the next certified grasp and its release are reachable.
    world, kin, step, calls = setup(monkeypatch, {3, 4})
    plans, info = rollout.plan_transport(world, kin, world, step, max_grasps=1)
    assert plans[0].grasp.kind == '1'
    assert info['expanded_candidate_search']
    assert info['initial_ranked_search']['feasible_plans'] == 0
    assert len(calls) == 5


def test_exhausts_remaining_goals(monkeypatch):
    # Pregrasp succeeds in both passes; only the second expanded release works.
    world, kin, step, calls = setup(monkeypatch, {1, 4, 7}, ngrasps=1, ngoals=2)
    plans, info = rollout.plan_transport(world, kin, world, step, max_goals=1)
    assert plans
    assert info['expanded_candidate_search']
    assert len(calls) == 7


def test_successful_ranked_search_does_not_expand(monkeypatch):
    world, kin, step, calls = setup(monkeypatch, {1, 2})
    plans, info = rollout.plan_transport(world, kin, world, step, max_grasps=1)
    assert plans
    assert len(calls) == 2
    assert 'expanded_candidate_search' not in info


def test_exhausted_search_fails_without_recursion(monkeypatch):
    world, kin, step, calls = setup(monkeypatch, set())
    with pytest.raises(rollout.StepFailed) as exc:
        rollout.plan_transport(world, kin, world, step, max_grasps=1)
    assert len(calls) == 4
    assert exc.value.evidence['rejections'] == {'pregrasp_ik:0': 1,
                                              'pregrasp_ik:1': 1,
                                              'pregrasp_ik:2': 1}
    assert 'initial_ranked_search' in exc.value.evidence
