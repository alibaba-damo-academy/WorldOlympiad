from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import rollout
from wo.expert.kin import Pose


@pytest.mark.parametrize('blocked', [None, 0.0, 1.0])
def test_overhead_checks_held_object_and_both_apertures(blocked):
    original = {'subject', 'table'}
    kin = SimpleNamespace(attached=None, ignore=original)
    seen = []
    def collision(q, gripper):
        assert kin.attached == ('subject', 'transform')
        assert kin.ignore == set()
        seen.append(gripper)
        return 'table' if gripper == blocked else None
    kin.in_collision = collision
    assert rollout.overhead_release_free(kin, 'subject', 'transform', [0]) == (blocked is None)
    assert seen[0] == 1.0
    if blocked != 1.0:
        assert seen == [1.0, 0.0]
    assert kin.attached is None and kin.ignore is original


def test_reorient_solves_suspended_pose_not_blocked_support(monkeypatch):
    pose = Pose.from_pq([0, 0, 1], [0, 0, 0, 1])
    world = SimpleNamespace(object_pose=lambda _: (pose.p, pose.q),
        aabb=lambda _: (np.zeros(3), np.ones(3)), arm_q=lambda: np.zeros(7),
        bodies={'subject': 1, 'target': 2}, cid=0, robot=3,
        set_arm=lambda _: None, set_gripper_state=lambda _: None,
        set_link_pose=lambda *a: None,
        p=SimpleNamespace(getClosestPoints=lambda *a, **kw: []))
    grasp = SimpleNamespace(world=lambda _: pose, kind='top', tcp_in_obj=pose, score=1)
    monkeypatch.setattr(rollout, 'sample_grasps', lambda *a: [grasp])
    monkeypatch.setattr(rollout, 'sample_goals', lambda *a, **kw:
                        ([SimpleNamespace(obj_pose=pose, score=1)], {}))
    calls = []
    kin = SimpleNamespace(attached=None, ignore=set())
    def ik(target, **kw):
        calls.append(target.p.copy())
        if len(calls) == 1:
            return np.zeros(7)
        if len(calls) < 4:
            return None
        assert kw['accept_solution'](np.ones(7))
        return np.ones(7)
    kin.ik = ik
    kin.in_collision = lambda *a, **kw: None
    kin.cartesian_path = lambda q, *a, **kw: [q]
    step = SimpleNamespace(subject='subject', target='target', primitive='reorient',
                           goal=SimpleNamespace(target='target'))
    plans, info = rollout.plan_transport(world, kin, world, step)
    assert plans[0].release_drop
    assert plans[0].overhead_rise >= rollout.LIFT_M
    assert info['overhead_rises_m'] == [plans[0].overhead_rise]
    assert calls[3][2] > calls[1][2]
    assert info['reorientation_overhead_releases'] == 1
    assert kin.attached is None and kin.ignore == set()


@pytest.mark.parametrize('clear_index', [0, 2, None])
def test_overhead_height_search_is_bounded_and_certified(clear_index):
    pose = Pose.from_pq([0, 0, 1], [0, 0, 0, 1])
    heights = []
    def ik(target, seed, accept_solution):
        heights.append(target.p[2])
        q = len(heights) - 1
        return q if accept_solution(q) else None
    accept = lambda q: clear_index is not None and q >= clear_index
    kin = SimpleNamespace(ik=ik)
    result = rollout.search_overhead_release(kin, pose, .1, None, accept)
    if clear_index is None:
        assert result is None
        assert len(heights) == 4
    else:
        assert result[0] == clear_index
        assert result[1] == pytest.approx(.1 + clear_index * rollout.LIFT_M)
        assert len(heights) == clear_index + 1
    assert heights == pytest.approx([1.1 + i * rollout.LIFT_M for i in range(len(heights))])
