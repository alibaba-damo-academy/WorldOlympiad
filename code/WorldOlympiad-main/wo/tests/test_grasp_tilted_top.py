"""Supplemental rim-clearance frames do not alter ordinary grasp searches."""
import numpy as np
import pybullet as p
from wo.expert import grasps as G
from wo.tests.test_grasp_receptacle import World


def test_tilted_top_frames_are_bounded_and_orthonormal():
    w = World()
    try:
        lo, hi = w.aabb('subject')
        ordinary = G._candidates(w, w.body, lo, hi, None)
        tilted = G._candidates(w, w.body, lo, hi, None, tilted_top=True)
        tops = [c for c, r, kind in ordinary if kind == 'top']
        assert len(tilted) == 4 * len(tops)
        for c, r, kind in tilted:
            assert kind == 'tilted_top'
            assert np.allclose(r.T @ r, np.eye(3))
            assert np.isclose(np.linalg.det(r), 1)
            assert r[2, 2] < -.7
            assert abs(r[2, 0]) < 1e-9
            assert any(np.allclose(c, pos) for pos in tops)
        assert G.sample_grasps(w, 'subject', tilted_top=True)
    finally:
        p.disconnect(w.cid)


def test_transport_tilt_fallback_only_after_checked_top_collision(monkeypatch):
    from wo.expert import rollout
    from wo.tests.test_transport_candidate_search import setup
    world, kin, step, _ = setup(monkeypatch, {1}, ngrasps=1)
    original_sampler = rollout.sample_grasps
    ordinary = original_sampler(world, 'subject')
    ordinary[0].kind = 'top'
    ordinary[0].width = 0.03
    monkeypatch.setattr(rollout, 'approach_fraction', lambda *a: 0.0)
    selections = []
    active = [False]
    def sampler(*args, **kwargs):
        active[0] = kwargs.get('tilted_top', False)
        selections.append(active[0])
        return ordinary
    monkeypatch.setattr(rollout, 'sample_grasps', sampler)
    def ik(*args, **kwargs):
        q = np.zeros(7)
        accept = kwargs.get('accept_solution')
        return q if accept is None or accept(q) else None
    kin.ik = ik
    def path(*args, **kwargs):
        if not active[0]:
            kwargs.get('diagnostics', {}).update(reason='collision', collision='cup')
            return None
        return [np.zeros(7)]
    kin.cartesian_path = path
    plans, info = rollout.plan_transport(world, kin, world, step)
    assert plans and selections == [False, False, True]
    assert info['tilted_top_fallback']
    assert info['ordinary_search']['rejections'] == {'approach_path': 1}
    assert kin.ignore == set()


def test_tilted_sampler_retains_antipodal_rejection(monkeypatch):
    w = World()
    try:
        calls = []
        def reject(*args, **kwargs):
            calls.append(kwargs['contact_reach'])
            return None
        monkeypatch.setattr(G, '_antipodal', reject)
        assert G.sample_grasps(w, 'subject', tilted_top=True) == []
        assert G.REACH_M in calls and G.PAD_HALF_OPEN_M in calls
    finally:
        p.disconnect(w.cid)
