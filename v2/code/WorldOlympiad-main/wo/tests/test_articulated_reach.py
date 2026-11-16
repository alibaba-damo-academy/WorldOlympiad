"""Placement fallback must use the articulation executor's rigid grasp frames."""
from types import SimpleNamespace
import numpy as np
from wo.expert import embodiment as e
from wo.expert import articulate as a
from wo.expert.kin import Pose


def test_articulation_reach_grasps_follow_joint_and_restore(monkeypatch):
    class World:
        support_id = 'table'
        bodies = {'cabinet': 1}
        articulation = {'door': {'body': 1, 'joint': 2, 'host': 'cabinet',
                                 'q_closed': 0., 'q_open': 1.}}
        cid = 0
        q = .23
        def joint_q(self, comp): return self.q
        def set_joint_q(self, comp, q): self.q = q
    w = World()
    w.p = SimpleNamespace(getAABB=lambda *args, **kw: ([1, 0, 1], [1.1, .1, 1.1]))
    monkeypatch.setattr(a, 'rigid_links', lambda *args: {2, 3})
    monkeypatch.setattr(a, 'link_pose', lambda *args: Pose(np.array([w.q, 0, 1]), np.eye(3)))
    calls = []
    offset = Pose(np.array([0, .04, 0]), np.eye(3))
    def sample(world, host, **kwargs):
        calls.append((host, world.q, kwargs))
        return [SimpleNamespace(tcp_in_obj=offset)]
    monkeypatch.setattr(e, 'sample_grasps', sample)
    spec = SimpleNamespace(steps=[])
    ordinary = e._reach_poses(w, spec, (0, 0, 0))
    assert not calls and w.q == .23
    poses = e._reach_poses(w, spec, (0, 0, 0), grasp_fallback=True)
    assert len(calls) == 1 and calls[0][:2] == ('cabinet', 0.)
    assert calls[0][2]['links'] == {2, 3}
    assert calls[0][2]['prefer_links'] == {3}
    assert w.q == .23
    for index, fraction in enumerate((0., .5, .85)):
        assert len(poses[index]) == len(ordinary[index]) + 1
        np.testing.assert_allclose(poses[index][-1].p, [fraction, .04, 1])


def test_articulation_reach_without_grasps_keeps_original_poses(monkeypatch):
    # No certified grasp must not vacuously remove the feature reach gate.
    class World:
        support_id = 'table'
        bodies = {}
        articulation = {'door': {'body': 1, 'joint': 2, 'host': 'cabinet',
                                 'q_closed': 0., 'q_open': 1.}}
        cid = 0
        q = 0.
        def joint_q(self, comp): return self.q
        def set_joint_q(self, comp, q): self.q = q
    w = World()
    w.p = SimpleNamespace(getAABB=lambda *args, **kw: ([1, 0, 1], [1.1, .1, 1.1]))
    monkeypatch.setattr(a, 'rigid_links', lambda *args: {2})
    monkeypatch.setattr(a, 'link_pose', lambda *args: Pose(np.zeros(3), np.eye(3)))
    monkeypatch.setattr(e, 'sample_grasps', lambda *args, **kw: [])
    poses = e._reach_poses(w, SimpleNamespace(steps=[]), (0, 0, 0), grasp_fallback=True)
    assert len(poses) == 3 and all(len(alts) == 4 for alts in poses)
