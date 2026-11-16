from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import rollout
from wo.expert.kin import Pose


def test_transport_preserves_checked_approach_rejection(monkeypatch):
    pose = Pose.from_pq([0, 0, 1], [0, 0, 0, 1])
    world = SimpleNamespace(object_pose=lambda _: (pose.p, pose.q),
                            aabb=lambda _: (np.zeros(3), np.ones(3)),
                            arm_q=lambda: np.zeros(7), bodies={'target': 1})
    grasp = SimpleNamespace(world=lambda _: pose, kind='top', width=0.03)
    monkeypatch.setattr(rollout, 'sample_grasps', lambda *a, **kw: [grasp])
    monkeypatch.setattr(rollout, 'approach_fraction', lambda *a: 0.0)
    monkeypatch.setattr(rollout, 'sample_goals', lambda *a, **kw: ([object()], {}))
    kin = SimpleNamespace(ik=lambda *a, **kw: np.zeros(7),
                          in_collision=lambda *a, **kw: None, attached=None, ignore=set())

    def path(*args, **kwargs):
        assert kwargs['gripper'] == 0.0
        assert 'check' not in kwargs
        assert kin.ignore == {'subject'}
        kwargs['diagnostics'].update(reason='collision', body='cup', progress=0.5)
        return None

    kin.cartesian_path = path
    def ik(*args, **kwargs):
        if 'accept_solution' in kwargs:
            assert kin.ignore == set()
            assert not kwargs['accept_solution'](np.zeros(7))
            assert kin.ignore == set()
            return None
        return np.zeros(7)
    kin.ik = ik
    step = SimpleNamespace(subject='subject', target='target',
                           goal=SimpleNamespace(target='target'))
    with pytest.raises(rollout.StepFailed) as exc:
        rollout.plan_transport(world, kin, world, step, _aperture=True, _tilted_top=True)
    assert exc.value.evidence['approach_failures'] == [
        {'grasp_kind': 'top', 'reason': 'collision', 'body': 'cup', 'progress': 0.5}]
    assert exc.value.evidence['rejections'] == {'approach_path': 1}
    assert kin.ignore == set()
