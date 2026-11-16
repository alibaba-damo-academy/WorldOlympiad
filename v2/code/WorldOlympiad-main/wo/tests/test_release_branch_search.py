from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import rollout
from wo.expert.kin import Pose


@pytest.mark.parametrize('block', ['none', 'collision', 'retreat'])
def test_release_fallback_certifies_endpoint_and_retreat(monkeypatch, block):
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
    options = []
    alternate = np.ones(7)
    def ik(*args, **kwargs):
        options.append(kwargs)
        if len(options) == 1:
            return np.zeros(7)
        if len(options) == 2:
            return None
        assert 'accept_solution' in kwargs
        assert kin.ignore == {'subject'}
        return alternate if kwargs['accept_solution'](alternate) else None
    kin = SimpleNamespace(ik=ik, attached=None, ignore=set())
    kin.in_collision = lambda q, **kw: ('table' if block == 'collision' and np.all(q == 1) else None)
    kin.cartesian_path = lambda q, *a, **kw: (None if block == 'retreat' and np.all(q == 1) else [q])
    step = SimpleNamespace(subject='subject', target='target', primitive='transport', goal=SimpleNamespace(target='target'))
    if block == 'none':
        plans, info = rollout.plan_transport(world, kin, world, step)
        assert np.array_equal(plans[0].q_release, alternate)
        assert info['release_branch_recoveries'] == 1
        assert info['recovered_release_rejections'] == ['release_ik']
    else:
        with pytest.raises(rollout.StepFailed) as exc:
            rollout.plan_transport(world, kin, world, step)
        assert exc.value.evidence['rejections']['release_ik'] == 1
    assert kin.ignore == set()
