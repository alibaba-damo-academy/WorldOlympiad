from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import push
from wo.expert.kin import Pose
from wo.expert.rollout import StepFailed
from wo.spec import Step, Predicate


class World:
    def __init__(self):
        self.bodies = {'book': 1, 'mat': 2}
        self.pos = np.array([0., 0., .02])
        self.p = self
        self.cid = 0
        self.pushes = 0

    def object_pose(self, oid):
        return (self.pos.copy() if oid == 'book' else np.array([.25, 0., .02])), np.array([0., 0., 0., 1.])

    def getBasePositionAndOrientation(self, body, **kwargs):
        return self.object_pose('book' if body == 1 else 'mat')

    def aabb(self, oid):
        return self.pos - [.02, .02, .02], self.pos + [.02, .02, .02]

    def arm_q(self):
        return np.zeros(7)

    def ee_pose(self):
        return [0., 0., .3], [0., 0., 0., 1.]

    def act(self, q):
        pass


@pytest.mark.parametrize('mode', ['valid_reuse', 'reenter', 'no_entry'])
def test_reused_stroke_failure_reenters_checked_path_without_undoing_push(monkeypatch, mode):
    world = World()
    calls = []
    checks = {}

    class Executor:
        target = np.zeros(7)
        def __init__(self, *args):
            pass
        def run(self, actions, **kwargs):
            pass
        def settle(self, q):
            pass

    def checked(kin, q, start, end, subject, target, ev, segment):
        checks[segment] = checks.get(segment, 0) + 1
        calls.append(('stroke', segment))
        if segment == 1 and mode != 'valid_reuse' and checks[segment] == 1:
            ev.setdefault('stroke_checks', []).append({'segment': segment, 'reason': 'ik'})
            return None
        return [q, q]

    def separate(*args):
        calls.append(('separate', world.pushes))
        np.testing.assert_allclose(world.pos, [.04, 0., .02])
        return True

    def entry(kin, q, start, end, subject, target, ev, segment):
        calls.append(('entry', segment))
        if mode == 'no_entry' and segment == 1:
            return None
        return start, end, q, [q], [q]

    def feedback(ex, d, subject, actions, before, direction, boxes):
        world.pos[0] += .04
        world.pushes += 1
        return {'executed_actions': 1}

    monkeypatch.setattr(push, 'Executor', Executor)
    monkeypatch.setattr(push, 'sample_goals', lambda *a, **k: ([SimpleNamespace(obj_pose=Pose(np.array([.25, 0., .02]), np.eye(3)))], {}))
    monkeypatch.setattr(push.predicates, 'evaluate', lambda *a, **k: (world.pushes >= 2, {}))
    monkeypatch.setattr(push, 'sync', lambda *a: None)
    monkeypatch.setattr(push, '_push_obstacles', lambda *a: [])
    monkeypatch.setattr(push, '_push_support_bounds', lambda *a: (np.array([-1., -1.]), np.array([1., 1.])))
    monkeypatch.setattr(push, '_contact_radius', lambda *a: .02)
    monkeypatch.setattr(push, '_needs_push_entry', lambda *a: False)
    monkeypatch.setattr(push, '_checked_push_line', checked)
    monkeypatch.setattr(push, '_plan_push_entry', entry)
    monkeypatch.setattr(push, '_separate_pusher', separate)
    monkeypatch.setattr(push, 'feedback_push', feedback)
    monkeypatch.setattr(push, 'fine_actions', lambda *a: [])
    monkeypatch.setattr(push, 'free_actions', lambda *a: [])
    kin = SimpleNamespace(ignore=set(), cartesian_path=lambda *a, **k: None)
    step = Step('push', 'book', Predicate('on_region', 'book', 'mat'), target='mat')
    if mode == 'no_entry':
        with pytest.raises(StepFailed, match='no checked entry and stroke frame'):
            push.run_push(world, world, kin, world, step)
        assert world.pushes == 1 and checks[1] == 1
        np.testing.assert_allclose(world.pos, [.04, 0., .02])
    else:
        evidence = push.run_push(world, world, kin, world, step)
        assert world.pushes == 2
        assert checks[1] == (1 if mode == 'valid_reuse' else 2)
        assert evidence.get('reuse_stroke_reentry', []) == ([] if mode == 'valid_reuse' else [1])
    if mode == 'valid_reuse':
        assert ('separate', 1) not in calls and ('entry', 1) not in calls
    else:
        assert calls.index(('stroke', 1)) < calls.index(('separate', 1)) < calls.index(('entry', 1))
