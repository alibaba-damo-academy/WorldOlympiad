"""Release diagnostic evidence must survive state restoration as plain JSON."""
import json
from types import SimpleNamespace

import numpy as np

from wo.expert import rollout


def test_release_snapshot_records_contact_links_poses_and_open_state(monkeypatch):
    cp = (0, 11, 99, -1, 8, (1., 2., 3.), (4., 5., 6.), (0., 0., 1.), -.001, 2.5)
    world = SimpleNamespace(
        bodies={'bead': 11, 'bowl': 12}, robot=99,
        object_pose=lambda oid: (np.array([1., 2., 3.]), np.array([0., 0., 0., 1.])),
        ee_pose=lambda: (np.zeros(3), np.array([0., 0., 0., 1.])),
        gripper_fraction=lambda: 0., contacts=lambda oid: [cp])
    monkeypatch.setattr(rollout.predicates, 'support_below', lambda d, s: 'robot')
    evidence = rollout.release_snapshot(world, 'bead', 'bowl')
    # Must own its values: later simulated-world restoration cannot erase it.
    world.bodies.clear()
    saved = json.loads(json.dumps(evidence))
    assert saved['gripper_fraction'] == 0.
    assert saved['support'] == 'robot'
    assert saved['subject']['position'] == [1., 2., 3.]
    assert saved['contacts'][0]['body'] == 'robot'
    assert saved['contacts'][0]['other_link'] == 8
    assert saved['contacts'][0]['normal_force_n'] == 2.5
    assert saved['contacts'][0]['distance_m'] == -.001


def test_release_snapshot_accepts_no_contact(monkeypatch):
    world = SimpleNamespace(bodies={'obj': 1, 'target': 2}, robot=3,
                            object_pose=lambda oid: ([0, 0, 1], [0, 0, 0, 1]),
                            ee_pose=lambda: ([0, 0, 1], [0, 0, 0, 1]),
                            gripper_fraction=lambda: 1., contacts=lambda oid: [])
    monkeypatch.setattr(rollout.predicates, 'support_below', lambda d, s: None)
    evidence = rollout.release_snapshot(world, 'obj', 'target')
    assert evidence['contacts'] == []
    assert evidence['support'] is None
    json.dumps(evidence)


def test_failed_grasp_preserves_before_and_after_close(monkeypatch):
    import pytest
    world = SimpleNamespace(gripper_fraction=lambda: 0., arm_q=lambda: [0.],
                            grasped=lambda s: False)
    kin = SimpleNamespace(in_collision=lambda *a, **kw: None,
                          plan=lambda *a, **kw: [[0.]])
    executor = SimpleNamespace(d=world, run=lambda *a, **kw: None, settle=lambda *a: None)
    monkeypatch.setattr(rollout, 'Executor', lambda *a: executor)
    monkeypatch.setattr(rollout, 'sync', lambda *a: None)
    monkeypatch.setattr(rollout, 'resolve_target', lambda *a: 'target')
    monkeypatch.setattr(rollout, 'free_actions', lambda x: x)
    monkeypatch.setattr(rollout, 'fine_actions', lambda x, y: x)
    snapshots = iter([{'contacts': [], 'gripper_fraction': 0.},
                      {'contacts': [{'other_link': 11}], 'gripper_fraction': 1.}])
    monkeypatch.setattr(rollout, 'release_snapshot', lambda *a: next(snapshots))
    plan = SimpleNamespace(grasp=SimpleNamespace(kind='top', width=.03),
                           approach_gripper=0., q_pre=[0.], q_grasp=[0.],
                           approach_path=[[0.]])
    with pytest.raises(rollout.StepFailed) as caught:
        rollout.execute_transport(world, object(), kin,
                                  SimpleNamespace(subject='subject'), plan)
    ev = caught.value.evidence
    assert ev['grasp_snapshots']['before_close']['contacts'] == []
    assert ev['grasp_snapshots']['after_close']['contacts'][0]['other_link'] == 11
    json.dumps(ev)
