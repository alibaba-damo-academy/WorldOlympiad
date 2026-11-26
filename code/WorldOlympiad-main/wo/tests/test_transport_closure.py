"""Contact-limited transport closure retains physical grasp checks."""
from types import SimpleNamespace
import numpy as np
from wo.expert.rollout import close_transport_gripper, stabilized_transport_actions, HOLD_FRAMES


def test_stabilized_transport_actions_repeats_each_waypoint():
    qs = [np.array([1.0]), np.array([2.0])]
    out = stabilized_transport_actions(qs, repeats=3)
    assert [float(q[0]) for q in out] == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


def test_stabilized_transport_actions_rejects_nonpositive_repeats():
    try:
        stabilized_transport_actions([np.zeros(1)], repeats=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_closure_ramps_from_measured_aperture_and_holds_full_command():
    calls = []
    world = SimpleNamespace(gripper_fraction=lambda: .25,
                            grasped=lambda subject: subject == 'object')
    ex = SimpleNamespace(d=world, run=lambda qs, **kw: calls.append((qs, kw)))
    q = np.zeros(7)
    trace = close_transport_gripper(ex, q, 'object')
    commands = [kw['gripper'] for _, kw in calls]
    assert len(trace) == 1
    assert commands[0] == commands[-1] == 0.25 + (1.0 - 0.25) / HOLD_FRAMES
    assert len(calls[-1][0]) == HOLD_FRAMES
    assert all(kw['phase'] == 'grasp' for _, kw in calls)
    assert all(item['bilateral'] for item in trace)
    assert all(item['measured_fraction'] == .25 for item in trace)


def test_closure_does_not_claim_or_skip_failed_bilateral_contact():
    calls = []
    ex = SimpleNamespace(d=SimpleNamespace(gripper_fraction=lambda: 0.,
                                          grasped=lambda subject: False),
                         run=lambda qs, **kw: calls.append(kw))
    trace = close_transport_gripper(ex, np.zeros(7), 'object')
    assert not any(item['bilateral'] for item in trace)
    assert calls[-1]['gripper'] == 1.


def test_closure_uses_loaded_contact_when_world_provides_it():
    calls = []
    world = SimpleNamespace(gripper_fraction=lambda: 0.,
                            grasped=lambda subject: True,
                            load_bearing_grasped=lambda subject: False)
    ex = SimpleNamespace(d=world, run=lambda qs, **kw: calls.append(kw))
    trace = close_transport_gripper(ex, np.zeros(7), 'object')
    assert trace and not trace[-1]['load_bearing']
    assert calls[-1]['gripper'] == 1.


def test_closure_retries_bounded_tightening_after_settling_slip():
    calls = []
    state = {'fraction': 0.0, 'checks': 0}
    def loaded(subject):
        state['checks'] += 1
        # Contact is found during the ramp, then is lost during the settling
        # hold, and becomes load-bearing again after tightening.
        return state['checks'] == 2 or state['fraction'] >= .8
    def fraction():
        return state['fraction']
    def run(qs, **kw):
        state['fraction'] = float(kw['gripper'])
        calls.append(kw)
    ex = SimpleNamespace(d=SimpleNamespace(gripper_fraction=fraction,
                                             grasped=lambda subject: True,
                                             load_bearing_grasped=loaded),
                         run=run)
    trace = close_transport_gripper(ex, np.zeros(7), 'object')
    assert any(item.get('retention_pass') for item in trace)
    assert trace[-1]['load_bearing']
