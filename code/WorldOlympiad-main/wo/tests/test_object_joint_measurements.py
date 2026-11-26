from types import SimpleNamespace
import pytest
from wo.sim.joint_state import measured_object_joints
from wo.models.cosmos import _publish_initial_joint_measurements


def physics(names=(b'slide', b'hinge', b'weld'), values=(.012, -.4, 0)):
    return SimpleNamespace(JOINT_REVOLUTE=0, JOINT_PRISMATIC=1,
        getNumJoints=lambda *a, **kw: 3,
        getJointInfo=lambda body, j, **kw: (j, names[j], (1, 0, 4)[j]),
        getJointState=lambda body, j, **kw: (values[j],))


def test_measures_all_passive_movable_joints():
    assert measured_object_joints(physics(), 10, 3) == {'slide': .012, 'hinge': -.4}


@pytest.mark.parametrize('p', [physics(values=(float('nan'), 0, 0)),
                              physics(names=(b'x', b'x', b'weld'))])
def test_invalid_measurements_fail_closed(p):
    with pytest.raises(ValueError):
        measured_object_joints(p, 10, 3)


def test_bundle_preserves_measurement_and_provenance_not_legacy_zero():
    records = {'b': {}, 'legacy': {}}
    _publish_initial_joint_measurements({'frames': [{'objects': {
        'b': {'joint_positions': {'slide': .012, 'hinge': -.4}}}}]}, records)
    assert records['b']['initial_joint_positions'] == {'slide': .012, 'hinge': -.4}
    assert records['b']['initial_joint_state_source'].endswith('objects.b.joint_positions')
    assert records['legacy'] == {}


def test_bundle_invalid_state_rejected():
    with pytest.raises(ValueError):
        _publish_initial_joint_measurements({'frames': [{'objects': {
            'b': {'joint_positions': {'slide': float('inf')}}}}]}, {'b': {}})
