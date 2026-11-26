from pathlib import Path

from autoresearch.cosmos_task_graph_predicates import NativeTaskGraphPredicates
from wo.tests.test_native_support import body


def test_all_expanded_contents_require_measured_release():
    graph = {
        'controller_program': [{'operator': 'place', 'subjects': ['beads']}],
        'success_program': [{'predicate': 'inside', 'subject': 'beads', 'object': 'bowl'}],
    }
    plan = {'bodies': [body('tray', 'bowl', [.3]*3, 0, True)] +
            [body('bead_' + str(i), 'beads', [.02]*3, 0, False) for i in range(3)]}
    v = NativeTaskGraphPredicates(graph, plan, Path('.'))

    def frame(t, contacts):
        return {'time_s': t, 'objects': {
            'bead_' + str(i): {'position': [i*.03, 0, 0],
                             'orientation_xyzw': [0, 0, 0, 1]}
            for i in range(3)}, 'robot_cup_contact_links': [],
            'robot_subject_contacts': contacts}

    released = {'bead_' + str(i): False for i in range(3)}
    assert not v.update(frame(0, released))
    assert v.update(frame(.6, released))
    assert not v.update(frame(1.2, {**released, 'bead_2': True}))
    # Missing data is NOT a released object, even if the cup is released.
    assert not v.update(frame(1.8, {'bead_0': False, 'bead_1': False}))
    assert not v.update(frame(2.4, released))
    assert v.update(frame(3.0, released))
    outside = frame(3.6, released)
    outside['objects']['bead_2']['position'] = [1, 0, 0]
    assert not v.update(outside)


def test_runner_records_per_subject_contacts_at_physics_cadence():
    source = Path('scripts/run_cosmos3_worldir_isaac_closed_loop.py').read_text()
    assert 'subject_contacts[name] |= _contact_active(sensor)' in source
    assert 'frame["robot_subject_contacts"] = subject_contacts' in source
    assert 'initial_frame["robot_subject_contacts"][name] = _contact_active(sensor)' in source
