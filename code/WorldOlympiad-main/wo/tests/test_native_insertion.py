from pathlib import Path

from autoresearch.cosmos_task_graph_predicates import NativeTaskGraphPredicates
from wo.models.cosmos import task_graph
from wo.spec import TaskSpec, TaskObject, Step, Predicate
from wo.tests.test_native_support import body


def test_insertion_live_depth_release_and_stability():
    spec = TaskSpec('insert', 'insertion', 'Insert peg',
                    (TaskObject('peg', 'peg', 'manipuland'),
                     TaskObject('socket', 'socket', 'destination')),
                    (Step('transport', 'peg', Predicate('inserted', 'peg', 'socket')),))
    graph = task_graph(spec)
    assert graph['success_program'][0]['predicate'] == 'inserted'
    assert graph['controller_program'][-1]['operator'] == 'insert'
    v = NativeTaskGraphPredicates(graph, {'bodies': [
        body('cup', 'peg', [.04, .04, .10], 0, False),
        body('tray', 'socket', [.08, .08, .05], 0, True)]}, Path('.'))

    def frame(t, x=0, z=.04, held=False):
        return {'time_s': t, 'cup': {'position': [x, 0, z],
                'orientation_xyzw': [0, 0, 0, 1],
                'linear_velocity': [0, 0, 0], 'angular_velocity': [0, 0, 0]},
                'robot_cup_contact_links': ['finger'] if held else []}

    assert not v.update(frame(0))
    assert v.update(frame(.6))
    ev = v.last_evidence['geometry']['0']['cup']
    assert ev['insertion_depth_m'] > ev['required_depth_m']
    assert not v.update(frame(1.2, z=.05))  # only 25mm of required 30mm
    assert not v.update(frame(1.8, x=.1))  # deep but outside socket
    assert not v.update(frame(2.4, held=True))
    missing = frame(3.0)
    del missing['robot_cup_contact_links']
    assert not v.update(missing)  # fail closed for unmeasured release
    assert not v.update(frame(3.6))
    assert v.update(frame(4.2))


def test_insertion_uses_collision_not_decorative_visual_bounds():
    graph = {'controller_program': [{'operator': 'insert', 'subjects': ['peg']}],
             'success_program': [{'predicate': 'inserted', 'subject': 'peg', 'object': 'socket'}]}
    peg = body('cup', 'peg', [.04, .04, .10], 0, False)
    target = body('tray', 'socket', [.08, .08, .05], 0, True)
    target['links'][0]['visuals'] = [{'kind': 'box', 'size': [.08, .08, 1.0]}]
    v = NativeTaskGraphPredicates(graph, {'bodies': [peg, target]}, Path('.'))
    states = {'cup': {'position': [0, 0, .1], 'orientation_xyzw': [0,0,0,1]},
              'tray': {'position': [0,0,0], 'orientation_xyzw': [0,0,0,1]}}
    passed, ev = v.inserted('cup', 'tray', states)
    assert not passed
    assert ev['insertion_depth_m'] < 0
