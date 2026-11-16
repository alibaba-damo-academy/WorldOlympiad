from pathlib import Path
import numpy as np
import pytest
from wo.models.cosmos import task_graph
from wo.spec import TaskSpec, TaskObject, Step, Predicate
from wo.tasks.scene import legacy_task
from autoresearch.cosmos_task_graph_predicates import NativeTaskGraphPredicates
from wo.tests.test_native_support import body


def spec(primitive, kind):
    return TaskSpec('motion', primitive, 'Move the block',
                    (TaskObject('block', 'block', 'manipuland'),
                     TaskObject('base', 'region', 'destination', 'yellow 18 cm x 15 cm'),
                     TaskObject('table', 'table', 'support')),
                    (Step(primitive, 'block', Predicate(kind, 'block', 'base')),))


def test_goal_regions_are_physical_markers():
    task = legacy_task(spec('push', 'on_region'))
    goal = next(o for o in task['objects'] if o['object_id'] == 'base')
    assert goal['category'] == 'mat'
    assert '18 cm x 15 cm' in goal['visual_description']
    assert '1 mm thick' in goal['visual_description']


def test_push_has_no_pick_and_rejects_lift_place():
    graph = task_graph(spec('push', 'on_region'))
    assert [o['operator'] for o in graph['controller_program']] == ['push']
    v = NativeTaskGraphPredicates(graph, {'bodies': [
        body('cup', 'block', [.04]*3, .07, False),
        body('tray', 'base', [.10]*3, 0, True)]}, Path('.'))
    def frame(t):
        return {'time_s': t, 'cup': {'position': [0,0,.07],
                'orientation_xyzw': [0,0,0,1]}, 'robot_cup_contact_links': []}
    assert not v.update(frame(0))
    assert v.update(frame(.6))
    v.lifted.add('cup')
    assert not v.update(frame(1.2))


@pytest.mark.parametrize('kind,axis', [('upright', [0,0,1]), ('inverted', [0,0,-1])])
def test_reorientation_requires_lift_and_declared_orientation(kind, axis):
    graph = task_graph(spec('reorient', kind))
    goal = graph['success_program'][0]
    assert goal['terminal_upright_axis_world'] == axis
    assert goal['reorientation_required']
    v = NativeTaskGraphPredicates(graph, {'bodies': [
        body('cup', 'block', [.04]*3, .07, False),
        body('tray', 'base', [.10]*3, 0, True)]}, Path('.'))
    q = [0,0,0,1] if kind == 'upright' else [1,0,0,0]
    def frame(t, quat=q):
        return {'time_s': t, 'cup': {'position': [0,0,.07],
                'orientation_xyzw': quat}, 'robot_cup_contact_links': []}
    assert not v.update(frame(0))
    assert not v.update(frame(.6))  # initially oriented is not a manipulation
    v.lifted.add('cup')
    assert not v.update(frame(1.2))
    assert v.update(frame(1.8))
    assert not v.update(frame(2.4, [0, np.sqrt(.5), 0, np.sqrt(.5)]))
