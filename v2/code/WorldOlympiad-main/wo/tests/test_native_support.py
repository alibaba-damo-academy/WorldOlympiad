from pathlib import Path

import pytest

from autoresearch.cosmos_task_graph_predicates import NativeTaskGraphPredicates
from wo.models.cosmos import task_graph
from wo.spec import Predicate, Step, TaskObject, TaskSpec
from wo.models.cosmos import Unsupported


def body(name, identity, size, z, static):
    geom = {'kind': 'box', 'size': size}
    return {'name': name, 'static': static,
            'metadata': {'task_object_id': identity},
            'pose': {'xyz': [0, 0, z], 'quat_xyzw': [0, 0, 0, 1]},
            'links': [{'visuals': [geom], 'collisions': [geom]}]}


@pytest.mark.parametrize('kind', ['on', 'stacked', 'on_region'])
def test_native_support_measurement(kind):
    spec = TaskSpec('support', 'stacking', 'Place block on base',
                    (TaskObject('block', 'block', 'manipuland'),
                     TaskObject('base', 'block', 'destination')),
                    (Step('transport', 'block', Predicate(kind, 'block', 'base')),))
    graph = task_graph(spec)
    verifier = NativeTaskGraphPredicates(graph, {'bodies': [
        body('cup', 'block', [.04, .04, .04], .07, False),
        body('tray', 'base', [.10, .10, .10], 0, True)]}, Path('.'))
    def frame(t, x=0, z=.07, held=False):
        return {'time_s': t, 'cup': {'position': [x, 0, z],
                                    'orientation_xyzw': [0, 0, 0, 1]},
                'robot_cup_contact_links': ['finger'] if held else []}
    assert not verifier.update(frame(0))  # stable dwell required
    assert verifier.update(frame(.6))
    assert not verifier.update(frame(1.2, x=.20))  # beside base
    assert not verifier.update(frame(1.8, z=.10))  # hovering
    assert not verifier.update(frame(2.4, held=True))  # no release


def test_unrepresented_native_primitive_is_not_applicable():
    spec = TaskSpec('art', 'articulation', 'Open door',
                    (TaskObject('door', 'door', 'manipuland'),),
                    (Step('articulate', 'door', Predicate('joint_open', 'door', None)),))
    with pytest.raises(Unsupported, match='cannot represent primitive'):
        task_graph(spec)
