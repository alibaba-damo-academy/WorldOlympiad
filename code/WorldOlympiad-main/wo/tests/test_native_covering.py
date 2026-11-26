from pathlib import Path
from autoresearch.cosmos_task_graph_predicates import NativeTaskGraphPredicates
from wo.tests.test_native_support import body


def test_covering_requires_actual_target_support_release_and_dwell():
    graph = {'controller_program': [{'operator': 'cover', 'subjects': ['lid']}],
             'success_program': [{'predicate': 'covering', 'subject': 'lid', 'object': 'box'}]}
    v = NativeTaskGraphPredicates(graph, {'bodies': [
        body('cup', 'lid', [.12, .12, .01], 0, False),
        body('tray', 'box', [.1, .1, .04], 0, True)]}, Path('.'))
    def frame(t, support='tray', touching=True, held=False, x=0):
        ev = {'authority': 'physx_per_contact_normal_force', 'supported_by': support,
              'touching_bodies': ['tray'] if touching else [],
              'upward_load_by_body_n': {support: 1}}
        return {'time_s': t, 'cup': {'position': [x, 0, .025],
                'orientation_xyzw': [0, 0, 0, 1], 'linear_velocity': [0, 0, 0],
                'angular_velocity': [0, 0, 0]}, 'contact_support': {'cup': ev},
                'robot_cup_contact_links': ['finger'] if held else []}
    assert not v.update(frame(0))
    assert v.update(frame(.6))
    assert v.last_evidence['geometry']['0']['cup']['target_covered_fraction'] == 1
    assert not v.update(frame(1.2, support='table'))
    assert not v.update(frame(1.8, touching=False))
    assert not v.update(frame(2.4, held=True))
    assert not v.update(frame(3., x=.1))
    missing = frame(3.6)
    del missing['contact_support']
    assert not v.update(missing)
    assert not v.update(frame(4.2))
    assert v.update(frame(4.8))


def test_native_reader_rejects_saturation_and_preserves_filter_identity():
    import numpy as np
    import pytest
    from autoresearch.cosmos_contact_support import read_contact_support
    class Array:
        def __init__(self, value): self.value = np.array(value)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    class View:
        def get_contact_data(self, dt):
            return tuple(Array(x) for x in ([2., 0.], [], [[0,0,1], [0,0,0]], [], [[1,0]], [[0,1]]))
    class Sensor: contact_physx_view = View()
    ev = read_contact_support(Sensor(), ['tray', 'table'], .01)
    assert ev['supported_by'] == 'tray'
    assert ev['upward_load_by_body_n'] == {'tray': 2.}
    class Saturated:
        def get_contact_data(self, dt):
            return tuple(Array(x) for x in ([2.], [], [[0,0,1]], [], [[1]], [[0]]))
    Sensor.contact_physx_view = Saturated()
    with pytest.raises(ValueError, match='saturated'):
        read_contact_support(Sensor(), ['tray'], .01)
