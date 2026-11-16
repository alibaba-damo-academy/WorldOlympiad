from wo.expert import embodiment as E
from wo.expert.grasps import _finger_sweep
import numpy as np
import pytest


class DummyWorld:
    def __init__(self, *a, **kw):
        pass
    def close(self):
        pass


def test_finger_sweep_stops_at_tcp_for_thin_supported_objects():
    base = np.array([0.2, -0.1, 0.8])
    approach = np.array([0.0, 0.0, -1.0])
    start, end = _finger_sweep(base, approach)
    assert np.allclose(end, base)
    assert np.allclose(start, base - 0.14 * approach)


def test_single_valid_grasp_does_not_trigger_unnecessary_rescaling(monkeypatch, tmp_path):
    scene = object()
    monkeypatch.setattr(E, 'World', DummyWorld)
    monkeypatch.setattr(E, 'prepare_bodies', lambda *a: None)
    monkeypatch.setattr(E, 'scaled_scene', lambda s, oid, scale, work: s)
    monkeypatch.setattr(E, 'sample_grasps', lambda *a: [object()])
    candidate, info = E.fit_gripper(scene, 'lid', tmp_path)
    assert candidate is scene
    assert info['scale'] == 1.0
    assert info['ladder'] == [{'scale': 1.0, 'grasps': 1}]


def test_sizing_excludes_neighbours_without_scaling(monkeypatch, tmp_path):
    calls = []
    class SizingWorld(DummyWorld):
        def __init__(self, *a, **kw):
            calls.append(kw)
            self.accessible = kw.get('include') == ['bead']
    monkeypatch.setattr(E, 'World', SizingWorld)
    monkeypatch.setattr(E, 'prepare_bodies', lambda *a: None)
    monkeypatch.setattr(E, 'scaled_scene', lambda s, oid, scale, work: s)
    monkeypatch.setattr(E, 'sample_grasps', lambda w, oid: [object()] if w.accessible else [])
    scene = object()
    candidate, info = E.fit_gripper(scene, 'bead', tmp_path)
    assert candidate is scene
    assert info['scale'] == 1.0
    assert calls == [{'mode': 'objects', 'include': ['bead']}]


def test_zero_valid_grasps_remains_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(E, 'World', DummyWorld)
    monkeypatch.setattr(E, 'prepare_bodies', lambda *a: None)
    monkeypatch.setattr(E, 'scaled_scene', lambda s, oid, scale, work: s)
    monkeypatch.setattr(E, 'sample_grasps', lambda *a: [])
    with pytest.raises(RuntimeError, match='no gripper-compatible scale'):
        E.fit_gripper(object(), 'lid', tmp_path)
