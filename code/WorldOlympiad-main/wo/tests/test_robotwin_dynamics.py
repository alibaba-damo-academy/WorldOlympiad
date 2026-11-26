import json
from types import SimpleNamespace
import pytest
import numpy as np
from wo.models import robotwin_dynamics as D


def row():
    return dict(mass_kg=.68, inertia_diagonal_kg_m2=[.0004, .0005, .0006],
                inertial_position=[.01, .02, .03], inertial_orientation_xyzw=[0, 0, 1, 0])


def test_apply_preserves_measured_frame_and_tensor():
    calls = []
    builder = SimpleNamespace(set_mass_and_inertia=lambda *args: calls.append(args))
    D.apply(builder, row(), lambda **kw: kw)
    assert calls == [(.68, {'p': [.01, .02, .03], 'q': [0, 0, 0, 1]}, [.0004, .0005, .0006])]


@pytest.mark.parametrize('field,value', [('mass_kg', 0), ('mass_kg', float('nan')),
    ('inertia_diagonal_kg_m2', [1, 0, 1]), ('inertial_orientation_xyzw', [0, 0, 0, 2]),
    ('inertial_position', [0, 0])])
def test_bad_measurement(field, value):
    r = row(); r[field] = value
    with pytest.raises(ValueError): D.validate_rigid(r)


def test_bind_archive_and_native_tamper(tmp_path, monkeypatch):
    ref = tmp_path / 'references/c'; ref.mkdir(parents=True)
    source = tmp_path / 'expert_checks/c/e/initialization.json'
    source.parent.mkdir(parents=True)
    data = {'measured_dynamics': {'cube': {'base_link': row()}}, 'dynamics_protocol': 'measured'}
    source.write_text(json.dumps(data))
    calls = []
    monkeypatch.setattr(D, 'load_bound_supplement', lambda r, s: (calls.append((r,s)) or data))
    objects = [{'object_id': 'cube', 'fixed': False, 'mass': 600}]
    binding = D.bind(ref, tmp_path / 'episode', objects)
    assert calls == [(ref, source)]
    assert objects[0]['mass'] == .68
    plan = dict(objects=objects, dynamics_evidence=binding)
    D.verify(plan)
    objects[0]['measured_dynamics'] = {**row(), 'mass_kg': 600}
    with pytest.raises(ValueError, match='plan mismatch'): D.verify(plan)
    objects[0]['measured_dynamics'] = row()
    archive = __import__('pathlib').Path(binding['archive'])
    archive.write_text('{}')
    with pytest.raises(ValueError, match='archive changed'): D.verify(plan)
    with pytest.raises(ValueError, match='immutable'): D.bind(ref, tmp_path / 'episode', objects)


def test_multilink_is_not_silently_collapsed(tmp_path, monkeypatch):
    ref = tmp_path / 'references/c'; ref.mkdir(parents=True)
    source = tmp_path / 'expert_checks/c/e/initialization.json'
    source.parent.mkdir(parents=True)
    data = {'measured_dynamics': {'cube': {'base_link': row()}}, 'dynamics_protocol': 'measured'}
    source.write_text(json.dumps(data))
    monkeypatch.setattr(D, 'load_bound_supplement', lambda *a: data)
    with pytest.raises(ValueError, match='multi-link'):
        D.bind(ref, tmp_path / 'episode', [{'object_id': 'cube', 'fixed': False, 'urdf': 'body.urdf'}])


def _static_row(friction=.5, restitution=.1, mass=0.0):
    return {'mass_kg': mass, 'lateral_friction': friction, 'restitution': restitution}


def test_flattened_static_links_accept_common_contact_material():
    rows = {'counter_top': _static_row(), 'counter_leg': _static_row()}
    assert D._bindable_single_actor_rows(rows) == 'counter_top'


def test_flattened_dynamic_links_are_rejected():
    with pytest.raises(ValueError, match='dynamic multi-link'):
        D._bindable_single_actor_rows({'a': _static_row(), 'b': _static_row(mass=1.0)})


def test_flattened_static_links_with_conflicting_material_are_rejected():
    with pytest.raises(ValueError, match='conflicting contact'):
        D._bindable_single_actor_rows({'a': _static_row(), 'b': _static_row(friction=.8)})


def test_single_dynamic_actor_is_not_mistaken_for_flattened_multilink(monkeypatch):
    class Component:
        def get_components(self): return [self]
        def get_collision_shapes(self): return []
        def set_mass(self, value): self.mass = value
        def set_inertia(self, value): self.inertia = value
        def set_cmass_local_pose(self, value): pass
        def get_mass(self): return .68
        def get_inertia(self):
            import numpy as np
            return np.array([.0004, .0005, .0006])
        def get_cmass_local_pose(self): return type('P', (), {
            'p': np.array([.01, .02, .03]), 'q': np.array([0, 0, 0, 1])})()
    class Shape:
        def get_physical_material(self):
            return type('M', (), {'static_friction': 1., 'dynamic_friction': 1., 'restitution': 0.})()
    class Handle:
        def get_components(self): return [self.component]
    handle = Handle(); handle.component = Component()
    class Physx: PhysxRigidBaseComponent = Component
    monkeypatch.setitem(__import__('sys').modules, 'sapien', type('S', (), {'physx': Physx})())
    scene = type('Scene', (), {'create_physical_material': lambda *a: None})()
    row_data = {**row(), 'lateral_friction': 1., 'restitution': 0.}
    D.apply_loaded(scene, handle, {'base_link': row_data}, lambda **kw: kw)
    assert handle.component.mass == .68
