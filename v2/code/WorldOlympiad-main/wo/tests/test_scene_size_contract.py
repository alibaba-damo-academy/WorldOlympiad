from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
import trimesh

from wo.spec import TaskObject
from wo.tasks.collision_envelope import conform_collision_envelopes
from wo.tasks.geometry_contract import visual_bounds, validate_size_contracts


def asset(tmp_path, size, suffix='.obj'):
    mesh = trimesh.creation.box(extents=size)
    if suffix == '.glb':
        # SDF Z-up -> glTF Y-up.
        mesh.vertices = mesh.vertices[:, [0, 2, 1]] * [1, 1, -1]
    path = tmp_path / ('visual' + suffix)
    mesh.export(path)
    sdf = tmp_path / 'asset.sdf'
    sdf.write_text(f'<sdf><model><link><visual><geometry><mesh><uri>{path.name}</uri>'
                   f'</mesh></geometry></visual><collision><geometry><mesh><uri>{path.name}</uri>'
                   '</mesh></geometry></collision></link></model></sdf>')
    return sdf


@pytest.mark.parametrize('suffix', ['.obj', '.glb'])
def test_bounds_use_physical_sdf_axes(tmp_path, suffix):
    sdf = asset(tmp_path, [.08, .0744, .0496], suffix)
    np.testing.assert_allclose(np.diff(visual_bounds(sdf), axis=0)[0], [.08, .0744, .0496], atol=1e-8)


def test_export_size_mismatch_is_not_silently_accepted(tmp_path):
    sdf = asset(tmp_path, [.08, .0744, .0496])
    obj = TaskObject('destination', 'socket', 'destination', geometry_constraints={
        'overall_dimensions_m': [.08, .08, .05]})
    body = SimpleNamespace(sdf_path=sdf, scale=1)
    scene = SimpleNamespace(support=SimpleNamespace(object_id='table'), objects={'destination': body})
    result = validate_size_contracts(SimpleNamespace(objects=[obj]), scene)
    assert not result['success']
    np.testing.assert_allclose(result['objects']['destination']['measured_m'], [.08, .0744, .0496])
    obj = TaskObject('destination', 'socket', 'destination', geometry_constraints={
        'overall_dimensions_m': [.08, .0744, .0496]})
    assert validate_size_contracts(SimpleNamespace(objects=[obj]), scene)['success']


def test_shifted_collision_fails_closed(tmp_path):
    path = tmp_path / 'mesh.obj'
    trimesh.creation.box(extents=[.08, .08, .05]).export(path)
    sdf = tmp_path / 'shifted.sdf'
    sdf.write_text(f'<sdf><model><link><visual><geometry><mesh><uri>{path.name}</uri>'
                   f'</mesh></geometry></visual><collision><pose>0 0 .075 0 0 0</pose>'
                   f'<geometry><mesh><uri>{path.name}</uri></mesh></geometry></collision>'
                   '</link></model></sdf>')
    obj = TaskObject('destination', 'socket', 'destination', geometry_constraints={
        'overall_dimensions_m': [.08, .08, .05]})
    scene = SimpleNamespace(support=SimpleNamespace(object_id='table'),
                            objects={'destination': SimpleNamespace(sdf_path=sdf, scale=1)})
    result = validate_size_contracts(SimpleNamespace(objects=[obj]), scene)
    assert not result['success']
    np.testing.assert_allclose(result['objects']['destination']['center_delta_m'], [0, 0, .075])


def test_missing_collision_fails_closed(tmp_path):
    path = tmp_path / 'visual.obj'
    trimesh.creation.box(extents=[.08, .08, .05]).export(path)
    sdf = tmp_path / 'visual_only.sdf'
    sdf.write_text(f'<sdf><model><link><visual><geometry><mesh><uri>{path.name}</uri>'
                   '</mesh></geometry></visual></link></model></sdf>')
    obj = TaskObject('destination', 'socket', 'destination', geometry_constraints={
        'overall_dimensions_m': [.08, .08, .05]})
    result = validate_size_contracts(SimpleNamespace(objects=[obj]),
                                     SimpleNamespace(support=SimpleNamespace(object_id='table'),
                                                    objects={'destination': SimpleNamespace(sdf_path=sdf, scale=1)}))
    assert not result['success']
    assert result['objects']['destination']['collision_m'] is None


def test_no_contract_leaves_legacy_assets_untouched():
    obj = TaskObject('destination', 'socket', 'destination')
    result = validate_size_contracts(SimpleNamespace(objects=[obj]), None)
    assert result['success'] and result['objects'] == {}


def test_scene_generation_conforms_inside_collision_envelope_and_rebinds_certificate(tmp_path):
    visual = tmp_path / 'visual.obj'
    collision = tmp_path / 'collision.obj'
    trimesh.creation.box(extents=[.24, .24, .07]).export(visual)
    trimesh.creation.box(extents=[.11885, .2371, .0699]).export(collision)
    sdf = tmp_path / 'bowl.sdf'
    sdf.write_text(f'''<sdf><model><link>
      <visual><geometry><mesh><uri>{visual.name}</uri></mesh></geometry></visual>
      <collision><pose>-.059715 0 -.00005 0 0 0</pose><geometry><mesh>
        <uri>{collision.name}</uri></mesh></geometry></collision>
      <collision><pose>.059135 0 -.00005 0 0 0</pose><geometry><mesh>
        <uri>{collision.name}</uri></mesh></geometry></collision>
      </link></model></sdf>''')
    obj = TaskObject('destination', 'bowl', 'destination', geometry_constraints={
        'overall_dimensions_m': [.24, .24, .07]})
    scene = SimpleNamespace(support=SimpleNamespace(object_id='table'),
                            objects={'destination': SimpleNamespace(sdf_path=sdf, scale=1)})
    generation = tmp_path / 'generation'
    generation.mkdir()
    certificate_path = generation / 'task_inventory_certificate.json'
    certificate_path.write_text(json.dumps({'matches': [{
        'task_object_id': 'destination', 'sdf_path': str(sdf),
        'sdf_sha256': hashlib.sha256(sdf.read_bytes()).hexdigest()}]}))
    manifest_path = generation / 'generation_manifest.json'
    manifest_path.write_text(json.dumps({'task_inventory_certificate_sha256': 'old'}))

    report = conform_collision_envelopes(SimpleNamespace(objects=[obj]), scene, generation)

    assert len(report['adjustments']) == 1
    assert len(report['adjustments'][0]['collision_meshes']) == 2
    result = validate_size_contracts(SimpleNamespace(objects=[obj]), scene)
    assert result['success']
    assert result['objects']['destination']['collision_m'] == pytest.approx([.24, .24, .07])
    assert result['objects']['destination']['center_delta_m'] == pytest.approx([0, 0, 0], abs=1e-8)
    bound = json.loads(certificate_path.read_text())['matches'][0]
    assert bound['sdf_sha256'] == hashlib.sha256(sdf.read_bytes()).hexdigest()
    assert json.loads(manifest_path.read_text())['task_inventory_certificate_sha256'] == \
        hashlib.sha256(certificate_path.read_bytes()).hexdigest()
    assert (generation / 'collision_envelope_adjustments.json').is_file()


def test_collision_envelope_affine_respects_sdf_pose_and_mesh_scale(tmp_path):
    visual = tmp_path / 'visual.obj'
    collision = tmp_path / 'collision.obj'
    trimesh.creation.box(extents=[.08, .08, .05]).export(visual)
    trimesh.creation.box(extents=[.078, .078, .0499]).export(collision)
    sdf = tmp_path / 'posed.sdf'
    sdf.write_text(f'<sdf><model><link><visual><geometry><mesh><uri>{visual.name}</uri></mesh></geometry></visual>'
                   f'<collision><pose>-.0001 0 0 0 0 1.5707963267948966</pose><geometry><mesh>'
                   f'<uri>{collision.name}</uri><scale>.99 .99 1</scale></mesh></geometry></collision>'
                   '</link></model></sdf>')
    obj = TaskObject('destination', 'object', 'destination', geometry_constraints={
        'overall_dimensions_m': [.08, .08, .05]})
    scene = SimpleNamespace(support=SimpleNamespace(object_id='table'),
                             objects={'destination': SimpleNamespace(sdf_path=sdf, scale=1)})
    generation = tmp_path / 'generation'
    generation.mkdir()
    cert = generation / 'task_inventory_certificate.json'
    cert.write_text(json.dumps({'matches': [{'task_object_id': 'destination',
                    'sdf_path': str(sdf), 'sdf_sha256': hashlib.sha256(sdf.read_bytes()).hexdigest()}]}))
    (generation / 'generation_manifest.json').write_text(json.dumps({}))

    conform_collision_envelopes(SimpleNamespace(objects=[obj]), scene, generation)

    result = validate_size_contracts(SimpleNamespace(objects=[obj]), scene)
    assert result['success'], result
    assert result['objects']['destination']['center_delta_m'] == pytest.approx([0, 0, 0], abs=1e-8)


def test_collision_conforming_never_shrinks_an_overhanging_collision(tmp_path):
    visual = tmp_path / 'visual.obj'
    collision = tmp_path / 'collision.obj'
    trimesh.creation.box(extents=[.08, .08, .05]).export(visual)
    trimesh.creation.box(extents=[.082, .08, .05]).export(collision)
    sdf = tmp_path / 'overhang.sdf'
    sdf.write_text(f'<sdf><model><link><visual><geometry><mesh><uri>{visual.name}</uri></mesh></geometry></visual>'
                   f'<collision><geometry><mesh><uri>{collision.name}</uri></mesh></geometry></collision>'
                   '</link></model></sdf>')
    obj = TaskObject('destination', 'socket', 'destination', geometry_constraints={
        'overall_dimensions_m': [.08, .08, .05]})
    scene = SimpleNamespace(support=SimpleNamespace(object_id='table'),
                             objects={'destination': SimpleNamespace(sdf_path=sdf, scale=1)})
    generation = tmp_path / 'generation'
    generation.mkdir()
    original = sdf.read_bytes()
    report = conform_collision_envelopes(SimpleNamespace(objects=[obj]), scene, generation)
    assert report['adjustments'] == []
    assert sdf.read_bytes() == original
    assert not validate_size_contracts(SimpleNamespace(objects=[obj]), scene)['success']
