import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from wo.spec import TaskObject
from wo.tasks.asset_size_normalization import normalize_asset_sizes
from wo.tasks.geometry_contract import collision_bounds, validate_size_contracts, visual_bounds


def _fixture(tmp_path, visual_extents=(.24, .24, .0672), collision_extents=(.238, .238, .067)):
    visual = tmp_path / 'visual.glb'
    collision = tmp_path / 'collision.obj'
    visual_mesh = trimesh.creation.box(extents=visual_extents)
    visual_mesh.apply_translation([.01, -.02, .08])
    # Encode the Z-up physical mesh in GLTF's Y-up coordinates.
    visual_mesh.vertices = visual_mesh.vertices[:, [0, 2, 1]] * [1, 1, -1]
    visual_mesh.export(visual)
    collision_mesh = trimesh.creation.box(extents=collision_extents)
    collision_mesh.apply_translation([.01, -.02, .08])
    collision_mesh.export(collision)
    sdf = tmp_path / 'bowl.sdf'
    sdf.write_text(f'<sdf><model><link name="body"><inertial><pose>0 0 0 0 0 0</pose><mass>1</mass>'
                   '<inertia><ixx>.01</ixx><iyy>.01</iyy><izz>.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>'
                   f'<visual name="v"><geometry><mesh><uri>{visual.name}</uri></mesh></geometry></visual>'
                   f'<collision name="c"><geometry><mesh><uri>{collision.name}</uri></mesh></geometry></collision>'
                   '</link></model></sdf>')
    obj = TaskObject('bowl', 'shallow bowl', 'destination', geometry_constraints={
        'overall_dimensions_m': [.24, .24, .07]})
    scene = SimpleNamespace(support=SimpleNamespace(object_id='counter'),
                            objects={'bowl': SimpleNamespace(sdf_path=sdf, scale=1.0)})
    generation = tmp_path / 'generation'
    generation.mkdir()
    certificate = generation / 'task_inventory_certificate.json'
    certificate.write_text(json.dumps({'matches': [{'task_object_id': 'bowl',
        'sdf_path': str(sdf), 'sdf_sha256': hashlib.sha256(sdf.read_bytes()).hexdigest()}]}))
    manifest = generation / 'generation_manifest.json'
    manifest.write_text(json.dumps({'passed': True}))
    return SimpleNamespace(objects=[obj]), scene, generation, sdf, certificate, manifest


def test_anisotropic_size_normalization_preserves_center_and_rebinds_hashes(tmp_path):
    spec, scene, generation, sdf, certificate, manifest = _fixture(tmp_path)
    before = validate_size_contracts(spec, scene)
    assert not before['success']
    old_center = np.mean(visual_bounds(sdf), axis=0)

    report = normalize_asset_sizes(spec, scene, generation)

    assert report['adjustments'][0]['axis_factors'] == pytest.approx([1, 1, .07 / .0672])
    after = validate_size_contracts(spec, scene)
    assert after['objects']['bowl']['measured_m'] == pytest.approx([.24, .24, .07], abs=1e-8)
    assert after['objects']['bowl']['center_delta_m'] == pytest.approx([0, 0, 0], abs=1e-8)
    assert np.ptp(collision_bounds(sdf), axis=0) == pytest.approx([.238, .238, .067 * .07 / .0672])
    cert = json.loads(certificate.read_text())
    assert cert['matches'][0]['sdf_sha256'] == hashlib.sha256(sdf.read_bytes()).hexdigest()
    bound_manifest = json.loads(manifest.read_text())
    assert bound_manifest['task_inventory_certificate_sha256'] == hashlib.sha256(certificate.read_bytes()).hexdigest()
    audit = generation / bound_manifest['asset_size_normalization']['path']
    assert bound_manifest['asset_size_normalization']['sha256'] == hashlib.sha256(audit.read_bytes()).hexdigest()
    assert np.mean(visual_bounds(sdf), axis=0) == pytest.approx(old_center)
    inertial = ET.parse(sdf).find('.//inertial')
    assert float(inertial.findtext('mass')) == pytest.approx(1.0)
    assert [float(v) for v in inertial.findtext('pose').split()[:3]] == pytest.approx(
        old_center * (1 - np.asarray([1, 1, .07 / .0672])), abs=1e-8)
    assert float(inertial.findtext('inertia/ixx')) != pytest.approx(.01)


def test_large_axis_correction_fails_closed_without_editing_asset(tmp_path):
    spec, scene, generation, sdf, certificate, _ = _fixture(tmp_path, visual_extents=(.24, .24, .04))
    original_sdf = sdf.read_bytes()
    original_certificate = certificate.read_bytes()
    with pytest.raises(ValueError, match='exceeds safe axis-wise limits'):
        normalize_asset_sizes(spec, scene, generation)
    assert sdf.read_bytes() == original_sdf
    assert certificate.read_bytes() == original_certificate


def test_rotated_mesh_instance_fails_closed_without_editing_asset(tmp_path):
    spec, scene, generation, sdf, certificate, _ = _fixture(tmp_path)
    tree = ET.parse(sdf)
    collision = tree.find('.//collision')
    ET.SubElement(collision, 'pose').text = '0 0 0 0 0 0.2'
    tree.write(sdf)
    original_sdf = sdf.read_bytes()
    original_certificate = certificate.read_bytes()
    with pytest.raises(ValueError, match='rotated collision poses'):
        normalize_asset_sizes(spec, scene, generation)
    assert sdf.read_bytes() == original_sdf
    assert certificate.read_bytes() == original_certificate
