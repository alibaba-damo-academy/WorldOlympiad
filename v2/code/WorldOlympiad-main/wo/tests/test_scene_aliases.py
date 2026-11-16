import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from wo.spec import TaskSpec
from wo.scene import canonicalize_support_aliases, SceneObject, _repair_reciprocal_bindings, resolve_scenecode
from wo.tasks.scene import archive_stale_task_recoveries
from wo.expert import embodiment as e


def test_reciprocal_lid_container_binding_is_corrected_without_coordinates():
    spec = TaskSpec.from_dict({
        'schema_version': 'wo_task_spec.v1', 'case_id': 'cover',
        'family': 'covering', 'instruction': 'cover box',
        'objects': [
            {'object_id': 'lid', 'category': 'food_container_lid', 'role': 'manipuland'},
            {'object_id': 'box', 'category': 'food_storage_box', 'role': 'destination'},
            {'object_id': 'table', 'category': 'countertop', 'role': 'support'}],
        'steps': [{'primitive': 'transport', 'subject': 'lid', 'target': 'box',
                   'goal': {'kind': 'covering', 'subject': 'lid', 'target': 'box'}}]})
    matches = {
        'lid': {'scene_name': 'food_storage_container', 'scene_object_id': 'container'},
        'box': {'scene_name': 'food_storage_lid', 'scene_object_id': 'lid'},
        'table': {'scene_name': 'counter', 'scene_object_id': 'table'},
    }
    corrections = _repair_reciprocal_bindings(spec, matches)
    assert corrections and matches['lid']['scene_name'] == 'food_storage_lid'
    assert matches['box']['scene_name'] == 'food_storage_container'


def test_resolve_preserves_explicit_reciprocal_inventory_members(tmp_path):
    asset = tmp_path / 'asset.sdf'
    asset.write_text('<sdf/>')
    objects = {}
    for i, oid in enumerate((
            'container', 'container_instance_0', 'lid', 'lid_instance_0', 'table')):
        objects[oid] = {
            'sdf_path': str(asset),
            'transform': {'translation': [i * .1, 0, .8 if oid != 'table' else 0],
                          'rotation_wxyz': [1, 0, 0, 0]},
        }
    house = tmp_path / 'house.json'
    house.write_text(json.dumps({'rooms': {'room': {'objects': objects}}}))
    spec = TaskSpec.from_dict({
        'schema_version': 'wo_task_spec.v1', 'case_id': 'cover_explicit',
        'family': 'covering', 'instruction': 'cover box',
        'objects': [
            {'object_id': 'lid', 'category': 'food_container_lid', 'role': 'manipuland',
             'inventory_object_id': 'lid', 'scene_member_id': 'lid_instance_0'},
            {'object_id': 'box', 'category': 'food_storage_box', 'role': 'destination',
             'inventory_object_id': 'box', 'scene_member_id': 'container_instance_0'},
            {'object_id': 'table', 'category': 'countertop', 'role': 'support'}],
        'steps': [{'primitive': 'transport', 'subject': 'lid', 'target': 'box',
                   'goal': {'kind': 'covering', 'subject': 'lid', 'target': 'box'}}]})
    (tmp_path / 'task_inventory_certificate.json').write_text(json.dumps({
        'house_state': str(house), 'matches': [
            {'task_object_id': 'lid', 'scene_object_id': 'lid', 'scene_name': 'food_storage_container',
             'status': 'matched', 'sdf_path': str(asset)},
            {'task_object_id': 'box', 'scene_object_id': 'container', 'scene_name': 'food_storage_lid',
             'status': 'matched', 'sdf_path': str(asset)},
            {'task_object_id': 'table', 'scene_object_id': 'table', 'status': 'matched',
             'sdf_path': str(asset)}]}))
    scene = resolve_scenecode(spec, tmp_path)
    assert scene.source['binding_corrections'] == []
    assert scene.source['instance_bindings']['lid']['scene_object_id'] == 'lid_instance_0'
    assert scene.source['instance_bindings']['box']['scene_object_id'] == 'container_instance_0'


def test_resolve_binds_certificate_relative_assets_from_generation(tmp_path):
    asset_dir = tmp_path / 'assets'
    asset_dir.mkdir()
    (asset_dir / 'box.sdf').write_text('<sdf/>')
    house = tmp_path / 'house.json'
    records = {
        oid: {'sdf_path': 'assets/box.sdf',
              'transform': {'translation': [0, 0, 0], 'rotation_wxyz': [1, 0, 0, 0]}}
        for oid in ('box', 'table')}
    house.write_text(json.dumps({'rooms': {'room': {'objects': records}}}))
    (tmp_path / 'task_inventory_certificate.json').write_text(json.dumps({
        'house_state': 'house.json',
        'matches': [
            {'task_object_id': 'box', 'scene_object_id': 'box', 'status': 'matched',
             'sdf_path': 'assets/box.sdf'},
            {'task_object_id': 'table', 'scene_object_id': 'table', 'status': 'matched',
             'sdf_path': 'assets/box.sdf'}]}))
    spec = TaskSpec.from_dict({
        'schema_version': 'wo_task_spec.v1', 'case_id': 'relative',
        'family': 'pick_place', 'instruction': 'move',
        'objects': [{'object_id': 'box', 'category': 'box', 'role': 'manipuland'},
                    {'object_id': 'table', 'category': 'countertop', 'role': 'support'}],
        'steps': [{'primitive': 'transport', 'subject': 'box', 'target': 'table',
                   'goal': {'kind': 'on', 'subject': 'box', 'target': 'table'}}]})
    scene = resolve_scenecode(spec, tmp_path)
    assert scene.objects['box'].sdf_path == asset_dir / 'box.sdf'


def test_reciprocal_repair_rejects_unrelated_lid_and_box():
    spec = TaskSpec.from_dict({
        'schema_version': 'wo_task_spec.v1', 'case_id': 'x', 'family': 'pick_place',
        'instruction': 'move', 'objects': [
            {'object_id': 'apple', 'category': 'apple', 'role': 'manipuland'},
            {'object_id': 'lid', 'category': 'food_container_lid', 'role': 'destination'}],
        'steps': [{'primitive': 'transport', 'subject': 'apple', 'target': 'lid',
                   'goal': {'kind': 'on_region', 'subject': 'apple', 'target': 'lid'}}]})
    matches = {
        'apple': {'scene_name': 'food_storage_lid', 'scene_object_id': 'lid'},
        'lid': {'scene_name': 'spoon', 'scene_object_id': 'spoon'},
    }
    assert _repair_reciprocal_bindings(spec, matches) == []


def test_reciprocal_repair_rejects_ambiguous_occurrences():
    spec = TaskSpec.from_dict({
        'schema_version': 'wo_task_spec.v1', 'case_id': 'x', 'family': 'covering',
        'instruction': 'cover', 'objects': [
            {'object_id': 'lid', 'category': 'food_container_lid', 'role': 'manipuland'},
            {'object_id': 'box_a', 'category': 'food_storage_box', 'role': 'destination'},
            {'object_id': 'box_b', 'category': 'food_storage_box', 'role': 'destination'}],
        'steps': [{'primitive': 'transport', 'subject': 'lid', 'target': 'box_a',
                   'goal': {'kind': 'covering', 'subject': 'lid', 'target': 'box_a'}}]})
    matches = {
        'lid': {'scene_name': 'food_storage_container', 'scene_object_id': 'container'},
        'box_a': {'scene_name': 'food_storage_lid', 'scene_object_id': 'lid_a'},
        'box_b': {'scene_name': 'food_storage_lid', 'scene_object_id': 'lid_b'},
    }
    assert _repair_reciprocal_bindings(spec, matches) == []


def test_reciprocal_repair_rejects_shared_word_but_different_domain():
    spec = TaskSpec.from_dict({
        'schema_version': 'wo_task_spec.v1', 'case_id': 'x', 'family': 'covering',
        'instruction': 'cover', 'objects': [
            {'object_id': 'lid', 'category': 'one_lid', 'role': 'manipuland'},
            {'object_id': 'box', 'category': 'one_box', 'role': 'destination'}],
        'steps': [{'primitive': 'transport', 'subject': 'lid', 'target': 'box',
                   'goal': {'kind': 'covering', 'subject': 'lid', 'target': 'box'}}]})
    matches = {
        'lid': {'scene_name': 'two_container', 'scene_object_id': 'container'},
        'box': {'scene_name': 'one_lid', 'scene_object_id': 'lid'},
    }
    # ``one`` alone is not enough to bridge a different scene domain.
    assert _repair_reciprocal_bindings(spec, matches) == []


def test_support_alias_is_not_a_second_table(tmp_path):
    spec = TaskSpec.from_dict({
        'schema_version': 'wo_task_spec.v1',
        'case_id': 'test', 'family': 'reorientation', 'instruction': 'upright bottle',
        'objects': [
            {'object_id': 'bottle', 'category': 'bottle', 'role': 'manipuland', 'description': 'bottle'},
            {'object_id': 'table', 'category': 'table', 'role': 'support', 'description': 'table'},
            {'object_id': 'surface', 'category': 'table_region', 'role': 'destination', 'description': 'surface'}],
        'steps': [{'primitive': 'reorient', 'subject': 'bottle', 'target': 'surface',
                   'goal': {'kind': 'upright', 'subject': 'bottle'}}]})
    def cert(surface):
        (tmp_path / 'task_inventory_certificate.json').write_text(json.dumps({'matches': [
            {'task_object_id': k, 'scene_object_id': v, 'status': 'matched'}
            for k, v in [('bottle', 'b'), ('table', 't'), ('surface', surface)]]}))
    cert('t')
    result = canonicalize_support_aliases(spec, tmp_path)
    assert [o.object_id for o in result.objects] == ['bottle', 'table']
    assert result.steps[0].target == 'table'
    cert('another_table')
    assert canonicalize_support_aliases(spec, tmp_path) == spec


def test_stale_checkpoint_archived_not_destroyed(tmp_path):
    path = tmp_path / 'robotics_scenecode_partial_recoveries' / 'batch' / 'case'
    path.mkdir(parents=True)
    (path / 'partial_recovery_manifest.json').write_text(json.dumps({'task_sha256': 'old'}))
    archive_stale_task_recoveries(tmp_path, 'batch', 'case', 'old')
    assert path.exists()
    archive_stale_task_recoveries(tmp_path, 'batch', 'case', 'new')
    assert not path.exists()
    assert len(list(path.parent.glob('case.stale_*'))) == 1


def test_canonical_rebase_lifts_floor_object(monkeypatch, tmp_path):
    from wo.tests.test_embodiment_layout import Scene
    obj = SceneObject('cube', 'manipuland', tmp_path / 'x.sdf', (0, 0, 0), (0, 0, 0, 1), False)
    scene = Scene(SimpleNamespace(object_id='table'), {'cube': obj}, {})
    class World:
        support_id = 'table'
        def __init__(self, *args, **kwargs): pass
        def aabb(self, oid):
            if oid == 'table': return np.array([-1, -1, 0]), np.array([1, 1, .75])
            return np.array([-.025, -.025, 0]), np.array([.025, .025, .05])
        def close(self): pass
    monkeypatch.setattr(e, 'World', World)
    monkeypatch.setattr(e, 'prepare_bodies', lambda *a: {})
    result, _ = e.droid_canonical_layout(scene, tmp_path)
    assert result.objects['cube'].position[2] == .75
