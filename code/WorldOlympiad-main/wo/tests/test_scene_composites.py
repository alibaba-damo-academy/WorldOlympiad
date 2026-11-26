import json
import math

from copy import deepcopy


def test_inventory_importer_share_occurrences(tmp_path):
    from autoresearch.robotics_scenecode_case_generator import _scene_inventory
    from wo.scene import _house_objects
    house = tmp_path / 'combined_house' / 'house_state.json'
    house.parent.mkdir()
    a = member()
    a['transform']['rotation_wxyz'] = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
    house.write_text(json.dumps({'rooms': {'0': {'objects': {
        'cube_0': member(), 'stack': {'metadata': {
            'composite_type': 'stack', 'member_assets': [a]}}}}}}))
    registry = tmp_path / 'room_0' / 'generated_assets' / 'asset_registry.json'
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({'rejected': {'sdf_path': 'bad.sdf'}}))
    inventory = _scene_inventory(house)
    imported = _house_objects(house)
    assert {i['object_id'] for i in inventory} == set(imported)
    assert len(imported) == 2
    assert imported['cube_0']['transform']['translation'] == [1, 2, 3]
    assert imported['stack::member_0::cube_0']['transform']['translation'] == pytest.approx([1, 2.1, 3])
    assert all(i['sdf_path_candidates'] == ['cube.sdf'] for i in inventory)


import pytest

from wo.scene_objects import physical_objects


def member(asset='cube_0'):
    return {
        'asset_id': asset, 'name': 'cube', 'sdf_path': 'cube.sdf',
        'transform': {'translation': [1, 2, 3], 'rotation_wxyz': [1, 0, 0, 0]},
        'internal_model_pose': {'translation': [.1, 0, 0], 'rotation_wxyz': [1, 0, 0, 0]},
    }


@pytest.mark.parametrize('kind', ['stack', 'pile', 'filled_container'])
def test_composite_world_pose_and_occurrences(kind):
    a = member()
    members = {'member_assets': [a, a]} if kind != 'filled_container' else {
        'container_asset': a, 'fill_assets': [a]}
    original = {'cube_0': member(), 'assembly': {
        'object_type': 'manipuland',
        'transform': {'translation': [100, 100, 100]},
        'bbox_min': [-10, -10, -10], 'description': 'composite description',
        'metadata': {'composite_type': kind, **members},
    }}
    snapshot = deepcopy(original)
    result = physical_objects(original)
    assert set(result) == {'cube_0', 'assembly::member_0::cube_0', 'assembly::member_1::cube_0'}
    for key in ('assembly::member_0::cube_0', 'assembly::member_1::cube_0'):
        assert result[key]['transform'] == a['transform']
        assert result[key]['internal_model_pose'] == a['internal_model_pose']
        assert result[key]['metadata']['source_asset_id'] == 'cube_0'
        assert result[key]['sdf_path'] == 'cube.sdf'
        assert 'bbox_min' not in result[key]
        assert 'description' not in result[key]
    assert original == snapshot
    result['assembly::member_0::cube_0']['transform']['translation'][0] = 7
    assert original == snapshot


def test_missing_transform_is_not_parent_pose():
    with pytest.raises(ValueError, match='lacks world transform'):
        physical_objects({'pile': {'metadata': {
            'composite_type': 'pile', 'member_assets': [{'sdf_path': 'a.sdf'}]}}})


def test_occurrence_collision_is_explicit():
    with pytest.raises(ValueError, match='ID collision'):
        physical_objects({'x::member_0::cube_0': member(), 'x': {
            'metadata': {'composite_type': 'pile', 'member_assets': [member()]}}})


def test_unknown_composite_preserved():
    raw = {'x': {'metadata': {'composite_type': 'future_type'}, 'sdf_path': 'x.sdf'}}
    assert physical_objects(raw) == raw
