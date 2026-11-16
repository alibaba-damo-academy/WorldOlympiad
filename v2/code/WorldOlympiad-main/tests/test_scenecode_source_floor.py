import hashlib
import json
from pathlib import Path

import pytest

from autoresearch.robotics_scenecode_case_generator import _prepare_source_floor
from wo.tasks.scene import source_floor_env


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def floor(tmp_path, monkeypatch):
    source = tmp_path / 'old/scene_000'
    (source/'room_geometry').mkdir(parents=True)
    (source/'floor_plans').mkdir()
    (source/'room_geometry/floor.sdf').write_text('<sdf/>')
    (source/'floor_plans/floor.gltf').write_bytes(b'geometry bytes')
    layout = {'house_prompt':'old prose', 'rooms':[{'id':'room','prompt':'old prose'}],
              'placement_valid':True, 'connectivity_valid':True,
              'room_geometries':{'room':{'width':4.8,'length':4.,'wall_height':2.5}}}
    (source/'house_layout.json').write_text(json.dumps(layout))
    manifest = tmp_path/'source.json'
    record = {'schema_version':'scenecode_source_floor.v1','source_job_id':1454,
              'source_scene_dir':str(source),'scene_prompt_sha256':'new-prompt-hash',
              'room_size_m':[4.8,4.],
              'source_files':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}}
    manifest.write_text(json.dumps(record))
    monkeypatch.setenv('SCENECODE_SOURCE_FLOOR_MANIFEST',str(manifest))
    monkeypatch.setenv('SCENECODE_SOURCE_FLOOR_MANIFEST_SHA256',sha(manifest))
    return tmp_path, source, manifest, record, layout


def test_default_unchanged_and_explicit_floor_preserves_geometry(floor, monkeypatch):
    root, source, manifest, record, old = floor
    monkeypatch.delenv('SCENECODE_SOURCE_FLOOR_MANIFEST')
    assert _prepare_source_floor(root,root/'unused','new prose','new-prompt-hash') is None
    monkeypatch.setenv('SCENECODE_SOURCE_FLOOR_MANIFEST',str(manifest))
    target = _prepare_source_floor(root,root/'stage','new prose','new-prompt-hash')/'scene_000'
    new = json.loads((target/'house_layout.json').read_text())
    expected=json.loads(json.dumps(old));expected['house_prompt']='new prose';expected['rooms'][0]['prompt']='new prose'
    assert new == expected
    assert json.loads((source/'house_layout.json').read_text()) == old
    for relative,digest in record['source_files'].items():
        if relative != 'house_layout.json': assert sha(target/relative) == digest
    receipt=json.loads((root/'stage/source_floor_input.json').read_text())
    assert receipt['source_files'] == record['source_files']
    assert receipt['changed_fields'] == ['house_prompt','rooms[0].prompt']
    assert receipt['source_room_geometry_width_length_m'] == [4.8,4.]
    assert receipt['current_scene_prompt_size_m'] == [4.8,4.]


def test_wrong_prompt_or_changed_resource_rejected(floor):
    root,source,manifest,record,_=floor
    with pytest.raises(ValueError,match='binding'):
        _prepare_source_floor(root,root/'a','new','other-hash')
    (source/'floor_plans/floor.gltf').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='inventory/hash'):
        _prepare_source_floor(root,root/'b','new','new-prompt-hash')
    assert not (root/'b').exists()


def test_nonfloor_asset_or_unfinished_layout_rejected(floor,monkeypatch):
    root,source,manifest,record,_=floor
    asset=source/'room_room/generated_assets/furniture/mesh.obj'
    asset.parent.mkdir(parents=True);asset.write_bytes(b'task asset')
    with pytest.raises(ValueError,match='task/furniture'):
        _prepare_source_floor(root,root/'a','new','new-prompt-hash')
    asset.unlink()
    value=json.loads((source/'house_layout.json').read_text());value['placement_valid']=False
    (source/'house_layout.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='completed'):
        _prepare_source_floor(root,root/'b','new','new-prompt-hash')


def test_run_scoped_sidecar_binds_current_task(tmp_path):
    task=tmp_path/'tasks/case.json';task.parent.mkdir();task.write_text('{}')
    assert source_floor_env(task)=={}
    folder=tmp_path/'source_floors';folder.mkdir()
    manifest=folder/'case.json'
    manifest.write_text(json.dumps({'schema_version':'scenecode_source_floor.v1','case_id':'case','task_sha256':sha(task)}))
    assert source_floor_env(task)['SCENECODE_SOURCE_FLOOR_MANIFEST_SHA256']==sha(manifest)
    task.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='current TaskSpec'): source_floor_env(task)
