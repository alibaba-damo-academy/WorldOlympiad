import json
import xml.etree.ElementTree as ET

import pytest

from wo.models.cosmos import _archive_physics_urdf, _sha
from autoresearch.robotics_runtime_urdf_rebind import bind_runtime_physics_urdfs


def fixture(tmp_path):
    expert = tmp_path / 'expert'
    expert.mkdir()
    mesh = expert / 'part.obj'
    mesh.write_text('v 0 0 0\nv 1 0 0\n')
    source = expert / 'body.urdf'
    source.write_text('<robot name="b"><link name="base"><inertial><mass value="0.4"/>'
                      '<inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/>'
                      '</inertial><collision><geometry>'
                      f'<mesh filename="{mesh}" scale="0.1 0.2 0.3"/>'
                      '</geometry></collision></link><link name="cap"/>'
                      '<joint name="slide" type="prismatic"><parent link="base"/>'
                      '<child link="cap"/><origin xyz="0 0 .1" rpy="0 .2 0"/>'
                      '<axis xyz="0 0 1"/><limit lower="0" upper=".02" effort="2" velocity="1"/>'
                      '</joint></robot>')
    bundle = tmp_path / 'bundle'
    archived = bundle / 'b' / 'physics.urdf'
    _archive_physics_urdf(source, archived)
    manifest = bundle / 'manifest.json'
    manifest.write_text('{}')
    digests = {p.relative_to(bundle).as_posix(): _sha(p) for p in bundle.rglob('*') if p.is_file()}
    return source, archived, manifest, digests


def test_authoritative_urdf_preserved_and_rebound(tmp_path):
    source, archived, manifest, digests = fixture(tmp_path)
    original = ET.parse(source)
    paths = bind_runtime_physics_urdfs(manifest, {'b': {'physics_urdf': 'b/physics.urdf'}},
                                       tmp_path / 'runtime', digests)
    result = ET.parse(paths['b'])
    for xpath in ('.//joint', './/inertial'):
        assert ET.tostring(result.find(xpath)) == ET.tostring(original.find(xpath))
    assert result.find('.//mesh').get('scale') == '0.1 0.2 0.3'
    assert archived.parent.joinpath(ET.parse(archived).find('.//mesh').get('filename')).is_file()
    assert result.find('.//mesh').get('filename').startswith(str(manifest.parent))
    receipt = json.loads((tmp_path / 'runtime/urdf_asset_binding_receipt.json').read_text())
    assert receipt['assets'][0]['sha256'] == _sha(archived)


@pytest.mark.parametrize('tamper', ['mesh', 'urdf'])
def test_tamper_fails_closed(tmp_path, tamper):
    _, archived, manifest, digests = fixture(tmp_path)
    path = archived if tamper == 'urdf' else archived.parent / ET.parse(archived).find('.//mesh').get('filename')
    path.write_text('tampered')
    with pytest.raises(ValueError, match='digest changed'):
        bind_runtime_physics_urdfs(manifest, {'b': {'physics_urdf': 'b/physics.urdf'}}, tmp_path / 'r', digests)


def test_missing_authoritative_mesh_is_not_substituted(tmp_path):
    source = tmp_path / 'b.urdf'
    source.write_text('<robot><link><visual><geometry><mesh filename="missing.obj"/></geometry></visual></link></robot>')
    with pytest.raises(FileNotFoundError):
        _archive_physics_urdf(source, tmp_path / 'out.urdf')
