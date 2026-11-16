from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from wo.models.cosmos import _archive_physics_sdf, _sha
from autoresearch.robotics_runtime_sdf_rebind import bind_runtime_physics_sdfs


def test_generated_hulls_are_archived_and_role_locally_bound(tmp_path):
    bundle = tmp_path / 'bundle'
    prepared = {}
    for role, content in [('bead1', 'v 0 0 0\n'), ('bead2', 'v 1 0 0\n')]:
        expert = tmp_path / 'expert' / role
        expert.mkdir(parents=True)
        mesh = expert / 'hull.obj'
        mesh.write_text(content)
        physics = expert / 'shape.sdf'
        physics.write_text('<sdf><model><link name="body"><collision><geometry>'
                           f'<mesh><uri>{mesh}</uri></mesh></geometry></collision>'
                           '</link></model></sdf>')
        dest = bundle / role
        dest.mkdir(parents=True)
        source = dest / 'source.sdf'
        source.write_text('<sdf><model><link name="body"/></model></sdf>')
        _archive_physics_sdf(physics, dest / 'physics.sdf')
        assert not Path(ET.parse(dest / 'physics.sdf').findtext('.//mesh/uri')).is_absolute()
        prepared[role] = {'source_sdf': f'{role}/source.sdf',
                          'physics_sdf': f'{role}/physics.sdf'}
    manifest = bundle / 'manifest.json'
    manifest.write_text('{}')
    digests = {p.relative_to(bundle).as_posix(): _sha(p)
               for p in bundle.rglob('*') if p.is_file()}
    paths = bind_runtime_physics_sdfs(manifest, prepared, tmp_path / 'runtime', digests)
    for role, path in paths.items():
        uri = Path(ET.parse(path).findtext('.//mesh/uri'))
        assert uri == bundle / role / 'expert_collision/0/hull.obj'
    (bundle / 'bead1/expert_collision/0/hull.obj').write_text('tampered')
    with pytest.raises(ValueError, match='digest changed'):
        bind_runtime_physics_sdfs(manifest, prepared, tmp_path / 'runtime2', digests)


def test_absolute_source_meshes_bound_for_visuals_and_collisions(tmp_path):
    external = tmp_path / 'external'
    external.mkdir()
    mesh = external / 'socket.obj'
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    source = external / 'socket.sdf'
    geom = f'<geometry><mesh><uri>{mesh}</uri></mesh></geometry>'
    source.write_text('<sdf><model name="socket"><link name="body">'
                      f'<visual name="v">{geom}</visual>'
                      f'<collision name="c">{geom}</collision>'
                      '</link></model></sdf>')
    bundle = tmp_path / 'bundle'
    dest = bundle / 'socket'
    dest.mkdir(parents=True)
    _archive_physics_sdf(source, dest / 'source.sdf', mesh_dir='source_meshes')
    _archive_physics_sdf(source, dest / 'physics.sdf')
    uris = [n.text for n in ET.parse(dest / 'source.sdf').findall('.//mesh/uri')]
    assert uris == ['source_meshes/0/socket.obj'] * 2
    assert (dest / uris[0]).read_bytes() == mesh.read_bytes()
    manifest = bundle / 'manifest.json'
    manifest.write_text('{}')
    digests = {p.relative_to(bundle).as_posix(): _sha(p)
               for p in bundle.rglob('*') if p.is_file()}
    paths = bind_runtime_physics_sdfs(manifest, {
        'socket': {'source_sdf': 'socket/source.sdf', 'physics_sdf': 'socket/physics.sdf'}
    }, tmp_path / 'runtime', digests)
    for node in ET.parse(paths['socket']).findall('.//mesh/uri'):
        assert Path(node.text).is_relative_to(bundle)
        assert Path(node.text).read_bytes() == mesh.read_bytes()


def test_gltf_external_buffers_and_images_archived(tmp_path):
    import json
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'buffer 0.bin').write_bytes(b'actual vertices')
    (tmp_path / 'texture.png').write_bytes(b'actual pixels')
    payload = {'buffers': [{'uri': 'buffer%200.bin'}],
               'images': [{'uri': '../texture.png'}, {'uri': 'data:image/png;base64,AA=='}]}
    mesh = source / 'mesh.gltf'
    mesh.write_text(json.dumps(payload))
    sdf = source / 'source.sdf'
    sdf.write_text('<sdf><model><link><visual><geometry><mesh><uri>mesh.gltf</uri>'
                   '</mesh></geometry></visual></link></model></sdf>')
    output = tmp_path / 'bundle' / 'source.sdf'
    _archive_physics_sdf(sdf, output)
    uri = ET.parse(output).find('.//mesh/uri').text
    archived = output.parent / uri
    result = json.loads(archived.read_text())
    assert (archived.parent / result['buffers'][0]['uri']).read_bytes() == b'actual vertices'
    assert (archived.parent / result['images'][0]['uri']).read_bytes() == b'actual pixels'
    assert result['images'][1] == payload['images'][1]
    assert json.loads(mesh.read_text()) == payload
    (source / 'buffer 0.bin').unlink()
    with pytest.raises(FileNotFoundError):
        _archive_physics_sdf(sdf, tmp_path / 'missing' / 'source.sdf')


def test_missing_physics_mesh_is_not_silently_replaced(tmp_path):
    physics = tmp_path / 'shape.sdf'
    physics.write_text('<sdf><model><link><collision><geometry><mesh>'
                       '<uri>missing.obj</uri></mesh></geometry></collision></link></model></sdf>')
    with pytest.raises(FileNotFoundError):
        _archive_physics_sdf(physics, tmp_path / 'out.sdf')
