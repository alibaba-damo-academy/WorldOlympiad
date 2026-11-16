import hashlib
import json
import xml.etree.ElementTree as ET

import trimesh

from wo.tasks.visual_buffer_recovery import recover_overwritten_visual_buffers


def test_shared_gltf_buffer_collision_rebinds_only_broken_visual_and_preserves_collision(tmp_path):
    directory = tmp_path / 'assets'
    directory.mkdir()
    mesh = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    canonical = directory / 'first_canonical.gltf'
    for name, payload in mesh.export(file_type='gltf').items():
        (directory / (canonical.name if name == 'model.gltf' else name)).write_bytes(payload)
    # A second independent export overwrites the first export's generic buffer.
    (directory / 'gltf_buffer_0.bin').write_bytes(b'bad')
    (directory / 'first.gltf').write_bytes(canonical.read_bytes())
    # Restore canonical with a different namespaced buffer URI.
    doc = json.loads(canonical.read_text())
    for buffer in doc['buffers']:
        original = directory / buffer['uri']
        original.write_bytes(b'x' * buffer['byteLength'])
        renamed = 'first_' + buffer['uri']
        (directory / renamed).write_bytes(original.read_bytes())
        buffer['uri'] = renamed
    # Real mesh data, not dummy bytes, for trimesh scene validation.
    source = mesh.export(file_type='gltf')
    for buffer in doc['buffers']:
        (directory / buffer['uri']).write_bytes(source[buffer['uri'].removeprefix('first_')])
    canonical.write_text(json.dumps(doc))
    (directory / 'gltf_buffer_0.bin').write_bytes(b'bad')
    sdf = directory / 'first.sdf'
    sdf.write_text('<sdf><model><link><visual><geometry><mesh><uri>first.gltf</uri></mesh></geometry></visual>'
                   '<collision><geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision></link></model></sdf>')
    (directory / 'collision.obj').write_text('v 0 0 0\n')
    cert = tmp_path / 'task_inventory_certificate.json'
    cert.write_text(json.dumps({'matches': [{'sdf_path': str(sdf), 'sdf_sha256': 'old'}]}))
    manifest = tmp_path / 'generation_manifest.json'
    manifest.write_text('{}')
    changes = recover_overwritten_visual_buffers(tmp_path)
    assert len(changes) == 1
    assert ET.parse(sdf).findtext('.//visual/geometry/mesh/uri') == 'first_canonical.gltf'
    assert ET.parse(sdf).findtext('.//collision/geometry/mesh/uri') == 'collision.obj'
    assert json.loads(cert.read_text())['matches'][0]['sdf_sha256'] == hashlib.sha256(sdf.read_bytes()).hexdigest()
    assert json.loads(manifest.read_text())['task_inventory_certificate_sha256'] == hashlib.sha256(cert.read_bytes()).hexdigest()
    assert recover_overwritten_visual_buffers(tmp_path) == []
