import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from wo.sim.sdf_urdf import sdf_to_urdf
from wo.models.cosmos import _archive_physics_urdf, _sha
from autoresearch.robotics_runtime_urdf_rebind import bind_runtime_physics_urdfs


def fixture(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'collision.obj').write_text('v 0 0 0\nv 1 0 0\n')
    (bundle / 'buffer.bin').write_bytes(b'1234')
    (bundle / 'visual.gltf').write_text(json.dumps({'buffers': [{'uri': 'buffer.bin'}]}))
    sdf = bundle / 'source.sdf'
    sdf.write_text('''<sdf version="1.7"><model name="b">
    <link name="base"><collision name="c"><geometry><mesh><uri>collision.obj</uri><scale>.2 .2 .2</scale></mesh></geometry></collision>
    <visual name="v"><pose>.01 .02 .03 0 0 .2</pose><geometry><mesh><uri>visual.gltf</uri><scale>.2 .2 .2</scale></mesh></geometry></visual></link>
    <link name="cap"><pose>0 0 .1 0 0 0</pose></link>
    <joint name="slide" type="prismatic"><parent>base</parent><child>cap</child><axis><xyz>0 0 1</xyz><limit><lower>0</lower><upper>.1</upper></limit></axis></joint>
    </model></sdf>''')
    original = sdf_to_urdf(sdf, tmp_path / 'original.urdf')
    _archive_physics_urdf(original, bundle / 'physics.urdf')
    manifest = bundle / 'manifest.json'
    manifest.write_text('{}')
    assets = {'b': {'source_sdf': 'source.sdf', 'physics_urdf': 'physics.urdf'}}
    digests = {p.relative_to(bundle).as_posix(): _sha(p) for p in bundle.rglob('*') if p.is_file()}
    return bundle, manifest, assets, digests


def test_visual_overlay_preserves_physics_and_binds_dependencies(tmp_path):
    bundle, manifest, assets, digests = fixture(tmp_path)
    before = ET.parse(bundle / 'physics.urdf')
    out = tmp_path / 'runtime'
    result = ET.parse(bind_runtime_physics_urdfs(manifest, assets, out, digests)['b'])
    for tag in ('joint', 'inertial'):
        assert [ET.tostring(x) for x in result.findall('.//' + tag)] == [ET.tostring(x) for x in before.findall('.//' + tag)]
    visual = result.find('.//visual')
    assert visual.find('origin').get('xyz') == '0.01 0.02 0.03'
    receipt = json.loads((out / 'urdf_asset_binding_receipt.json').read_text())
    binding = receipt['assets'][0]['visual_binding']
    assert binding['meshes'][0]['resources'][0]['sha256'] == _sha(bundle / 'buffer.bin')


@pytest.mark.parametrize('change', ['scale', 'frame', 'dependency', 'existing_visual'])
def test_visual_overlay_fails_closed(tmp_path, change):
    bundle, manifest, assets, digests = fixture(tmp_path)
    sdf = bundle / 'source.sdf'
    physics = bundle / 'physics.urdf'
    if change == 'scale':
        sdf.write_text(sdf.read_text().replace('.2 .2 .2', '.3 .3 .3'))
        digests['source.sdf'] = _sha(sdf)
    elif change == 'frame':
        sdf.write_text(sdf.read_text().replace('0 0 .1 0 0 0', '0 0 .2 0 0 0'))
        digests['source.sdf'] = _sha(sdf)
    elif change == 'dependency':
        (bundle / 'buffer.bin').write_bytes(b'tampered')
    else:
        tree = ET.parse(physics)
        ET.SubElement(tree.find('link'), 'visual')
        tree.write(physics)
        digests['physics.urdf'] = _sha(physics)
    with pytest.raises(ValueError):
        bind_runtime_physics_urdfs(manifest, assets, tmp_path / 'runtime', digests)
