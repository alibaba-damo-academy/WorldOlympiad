from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from wo.sim.convex_collision import simplify_convex_collision


def fixture(tmp_path, visual):
    visual.export(tmp_path / 'visual.obj')
    parts = []
    for i, x in enumerate((-.2, .2)):
        mesh = trimesh.creation.box(extents=(.6, 1, 1))
        mesh.export(tmp_path / f'part{i}.obj')
        parts.append(f'<collision name="p{i}"><pose>{x} 0 0 0 0 0</pose>'
                     f'<geometry><mesh><uri>part{i}.obj</uri></mesh></geometry></collision>')
    src = tmp_path / 'source.sdf'
    src.write_text('<sdf><model><link name="base"><inertial><mass>0.002</mass></inertial>'
                   '<visual><geometry><mesh><uri>visual.obj</uri></mesh></geometry></visual>'
                   + ''.join(parts) + '</link></model></sdf>')
    dst = tmp_path / 'physics.sdf'
    dst.write_text(src.read_text())
    return src, dst


def test_convex_solid_contains_every_original_collision_vertex(tmp_path):
    src, dst = fixture(tmp_path, trimesh.creation.box())
    result = simplify_convex_collision(src, dst)
    assert result['original_collision_count'] == 2
    root = ET.parse(dst)
    assert len(root.findall('.//collision')) == 1
    assert root.find('.//mass').text == '0.002'
    hull = trimesh.load(root.find('.//collision//uri').text, force='mesh')
    np.testing.assert_allclose(hull.bounds, [[-.5, -.5, -.5], [.5, .5, .5]])
    saved = dst.read_bytes()
    assert simplify_convex_collision(src, dst) is None
    assert dst.read_bytes() == saved


def test_concave_visual_preserves_cavities(tmp_path):
    # Two separated solids must NOT become a solid bridge.
    a = trimesh.creation.box(); b = a.copy(); b.apply_translation((2, 0, 0))
    src, dst = fixture(tmp_path, trimesh.util.concatenate([a, b]))
    before = dst.read_bytes()
    assert simplify_convex_collision(src, dst) is None
    assert dst.read_bytes() == before


def test_open_surface_not_certified_as_convex_solid(tmp_path):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    src, dst = fixture(tmp_path, mesh)
    assert simplify_convex_collision(src, dst) is None
    assert len(ET.parse(dst).findall('.//collision')) == 2


def test_articulated_prepared_geometry_is_untouched(tmp_path):
    src, dst = fixture(tmp_path, trimesh.creation.box())
    root = ET.parse(dst)
    ET.SubElement(root.find('model'), 'link', name='moving')
    root.write(dst)
    before = dst.read_bytes()
    assert simplify_convex_collision(src, dst) is None
    assert dst.read_bytes() == before


def test_truncated_gltf_texture_does_not_block_geometry_simplification(tmp_path):
    import json
    from io import BytesIO

    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    visual = trimesh.creation.box()
    visual.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(visual.vertices), 2)),
        material=PBRMaterial(baseColorTexture=Image.new('RGB', (4, 4), 'red')))
    exported = visual.export(tmp_path / 'visual.gltf')
    for name, data in exported.items():
        (tmp_path / name).write_bytes(data)

    # Damage only the embedded, render-only PNG payload, like an interrupted
    # texture write. Mesh buffers and SDF collision geometry remain intact.
    header = json.loads((tmp_path / 'visual.gltf').read_text())
    image_view = header['images'][0]['bufferView']
    image_buffer = header['bufferViews'][image_view]['buffer']
    texture_path = tmp_path / header['buffers'][image_buffer]['uri']
    texture_bytes = texture_path.read_bytes()
    texture_path.write_bytes(texture_bytes[:24] + b'\x00' * (len(texture_bytes) - 24))
    try:
        Image.open(BytesIO(texture_path.read_bytes())).load()
    except OSError:
        pass
    else:
        raise AssertionError('fixture texture was not truncated')

    part = trimesh.creation.box(extents=(.6, 1, 1))
    part.export(tmp_path / 'part.obj')
    collisions = ''.join(
        f'<collision name="p{i}"><pose>{x} 0 0 0 0 0</pose>'
        '<geometry><mesh><uri>part.obj</uri></mesh></geometry></collision>'
        for i, x in enumerate((-.2, .2)))
    sdf = ('<sdf><model><link name="base"><inertial><mass>0.002</mass></inertial>'
           '<visual><geometry><mesh><uri>visual.gltf</uri></mesh></geometry></visual>'
           + collisions + '</link></model></sdf>')
    src = tmp_path / 'source.sdf'
    dst = tmp_path / 'physics.sdf'
    src.write_text(sdf)
    dst.write_text(sdf)

    result = simplify_convex_collision(src, dst)
    assert result is not None
    assert result['original_collision_count'] == 2
    assert len(ET.parse(dst).findall('.//collision')) == 1
