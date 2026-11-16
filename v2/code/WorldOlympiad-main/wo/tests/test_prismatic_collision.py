import json
from io import BytesIO
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

from wo.sim.prismatic_collision import partition_prismatic_mesh, preserve_prismatic_cavities


def test_box_partition_preserves_volume():
    box = trimesh.creation.box(extents=[.08, .08, .05])
    pieces = partition_prismatic_mesh(box)
    assert pieces
    assert np.isclose(sum(p.volume for p in pieces), box.volume)


def test_annular_extrusion_preserves_hole():
    # Watertight square ring: each side is a rectangular extrusion.
    outer = np.array([[-2,-2], [2,-2], [2,2], [-2,2]])
    inner = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
    v = np.vstack([np.c_[outer, np.zeros(4)], np.c_[inner, np.zeros(4)],
                   np.c_[outer, np.ones(4)], np.c_[inner, np.ones(4)]])
    faces = []
    for i in range(4):
        j = (i+1)%4
        for quad in [(i,j,j+8,i+8), (i+4,i+12,j+12,j+4),
                     (i+8,j+8,j+12,i+12), (i+4,j+4,j,i)]:
            a,b,c,d = quad; faces.extend([(a,b,c), (a,c,d)])
    mesh = trimesh.Trimesh(v, faces, process=True)
    mesh.fix_normals()
    pieces = partition_prismatic_mesh(mesh)
    assert pieces
    assert np.isclose(sum(p.volume for p in pieces), 12)
    # No prism may cover the hole's center at mid-height.
    for p in pieces:
        assert np.any(p.face_normals @ np.array([0,0,.5]) -
                      np.einsum('ij,ij->i',p.face_normals,p.triangles_center) > 1e-8)


def test_sloping_and_open_meshes_rejected():
    assert partition_prismatic_mesh(trimesh.creation.icosphere()) is None
    m = trimesh.creation.box(); m.update_faces(np.arange(len(m.faces)-1))
    assert partition_prismatic_mesh(m) is None


def test_truncated_gltf_texture_does_not_block_collision_mesh_inspection(tmp_path):
    visual = trimesh.creation.box()
    visual.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(visual.vertices), 2)),
        material=PBRMaterial(baseColorTexture=Image.new('RGB', (4, 4), 'red')))
    exported = visual.export(tmp_path / 'visual.gltf')
    for name, data in exported.items():
        (tmp_path / name).write_bytes(data)

    # A damaged render-only PNG must not prevent reading the intact geometry.
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

    collision = trimesh.creation.box().export(tmp_path / 'collision.obj')
    src = tmp_path / 'source.sdf'
    prepared = tmp_path / 'prepared.sdf'
    sdf = ('<sdf><model><link name="base"><visual><geometry><mesh>'
           '<uri>visual.gltf</uri></mesh></geometry></visual><collision>'
           '<geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision>'
           '</link></model></sdf>')
    src.write_text(sdf)
    prepared.write_text(sdf)
    # Convex visuals need no special cavity partition, but must load without
    # decoding the malformed texture (the formerly failing production path).
    assert preserve_prismatic_cavities(src, prepared) is None
    assert len(ET.parse(prepared).findall('.//collision')) == 1
