"""Conservative simplification of redundant decompositions of convex solids.

SceneCode sometimes decomposes even a sphere into dozens of overlapping hulls.
Their competing contact manifolds can retain a light object inside an open jaw.
Only a single, watertight, convex visual solid qualifies. The replacement hull
contains ALL existing collision vertices: no cavities are removed from concave
visual objects, and no existing collision material is excluded.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def _resolve(base: Path, uri: str) -> Path:
    path = Path(uri.removeprefix("file://"))
    return path if path.is_absolute() else base / path


def simplify_convex_collision(source: Path, prepared: Path) -> dict | None:
    if prepared.suffix != '.sdf':
        return None
    src = ET.parse(source).getroot()
    tree = ET.parse(prepared)
    links = tree.findall('.//link')
    visuals = src.findall('.//visual')
    if len(links) != 1 or len(visuals) != 1:
        return None
    collisions = links[0].findall('collision')
    if len(collisions) < 2:
        return None
    visual_uri = visuals[0].find('geometry/mesh/uri')
    if visual_uri is None:
        return None
    # Collision simplification uses only geometry. Avoid decoding render-only
    # materials/textures: generated GLTF texture payloads can be malformed or
    # truncated without invalidating the mesh or its collision geometry.
    visual = trimesh.load(_resolve(source.parent, visual_uri.text), force='mesh',
                          skip_materials=True)
    # glTF UV/normal seams are duplicated vertices, not physical boundaries.
    visual.merge_vertices(merge_tex=True, merge_norm=True)
    visual.update_faces(visual.nondegenerate_faces())
    visual.remove_unreferenced_vertices()
    if not visual.is_watertight or not visual.is_convex:
        return None
    vertices = []
    for collision in collisions:
        mesh_node = collision.find('geometry/mesh')
        if mesh_node is None or mesh_node.find('uri') is None:
            return None
        mesh = trimesh.load(_resolve(prepared.parent, mesh_node.find('uri').text), force='mesh')
        points = np.array(mesh.vertices, copy=True)
        scale = mesh_node.find('scale')
        if scale is not None:
            points *= np.asarray([float(v) for v in scale.text.split()])
        pose = collision.find('pose')
        if pose is not None:
            if pose.get('relative_to'):
                return None
            values = [float(v) for v in pose.text.split()]
            if len(values) != 6:
                return None
            rotation = trimesh.transformations.euler_matrix(*values[3:])[:3, :3]
            points = points @ rotation.T + values[:3]
        vertices.append(points)
    points = np.vstack(vertices)
    hull = trimesh.convex.convex_hull(points)
    # Explicit containment certificate; never shrink the original union.
    if np.max(points @ hull.face_normals.T -
              np.einsum('ij,ij->i', hull.triangles_center, hull.face_normals)) > 1e-8:
        raise ValueError('replacement hull excludes original collision vertices')
    path = prepared.with_name(prepared.stem + '_convex_solid.obj')
    hull.export(path)
    new = ET.Element('collision', name='convex_solid')
    surface = collisions[0].find('surface')
    if surface is not None:
        new.append(ET.fromstring(ET.tostring(surface)))
    mesh = ET.SubElement(ET.SubElement(new, 'geometry'), 'mesh')
    ET.SubElement(mesh, 'uri').text = str(path.resolve())
    for collision in collisions:
        links[0].remove(collision)
    links[0].append(new)
    tree.write(prepared, encoding='utf-8', xml_declaration=True)
    return {'policy': 'convex_visual_conservative_collision_hull',
            'original_collision_count': len(collisions),
            'replacement_collision_count': 1,
            'original_vertices': len(points), 'hull_volume_m3': float(hull.volume)}
