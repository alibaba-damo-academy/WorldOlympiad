"""Conservative convex decomposition for closed shell collision meshes.

A single concave mesh must never be passed through a convex-hull fallback: that
closes cavities.  This exporter emits one thin convex prism per source triangle.
The prisms retain every source face (including inner cavity faces), so the
result is conservative locally without filling the vessel.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from .convex_collision import _resolve


def _indent_xml(element, level=0):
    """Pretty-print generated SDF so native XML readers get bounded lines."""
    pad = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = pad + "  "
        for child in element:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not element.tail or not element.tail.strip()):
        element.tail = "\n" + "  " * (level - 1)


def _mesh_from_collision(source: Path):
    root = ET.parse(source).getroot()
    model = root.find('.//model')
    link = model.find('link') if model is not None else None
    collisions = link.findall('collision') if link is not None else []
    if len(collisions) != 1:
        return None
    geom = collisions[0].find('geometry/mesh')
    if geom is None or geom.find('uri') is None:
        return None
    mesh = trimesh.load(_resolve(source.parent, geom.find('uri').text), force='mesh')
    scale = geom.find('scale')
    if scale is not None:
        mesh.vertices *= np.asarray([float(x) for x in scale.text.split()])
    pose = collisions[0].find('pose')
    if pose is not None and (pose.get('relative_to') or len(pose.text.split()) not in (3, 6)):
        return None
    if pose is not None:
        v = [float(x) for x in pose.text.split()]
        if len(v) == 6:
            mesh.apply_transform(trimesh.transformations.euler_matrix(*v[3:]))
        mesh.apply_translation(v[:3])
    return mesh, collisions[0]


def preserve_concave_shell(source: Path, prepared: Path):
    """Replace a concave single collision with face-preserving convex prisms."""
    if prepared.suffix != '.sdf':
        return None
    marker = prepared.with_suffix('.concave.json')
    if marker.exists():
        return json.loads(marker.read_text())
    result = _mesh_from_collision(source)
    if result is None:
        return None
    mesh, source_collision = result
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    # Face-preserving prisms are intentionally valid for shells whose exporter
    # left coincident seam vertices: watertightness is not a safe admission
    # criterion here, while convex input still needs the ordinary path. Keep a
    # finite guard for pathological meshes without silently convexifying them.
    if mesh.is_convex or len(mesh.faces) > 10000:
        return None
    tree = ET.parse(prepared)
    link = tree.find('.//link')
    collisions = link.findall('collision') if link is not None else []
    if link is None or not collisions:
        return None
    surface = source_collision.find('surface')
    scale = max(float(np.max(mesh.extents)), 1e-3)
    eps = max(scale * 2e-5, 2e-7)
    outputs = []
    for i, (face, normal) in enumerate(zip(mesh.faces, mesh.face_normals)):
        tri = mesh.vertices[face]
        pts = np.vstack((tri - eps * normal, tri + eps * normal))
        piece = trimesh.convex.convex_hull(pts)
        path = prepared.with_name(f'{prepared.stem}_shell_{i:05d}.obj')
        piece.export(path)
        node = ET.Element('collision', name=f'concave_shell_{i:05d}')
        if surface is not None:
            node.append(ET.fromstring(ET.tostring(surface)))
        uri = ET.SubElement(ET.SubElement(node, 'geometry'), 'mesh').find('uri')
        if uri is None:
            uri = ET.SubElement(node.find('geometry/mesh'), 'uri')
        uri.text = str(path.resolve())
        link.append(node)
        outputs.append(path.name)
    for collision in collisions:
        link.remove(collision)
    # ElementTree otherwise serializes every generated collision on one line.
    # Large, valid SDF documents then exceed the simulator XML parser's line
    # handling limits even though the XML is well formed.  Keep the physical
    # decomposition unchanged, but emit bounded lines for native loaders.
    _indent_xml(tree.getroot())
    tree.write(prepared, encoding='utf-8', xml_declaration=True)
    evidence = {'policy': 'face_preserving_convex_shell_decomposition',
                'source_faces': int(len(mesh.faces)), 'pieces': len(outputs),
                'thickness_m': float(2 * eps), 'pieces_files': outputs}
    marker.write_text(json.dumps(evidence, indent=2))
    return evidence
