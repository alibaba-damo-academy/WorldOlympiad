"""Exact convex partition for watertight vertical, stepped extrusions.

Unlike approximate convex decomposition, triangle prisms preserve holes. Only
solids with vertical walls, horizontal roofs and no downward undercuts qualify.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from .convex_collision import _resolve


def partition_prismatic_mesh(mesh):
    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        return None
    z = mesh.vertices[:, 2]
    levels = np.unique(np.round(z, 7))
    if len(levels) < 2 or len(levels) > 16:
        return None
    normals = mesh.face_normals
    horizontal = np.abs(normals[:, 2]) > 1 - 1e-7
    vertical = np.abs(normals[:, 2]) < 1e-7
    if not np.all(horizontal | vertical):
        return None
    triangles = mesh.triangles
    downward = horizontal & (normals[:, 2] < 0)
    if np.any(np.abs(triangles[downward, :, 2] - levels[0]) > 2e-7):
        return None  # overhangs need a different partition
    roofs = triangles[horizontal & (normals[:, 2] > 0)]
    pieces = []
    for low, high in zip(levels[:-1], levels[1:]):
        for roof in roofs:
            if roof[0, 2] < high - 2e-7:
                continue
            lower = roof.copy(); lower[:, 2] = low
            upper = roof.copy(); upper[:, 2] = high
            hull = trimesh.convex.convex_hull(np.vstack([lower, upper]))
            pieces.append(hull)
    volume = sum(p.volume for p in pieces)
    if not pieces or not np.isclose(volume, mesh.volume, rtol=1e-5, atol=1e-12):
        return None
    return pieces


def preserve_prismatic_cavities(source: Path, prepared: Path):
    if prepared.suffix != '.sdf':
        return None
    marker = prepared.with_suffix('.prismatic.json')
    if marker.exists():
        import json
        return json.loads(marker.read_text())
    src = ET.parse(source)
    tree = ET.parse(prepared)
    visuals = src.findall('.//visual')
    links = tree.findall('.//link')
    if len(visuals) != 1 or len(links) != 1:
        return None
    visual = visuals[0]
    # Nontrivial mesh/visual transforms require a separate certified conversion.
    if visual.find('pose') is not None or visual.find('geometry/mesh/scale') is not None:
        return None
    uri = visual.find('geometry/mesh/uri')
    if uri is None:
        return None
    path = _resolve(source.parent, uri.text)
    # Collision partitioning uses only geometric topology; generated glTF
    # textures can be truncated without affecting the collision mesh.
    mesh = trimesh.load(path, force='mesh', skip_materials=True)
    if path.suffix.lower() in ('.gltf', '.glb'):
        # glTF exporter is Y-up; collision OBJ/SDF are Z-up.
        mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, -1, 1]
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if mesh.is_convex:
        return None  # keep the existing conservative solid policy
    pieces = partition_prismatic_mesh(mesh)
    if pieces is None:
        return None
    collisions = links[0].findall('collision')
    if not collisions:
        return None
    surface = collisions[0].find('surface')
    for i, piece in enumerate(pieces):
        output = prepared.with_name(prepared.stem + f'_prism_{i}.obj')
        piece.export(output)
        node = ET.SubElement(links[0], 'collision', name=f'certified_prism_{i}')
        if surface is not None:
            node.append(ET.fromstring(ET.tostring(surface)))
        ET.SubElement(ET.SubElement(ET.SubElement(node, 'geometry'), 'mesh'), 'uri').text = str(output.resolve())
    for collision in collisions:
        links[0].remove(collision)
    tree.write(prepared, encoding='utf-8', xml_declaration=True)
    evidence = {'policy': 'exact_vertical_triangle_prisms', 'pieces': len(pieces),
                'visual_volume_m3': float(mesh.volume),
                'partition_volume_m3': float(sum(p.volume for p in pieces))}
    import json
    marker.write_text(json.dumps(evidence, indent=2))
    return evidence
