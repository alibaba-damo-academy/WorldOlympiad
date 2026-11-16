"""Resolve missing generated inertial frames at the physics projection boundary."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from .convex_collision import _resolve


def repair_missing_inertial_poses(sdf: Path) -> list[dict]:
    """Place incomplete-frame inertials at the measured collision-mesh COM.

    This is deliberately limited to a missing pose; authored poses and tensors
    are immutable.  Collision bounds are in the native SDF frame, so this also
    prevents a translated mesh from silently receiving a link-origin COM.
    """
    tree = ET.parse(sdf)
    repairs = []
    for link in tree.findall('.//link'):
        inertial = link.find('inertial')
        if inertial is None or inertial.find('pose') is not None:
            continue
        points = []
        for collision in link.findall('collision'):
            geom = collision.find('geometry/mesh')
            if geom is None or geom.find('uri') is None:
                continue
            mesh = trimesh.load(_resolve(sdf.parent, geom.find('uri').text), force='mesh')
            scale = geom.find('scale')
            if scale is not None:
                mesh.vertices *= np.asarray([float(v) for v in scale.text.split()])
            points.append(np.asarray(mesh.vertices))
        if not points:
            continue
        vertices = np.vstack(points)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        pose = ET.Element('pose')
        pose.text = ' '.join(f'{float(v):.12g}' for v in (*center, 0.0, 0.0, 0.0))
        inertial.insert(1 if len(inertial) else 0, pose)
        repairs.append({'link': link.get('name', ''), 'center_m': center.tolist(),
                        'policy': 'collision_aabb_center_for_missing_inertial_pose'})
    if repairs:
        tree.write(sdf, encoding='utf-8', xml_declaration=True)
    return repairs
