"""Fail closed on explicit physical-size contracts at the scene export boundary.

This checks exported geometry, not descriptions or inventory labels. It does not
resize assets or weaken task predicates. Functional-cavity metrology is separate.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def visual_bounds(sdf_path):
    path = Path(sdf_path)
    root = ET.parse(path)
    links = root.findall('.//link')
    if len(links) != 1:
        raise ValueError('size contract metrology requires a single-link asset')
    points = []
    for visual in links[0].findall('visual'):
        node = visual.find('geometry/mesh')
        if node is None:
            raise ValueError('size contract metrology requires mesh visuals')
        uri = node.findtext('uri', '').removeprefix('file://')
        mesh_path = Path(uri)
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = np.array(mesh.vertices, copy=True)
        # SceneCode exports visual glTF in Y-up and collision OBJ in Z-up.
        if mesh_path.suffix.lower() in {'.gltf', '.glb'}:
            vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
        scale = node.findtext('scale')
        if scale:
            vertices *= np.asarray([float(v) for v in scale.split()])
        pose = visual.find('pose')
        if pose is not None:
            if pose.get('relative_to'):
                raise ValueError('unsupported relative visual pose in size metrology')
            values = [float(v) for v in pose.text.split()]
            rotation = trimesh.transformations.euler_matrix(*values[3:])[:3, :3]
            vertices = vertices @ rotation.T + values[:3]
        points.append(vertices)
    if not points:
        raise ValueError('asset has no visual geometry for size contract')
    vertices = np.vstack(points)
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def collision_bounds(sdf_path, *, rotation=None):
    """Measure native collision meshes in the SDF's Z-up frame."""
    path = Path(sdf_path)
    root = ET.parse(path)
    links = root.findall('.//link')
    if len(links) != 1:
        raise ValueError('size contract metrology requires a single-link asset')
    points = []
    for collision in links[0].findall('collision'):
        node = collision.find('geometry/mesh')
        if node is None:
            raise ValueError('size contract metrology requires mesh collisions')
        uri = node.findtext('uri', '').removeprefix('file://')
        mesh_path = Path(uri)
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = np.array(mesh.vertices, copy=True)
        scale = node.findtext('scale')
        if scale:
            vertices *= np.asarray([float(v) for v in scale.split()])
        pose = collision.find('pose')
        if pose is not None:
            if pose.get('relative_to'):
                raise ValueError('unsupported relative collision pose in size metrology')
            values = [float(v) for v in pose.text.split()]
            pose_rotation = trimesh.transformations.euler_matrix(*values[3:])[:3, :3]
            vertices = vertices @ pose_rotation.T + values[:3]
        points.append(vertices)
    if not points:
        raise ValueError('asset has no collision geometry for size metrology')
    vertices = np.vstack(points)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)):
            raise ValueError("collision measurement needs a finite proper rotation")
        vertices = vertices @ rotation.T
    return np.array([vertices.min(axis=0), vertices.max(axis=0)])


def validate_size_contracts(spec, scene):
    records = {}
    for obj in spec.objects:
        expected = obj.geometry_constraints.get('overall_dimensions_m')
        if expected is None:
            continue
        expected = np.asarray(expected, dtype=float)
        if expected.shape != (3,) or not np.all(np.isfinite(expected)) or np.any(expected <= 0):
            raise ValueError('overall_dimensions_m must be three finite positive sizes')
        asset = scene.support if obj.object_id == scene.support.object_id else scene.objects[obj.object_id]
        visual = np.diff(visual_bounds(asset.sdf_path), axis=0)[0] * asset.scale
        # A published physical contract is never satisfied by visual geometry
        # alone.  Missing collision is an admission failure (visual-only
        # fixtures may still be used when no contract is declared).
        xml_root = ET.parse(asset.sdf_path)
        has_collision = bool(xml_root.findall('.//link/collision'))
        collision = (np.diff(collision_bounds(asset.sdf_path), axis=0)[0] * asset.scale
                     if has_collision else None)
        visual_passed = bool(np.all(np.abs(visual - expected) <= 0.0005))
        collision_passed = (has_collision and
                            bool(np.all(np.abs(collision - expected) <= 0.0005)))
        visual_center = (visual_bounds(asset.sdf_path)[0] + visual_bounds(asset.sdf_path)[1]) / 2
        collision_center = (collision_bounds(asset.sdf_path)[0] + collision_bounds(asset.sdf_path)[1]) / 2 if has_collision else None
        center_delta = None if collision_center is None else (collision_center - visual_center) * asset.scale
        center_passed = (center_delta is not None and
                         bool(np.all(np.abs(center_delta) <= 0.0005)))
        records[obj.object_id] = {'expected_m': expected.tolist(), 'measured_m': visual.tolist(),
                                  'collision_m': None if collision is None else collision.tolist(),
                                  'center_delta_m': None if center_delta is None else center_delta.tolist(),
                                  'tolerance_m': 0.0005,
                                  'passed': visual_passed and collision_passed and center_passed}
    return {'schema_version': 'wo_scene_size_validation.v1', 'objects': records,
            'success': all(r['passed'] for r in records.values())}
