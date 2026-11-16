"""Normalize small axis-wise export errors for explicit generated-asset size contracts.

The transform is applied to the exported SDF mesh instances (visual and collision),
not task prose or authored dimensions.  It is deliberately narrow: only single-link,
mesh-only assets with unrotated, link-relative instances and small corrections are
supported.  The unchanged physical-size validator remains the acceptance gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from .geometry_contract import visual_bounds

_MAX_AXIS_FACTOR = 1.10
_MIN_AXIS_FACTOR = 1.0 / _MAX_AXIS_FACTOR


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh_instances(root: ET.Element, sdf_path: Path):
    links = root.findall('.//link')
    if len(links) != 1:
        raise ValueError('size normalization requires a single-link asset')
    link = links[0]
    visuals = link.findall('visual')
    collisions = link.findall('collision')
    if not visuals or not collisions:
        raise ValueError('size normalization requires visual and collision meshes')
    instances = []
    for kind, nodes in (('visual', visuals), ('collision', collisions)):
        for instance in nodes:
            geometry = instance.find('geometry')
            mesh = geometry.find('mesh') if geometry is not None else None
            if mesh is None or mesh.find('uri') is None:
                raise ValueError(f'size normalization supports mesh {kind} geometry only')
            uri = mesh.findtext('uri', '').removeprefix('file://')
            source = Path(uri)
            if not source.is_absolute():
                source = sdf_path.parent / source
            if not source.is_file():
                raise FileNotFoundError(f'missing generated {kind} mesh: {source}')
            pose = instance.find('pose')
            if pose is not None and pose.get('relative_to'):
                raise ValueError(f'size normalization does not support relative {kind} poses')
            pose_values = np.zeros(6, dtype=float)
            if pose is not None:
                raw = [float(v) for v in (pose.text or '').split()]
                if len(raw) not in (3, 6) or not np.isfinite(raw).all():
                    raise ValueError(f'invalid {kind} pose')
                pose_values[:len(raw)] = raw
            if not np.allclose(pose_values[3:], 0, atol=1e-10):
                raise ValueError(f'size normalization does not support rotated {kind} poses')
            scale = mesh.find('scale')
            mesh_scale = np.ones(3, dtype=float)
            if scale is not None:
                try:
                    mesh_scale = np.asarray([float(v) for v in (scale.text or '').split()])
                except ValueError as exc:
                    raise ValueError(f'invalid {kind} mesh scale') from exc
                if (mesh_scale.shape != (3,) or not np.isfinite(mesh_scale).all()
                        or np.any(mesh_scale <= 0)):
                    raise ValueError(f'invalid {kind} mesh scale')
            instances.append((kind, instance, mesh, pose, pose_values, scale, mesh_scale))
    return instances


def _transform_inertial(link: ET.Element, center: np.ndarray, factors: np.ndarray) -> dict | None:
    inertial = link.find('inertial')
    if inertial is None:
        return None
    mass_node = inertial.find('mass')
    inertia = inertial.find('inertia')
    if mass_node is None or inertia is None:
        raise ValueError('size normalization requires complete inertial mass properties')
    mass = float(mass_node.text)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('size normalization requires a finite positive inertial mass')
    pose = inertial.find('pose')
    values = np.zeros(6, dtype=float)
    if pose is not None:
        if pose.get('relative_to'):
            raise ValueError('size normalization does not support relative inertial poses')
        raw = [float(v) for v in (pose.text or '').split()]
        if len(raw) not in (3, 6) or not np.isfinite(raw).all():
            raise ValueError('invalid inertial pose')
        values[:len(raw)] = raw
    fields = ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')
    vals = {name: float(inertia.findtext(name, 'nan')) for name in fields}
    if not np.isfinite(list(vals.values())).all():
        raise ValueError('size normalization requires finite inertia tensor components')
    tensor = np.array([[vals['ixx'], vals['ixy'], vals['ixz']],
                       [vals['ixy'], vals['iyy'], vals['iyz']],
                       [vals['ixz'], vals['iyz'], vals['izz']]])
    # Convert inertia to its second-moment tensor, transform in link coordinates,
    # then convert back while retaining the same mass and inertial-frame axes.
    second = np.eye(3) * (np.trace(tensor) / 2.0) - tensor
    rotation = trimesh.transformations.euler_matrix(*values[3:])[:3, :3]
    second_link = rotation @ second @ rotation.T
    second_link = factors[:, None] * second_link * factors[None, :]
    tensor_link = np.eye(3) * np.trace(second_link) - second_link
    tensor_new = rotation.T @ tensor_link @ rotation
    old_values = values.copy()
    values[:3] = center + (values[:3] - center) * factors
    for name, value in zip(fields, (tensor_new[0, 0], tensor_new[1, 1], tensor_new[2, 2],
                                   tensor_new[0, 1], tensor_new[0, 2], tensor_new[1, 2])):
        inertia.find(name).text = f'{value:.12g}'
    if pose is None:
        pose = ET.Element('pose')
        inertial.insert(0, pose)
    pose.text = ' '.join(f'{v:.12g}' for v in values)
    return {'mass_kg': mass, 'old_pose': old_values.tolist(),
            'new_pose': values.tolist(), 'old_inertia': vals,
            'new_inertia': {name: float(value) for name, value in zip(fields,
                (tensor_new[0, 0], tensor_new[1, 1], tensor_new[2, 2],
                 tensor_new[0, 1], tensor_new[0, 2], tensor_new[1, 2]))}}


def _bind_changed_sdfs(generation_dir: Path, changed: list[dict]) -> None:
    certificate_path = generation_dir / 'task_inventory_certificate.json'
    manifest_path = generation_dir / 'generation_manifest.json'
    if not certificate_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError('size normalization requires the current inventory certificate and generation manifest')
    certificate = json.loads(certificate_path.read_text())
    expected = {Path(record['sdf_path']).resolve(): record['sdf_sha256'] for record in changed}
    updated = set()
    for match in certificate.get('matches', []):
        raw = Path(match.get('sdf_path', ''))
        path = raw if raw.is_absolute() else generation_dir / raw
        resolved = path.resolve()
        if resolved in expected:
            match['sdf_sha256'] = expected[resolved]
            updated.add(resolved)
    if updated != set(expected):
        raise ValueError('modified size-normalized assets are not all bound by the inventory certificate')
    certificate_path.write_text(json.dumps(certificate, indent=2) + '\n')
    manifest = json.loads(manifest_path.read_text())
    manifest['task_inventory_certificate_sha256'] = _digest(certificate_path)
    report_path = generation_dir / 'asset_size_normalization.json'
    report_path.write_text(json.dumps({'schema_version': 'wo_asset_size_normalization.v1',
                                      'adjustments': changed}, indent=2) + '\n')
    manifest['asset_size_normalization'] = {'path': report_path.name, 'sha256': _digest(report_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


def normalize_asset_sizes(spec, scene, generation_dir: Path) -> dict:
    """Apply only a small anisotropic correction to explicit overall dimensions.

    Axis ratios outside [1/1.10, 1.10] and unsupported SDF transforms fail closed.
    The resulting asset is still independently checked by validate_size_contracts.
    """
    generation_dir = Path(generation_dir)
    changed = []
    planned_writes = []
    for declaration in spec.objects:
        expected = declaration.geometry_constraints.get('overall_dimensions_m')
        if expected is None:
            continue
        target = np.asarray(expected, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or np.any(target <= 0):
            raise ValueError('overall_dimensions_m must be three finite positive sizes')
        asset = (scene.support if declaration.object_id == scene.support.object_id
                 else scene.objects[declaration.object_id])
        sdf_path = Path(asset.sdf_path)
        root = ET.parse(sdf_path).getroot()
        measured = np.diff(visual_bounds(sdf_path), axis=0)[0] * float(asset.scale)
        factors = target / measured
        if np.all(np.abs(measured - target) <= 0.0005):
            continue
        if (not np.isfinite(factors).all() or np.any(factors < _MIN_AXIS_FACTOR)
                or np.any(factors > _MAX_AXIS_FACTOR)):
            raise ValueError(f'physical size correction exceeds safe axis-wise limits for '
                             f'{declaration.object_id}: measured={measured.tolist()}, '
                             f'target={target.tolist()}, factors={factors.tolist()}')
        instances = _mesh_instances(root, sdf_path)
        bounds = visual_bounds(sdf_path)
        center = bounds.mean(axis=0)
        inertial_record = _transform_inertial(root.findall('.//link')[0], center, factors)
        for kind, instance, mesh, pose, pose_values, scale, mesh_scale in instances:
            # Uniform model-instance scale is applied by resolve_scenecode after SDF
            # loading; the same link-frame affine is therefore independent of it.
            if scale is None:
                scale = ET.SubElement(mesh, 'scale')
            scale.text = ' '.join(f'{v:.12g}' for v in mesh_scale * factors)
            translation = pose_values[:3]
            pose_values[:3] = center + (translation - center) * factors
            if pose is None:
                pose = ET.Element('pose')
                instance.insert(0, pose)
            pose.text = ' '.join(f'{v:.12g}' for v in pose_values)
        tree = ET.ElementTree(root)
        serialized = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        planned_writes.append((sdf_path, serialized))
        changed.append({'object_id': declaration.object_id, 'sdf_path': str(sdf_path),
                        'expected_m': target.tolist(), 'measured_before_m': measured.tolist(),
                        'axis_factors': factors.tolist(),
                        'visual_center_m': center.tolist(),
                        'inertial_update': inertial_record})
    if changed:
        certificate_path = generation_dir / 'task_inventory_certificate.json'
        manifest_path = generation_dir / 'generation_manifest.json'
        if not certificate_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError('size normalization requires the current inventory certificate and generation manifest')
        certificate = json.loads(certificate_path.read_text())
        bound = {}
        for match in certificate.get('matches', []):
            raw = Path(match.get('sdf_path', ''))
            path = raw if raw.is_absolute() else generation_dir / raw
            bound[path.resolve()] = match.get('sdf_sha256')
        for sdf_path, _ in planned_writes:
            if bound.get(sdf_path.resolve()) != _digest(sdf_path):
                raise ValueError(f'SDF is not bound to its current inventory hash: {sdf_path}')
        for sdf_path, serialized in planned_writes:
            tmp_path = sdf_path.with_name(sdf_path.name + '.normalizing')
            tmp_path.write_bytes(serialized)
            os.replace(tmp_path, sdf_path)
        for record in changed:
            record['sdf_sha256'] = _digest(Path(record['sdf_path']))
        _bind_changed_sdfs(generation_dir, changed)
    return {'schema_version': 'wo_asset_size_normalization.v1', 'adjustments': changed}
