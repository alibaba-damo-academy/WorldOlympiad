"""Rebind authoritative floating expert URDFs, preserving every joint and inertia.

No SDF substitution, joint locking or mesh reconstruction is permitted here.
The native compiler consumes the returned URDF with its measured link physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse

from .robotics_runtime_sdf_rebind import _inside, _source_path
from .robotics_views import sha256_file
from .io import write_json


def bind_runtime_physics_urdfs(reference_manifest: Path, prepared_assets: dict,
                               output_dir: Path, bundle_digests: dict[str, str]) -> dict[str, Path]:
    root = reference_manifest.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, derived = [], {}

    def verified(path: Path) -> tuple[Path, str, str]:
        path = _inside(root, path)
        relative = path.relative_to(root).as_posix()
        expected = bundle_digests.get(relative)
        if not expected or expected != sha256_file(path):
            raise ValueError(f'URDF bundle digest changed or missing: {relative}')
        return path, relative, expected

    def mesh_resources(mesh):
        resources = []
        if mesh.suffix.lower() == '.gltf':
            payload = json.loads(mesh.read_text())
            for collection in ('buffers', 'images'):
                for resource in payload.get(collection, []):
                    uri = resource.get('uri')
                    if not uri or uri.startswith('data:'):
                        continue
                    if urlparse(uri).scheme or Path(uri).is_absolute():
                        raise ValueError('archived glTF requires local relative resources')
                    _, dependency, digest = verified(mesh.parent / unquote(uri))
                    resources.append({'bundle_resource': dependency, 'sha256': digest})
        return resources

    def geometry_signature(element):
        # Compare converter output with archived physics, not topology alone.
        # This validates cumulative scale and the resolved link/joint frames.
        import numpy as np
        attrs = {}
        for key, value in element.attrib.items():
            if key == 'name':
                continue
            if key == 'filename':
                _, _, attrs[key] = verified(Path(value))
            else:
                try:
                    attrs[key] = tuple(float(x) for x in value.split())
                except ValueError:
                    attrs[key] = value
        return element.tag, attrs, [geometry_signature(c) for c in element]

    def equivalent(a, b):
        import numpy as np
        if a[0] != b[0] or a[1].keys() != b[1].keys() or len(a[2]) != len(b[2]):
            return False
        for key, value in a[1].items():
            other = b[1][key]
            if isinstance(value, tuple):
                if not isinstance(other, tuple) or len(value) != len(other) or not np.allclose(value, other, rtol=1e-8, atol=1e-10):
                    return False
            elif value != other:
                return False
        return all(equivalent(x, y) for x, y in zip(a[2], b[2]))

    for role, asset in prepared_assets.items():
        if not asset.get('physics_urdf'):
            continue
        if Path(role).name != role or role in {'.', '..'}:
            raise ValueError('URDF asset role must be a simple name')
        source, relative, digest = verified(_source_path(root, asset['physics_urdf']))
        tree = ET.parse(source)
        if tree.getroot().tag != 'robot':
            raise ValueError('expert physics URDF requires a robot root')
        bindings = []
        for node in tree.findall('.//mesh'):
            filename = node.get('filename')
            if not filename or Path(filename).is_absolute() or urlparse(filename).scheme:
                raise ValueError('archived URDF mesh requires bundle-relative filename')
            mesh, mesh_relative, mesh_digest = verified(source.parent / filename)
            resources = []
            if mesh.suffix.lower() == '.gltf':
                payload = json.loads(mesh.read_text())
                for collection in ('buffers', 'images'):
                    for resource in payload.get(collection, []):
                        uri = resource.get('uri')
                        if not uri or uri.startswith('data:'):
                            continue
                        if urlparse(uri).scheme or Path(uri).is_absolute():
                            raise ValueError('archived glTF requires local relative resources')
                        _, dependency, dependency_digest = verified(mesh.parent / unquote(uri))
                        resources.append({'bundle_resource': dependency, 'sha256': dependency_digest})
            node.set('filename', str(mesh))
            bindings.append({'bundle_mesh': mesh_relative, 'sha256': mesh_digest,
                             'resources': resources})
        visual_binding = None
        if asset.get('source_sdf'):
            # Expert URDF conversion deliberately strips visuals. Restore only
            # content-bound source visuals in the converter's exact link frame;
            # never change collisions, inertias, joints or their limits.
            import numpy as np
            from scipy.spatial.transform import Rotation
            from wo.sim.sdf_urdf import sdf_visuals
            visual_source, visual_relative, visual_digest = verified(
                _source_path(root, asset['source_sdf']))
            source_tree = ET.parse(visual_source)
            source_links = source_tree.findall('.//model/link')
            runtime_links = {link.get('name'): link for link in tree.findall('link')}
            if {link.get('name') for link in source_links} != set(runtime_links):
                raise ValueError('source visual and expert URDF link identities differ')
            source_joints = {j.get('name'): (j.get('type'), j.findtext('parent'), j.findtext('child'))
                             for j in source_tree.findall('.//model/joint')}
            runtime_joints = {j.get('name'): (j.get('type'), j.find('parent').get('link'),
                                             j.find('child').get('link')) for j in tree.findall('joint')}
            if source_joints != runtime_joints:
                raise ValueError('source visual and expert URDF joint topology differs')
            from wo.sim.sdf_urdf import sdf_to_urdf
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                converted = ET.parse(sdf_to_urdf(visual_source, Path(directory) / 'validation.urdf'))
            for expected in converted.findall('link'):
                actual = runtime_links[expected.get('name')]
                expected_collisions = expected.findall('collision')
                actual_collisions = actual.findall('collision')
                if len(expected_collisions) != len(actual_collisions) or not all(
                    equivalent(geometry_signature(x), geometry_signature(y))
                    for x, y in zip(expected_collisions, actual_collisions)):
                    raise ValueError('source visual scale/frame differs from authoritative collisions')
            actual_joints = {j.get('name'): j for j in tree.findall('joint')}
            for expected in converted.findall('joint'):
                actual = actual_joints[expected.get('name')]
                for tag in ('origin', 'axis'):
                    a, b = expected.find(tag), actual.find(tag)
                    if (a is None) != (b is None) or (a is not None and not equivalent(
                        geometry_signature(a), geometry_signature(b))):
                        raise ValueError('source visual joint frame differs from authoritative physics')
            visual_data = sdf_visuals(visual_source)
            visual_meshes = []
            for source_link in source_links:
                name = source_link.get('name')
                measured = visual_data['links'][name]
                declared = source_link.findall('visual')
                if len(declared) != len(measured['visuals']):
                    raise ValueError('source visual geometry was not fully represented')
                link = runtime_links[name]
                if link.findall('visual'):
                    raise ValueError('expert URDF already has visuals; source overlay is ambiguous')
                for original, geometry in zip(declared, measured['visuals'], strict=True):
                    mesh, mesh_relative, mesh_digest = verified(Path(geometry['file']))
                    transform = np.asarray(measured['urdf_T_link']) @ np.asarray(geometry['link_T_visual'])
                    visual = ET.SubElement(link, 'visual', name=original.get('name', 'visual'))
                    ET.SubElement(visual, 'origin', xyz=' '.join(map(str, transform[:3, 3])),
                        rpy=' '.join(map(str, Rotation.from_matrix(transform[:3, :3]).as_euler('xyz'))))
                    node = ET.SubElement(visual, 'geometry')
                    ET.SubElement(node, 'mesh', filename=str(mesh), scale=' '.join(map(str, geometry['scale'])))
                    visual_meshes.append({'bundle_mesh': mesh_relative, 'sha256': mesh_digest,
                                          'resources': mesh_resources(mesh)})
            visual_binding = {'source_sdf': visual_relative, 'sha256': visual_digest,
                              'meshes': visual_meshes, 'frame_protocol': 'sdf_to_urdf_link_frame'}
        target = output_dir / f'{role}.urdf'
        tree.write(target, encoding='unicode', xml_declaration=True)
        derived[role] = target
        records.append({'role': role, 'physics_urdf': relative, 'sha256': digest,
                        'runtime_urdf': target.name, 'runtime_sha256': sha256_file(target),
                        'mesh_bindings': bindings, 'visual_binding': visual_binding})
    write_json(output_dir / 'urdf_asset_binding_receipt.json', {
        'schema_version': 'robotics_runtime_urdf_rebind.v1',
        'reference_manifest_sha256': sha256_file(reference_manifest), 'assets': records})
    return derived
