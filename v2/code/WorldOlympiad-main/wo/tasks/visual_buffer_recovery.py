"""Recover overwritten shared glTF buffers using independently exported canonical meshes.

SceneCode may put several trimesh glTF exports in one directory; each export
uses gltf_buffer_N.bin, so the later asset overwrites the earlier one's data.
Only redirect a broken visual if a valid same-asset canonical projection exists.
Never alter collisions or a mesh that already passes buffer validation.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_gltf(path):
    try:
        data = json.loads(path.read_text())
        for buffer in data.get('buffers', []):
            uri = buffer['uri']
            if uri.startswith('data:'):
                continue
            if Path(uri).name != uri or (path.parent / uri).stat().st_size != buffer['byteLength']:
                return False
        return bool(data.get('meshes'))
    except (OSError, KeyError, ValueError):
        return False


def recover_overwritten_visual_buffers(generation):
    """Rebind corrupt visual-only glTFs to same-asset canonical exports, audibly.

    The canonical mesh is SceneCode's separate projection of the same code-object
    model. Fail closed if it is absent or cannot be loaded as an actual mesh.
    """
    import trimesh

    generation = Path(generation)
    certificate_path = generation / 'task_inventory_certificate.json'
    manifest_path = generation / 'generation_manifest.json'
    certificate = json.loads(certificate_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    changed = []
    for match in certificate.get('matches', []):
        raw = Path(match.get('sdf_path', ''))
        sdf = raw if raw.is_absolute() else generation / raw
        if not sdf.is_file():
            continue
        root = ET.parse(sdf)
        modifications = []
        for uri in root.findall('.//visual/geometry/mesh/uri'):
            name = (uri.text or '').strip()
            if not name.endswith('.gltf'):
                continue
            path = sdf.parent / name
            if _valid_gltf(path):
                continue
            canonical = path.with_name(path.stem + '_canonical.gltf')
            if not _valid_gltf(canonical):
                raise ValueError(f'corrupt visual has no valid canonical projection: {path}')
            scene = trimesh.load(str(canonical), force='scene', process=False)
            if not scene.graph.nodes_geometry or any(len(m.faces) == 0 for m in scene.geometry.values()):
                raise ValueError(f'canonical projection is empty: {canonical}')
            uri.text = canonical.name
            modifications.append({'original': name, 'canonical': canonical.name,
                                  'canonical_sha256': _sha(canonical)})
        if modifications:
            root.write(sdf, encoding='unicode')
            match['sdf_sha256'] = _sha(sdf)
            changed.append({'sdf_path': str(sdf), 'sdf_sha256': _sha(sdf),
                            'visual_rebindings': modifications})
    if changed:
        certificate_path.write_text(json.dumps(certificate, indent=2) + '\n')
        manifest['task_inventory_certificate_sha256'] = _sha(certificate_path)
        report = generation / 'visual_buffer_recovery.json'
        report.write_text(json.dumps({'schema_version': 'wo_visual_buffer_recovery.v1',
                                     'changes': changed}, indent=2) + '\n')
        manifest['visual_buffer_recovery'] = {'path': report.name, 'sha256': _sha(report)}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return changed
