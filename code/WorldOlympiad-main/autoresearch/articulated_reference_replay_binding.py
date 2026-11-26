"""Verify visual-only revisions without weakening native replay physics identity."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from .io import sha256_file


def physical_signature(path: Path) -> str:
    path=Path(path).resolve();root=ET.parse(path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag=='visual':parent.remove(child)
    for node in root.findall('.//uri'):
        mesh=Path(node.text.strip());mesh=mesh if mesh.is_absolute() else path.parent/mesh
        node.text=sha256_file(mesh.resolve())
    for node in root.iter():
        if node.text:node.text=' '.join(node.text.split())
        node.tail=None
    return hashlib.sha256(ET.tostring(root)).hexdigest()


def validate_reference_binding(reference: dict, config: dict) -> dict:
    expected=reference['binding']['host_sdf_sha256'];native=Path(config['source_sdf'])
    if sha256_file(native)!=config['source_sdf_sha256']:
        raise ValueError('native replay asset changed')
    if expected==config['source_sdf_sha256']:
        return {'policy':'exact_reference_sdf_sha256','passed':True}
    manifest_path=Path(config['reference_manifest']).resolve()
    if sha256_file(manifest_path)!=config['reference_manifest_sha256']:
        raise ValueError('reference manifest changed before native replay')
    manifest=json.loads(manifest_path.read_text());root=manifest_path.parent
    originals=[p for p in (root/'simulation_assets/source_sdf/host').glob('*.sdf') if sha256_file(p)==expected]
    if len(originals)!=1:
        raise ValueError('reference replay lacks its exact original bundled SDF')
    certificate=root/manifest['visual_physics_alignment_certificate']
    receipt=json.loads(certificate.read_text())
    if (sha256_file(certificate)!=config['runtime_alignment_certificate_sha256']
            or sha256_file(certificate)!=manifest['visual_physics_alignment_certificate_sha256']
            or receipt.get('passed') is not True
            or receipt.get('visual_sdf_sha256')!=config['source_sdf_sha256']):
        raise ValueError('reference replay visual revision is not certified')
    old_signature=physical_signature(originals[0]);new_signature=physical_signature(native)
    if old_signature!=new_signature:
        raise ValueError('reference/native collision, inertia, joint or other physics differs')
    return {'policy':'certified_visual_only_revision_same_physical_xml.v1','passed':True,
            'original_sdf_sha256':expected,'native_sdf_sha256':sha256_file(native),
            'physical_signature_sha256':old_signature,'alignment_certificate_sha256':sha256_file(certificate)}
