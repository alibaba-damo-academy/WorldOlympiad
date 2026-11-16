"""Bind score videos and CAE to one version of the expert reference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RECEIPT = 'reference_binding.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_task(ref: Path, expected_task: Path | None = None) -> dict:
    cert = json.loads((ref / 'certificate.json').read_text())
    if cert.get('success') is not True:
        raise ValueError('reference binding requires a successful expert certificate')
    task = Path(cert['task']).resolve()
    if expected_task is not None and task != expected_task.resolve():
        raise ValueError('reference version mismatch: certificate belongs to another task')
    if cert.get('task_sha256') != digest(task):
        raise ValueError('reference version mismatch: task changed after expert execution')
    # A task may deliberately reuse a published SceneCode generation during
    # development. Bind the declared source from the certificate's own task
    # even when callers omit expected_task; otherwise verify_task(ref) could
    # accept a certificate whose generation was silently substituted.
    payload = json.loads(task.read_text())
    declared = payload.get('generation')
    produced = cert.get('generation')
    if declared:
        if not produced:
            raise ValueError('reference version mismatch: certificate omits declared generation source')
        if Path(declared).resolve() != Path(produced).resolve():
            raise ValueError('reference version mismatch: generation source differs')
        # Reused SceneCode generations must carry the export-time physical
        # size result when the task declares an explicit size contract. Legacy
        # tasks without that contract remain admissible: absence of a report is
        # not evidence that their (undeclared) geometry is invalid.
        sized_objects = [
            obj for obj in payload.get('objects', [])
            if obj.get('geometry_constraints', {}).get('overall_dimensions_m') is not None
        ]
        if sized_objects:
            validation = Path(declared) / 'physical_size_validation.json'
            if not validation.is_file():
                raise ValueError('reference admission requires physical size validation for declared generation')
            report = json.loads(validation.read_text())
            records = report.get('objects', {})
            for obj in sized_objects:
                expected = obj['geometry_constraints']['overall_dimensions_m']
                record = records.get(obj.get('object_id'))
                measured = (record or {}).get('measured_m')
                tolerance = float((record or {}).get('tolerance_m', 0.0005))
                if measured is None or len(measured) != 3 or any(
                        abs(float(a) - float(b)) > tolerance
                        for a, b in zip(measured, expected)):
                    raise ValueError('reference admission rejects generation with failed physical size validation')
    if expected_task is not None:
        expected_payload = json.loads(expected_task.read_text())
        expected_generation = expected_payload.get('generation')
        if expected_generation and (not declared or Path(expected_generation).resolve() != Path(declared).resolve()):
            raise ValueError('reference version mismatch: generation source differs')
    return cert


def capture(ref: Path, view: str = 'exterior_left') -> dict:
    ref = ref.resolve()
    cert_path = ref / 'certificate.json'
    cert = verify_task(ref)
    # The three-track scorer sees the video, while CAE sees the trajectory and
    # articulated URDFs. Bind both sources, including the task definition.
    files = [cert_path, ref / 'trajectory.json', ref / 'replay' / f'{view}.mp4']
    files.extend(ref / name for name in ('scene_inputs.json', 'replay_plan.json') if (ref / name).is_file())
    files.extend(sorted(p for p in (ref / 'physics_sdf').rglob('*')
                        if p.is_file() and p.suffix in {'.urdf', '.sdf'}))
    task = Path(cert['task'])
    return {'schema_version': 'wo_reference_binding.v1', 'reference': str(ref),
            'view': view, 'files': {str(p.relative_to(ref)): digest(p) for p in files},
            'task': str(task.resolve()), 'task_sha256': digest(task)}


def write_binding(ref: Path, case: Path, view: str = 'exterior_left') -> dict:
    receipt = capture(ref, view)
    manifest_path = case / 'worldolympiad_case.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        copied = case / manifest['reference_video']
        if digest(copied) != receipt['files'][f'replay/{view}.mp4']:
            raise ValueError('reference changed while building the score case')
    (case / RECEIPT).write_text(json.dumps(receipt, indent=1))
    return receipt


def verify_episode_binding(episode: Path, ref: Path) -> dict:
    path = episode / 'episode_reference_binding.json'
    if not path.exists():
        return {'status': 'legacy_unbound'}
    receipt = json.loads(path.read_text())
    if receipt.get('schema_version') != 'wo_episode_reference_binding.v1':
        raise ValueError('unsupported episode reference binding schema')
    if receipt.get('state') != 'complete':
        raise ValueError('bound native model execution did not complete')
    saved = receipt['reference_binding']
    if saved != capture(ref, saved['view']):
        raise ValueError('reference version mismatch: model execution used a different expert reference')
    return {'status': 'verified', **saved}


def verify_binding(episode: Path, ref: Path, *, case: Path | None = None) -> dict:
    verify_episode_binding(episode, ref)
    case = case if case is not None else episode / 'case'
    receipt_path = case / RECEIPT
    if receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if saved.get('schema_version') != 'wo_reference_binding.v1':
            raise ValueError('unsupported reference binding schema')
        current = capture(ref, saved['view'])
        if saved != current:
            raise ValueError('reference version mismatch: scoring and CAE must use the same expert reference')
        return {'status': 'verified', **saved}
    # Historical score cases have a copied reference video but no trajectory
    # receipt. Never silently combine that old video score with a new reference.
    manifest = json.loads((case / 'worldolympiad_case.json').read_text())
    view = manifest.get('reference_view_id', 'exterior_left')
    copied = case / manifest['reference_video']
    current = ref / 'replay' / f'{view}.mp4'
    if digest(copied) != digest(current):
        raise ValueError('reference version mismatch: saved scoring video differs from the current CAE reference')
    return {'status': 'legacy_video_verified', 'reference': str(ref.resolve()),
            'view': view, 'video_sha256': digest(current),
            'limitation': 'Historical case has no reference trajectory hash; final runs require a binding receipt.'}
