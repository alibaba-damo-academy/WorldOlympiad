"""Run a model stage only after reference readiness and bind its inputs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

from ..eval.reference_binding import capture, verify_episode_binding, verify_task

RECEIPT = 'episode_reference_binding.json'


def latest_reference_jobs(run: Path) -> dict[str, str]:
    # Reuse the worker's existing completion state; no new render gate or retry loop.
    database = run.resolve().parents[1] / 'wo_agent/jobs.sqlite3'
    latest = {}
    if database.is_file():
        with sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True) as con:
            for raw, status in con.execute("select args,status from jobs where kind='reference' order by id"):
                args = json.loads(raw)
                if args.get('run') == run.name:
                    latest[args['case']] = status
    receipt_path = run / 'phase1_import_receipt.json'
    if receipt_path.is_file():
        try:
            from .frozen_import import verify_frozen_import
            receipt = json.loads(receipt_path.read_text())
            case = receipt.get('case_id')
            if case and case not in latest:
                verify_frozen_import(run, case)
                latest[case] = 'done'
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return latest


def declared_task_paths(run: Path) -> list[Path]:
    """Resolve the run's evaluation cases; absent scope retains legacy all-task behavior.

    Unselected task files remain preserved in the run, but only cases explicitly
    declared by the run manifest participate in episode admission and reporting.
    """
    available = {path.stem: path for path in sorted((run / 'tasks').glob('*.json'))}
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if not isinstance(manifest, dict):
        raise ValueError('run manifest must be a JSON object')
    cases = manifest.get('cases')
    if cases is None:
        return list(available.values())
    if (not isinstance(cases, list) or not cases or
            any(not isinstance(case, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', case)
                for case in cases) or len(set(cases)) != len(cases)):
        raise ValueError('run manifest cases must be a non-empty list of unique safe case ids')
    unknown = sorted(set(cases) - set(available))
    if unknown:
        raise ValueError('run manifest declares cases without TaskSpecs: ' + ', '.join(unknown))
    return [available[case] for case in cases]


def missing_references(run: Path) -> list[str]:
    tasks = declared_task_paths(run)
    if not tasks:
        raise ValueError('all-experts-first gate: no declared tasks')
    missing = []
    jobs = latest_reference_jobs(run)
    for task in tasks:
        ref = run / 'references' / task.stem
        try:
            verify_task(ref, task)
            ok = (ref / 'replay/exterior_left.mp4').is_file() and jobs.get(task.stem) == 'done'
        except (OSError, ValueError, KeyError):
            ok = False
        if not ok:
            missing.append(task.stem)
    return missing


def require_references_ready(run: Path) -> None:
    missing = missing_references(run)
    if missing:
        raise ValueError('all-experts-first gate: incomplete references: ' + ', '.join(missing))


def require_submission_references(run: Path, connection: sqlite3.Connection, after: list[int]) -> None:
    """Queue an episode only when all references are ready or explicitly awaited.

    Pending reference jobs must cover every missing case through the declared
    dependency graph. The existing execution-time gate still validates their
    actual artifacts after dependencies complete.
    """
    closure, pending = {}, list(after)
    while pending:
        jid = pending.pop()
        if jid in closure:
            continue
        row = connection.execute('select * from jobs where id=?', (jid,)).fetchone()
        if row is None:
            raise ValueError(f'episode dependency #{jid} does not exist')
        row = dict(row)
        if row['status'] in ('failed', 'cancelled'):
            raise ValueError(f"episode dependency #{jid} is {row['status']}; use current successful dependencies")
        closure[jid] = row
        pending.extend(json.loads(row['after']))
    missing = missing_references(run)
    if not missing:
        return
    latest = {}
    for row in connection.execute("select * from jobs where kind='reference' order by id"):
        args = json.loads(row['args'])
        if args.get('run') == run.name:
            latest[args['case']] = row
    promised = {case for case, row in latest.items()
                if row['id'] in closure and row['status'] in ('queued', 'running')}
    uncovered = sorted(set(missing) - promised)
    if uncovered:
        raise ValueError('all-experts-first gate: incomplete references without pending '
                         'dependencies: ' + ', '.join(uncovered))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _render_source_hash() -> dict[str, str]:
    from ..render import REPO
    paths = (REPO / 'wo/render.py', REPO / 'wo/render_plan.py',
             REPO / 'wo/blender_replay.py',
             REPO / 'deployment/ziplab5090/scenecode_robotics_render_settings.py')
    return {str(path.relative_to(REPO)): _sha256(path) for path in paths}


def verify_evaluation_view(episode_dir: Path, ref: Path) -> dict:
    """Verify the scored exterior view is rendered from this native trajectory."""
    episode_dir, ref = Path(episode_dir).resolve(), Path(ref).resolve()
    receipt_path = episode_dir / 'native_evaluation_view.json'
    receipt = json.loads(receipt_path.read_text())
    episode = json.loads((episode_dir / 'episode.json').read_text())
    verify_episode_binding(episode_dir, ref)
    binding = capture(ref)
    view = binding['view']
    trajectory = episode_dir / 'native_run/simulation/trajectory.json'
    videos = episode.get('videos') or {}
    video_value = videos.get(view)
    if not video_value:
        raise ValueError(f'episode has no baseline replay video for required view {view!r}')
    video = Path(video_value)
    if not video.is_absolute():
        video = episode_dir / video
    plan_path = episode_dir / receipt.get('plan', '')
    if not trajectory.is_file() or not video.is_file() or not video.stat().st_size:
        raise FileNotFoundError('native replay inputs or exact-view video are missing')
    if receipt.get('schema_version') != 'wo_native_evaluation_view.v1':
        raise ValueError('unsupported native evaluation view receipt')
    if receipt.get('view') != view or receipt.get('video') != str(video.relative_to(episode_dir)):
        raise ValueError('native evaluation view receipt does not match the reference view')
    if receipt.get('native_trajectory_sha256') != _sha256(trajectory):
        raise ValueError('native trajectory changed after evaluation-view rendering')
    if receipt.get('video_sha256') != _sha256(video):
        raise ValueError('evaluation-view video changed after rendering')
    if receipt.get('reference_binding') != capture(ref, view):
        raise ValueError('evaluation-view video is bound to another expert reference')
    if receipt.get('renderer_sources') != _render_source_hash():
        raise ValueError('evaluation-view renderer source changed after rendering')
    if not plan_path.is_file() or receipt.get('plan_sha256') != _sha256(plan_path):
        raise ValueError('evaluation-view render plan is missing or changed')
    plan = json.loads(plan_path.read_text())
    if list(plan.get('views', {})) != [view] or len(plan.get('frames', [])) != receipt.get('frame_count'):
        raise ValueError('evaluation-view plan has the wrong camera or frame count')
    reference_plan_path = ref / 'replay_plan.json'
    if reference_plan_path.is_file():
        reference_plan = json.loads(reference_plan_path.read_text())
        if plan['views'].get(view) != (reference_plan.get('views') or {}).get(view):
            raise ValueError('native replay camera resolution/FOV differs from the frozen reference')
    return receipt


def ensure_evaluation_view(episode_dir: Path, ref: Path, render_gpu_id: int | None) -> dict:
    """Export the baseline-matched view from the native simulated trajectory.

    Existing camera captures are preserved. Scoring receives a deterministic
    replay through the same renderer and camera contract used by the reference.
    render_gpu_id is retained for caller compatibility and does not bind EGL.
    """
    episode_dir, ref = Path(episode_dir).resolve(), Path(ref).resolve()
    verify_episode_binding(episode_dir, ref)
    binding = capture(ref)
    receipt_path = episode_dir / 'native_evaluation_view.json'
    if receipt_path.is_file():
        try:
            return verify_evaluation_view(episode_dir, ref)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    trajectory = episode_dir / 'native_run/simulation/trajectory.json'
    if not trajectory.is_file() or not trajectory.stat().st_size:
        raise FileNotFoundError('baseline-matched rendering requires the native simulator trajectory')
    from ..render import render
    target = episode_dir / 'native_replay'
    if target.exists():
        target = episode_dir / f'native_replay_attempt_{time.time_ns()}'
    render(ref, trajectory=trajectory, out=target, views=(binding['view'],), gpu_id=render_gpu_id)
    video = target / f"{binding['view']}.mp4"
    plan_path = target.with_name(target.name + '_plan.json')
    plan = json.loads(plan_path.read_text())
    if not video.is_file() or not video.stat().st_size:
        raise FileNotFoundError('baseline renderer did not produce the required native episode view')
    episode_path = episode_dir / 'episode.json'
    episode = json.loads(episode_path.read_text())
    capture_meta = episode.get('capture') or (episode.get('native_result') or {}).get('capture') or {}
    videos = dict(capture_meta.get('videos') or {})
    videos.update(episode.get('videos') or {})
    rel_video = str(video.relative_to(episode_dir))
    videos[binding['view']] = rel_video
    episode['videos'] = videos
    episode_path.write_text(json.dumps(episode, indent=1))
    receipt = {'schema_version': 'wo_native_evaluation_view.v1', 'view': binding['view'],
               'video': rel_video, 'video_sha256': _sha256(video),
               'native_trajectory': str(trajectory), 'native_trajectory_sha256': _sha256(trajectory),
               'reference_binding': binding, 'plan': str(plan_path.relative_to(episode_dir)),
               'plan_sha256': _sha256(plan_path), 'frame_count': len(plan['frames']),
               'renderer_device_selection': 'host-configured EGL default', 'renderer_sources': _render_source_hash()}
    receipt_path.write_text(json.dumps(receipt, indent=1))
    return verify_evaluation_view(episode_dir, ref)


def render_existing_episode_view(run: Path, ref: Path, out: Path, render_gpu_id: int | None) -> dict:
    """Add only the baseline camera replay to an already completed native episode."""
    run, ref, out = Path(run).resolve(), Path(ref).resolve(), Path(out).resolve()
    expected = run / 'tasks' / f'{ref.name}.json'
    if not expected.is_file():
        raise FileNotFoundError(f'run has no bound TaskSpec for {ref.name}')
    verify_task(ref, expected)
    require_references_ready(run)
    episode = json.loads((out / 'episode.json').read_text())
    if Path(episode['reference']).resolve() != ref:
        raise ValueError('existing native episode belongs to a different expert reference')
    verify_episode_binding(out, ref)
    return ensure_evaluation_view(out, ref, render_gpu_id)


def _episode_output_lock(out: Path):
    """Acquire an interprocess lock for one native episode output path.

    Retries and duplicate worker deliveries can otherwise archive or write the
    same output directory concurrently, corrupting both the attempt history and
    the native rollout. Keep the lock file outside the archived output tree.
    """
    out = Path(out).resolve()
    lock_dir = out.parent / '.episode_locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f'{out.name}.lock'
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'episode output is already active: {out}') from exc
        return fd
    except Exception:
        os.close(fd)
        raise


def run_bound(run: Path, ref: Path, out: Path, command: list[str], *, render_gpu_id: int | None = None, runner=subprocess.run) -> dict:
    fd = _episode_output_lock(out)
    try:
        return _run_bound_locked(run, ref, out, command, render_gpu_id=render_gpu_id, runner=runner)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _run_bound_locked(run: Path, ref: Path, out: Path, command: list[str], *, render_gpu_id: int | None = None, runner=subprocess.run) -> dict:
    require_references_ready(run)
    # Final native admission must use the current source geometry, not merely
    # the reference's historical/copyable validation metadata.
    task = Path(json.loads((ref / 'certificate.json').read_text())['task'])
    from ..tasks.scene import validate_source
    validate_source(task)
    before = capture(ref)
    # A retry must produce fresh evidence. Preserve the entire previous attempt
    # before invoking an adapter, so old output cannot be accepted after a crash.
    archived = None
    if out.exists():
        history = out.parent / '.episode_history' / out.name
        history.mkdir(parents=True, exist_ok=True)
        archived = history / f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
        shutil.move(str(out), str(archived))
    out.mkdir(parents=True)
    receipt = {'schema_version': 'wo_episode_reference_binding.v1', 'state': 'running',
               'reference_binding': before, 'started': time.time(),
               'previous_attempt': str(archived) if archived else None}
    path = out / RECEIPT
    path.write_text(json.dumps(receipt, indent=1))
    try:
        result = runner(command, check=True)
        episode = json.loads((out / 'episode.json').read_text())
        if Path(episode['reference']).resolve() != ref.resolve():
            raise ValueError('model episode declared a different reference')
        if capture(ref) != before:
            raise ValueError('reference changed during native model execution')
        # The baseline-view renderer verifies this receipt before using the
        # trajectory. Mark the native adapter result complete first; a valid
        # task_success=false rollout is still a complete model execution. Any
        # subsequent rendering/validation exception is caught below and turns
        # the overall attempt receipt back to failed.
        receipt.update(state='complete', finished=time.time())
        path.write_text(json.dumps(receipt, indent=1))
        ensure_evaluation_view(out, ref, render_gpu_id)
    except Exception as exc:
        receipt.update(state='failed', finished=time.time(), error=str(exc))
        path.write_text(json.dumps(receipt, indent=1))
        raise
    receipt.update(state='complete', finished=time.time())
    path.write_text(json.dumps(receipt, indent=1))
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--render-gpu-id', type=int)
    ap.add_argument('--render-existing-only', action='store_true',
                    help='render the baseline-matched view for an already complete native episode')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.render_existing_only:
        if command:
            ap.error('--render-existing-only cannot be combined with an adapter command')
        render_existing_episode_view(args.run_root, args.ref, args.out, args.render_gpu_id)
        return
    if not command:
        ap.error('native adapter command required after --')
    run_bound(args.run_root, args.ref, args.out, command, render_gpu_id=args.render_gpu_id)


if __name__ == '__main__':
    main()
