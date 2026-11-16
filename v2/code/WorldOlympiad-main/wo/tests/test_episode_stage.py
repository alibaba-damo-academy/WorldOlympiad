import json
import hashlib
import subprocess

import pytest

from wo.agent import episode_stage as module
from wo.agent.episode_stage import RECEIPT, require_references_ready, run_bound
from wo.agent import stages
from wo.eval.reference_binding import verify_episode_binding


def fixture(tmp_path):
    run = tmp_path / 'run'
    (run / 'tasks').mkdir(parents=True)
    task = run / 'tasks/a.json'
    task.write_text(json.dumps({
        'schema_version': 'wo_task_spec.v1',
        'case_id': 'a',
        'family': 'pick_place',
        'instruction': 'Place the apple on the coaster.',
        'objects': [
            {'object_id': 'apple', 'category': 'apple', 'role': 'manipuland'},
            {'object_id': 'coaster', 'category': 'coaster', 'role': 'destination'},
            {'object_id': 'table', 'category': 'table', 'role': 'support'},
        ],
        'steps': [{
            'primitive': 'transport',
            'subject': 'apple',
            'target': 'coaster',
            'goal': {'kind': 'on', 'subject': 'apple', 'target': 'coaster'},
        }],
    }))
    ref = run / 'references/a'
    (ref / 'replay').mkdir(parents=True)
    (ref / 'certificate.json').write_text(json.dumps({'success': True, 'task': str(task), 'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(), 'frame_count': 1}))
    (ref / 'trajectory.json').write_text('{"frames":[]}')
    for view in ('exterior_left', 'exterior_right', 'top', 'wrist'):
        (ref / 'replay' / f'{view}.mp4').write_bytes(b'video')
    out = run / 'a/model_s0'
    return run, ref, out


def write_episode(out, ref):
    (out / 'episode.json').write_text(json.dumps({'reference': str(ref), 'task_success': False}))


def test_missing_reference_never_invokes_model_or_mutates_output(tmp_path, monkeypatch):
    run, ref, out = fixture(tmp_path)
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda _: {'a': 'done'})
    (run / 'tasks/b.json').write_text('{}')
    called = []
    with pytest.raises(ValueError, match='incomplete references: b'):
        run_bound(run, ref, out, ['model'], runner=lambda *a, **kw: called.append(a))
    assert called == []
    assert not out.exists()


def test_failed_certificate_and_absent_video_block_model(tmp_path):
    run, ref, out = fixture(tmp_path)
    (ref / 'certificate.json').write_text('{"success":false}')
    with pytest.raises(ValueError, match='all-experts-first'):
        require_references_ready(run)
    (ref / 'certificate.json').write_text('{"success":true}')
    (ref / 'replay/exterior_left.mp4').unlink()
    with pytest.raises(ValueError, match='all-experts-first'):
        require_references_ready(run)


def test_retry_preserves_previous_attempt_and_binds_new_output(tmp_path, monkeypatch):
    run, ref, out = fixture(tmp_path)
    out.mkdir(parents=True)
    (out / 'old.log').write_text('previous failed evidence')
    write_episode(out, ref)
    command = ['native', '--seed', '3']
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda _: {'a': 'done'})
    monkeypatch.setattr(module, 'ensure_evaluation_view', lambda *_: {'view': 'exterior_left'})
    observed = []
    def runner(argv, **kwargs):
        observed.append((argv, kwargs))
        assert not (out / 'episode.json').exists()
        write_episode(out, ref)
    result = run_bound(run, ref, out, command, render_gpu_id=0, runner=runner)
    assert observed == [(command, {'check': True})]
    from pathlib import Path
    assert (Path(result['previous_attempt']) / 'old.log').read_text() == 'previous failed evidence'
    assert result['state'] == 'complete'
    assert verify_episode_binding(out, ref)['status'] == 'verified'


def test_reference_drift_during_execution_is_not_certified(tmp_path, monkeypatch):
    run, ref, out = fixture(tmp_path)
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda _: {'a': 'done'})
    def runner(*args, **kwargs):
        write_episode(out, ref)
        (ref / 'trajectory.json').write_text('{"frames":[1]}')
    with pytest.raises(ValueError, match='reference changed during'):
        run_bound(run, ref, out, ['native'], runner=runner)
    assert json.loads((out / RECEIPT).read_text())['state'] == 'failed'
    with pytest.raises(ValueError, match='did not complete'):
        verify_episode_binding(out, ref)


def test_adapter_crash_cannot_reuse_an_old_successful_episode(tmp_path, monkeypatch):
    run, ref, out = fixture(tmp_path)
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda _: {'a': 'done'})
    out.mkdir(parents=True)
    write_episode(out, ref)
    def crash(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ['native'])
    with pytest.raises(subprocess.CalledProcessError):
        run_bound(run, ref, out, ['native'], runner=crash)
    assert not (out / 'episode.json').exists()
    assert json.loads((out / RECEIPT).read_text())['state'] == 'failed'


def test_complete_unsuccessful_native_rollout_is_bindable_before_view_render(tmp_path, monkeypatch):
    run, ref, out = fixture(tmp_path)
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda _: {'a': 'done'})
    observed = []

    def render_view(episode_dir, bound_ref, _gpu):
        # The native result (including task_success=false) is complete and
        # reference-bound before the renderer verifies/consumes its trajectory.
        assert json.loads((episode_dir / 'episode.json').read_text())['task_success'] is False
        observed.append(verify_episode_binding(episode_dir, bound_ref)['status'])
        return {'view': 'exterior_left'}

    monkeypatch.setattr(module, 'ensure_evaluation_view', render_view)
    run_bound(run, ref, out, ['native'], render_gpu_id=0,
              runner=lambda *a, **kw: write_episode(out, ref))
    assert observed == ['verified']
    assert json.loads((out / RECEIPT).read_text())['state'] == 'complete'
    assert json.loads((out / 'episode.json').read_text())['task_success'] is False


def test_scoring_refuses_a_later_reference_change(tmp_path, monkeypatch):
    run, ref, out = fixture(tmp_path)
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda _: {'a': 'done'})
    monkeypatch.setattr(module, 'ensure_evaluation_view', lambda *_: {'view': 'exterior_left'})
    run_bound(run, ref, out, ['native'], render_gpu_id=0, runner=lambda *a, **kw: write_episode(out, ref))
    (ref / 'replay/exterior_left.mp4').write_bytes(b'new-video')
    with pytest.raises(ValueError, match='reference version mismatch'):
        verify_episode_binding(out, ref)


def test_stage_retains_native_adapter_seed_window_and_port():
    cmd = stages.STAGES['episode'].command({'run':'bench_v1','case':'a','model':'gr00t','seed':3,'window_s':60},2)
    assert 'wo.agent.episode_stage' in cmd
    assert 'wo.models.gr00t' in cmd
    assert cmd[cmd.index('--seed') + 1] == '3'
    assert cmd[cmd.index('--window-s') + 1] == '60'
    assert cmd[cmd.index('--port') + 1] == '28002'
    assert cmd[cmd.index('--render-gpu-id') + 1] == '0'


def test_changed_task_cannot_reuse_successful_reference(tmp_path, monkeypatch):
    run, ref, out = fixture(tmp_path)
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda _: {'a': 'done'})
    (run / 'tasks/a.json').write_text('{"new_task":true}')
    called = []
    with pytest.raises(ValueError, match='all-experts-first'):
        run_bound(run, ref, out, ['model'], runner=lambda *a, **kw: called.append(a))
    assert called == [] and not out.exists()


@pytest.mark.parametrize('state', ['queued', 'running', 'failed'])
def test_existing_worker_state_prevents_reusing_old_video(tmp_path, monkeypatch, state):
    run, ref, out = fixture(tmp_path)
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': state})
    called = []
    with pytest.raises(ValueError, match='incomplete references: a'):
        run_bound(run, ref, out, ['model'], runner=lambda *a, **kw: called.append(a))
    assert not called and not out.exists()


def test_job_state_lookup_is_read_only_and_scoped_to_run(tmp_path):
    import sqlite3
    from wo.agent.episode_stage import latest_reference_jobs
    run = tmp_path / 'outputs/wo_runs/bench'
    run.mkdir(parents=True)
    database = tmp_path / 'outputs/wo_agent/jobs.sqlite3'
    database.parent.mkdir()
    with sqlite3.connect(database) as con:
        con.execute('create table jobs (id integer, kind text, args text, status text)')
        con.executemany('insert into jobs values (?,?,?,?)', [
            (1, 'reference', '{"run":"bench","case":"a"}', 'done'),
            (2, 'reference', '{"run":"bench","case":"a"}', 'running'),
            (3, 'reference', '{"run":"another","case":"a"}', 'done')])
    before = database.read_bytes()
    assert latest_reference_jobs(run) == {'a': 'running'}
    assert database.read_bytes() == before
