import importlib
import json
import os
from pathlib import Path
import sqlite3
import pytest

# Keep this module's imports independent of the host's default private workspace path.
_workspace_config = Path(__file__).resolve().parents[4] / 'config/two-interface.json'
if _workspace_config.is_file():
    os.environ.setdefault('WO_EVAL_CONFIG', str(_workspace_config))
from wo.agent import episode_stage as module, jobs, tools


@pytest.fixture
def queue():
    con=sqlite3.connect(':memory:', isolation_level=None)
    con.row_factory=sqlite3.Row
    con.executescript(jobs.SCHEMA)
    yield con
    con.close()


def insert(con, kind='reference', *, case='apple', run='nine', status='queued', after=()):
    return con.execute('insert into jobs(kind,args,status,after,created,timeout_s) values(?,?,?,?,0,100)',
        (kind,json.dumps({'run':run,'case':case}),status,json.dumps(list(after)))).lastrowid


def test_blocks_missing_global_references_even_if_subject_reference_is_done(queue,monkeypatch):
    monkeypatch.setattr(module,'missing_references',lambda run:['oven','box'])
    ref=insert(queue,status='done')
    check=insert(queue,'scene_check',status='done',after=[ref])
    with pytest.raises(ValueError,match='box, oven'):
        module.require_submission_references(Path('/run/nine'),queue,[ref,check])


def test_all_missing_references_may_be_awaited_through_transitive_dependencies(queue,monkeypatch):
    monkeypatch.setattr(module,'missing_references',lambda run:['apple','box'])
    a=insert(queue,case='apple')
    b=insert(queue,case='box',status='running')
    check=insert(queue,'scene_check',after=[a,b])
    module.require_submission_references(Path('/run/nine'),queue,[check])


@pytest.mark.parametrize('wrong_run,old_attempt',[(True,False),(False,True)])
def test_other_run_or_obsolete_reference_does_not_cover_current_missing_case(queue,monkeypatch,wrong_run,old_attempt):
    monkeypatch.setattr(module,'missing_references',lambda run:['apple'])
    parent=insert(queue,run='other' if wrong_run else 'nine')
    if old_attempt:insert(queue,case='apple')
    with pytest.raises(ValueError,match='apple'):
        module.require_submission_references(Path('/run/nine'),queue,[parent])


@pytest.mark.parametrize('status',['failed','cancelled'])
def test_doomed_explicit_dependency_is_rejected_before_insertion(queue,monkeypatch,status):
    monkeypatch.setattr(module,'missing_references',lambda run:[])
    parent=insert(queue,status=status)
    with pytest.raises(ValueError,match=status):
        module.require_submission_references(Path('/run/nine'),queue,[parent])


def test_ready_scope_keeps_existing_episode_submission_path(queue,monkeypatch):
    monkeypatch.setattr(module,'missing_references',lambda run:[])
    parent=insert(queue,status='done')
    module.require_submission_references(Path('/run/nine'),queue,[parent])


def test_manifest_case_scope_limits_episode_reference_gate_without_deleting_legacy_tasks(tmp_path, monkeypatch):
    (tmp_path / 'tasks').mkdir()
    for case in ('selected', 'legacy'):
        (tmp_path / 'tasks' / f'{case}.json').write_text(json.dumps({'case_id': case}))
    (tmp_path / 'manifest.json').write_text(json.dumps({'cases': ['selected']}))
    replay = tmp_path / 'references' / 'selected' / 'replay'
    replay.mkdir(parents=True)
    (replay / 'exterior_left.mp4').write_bytes(b'current replay')
    monkeypatch.setattr(module, 'verify_task', lambda ref, task: None)
    monkeypatch.setattr(module, 'latest_reference_jobs', lambda run: {'selected': 'done'})

    assert [p.stem for p in module.declared_task_paths(tmp_path)] == ['selected']
    assert module.missing_references(tmp_path) == []
    # Without an explicit scope, legacy behavior still requires every task.
    (tmp_path / 'manifest.json').write_text(json.dumps({}))
    assert module.missing_references(tmp_path) == ['legacy']


def test_manifest_case_scope_fails_closed_for_undeclared_task_file(tmp_path):
    (tmp_path / 'tasks').mkdir()
    (tmp_path / 'manifest.json').write_text(json.dumps({'cases': ['missing']}))
    with pytest.raises(ValueError, match='without TaskSpecs'):
        module.declared_task_paths(tmp_path)


def test_agent_receives_actionable_gate_message_without_creating_failed_job(queue,monkeypatch):
    monkeypatch.setattr(importlib, 'reload', lambda imported: imported)
    monkeypatch.setattr(module,'missing_references',lambda run:['oven'])
    monkeypatch.setattr(tools,'run_dir',lambda run:Path('/run/nine'))
    monkeypatch.setattr(tools.J,'connect',lambda:queue)
    monkeypatch.setattr(tools.J,'submit',lambda *a,**kw:pytest.fail('doomed episode entered queue'))
    result=tools.submit('episode',json.dumps({'run':'nine','case':'apple','model':'cosmos3','seed':0}))
    assert result.startswith('not queued (episode): all-experts-first gate:')
    assert 'oven' in result
    assert queue.execute("select count(*) from jobs where kind='episode'").fetchone()[0]==0


def test_episode_submit_refreshes_admission_gate_for_scoped_runs(tmp_path, queue, monkeypatch):
    run = tmp_path / 'run'
    (run / 'tasks').mkdir(parents=True)
    for case in ('selected', 'historical'):
        (run / 'tasks' / f'{case}.json').write_text(json.dumps({'case_id': case}))
    (run / 'manifest.json').write_text(json.dumps({'cases': ['selected']}))
    replay = run / 'references' / 'selected' / 'replay'
    replay.mkdir(parents=True)
    (replay / 'exterior_left.mp4').write_bytes(b'current replay')

    real_reload = importlib.reload
    refreshed = []
    def refresh_gate(imported):
        current = real_reload(imported)
        refreshed.append(current.__file__)
        monkeypatch.setattr(current, 'verify_task', lambda ref, task: None)
        monkeypatch.setattr(current, 'latest_reference_jobs', lambda path: {'selected': 'done'})
        return current
    monkeypatch.setattr(importlib, 'reload', refresh_gate)
    monkeypatch.setattr(tools, 'run_dir', lambda name: run)
    monkeypatch.setattr(tools.J, 'connect', lambda: queue)
    monkeypatch.setattr(tools.J, 'submit', lambda *args, **kwargs: 321)

    result = tools.submit('episode', json.dumps(
        {'run': 'fresh', 'case': 'selected', 'model': 'cosmos3', 'seed': 0}))
    assert result == 'queued job #321 (episode)'
    assert refreshed and refreshed[0].endswith('episode_stage.py')
