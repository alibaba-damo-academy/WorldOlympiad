import json
from wo.eval.report import coverage


def successful_reference(run):
    from wo.eval.reference_binding import digest
    task = run / 'tasks' / 'a.json'
    ref = run / 'references' / 'a'
    ref.mkdir(parents=True, exist_ok=True)
    (ref / 'certificate.json').write_text(json.dumps({
        'success': True, 'task': str(task.resolve()), 'task_sha256': digest(task)}))
    (ref / 'replay').mkdir(exist_ok=True)
    (ref / 'replay' / 'exterior_left.mp4').write_bytes(b'replay fixture')


def test_matrix_audit_keeps_missing_models_pending(tmp_path):
    (tmp_path / 'tasks').mkdir()
    (tmp_path / 'tasks' / 'a.json').write_text(json.dumps({'case_id': 'a', 'family': 'stacking'}))
    rows = [{'case_id': 'a', 'model': 'cosmos3_nano_droid', 'seed': 0,
             'status': 'complete'}]
    # Without an explicit --seeds the audit covers the seeds the run has, so a
    # one-seed benchmark is not reported as a half-finished two-seed one.
    audit = coverage(tmp_path, rows)
    assert audit['seeds'] == [0]
    assert audit['expected_episodes'] == 4
    assert len(audit['pending_matrix_entries']) == 3
    assert audit['missing_references'] == [{'case_id': 'a', 'family': 'stacking', 'status': 'pending'}]
    assert not audit['matrix_complete']
    assert all(x['status'] == 'pending' for x in audit['pending_matrix_entries'])
    # An explicit request still audits the wider matrix.
    wide = coverage(tmp_path, rows, seeds=(0, 1))
    assert wide['seeds'] == [0, 1]
    assert wide['expected_episodes'] == 8
    assert len(wide['pending_matrix_entries']) == 7


def test_run_manifest_declares_the_matrix_to_audit(tmp_path):
    from wo.eval.report import run_manifest

    (tmp_path / 'manifest.json').write_text(json.dumps(
        {"models": ["m1", "m2"], "seeds": [3, 5], "other": "ignored"}))
    m = run_manifest(tmp_path)
    assert m["models"] == ["m1", "m2"]
    assert m["seeds"] == [3, 5]
    # Missing or malformed manifests are empty dicts, never crashes.
    assert run_manifest(tmp_path / "absent") == {}
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text("{not json")
    assert run_manifest(tmp_path / "bad") == {}


def test_coverage_prefers_manifest_then_explicit_then_data(tmp_path, monkeypatch):
    from wo.eval.report import coverage
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': 'done'})

    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "a.json").write_text(json.dumps(
        {"case_id": "a", "family": "stacking"}))
    ref = tmp_path / "references" / "a"
    ref.mkdir(parents=True)
    successful_reference(tmp_path)
    rows = [{"case_id": "a", "model": "m1", "seed": 0, "status": "complete"}]
    # No manifest, no CLI: infer from data.
    audit = coverage(tmp_path, rows, models=("m1", "m2"))
    assert audit["seeds"] == [0] and audit["models"] == ["m1", "m2"]
    assert audit["expected_episodes"] == 2
    # Manifest present: it wins.
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"models": ["m1", "m2", "m3"], "seeds": [0, 1]}))
    audit2 = coverage(tmp_path, rows)
    assert audit2["seeds"] == [0, 1] and audit2["models"] == ["m1", "m2", "m3"]
    assert audit2["expected_episodes"] == 6
    # Explicit CLI overrides manifest.
    audit3 = coverage(tmp_path, rows, models=("m1",), seeds=(7,))
    assert audit3["seeds"] == [7] and audit3["models"] == ["m1"]
    assert audit3["expected_episodes"] == 1


def test_manifest_case_scope_excludes_historical_task_and_episode(tmp_path, monkeypatch):
    from wo.eval.report import coverage, select_declared_cases

    (tmp_path / 'tasks').mkdir()
    (tmp_path / 'tasks' / 'a.json').write_text(json.dumps({'case_id': 'a', 'family': 'pick_place'}))
    (tmp_path / 'tasks' / 'legacy.json').write_text(json.dumps({'case_id': 'legacy', 'family': 'pick_place'}))
    (tmp_path / 'manifest.json').write_text(json.dumps({'cases': ['a'], 'models': ['m'], 'seeds': [0]}))
    successful_reference(tmp_path)
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': 'done'})
    rows = [{'case_id': 'a', 'model': 'm', 'seed': 0, 'status': 'complete'},
            {'case_id': 'legacy', 'model': 'm', 'seed': 0, 'status': 'complete'}]
    kept, excluded = select_declared_cases(rows, ['a'])
    assert [row['case_id'] for row in kept] == ['a']
    assert [row['case_id'] for row in excluded] == ['legacy']
    audit = coverage(tmp_path, kept)
    assert audit['cases'] == 1 and audit['expected_episodes'] == 1
    assert audit['matrix_complete']


def test_requested_seeds_prefers_explicit_and_falls_back_to_zero(tmp_path):
    from wo.eval.report import requested_seeds

    assert requested_seeds([{'seed': 1}, {'seed': 0}, {'seed': 1}]) == [0, 1]
    assert requested_seeds([{'seed': 3}], (0, 1)) == [0, 1]
    # A run with no measured episodes still audits one seed rather than crashing.
    assert requested_seeds([]) == [0]


def test_matrix_requires_successful_reference_and_na_reason(tmp_path, monkeypatch):
    (tmp_path / 'tasks').mkdir()
    (tmp_path / 'tasks' / 'a.json').write_text(json.dumps({'case_id': 'a', 'family': 'hanging'}))
    ref = tmp_path / 'references' / 'a'
    ref.mkdir(parents=True)
    successful_reference(tmp_path)
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': 'done'})
    rows = [{'case_id': 'a', 'model': 'm', 'seed': 0, 'status': 'not_applicable'}]
    assert not coverage(tmp_path, rows, models=('m',), seeds=(0,))['matrix_complete']
    rows[0]['reason'] = 'native simulator lacks articulated importer'
    assert coverage(tmp_path, rows, models=('m',), seeds=(0,))['matrix_complete']
    (ref / 'certificate.json').write_text(json.dumps({'success': False}))
    assert not coverage(tmp_path, rows, models=('m',), seeds=(0,))['matrix_complete']


def test_coverage_rejects_changed_task_and_unfinished_reference(tmp_path, monkeypatch):
    (tmp_path / 'tasks').mkdir()
    task = tmp_path / 'tasks' / 'a.json'
    task.write_text(json.dumps({'case_id': 'a', 'family': 'stacking'}))
    successful_reference(tmp_path)
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': 'done'})
    rows = [{'case_id': 'a', 'model': 'm', 'seed': 0, 'status': 'complete'}]
    def audit():
        return coverage(tmp_path, rows, models=('m',), seeds=(0,))
    assert audit()['matrix_complete']
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': 'running'})
    assert audit()['missing_references'][0]['status'] == 'pending'
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': 'failed'})
    assert audit()['missing_references'][0]['status'] == 'failed'
    monkeypatch.setattr('wo.agent.episode_stage.latest_reference_jobs', lambda run: {'a': 'done'})
    assert audit()['matrix_complete']
    task.write_text(task.read_text() + '\n')
    assert not audit()['matrix_complete']
    successful_reference(tmp_path)
    assert audit()['matrix_complete']
    (tmp_path / 'references/a/certificate.json').write_text('{broken')
    assert not audit()['matrix_complete']


def test_episodes_outside_the_declared_seeds_stay_out_of_the_table():
    from wo.eval.report import select_declared

    rows = [{"case_id": "a", "seed": 0}, {"case_id": "a", "seed": 1}, {"case_id": "b", "seed": 0}]
    kept, gone = select_declared(rows, (0,))
    assert [r["case_id"] for r in kept] == ["a", "b"] and all(r["seed"] == 0 for r in kept)
    assert gone == [{"case_id": "a", "seed": 1}]
    # Nothing declared: nothing is hidden.
    assert select_declared(rows, ()) == (rows, [])
    assert select_declared(rows, None) == (rows, [])
