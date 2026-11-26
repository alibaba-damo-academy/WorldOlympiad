import json
import sys

from wo.eval import report


def test_report_preserves_all_tracks_and_markdown(tmp_path, monkeypatch):
    epdir = tmp_path / 'case' / 'model_s0'
    (epdir / 'case').mkdir(parents=True)
    (epdir / 'episode.json').write_text(json.dumps({
        'case_id': 'case', 'model': 'model', 'seed': 0, 'task_family': 'stacking',
        'task_success': True}))
    physical = {'score': .8, 'questions': [{'compliance': True, 'confidence': .9}],
                'dimensions': {'gravity': .7}}
    geometry = {'final_score_normalized': .6, 'components': {'shape': .5}}
    (epdir / 'case' / 'score.json').write_text(json.dumps({
        'physical': physical, 'three_d': geometry, 'interaction': {'score': .4}}))
    cae = {'status': 'scored', 'score': .8, 'source': 'simulator_execution',
           'entities': {'ee': {'track': .9, 'speed': .7},
                        'object': {'track': .8, 'speed': .6}}}
    monkeypatch.setattr(report, 'episode_cae', lambda _: cae)
    monkeypatch.setattr(report, 'episode_probes', lambda _: {
        'status': 'scored', 'score': .8, 'source': 'simulator_execution'})
    row = report.episode_result(epdir)
    assert row['cae_details'] == cae
    assert row['physical_details'] == physical
    assert row['geometry_details'] == geometry
    assert abs(row['interaction'] - .6) < 1e-6
    assert abs(row['worldolympiad2_score'] - 2 / 3) < 1e-6
    out = tmp_path / 'results.json'
    monkeypatch.setattr(sys, 'argv', ['report', '--runs', str(tmp_path), '--out', str(out)])
    report.main()
    text = out.with_suffix('.md').read_text()
    assert '\n## Per-episode track details\n' in text
    assert 'compliance' in text and 'confidence' in text and 'speed' in text


def test_report_links_task_reference_episode_and_score_provenance(tmp_path, monkeypatch):
    from wo.eval.report_v3 import project

    run = tmp_path
    task = run / 'tasks' / 'case.json'
    task.parent.mkdir(parents=True)
    task.write_text('{"case_id":"case"}')
    ref = run / 'references' / 'case'
    ref.mkdir(parents=True)
    certificate = ref / 'certificate.json'
    certificate.write_text(json.dumps({
        'case_id': 'case', 'success': True, 'task': str(task), 'task_sha256': 'task-hash'}))

    epdir = run / 'case' / 'model_s0'
    (epdir / 'case').mkdir(parents=True)
    (epdir / 'episode.json').write_text(json.dumps({
        'case_id': 'case', 'model': 'model', 'seed': 0, 'reference': str(ref),
        'task_success': False}))
    (epdir / 'episode_reference_binding.json').write_text(json.dumps({
        'state': 'complete', 'reference_binding': {
            'task': str(task), 'task_sha256': 'task-hash'}}))
    score = epdir / 'case' / 'score.json'
    score.write_text('{"physical":{"score":0.5}}')
    receipt = epdir / 'case' / 'three_track_receipt.json'
    receipt.write_text(json.dumps({'score_sha256': 'score-hash'}))
    monkeypatch.setattr(report, 'episode_cae', lambda _: {'status': 'pending', 'score': None})
    monkeypatch.setattr(report, 'episode_probes', lambda _: {'status': 'pending', 'score': None})

    row = report.episode_result(epdir)
    provenance = row['provenance']
    assert provenance['task'] == {'case_id': 'case', 'path': str(task.resolve()), 'sha256': 'task-hash'}
    assert provenance['reference']['path'] == str(ref.resolve())
    assert provenance['reference']['certificate'] == str(certificate.resolve())
    assert provenance['episode']['path'] == str(epdir.resolve())
    assert provenance['score']['path'] == str(score.resolve())
    assert provenance['score']['receipt'] == str(receipt.resolve())
    assert provenance['score']['sha256'] == 'score-hash'
    assert project(row)['provenance'] == provenance


def test_native_episode_case_id_comes_from_bound_reference(tmp_path, monkeypatch):
    epdir = tmp_path / 'apple_case' / 'lingbot_va_s0'
    epdir.mkdir(parents=True)
    ref = tmp_path / 'references' / 'apple_case'
    ref.mkdir(parents=True)
    (ref / 'certificate.json').write_text(json.dumps({'case_id': 'apple_case'}))
    (epdir / 'episode.json').write_text(json.dumps({
        'model': 'lingbot_va', 'seed': 0, 'reference': str(ref),
        'task_success': False}))
    monkeypatch.setattr(report, 'episode_cae', lambda _: {
        'status': 'pending', 'score': None, 'reason': 'test'})
    monkeypatch.setattr(report, 'episode_probes', lambda _: {
        'status': 'pending', 'score': None, 'reason': 'test'})

    row = report.episode_result(epdir)

    assert row['case_id'] == 'apple_case'
    assert row['model'] == 'lingbot_va'
    assert row['task_success'] is False


def test_missing_success_is_not_a_failure():
    rows = [{'model': 'm', 'family': 'f', 'task_success': None},
            {'model': 'm', 'family': 'f', 'task_success': True}]
    aggregate = report.aggregates(rows)['by_model']['m']
    assert aggregate['task_success_rate'] == 1
    assert aggregate['success_measured'] == 1
