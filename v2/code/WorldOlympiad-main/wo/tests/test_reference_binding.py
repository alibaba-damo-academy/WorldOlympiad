import json
import hashlib
from pathlib import Path

import pytest

from wo.eval.reference_binding import capture, verify_binding, write_binding


def setup_reference(tmp_path):
    ref = tmp_path / 'ref'
    (ref / 'replay').mkdir(parents=True)
    task = tmp_path / 'task.json'
    task.write_text('{"instruction":"move all three objects"}')
    (ref / 'certificate.json').write_text(json.dumps({'success': True, 'task': str(task), 'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(), 'frame_count': 1}))
    (ref / 'trajectory.json').write_text('{"frames":[]}')
    (ref / 'replay/exterior_left.mp4').write_bytes(b'video-version-one')
    ep = tmp_path / 'episode'
    (ep / 'case').mkdir(parents=True)
    return ref, ep, task


def test_receipt_accepts_exact_reference(tmp_path):
    ref, ep, _ = setup_reference(tmp_path)
    write_binding(ref, ep / 'case')
    assert verify_binding(ep, ref)['status'] == 'verified'


@pytest.mark.parametrize('changed', ['trajectory', 'video', 'task', 'certificate', 'urdf'])
def test_receipt_rejects_independent_reference_changes(tmp_path, changed):
    ref, ep, task = setup_reference(tmp_path)
    write_binding(ref, ep / 'case')
    if changed == 'trajectory':
        (ref / 'trajectory.json').write_text('{"frames":[1]}')
    elif changed == 'video':
        (ref / 'replay/exterior_left.mp4').write_bytes(b'video-version-two')
    elif changed == 'task':
        task.write_text('{"instruction":"move one object"}')
    elif changed == 'certificate':
        (ref / 'certificate.json').write_text(json.dumps({'success': True, 'task': str(task), 'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(), 'frame_count': 1, 'revision': 2}))
    else:
        (ref / 'physics_sdf').mkdir()
        (ref / 'physics_sdf/door.urdf').write_text('<robot/>')
    with pytest.raises(ValueError, match='reference version mismatch'):
        verify_binding(ep, ref)


def test_unsuccessful_reference_cannot_be_bound(tmp_path):
    ref, _, task = setup_reference(tmp_path)
    (ref / 'certificate.json').write_text(json.dumps({'success': False, 'task': str(task)}))
    with pytest.raises(ValueError, match='successful expert'):
        capture(ref)


def _size_ok(source):
    source.mkdir(parents=True, exist_ok=True)
    (source / 'physical_size_validation.json').write_text(json.dumps({'success': True}))


def test_verify_task_rejects_different_declared_generation(tmp_path):
    source_a = tmp_path / 'source-a'
    source_b = tmp_path / 'source-b'
    _size_ok(source_a)
    _size_ok(source_b)
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'generation': str(source_a)}))
    ref = tmp_path / 'ref'
    ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps({
        'success': True, 'task': str(task), 'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(),
        'generation': str(source_b)}))
    with pytest.raises(ValueError, match='generation source'):
        from wo.eval.reference_binding import verify_task
        verify_task(ref, task)


def test_verify_task_without_expected_task_rejects_generation_substitution(tmp_path):
    source_a = tmp_path / 'source-a'; source_b = tmp_path / 'source-b'
    _size_ok(source_a); _size_ok(source_b)
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'generation': str(source_a)}))
    ref = tmp_path / 'ref'; ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps({
        'success': True, 'task': str(task),
        'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(),
        'generation': str(source_b)}))
    from wo.eval.reference_binding import verify_task
    with pytest.raises(ValueError, match='generation source'):
        verify_task(ref)


def test_verify_task_rejects_missing_generation_certificate(tmp_path):
    source = tmp_path / 'source'
    _size_ok(source)
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'generation': str(source)}))
    ref = tmp_path / 'ref'
    ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps({
        'success': True, 'task': str(task),
        'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest()}))
    from wo.eval.reference_binding import verify_task
    with pytest.raises(ValueError, match='omits declared generation'):
        verify_task(ref, task)


def test_verify_task_accepts_measured_size_for_revised_task(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'physical_size_validation.json').write_text(json.dumps({
        'success': True, 'objects': {'box': {
            'measured_m': [0.14, 0.14, 0.162399992], 'tolerance_m': 0.0005}}}))
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'generation': str(source), 'objects': [{
        'object_id': 'box', 'geometry_constraints': {
            'overall_dimensions_m': [0.14, 0.14, 0.1624]}}]}))
    ref = tmp_path / 'ref'
    ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps({
        'success': True, 'task': str(task),
        'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(),
        'generation': str(source)}))
    from wo.eval.reference_binding import verify_task
    assert verify_task(ref, task)['success'] is True


def test_verify_task_accepts_same_declared_generation(tmp_path):
    source = tmp_path / 'source'
    _size_ok(source)
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'generation': str(source)}))
    ref = tmp_path / 'ref'
    ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps({
        'success': True, 'task': str(task), 'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(),
        'generation': str(source)}))
    from wo.eval.reference_binding import verify_task
    assert verify_task(ref, task)['success'] is True


def test_verify_task_accepts_legacy_generation_without_size_contract(tmp_path):
    """Old successful tasks did not emit physical_size_validation.json."""
    source = tmp_path / 'legacy-source'
    source.mkdir()
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'generation': str(source), 'objects': [
        {'object_id': 'apple', 'geometry_constraints': {}}
    ]}))
    ref = tmp_path / 'ref'
    ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps({
        'success': True, 'task': str(task),
        'task_sha256': hashlib.sha256(task.read_bytes()).hexdigest(),
        'generation': str(source)}))
    from wo.eval.reference_binding import verify_task
    assert verify_task(ref, task)['success'] is True


def legacy_case(ref, ep):
    (ep / 'case/ref.mp4').write_bytes((ref / 'replay/exterior_left.mp4').read_bytes())
    (ep / 'case/worldolympiad_case.json').write_text(json.dumps({
        'reference_video': 'ref.mp4', 'reference_view_id': 'exterior_left'}))


def test_legacy_matching_video_reports_its_limitation(tmp_path):
    ref, ep, _ = setup_reference(tmp_path)
    legacy_case(ref, ep)
    result = verify_binding(ep, ref)
    assert result['status'] == 'legacy_video_verified'
    assert 'no reference trajectory hash' in result['limitation']


def test_legacy_changed_video_is_rejected(tmp_path):
    ref, ep, _ = setup_reference(tmp_path)
    legacy_case(ref, ep)
    (ref / 'replay/exterior_left.mp4').write_bytes(b'new-reference')
    with pytest.raises(ValueError, match='reference version mismatch'):
        verify_binding(ep, ref)


def test_unscored_episode_has_no_reference_binding(tmp_path):
    ref, ep, _ = setup_reference(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_binding(ep, ref)


def test_cae_rejects_mixed_reference_before_loading_tracks(tmp_path, monkeypatch):
    from wo.eval import cae_episodes
    ref, ep, _ = setup_reference(tmp_path)
    legacy_case(ref, ep)
    (ep / 'episode.json').write_text(json.dumps({'reference': str(ref)}))
    (ep / 'task_graph.json').write_text('{"success_program":[]}')
    (ref / 'replay/exterior_left.mp4').write_bytes(b'new-reference')
    def tracks(*args, **kwargs):
        pytest.fail('mixed reference must be refused before measuring CAE')
    monkeypatch.setattr(cae_episodes, 'entity_tracks', tracks)
    with pytest.raises(ValueError, match='reference version mismatch'):
        cae_episodes.episode_cae(ep)


def test_report_with_mixed_reference_has_no_total(tmp_path):
    from wo.eval import report
    ref, ep, _ = setup_reference(tmp_path)
    legacy_case(ref, ep)
    (ep / 'episode.json').write_text(json.dumps({
        'case_id': 'case', 'model': 'model', 'reference': str(ref), 'task_success': True}))
    (ep / 'task_graph.json').write_text('{"success_program":[]}')
    (ep / 'case/score.json').write_text(json.dumps({
        'physical': {'score': .8}, 'three_d': {'final_score_normalized': .6},
        'interaction': {'score': .4}}))
    (ref / 'replay/exterior_left.mp4').write_bytes(b'new-reference')
    result = report.episode_result(ep)
    assert result['status'] == 'pending'
    assert result['worldolympiad2_score'] is None
    assert 'reference version mismatch' in result['cae_details']['reason']


def test_new_binding_rejects_task_changed_since_certificate(tmp_path):
    ref, ep, task = setup_reference(tmp_path)
    task.write_text('{"instruction":"a different task"}')
    with pytest.raises(ValueError, match='task changed after expert execution'):
        capture(ref)
