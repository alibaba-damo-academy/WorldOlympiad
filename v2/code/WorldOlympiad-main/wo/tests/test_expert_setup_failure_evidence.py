import hashlib
import json
from types import SimpleNamespace

import pytest

from wo.expert import run


def setup_inputs(tmp_path, monkeypatch):
    task = tmp_path/'task.json'
    task.write_text('{"case_id":"case"}')
    support = tmp_path/'support.sdf'; support.write_text('real source bytes')
    obj = SimpleNamespace(object_id='support', sdf_path=support)
    scene = SimpleNamespace(support=obj, objects={})
    spec = SimpleNamespace(case_id='case', family='articulation', steps=[])
    monkeypatch.setattr(run.TaskSpec, 'load', lambda _:spec)
    monkeypatch.setattr(run, 'canonicalize_support_aliases', lambda s,g:s)
    monkeypatch.setattr(run, 'resolve_scenecode', lambda s,g:scene)
    return task, scene, tmp_path/'reference'


def test_placement_failure_writes_current_failure_with_source_binding(tmp_path, monkeypatch):
    task, scene, out = setup_inputs(tmp_path, monkeypatch)
    def fail(*args, **kw):
        raise RuntimeError('no reachable base')
    monkeypatch.setattr(run, 'place_robot', fail)
    with pytest.raises(RuntimeError, match='no reachable base'):
        run.run_case(task, tmp_path/'generation', out)
    cert = json.loads((out/'certificate.json').read_text())
    assert cert['success'] is False
    assert cert['frame_count'] == 0 and cert['steps'] == []
    assert cert['task_sha256'] == hashlib.sha256(task.read_bytes()).hexdigest()
    assert cert['source_assets']['support']['sdf_sha256'] == hashlib.sha256(scene.support.sdf_path.read_bytes()).hexdigest()
    assert cert['failure']['stage'] == 'robot_placement'
    assert not (out/'trajectory.json').exists()


def test_failed_retry_keeps_previous_certificate_separate_and_uses_start_snapshot(tmp_path, monkeypatch):
    task, scene, out = setup_inputs(tmp_path, monkeypatch)
    out.mkdir(); previous = '{"success":false,"old_attempt":true}'
    (out/'certificate.json').write_text(previous)
    initial = task.read_bytes()
    def fail(*args, **kw):
        task.write_text('{"case_id":"changed_during_run"}')
        raise RuntimeError('current placement failure')
    monkeypatch.setattr(run, 'place_robot', fail)
    with pytest.raises(RuntimeError):
        run.run_case(task, tmp_path/'generation', out)
    cert = json.loads((out/'certificate.json').read_text())
    assert cert['failure']['detail'] == 'current placement failure'
    assert cert['task_sha256'] == hashlib.sha256(initial).hexdigest()
    assert cert['task_sha256'] != hashlib.sha256(task.read_bytes()).hexdigest()
    archived = list((out/'.history').glob('*/certificate.json'))
    assert len(archived) == 1 and archived[0].read_text() == previous


def test_successful_placement_does_not_create_a_setup_failure(tmp_path, monkeypatch):
    task, scene, out = setup_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(run, 'place_robot', lambda *a,**kw:(scene, {}))
    monkeypatch.setattr(run, 'prepare_bodies', lambda *a:{})
    class StopAtRollout(Exception):
        pass
    def stop(*args, **kw):
        raise StopAtRollout()
    monkeypatch.setattr(run, 'World', stop)
    with pytest.raises(StopAtRollout):
        run.run_case(task, tmp_path/'generation', out)
    assert not (out/'certificate.json').exists()
