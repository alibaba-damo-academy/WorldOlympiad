"""Authoring must not validate with schema classes cached by the tool server."""
import json
from pathlib import Path

import pytest

from wo.agent import tools


def payload():
    return {
        'case_id': 'members', 'family': 'transfer', 'instruction': 'Move bead.',
        'objects': [
            {'object_id': 'bead', 'category': 'bead', 'role': 'manipuland',
             'inventory_object_id': 'beads', 'scene_member_id': 'bead_0',
             'initial_container': 'cup'},
            {'object_id': 'cup', 'category': 'cup', 'role': 'destination'},
            {'object_id': 'bowl', 'category': 'bowl', 'role': 'destination'}],
        'steps': [{'primitive': 'transport', 'subject': 'bead', 'target': 'bowl',
                   'goal': {'kind': 'inside', 'subject': 'bead', 'target': 'bowl'}}],
    }


def setup_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, 'MAIN', Path(__file__).resolve().parents[2])
    monkeypatch.setattr(tools, 'run_dir', lambda run: tmp_path)
    # Simulate the pre-promotion class reference, without reloading modules or
    # mutating schema state used by other tests.
    monkeypatch.setattr(tools, 'TaskSpec', object())


def test_author_uses_fresh_schema_and_preserves_generation(monkeypatch, tmp_path):
    setup_tools(monkeypatch, tmp_path)
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    out = tasks / 'members.json'
    out.write_text(json.dumps({'generation': '/existing/scene'}))
    result = tools.author_task('test', json.dumps(payload()))
    saved = json.loads(out.read_text())
    assert saved['schema_version'] == 'wo_task_spec.v1'
    assert saved['objects'][0]['scene_member_id'] == 'bead_0'
    assert saved['generation'] == '/existing/scene'
    assert 'steps=1' in result


def test_invalid_current_schema_does_not_overwrite(monkeypatch, tmp_path):
    setup_tools(monkeypatch, tmp_path)
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    out = tasks / 'members.json'
    out.write_text('original')
    bad = payload()
    del bad['objects'][0]['inventory_object_id']
    with pytest.raises(ValueError, match='must be declared together'):
        tools.author_task('test', json.dumps(bad))
    assert out.read_text() == 'original'
