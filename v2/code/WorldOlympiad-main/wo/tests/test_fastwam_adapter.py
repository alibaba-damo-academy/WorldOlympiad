import json
from types import SimpleNamespace

import pytest

from wo.models import fastwam as fw


def test_server_command_native_weights():
    cmd = fw.server_command(29058, 4)
    assert cmd[cmd.index('--checkpoint') + 1].endswith('robotwin_uncond_3cam_384.pt')
    assert cmd[cmd.index('--dataset-stats') + 1].endswith('robotwin_uncond_3cam_384_dataset_stats.json')
    assert cmd[-1] == '4'
    assert 'wo.models.fastwam_server' in cmd


@pytest.mark.parametrize('rc', [0, 1])
def test_client_protocol_and_owned_cleanup(tmp_path, monkeypatch, rc):
    calls = []
    proc = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: calls.append(('wait', timeout)))
    monkeypatch.setattr(fw, 'start_server', lambda *a, **kw: proc)
    monkeypatch.setattr(fw.rt, '_wait_port', lambda *a: None)
    monkeypatch.setattr(fw.os, 'killpg', lambda *a: calls.append(('kill', a)))

    def client(out, args, **kw):
        calls.append(('client', args))
        capture = out / 'native_run/capture'
        capture.mkdir(parents=True)
        (capture / 'capture_metadata.json').write_text('{"task_success": false}')
        (out / 'logs/client.log').write_text('infra problem')
        return rc

    monkeypatch.setattr(fw.rt, 'run_client', client)
    if rc:
        with pytest.raises(RuntimeError, match='infra problem'):
            fw.execute(tmp_path, ['--stock-task', 'demo'], gpu=0, port=29058, seed=0)
    else:
        assert fw.execute(tmp_path, ['--stock-task', 'demo'], gpu=0, port=29058, seed=0)['task_success'] is False
    args = next(c[1] for c in calls if c[0] == 'client')
    assert args[args.index('--policy') + 1] == 'fastwam'
    assert ('kill', (123, 15)) in calls


def test_episode_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(fw.rt, 'build_plan', lambda *a, **kw: ({'prompt': 'test'}, object()))
    monkeypatch.setattr(fw, 'execute', lambda *a, **kw: {})
    monkeypatch.setattr(fw.rt, 'episode_record', lambda *a: {'model': 'lingbot_va', 'task_success': False})
    result = fw.run_episode(tmp_path / 'ref', tmp_path / 'out')
    assert result['model'] == 'fastwam'
    assert result['action_type'] == 'qpos'
    assert json.loads((tmp_path / 'out/episode.json').read_text()) == result
