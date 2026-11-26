import json
import sys
from types import SimpleNamespace

import pytest

from wo.models import fastwam as fw
from wo.agent import stages
from wo.models import policy_check


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
            fw.execute(tmp_path, ['--stock-task', 'demo'], gpu=0, simulator_gpu=1, port=29058, seed=0)
    else:
        assert fw.execute(tmp_path, ['--stock-task', 'demo'], gpu=0, simulator_gpu=1, port=29058, seed=0)['task_success'] is False
    args = next(c[1] for c in calls if c[0] == 'client')
    assert args[args.index('--policy') + 1] == 'fastwam'
    assert ('kill', (123, 15)) in calls


def test_native_wam_direct_entry_requires_preflight_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(fw, 'start_server', lambda *_a, **_k:
                        pytest.fail('server must not start on a shared GPU'))
    with pytest.raises(ValueError, match='selected by resource preflight'):
        fw.execute(tmp_path, ['--stock-task', 'demo'], gpu=0,
                   simulator_gpu=None, port=29058, seed=0)


def test_episode_identity(tmp_path, monkeypatch):
    binding = tmp_path / 'binding'
    binding.mkdir()
    task, state = binding / 'task.json', binding / 'state.json'
    task.write_text('{}')
    state.write_text('{}')
    monkeypatch.setattr(fw.rt, '_reference_bridge_inputs', lambda *a: (task, state))
    def execute(out, args, **kw):
        assert args[args.index('--task-spec') + 1] == str(task)
        assert args[args.index('--reference-state') + 1] == str(state)
        assert args[args.index('--out') + 1] == str(out)
        (out / 'episode.json').write_text(json.dumps({'model': 'fastwam',
                                                       'task_success': False}))
    monkeypatch.setattr(fw, 'execute', execute)
    result = fw.run_episode(tmp_path / 'ref', tmp_path / 'out')
    assert result['model'] == 'fastwam'
    assert result['action_type'] == 'qpos'
    assert json.loads((tmp_path / 'out/episode.json').read_text()) == result


def test_episode_stage_routes_model_and_simulator_cards_separately():
    args = {'run': 'r', 'case': 'c', 'model': 'fastwam',
            'gpu': 2, 'simulator_gpu': 5}
    command = stages.STAGES['episode'].command(args, 1)
    assert command[command.index('--gpu') + 1] == '2'
    assert command[command.index('--simulator-gpu') + 1] == '5'
    args.pop('gpu')
    args.pop('simulator_gpu')
    command = stages.STAGES['episode'].command(args, 1)
    assert '--gpu' not in command
    assert '--simulator-gpu' not in command


def test_policy_check_uses_selected_gpu_pair(tmp_path, monkeypatch):
    monkeypatch.setenv('WO_TESTED_MODEL_GPU', '2')
    monkeypatch.setenv('WO_SIMULATOR_GPU', '5')
    out = tmp_path / 'check'
    monkeypatch.setattr(sys, 'argv', ['policy_check', '--model', 'fastwam',
                                     '--stock-task', 'stack_blocks_two', '--out', str(out),
                                     '--port', '29058'])
    def run(command, **kwargs):
        assert command[command.index('--gpu') + 1] == '2'
        assert command[command.index('--simulator-gpu') + 1] == '5'
        capture = out / 'native_run/capture'
        capture.mkdir(parents=True)
        (capture / 'capture_metadata.json').write_text('{}')
    monkeypatch.setattr(policy_check.subprocess, 'run', run)
    monkeypatch.setattr(policy_check, 'policy_check_receipt', lambda *_a: {'control_only': True})
    policy_check.main()
    assert json.loads((out / 'policy_check.json').read_text())['control_only'] is True
