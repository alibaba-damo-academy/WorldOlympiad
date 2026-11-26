import json
from wo.models import gr00t


def test_n16_server_contract():
    command = gr00t.server_command(28123)
    assert 'Isaac-GR00T-n1d6' in command[1]
    # run_gr00t_server.py validates this against an EmbodimentTag enum, so the
    # tag must be the enum member's own spelling; lowercase is rejected at startup.
    assert command[command.index('--embodiment-tag') + 1] == 'OXE_DROID'
    assert '--use-sim-policy-wrapper' in command
    assert command[-1] == '28123'


def test_identity_hashes_loaded_weights_and_code(tmp_path):
    checkpoint = tmp_path / 'weights'
    source = tmp_path / 'source'
    checkpoint.mkdir()
    (source / 'gr00t').mkdir(parents=True)
    (source / 'gr00t/policy.py').write_text('pass')
    (checkpoint / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'shard.safetensors'}}))
    (checkpoint / 'shard.safetensors').write_bytes(b'weights')
    for name in ('config.json', 'processor_config.json', 'statistics.json', 'embodiment_id.json'):
        (checkpoint / name).write_text('{}')
    dependencies = {'flash_attn': '2.8.3', 'flash_attn_binary_sha256': 'binary-a'}
    path = gr00t.runtime_identity(tmp_path / 'out', source, checkpoint, dependencies=dependencies)
    first = json.loads(path.read_text())
    assert first['identity']['profile'] == 'gr00t_n1.6_droid'
    assert first['identity']['checkpoint_revision'] is None
    assert first['identity']['use_sim_policy_wrapper'] is True
    (checkpoint / 'shard.safetensors').write_bytes(b'changed weights')
    second = json.loads(gr00t.runtime_identity(tmp_path / 'out', source, checkpoint,
                                               dependencies=dependencies).read_text())
    assert first['identity_sha256'] != second['identity_sha256']

    assert second['identity']['runtime_dependencies'] == dependencies
    # Replacing the CUDA extension must change identity even with identical weights.
    changed = {**dependencies, 'flash_attn_binary_sha256': 'binary-b'}
    third = json.loads(gr00t.runtime_identity(tmp_path / 'out', source, checkpoint,
                                              dependencies=changed).read_text())
    assert second['identity_sha256'] != third['identity_sha256']


def test_dependencies_come_from_the_policy_interpreter(monkeypatch):
    calls = []
    expected = {'torch': '2.7.1+cu128', 'flash_attn': '2.8.3'}
    def probe(command, **kwargs):
        calls.append(command)
        return json.dumps(expected)
    monkeypatch.setattr(gr00t.subprocess, 'check_output', probe)
    assert gr00t.runtime_dependencies('/policy/python') == expected
    assert calls[0][:2] == ['/policy/python', '-c']
