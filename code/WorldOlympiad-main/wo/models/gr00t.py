"""GR00T N1.6-DROID in native Isaac, using the n1d6 ZMQ protocol.

Shares scene preparation and simulator capture with Cosmos, NOT its policy.
Bulk benchmark execution must follow a checked native positive control.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from . import cosmos as c

SOURCE = c.W / 'code/baselines/Isaac-GR00T-n1d6'
CHECKPOINT = c.W / 'models/GR00T-N1.6-DROID.fetch'
PYTHON = c.W / 'envs/groot/bin/python'


def runtime_dependencies(python=PYTHON):
    """Inspect the policy interpreter, rather than the simulator interpreter."""
    code = r'''
import hashlib
import importlib.util
import importlib.metadata
import json
import sys
from pathlib import Path
import torch

binary = importlib.util.find_spec("flash_attn_2_cuda")
if binary is None or binary.origin is None:
    raise RuntimeError("GR00T requires the FlashAttention 2 CUDA extension")
path = Path(binary.origin)
digest = hashlib.sha256()
with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "transformers": importlib.metadata.version("transformers"),
    "flash_attn": importlib.metadata.version("flash-attn"),
    "flash_attn_binary_sha256": digest.hexdigest(),
}))
'''
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))

def runtime_identity(out, source=SOURCE, checkpoint=CHECKPOINT, *, dependencies=None):
    """Hash actual loaded code and every weight shard; no invented upstream revision."""
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    files = sorted(set(index['weight_map'].values()))
    records = {name: c._sha(checkpoint / name) for name in files}
    for name in ('config.json', 'processor_config.json', 'statistics.json', 'embodiment_id.json'):
        records[name] = c._sha(checkpoint / name)
    code = {str(p.relative_to(source)): c._sha(p) for p in sorted((source / 'gr00t').rglob('*.py'))}
    if not code or not files:
        raise ValueError('empty GR00T source or checkpoint')
    identity = {'profile': 'gr00t_n1.6_droid', 'checkpoint': str(checkpoint),
                'checkpoint_files': records, 'source': str(source), 'source_files': code,
                'model_config_sha256': records['config.json'],
                'checkpoint_revision': None, 'framework_revision': None,
                'revision_note': 'local content hashes; no unverified git/HF revision claimed',
                'use_sim_policy_wrapper': True, 'execution_horizon': 8,
                'runtime_dependencies': dependencies or {}}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path = out / 'model/runtime_identity.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': 'wam_runtime_identity.v1',
                                'identity': identity, 'identity_sha256': digest}, indent=2))
    return path


def server_command(port):
    return [str(PYTHON), str(SOURCE / 'gr00t/eval/run_gr00t_server.py'),
            '--model-path', str(CHECKPOINT), '--embodiment-tag', 'OXE_DROID',
            '--use-sim-policy-wrapper', '--host', '127.0.0.1', '--port', str(port)]


def with_policy_server(logs, env, port, callback):
    with (logs / 'policy_server.log').open('w') as log:
        server = subprocess.Popen(server_command(port), cwd=SOURCE,
                                  env=dict(env, PYTHONPATH=str(SOURCE)),
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            c._wait_port(port, 900, server)
            return callback()
        finally:
            if server.poll() is None:
                os.killpg(server.pid, 15)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, 9)
                server.wait()


def run_episode(ref, out, *, gpu=0, port=28000, prompt=None, window_s=60, seed=0):
    cert = json.loads((ref / 'certificate.json').read_text())
    if c.is_articulated(c.TaskSpec.load(Path(cert['task']))):
        return c.run_articulated_episode(ref, out, gpu=gpu, port=port, prompt=prompt,
                                        window_s=window_s, seed=seed, policy_model='gr00t')
    out.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    manifest, graph, evaluation, spec = c.build_bundle(ref, out, window_s=window_s)
    prompt = prompt or spec.instruction
    env, env_r, env_i = c._envs(gpu)
    identity = runtime_identity(out, dependencies=runtime_dependencies())
    backend = out / 'native_backend'
    for derived in (backend, out / 'native_run'):
        shutil.rmtree(derived, ignore_errors=True)
    c._run([c.PY, c.R / 'scripts/build_task_graph_worldir.py', '--reference-manifest', manifest,
            '--task-graph', graph, '--trajectory', manifest.parent / 'trajectory.json',
            '--robot-urdf', c.ROBOT_URDF, '--prompt', prompt, '--scene-id', spec.case_id,
            '--output-dir', backend], logs / 'worldir.log', env_r)
    usd = backend / 'scene/task_graph_worldir.usd'
    c._run([c.ISAAC, c.R / 'scripts/materialize_worldir_isaac_usd.py', '--plan',
            backend / 'isaac/isaac_compile_plan.json', '--output', usd], logs / 'usd.log', env_i)
    with (logs / 'policy_server.log').open('w') as log:
        server = subprocess.Popen(server_command(port), cwd=SOURCE,
                                  env=dict(env, PYTHONPATH=str(SOURCE)),
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            c._wait_port(port, 900, server)
            c._run([c.ISAAC, c.R / 'scripts/run_cosmos3_worldir_isaac_closed_loop.py',
                    '--policy-model', 'gr00t', '--world-repo', c.R, '--world-usd', usd,
                    '--worldir-plan', backend / 'isaac/isaac_compile_plan.json',
                    '--reference-trajectory', manifest.parent / 'trajectory.json',
                    '--robot-usd', c.ROBOT_USD, '--robot-usd-sha256', c.ROBOT_USD_SHA,
                    '--runtime-identity', identity, '--output-dir', out / 'native_run',
                    '--prompt', prompt, '--seed-schedule-seed', str(seed),
                    '--policy-server-host', '127.0.0.1', '--policy-server-port', str(port),
                    '--official-robolab-task', 'worldolympiad/task_graph_native_episode',
                    '--task-graph', graph, '--official-robolab-task-sha256', c._sha(graph),
                    '--evaluation-input-manifest', evaluation], logs / 'episode.log', env_i)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, 15)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, 9)
                server.wait()
    frames = json.loads(evaluation.read_text())['simulator_state']['evaluation_window']['frame_count']
    c._run([c.PY, '-m', 'wo.models.native_isaac_export',
            '--capture', out / 'native_run/capture', '--output', out / 'native_run/export',
            '--frame-count', str(frames)], logs / 'export.log', env_r)
    return c._episode_result(out, ref, spec, prompt, frames, seed, model='gr00t_n1.6_droid')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', type=int, default=int(os.environ.get('WO_TESTED_MODEL_GPU', '0')))
    ap.add_argument('--port', type=int, default=28000)
    ap.add_argument('--window-s', type=float, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prompt')
    a = ap.parse_args()
    try:
        result = run_episode(a.ref, a.out, gpu=a.gpu, port=a.port, window_s=a.window_s,
                             seed=a.seed, prompt=a.prompt)
    except c.Unsupported as exc:
        # Only a genuine native importer/representation limitation is NA.
        cert = json.loads((a.ref / 'certificate.json').read_text())
        result = {'schema_version': 'wo_model_episode.v1', 'model': 'gr00t_n1.6_droid',
                  'case_id': cert['case_id'], 'reference': str(a.ref), 'seed': a.seed,
                  'window_s': a.window_s, 'task_success': None,
                  'status': 'not_applicable', 'reason': str(exc),
                  'termination_reason': 'not_applicable', 'not_applicable_reason': str(exc), 'videos': {}}
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'episode.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
