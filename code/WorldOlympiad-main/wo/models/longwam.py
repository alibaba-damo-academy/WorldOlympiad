"""Official Long-WAM RoboTwin IDM policy in the shared native evaluation harness."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
from . import fastwam as native
from . import lingbot_va as rt

W = rt.W
SOURCE_COMMIT = 'c6a27e771e803b1c12ffa28b82ba69b42472020e'
WEIGHT_REVISION = 'f456378dec6c4d38ee74717f5d71c0e7490bcc78'
SOURCE = W / 'code/baselines/LongLive-c6a27e/Long-WAM'
WEIGHTS = W / 'models/longwam/robotwin2_idm_f456378'
PYTHON = W / 'envs/longwam/bin/python'


def start_server(out, *, gpu, port, seed):
    for path in [PYTHON, WEIGHTS / 'model.pt', WEIGHTS / 'config.yaml', WEIGHTS / 'dataset_stats.json']:
        if not path.is_file():
            raise FileNotFoundError(path)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    env = dict(rt.client_env(gpu), LONGWAM_ROOT=str(SOURCE),
               LONGWAM_MODEL_ROOT=str(W / 'models/fastwam-assets'),
               DIFFSYNTH_MODEL_BASE_PATH=str(W / 'models/fastwam-assets'),
               PYTHONPATH=f'{rt.ROOT}:{SOURCE / "src"}')
    with (out / 'logs/policy_server.log').open('w') as log:
        return subprocess.Popen([str(PYTHON), '-u', '-m', 'wo.models.longwam_server',
            '--source', str(SOURCE), '--checkpoint-dir', str(WEIGHTS), '--out', str(out / 'policy'),
            '--port', str(port), '--seed', str(seed)], cwd=SOURCE, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def execute(out, args, *, gpu=0, simulator_gpu=None, port=29059, seed=0):
    meta = native.execute(out, args, gpu=gpu, simulator_gpu=simulator_gpu, port=port, seed=seed,
                          server_factory=start_server, policy='longwam')
    progress = json.loads((out / 'policy/progress.json').read_text())
    if progress['observation_steps'] != meta['capture']['native_actions']:
        raise RuntimeError('Long-WAM did not receive every native control-step observation')
    meta['transport_requests'] = meta['capture']['model_requests']
    meta['model_requests'] = progress['inference_chunks']
    meta['history_protocol'] = progress
    native_out = Path(args[args.index('--out') + 1]) if '--out' in args else out / 'native_run'
    (native_out / 'capture/capture_metadata.json').write_text(json.dumps(meta, indent=1))
    return meta


def run_episode(ref, out, *, gpu=0, simulator_gpu=None, port=29059, seed=0, window_s=None, prompt=None):
    out.mkdir(parents=True, exist_ok=True)
    task_spec, state = rt._reference_bridge_inputs(ref, out)
    args = ['--ref', str(Path(ref).resolve()), '--task-spec', str(task_spec),
            '--reference-state', str(state), '--out', str(out), '--seed', str(seed)]
    if window_s is not None:
        args += ['--window-s', str(window_s)]
    if prompt:
        args += ['--prompt', prompt]
    meta = execute(out, args, gpu=gpu, simulator_gpu=simulator_gpu, port=port, seed=seed)
    result = rt.episode_record(out, ref, None, prompt, seed)
    if result.get('model') != 'longwam':
        raise ValueError('native bridge episode receipt is not for longwam')
    result.update(native_result=meta, capture=meta['capture'], action_type='qpos', policy_identity={
        'name': 'Long-WAM RoboTwin2.0-IDM', 'source_commit': SOURCE_COMMIT,
        'checkpoint_revision': WEIGHT_REVISION, 'inference': 'official select_action, sync/reference'},
        imagination={'status': 'not_exported', 'reason':
            'Official IDM control uses partial-denoised future latents and returns actions only.'})
    (out / 'episode.json').write_text(json.dumps(result, indent=1))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=int(os.environ.get('WO_TESTED_MODEL_GPU', '0')))
    p.add_argument('--simulator-gpu', type=int,
                   default=(int(os.environ['WO_SIMULATOR_GPU'])
                            if 'WO_SIMULATOR_GPU' in os.environ else None))
    p.add_argument('--port', type=int, default=29059)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--window-s', type=float)
    p.add_argument('--prompt')
    p.add_argument('--stock-task')
    a = p.parse_args()
    if a.stock_task:
        args = ['--stock-task', a.stock_task, '--seed', str(a.seed)]
        if a.window_s is not None: args += ['--window-s', str(a.window_s)]
        if a.prompt: args += ['--prompt', a.prompt]
        result = execute(a.out, args, gpu=a.gpu, simulator_gpu=a.simulator_gpu,
                         port=a.port, seed=a.seed)
    else:
        if a.ref is None: p.error('--ref is required without --stock-task')
        result = run_episode(a.ref, a.out, gpu=a.gpu, simulator_gpu=a.simulator_gpu,
                             port=a.port, seed=a.seed,
                             window_s=a.window_s, prompt=a.prompt)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
