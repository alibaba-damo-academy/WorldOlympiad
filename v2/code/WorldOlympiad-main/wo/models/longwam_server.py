"""Loopback transport around the official Long-WAM RoboTwin policy callbacks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from wo.models.fastwam_server import decode_observation, serve


def infer_payload(policy, payload):
    from longwam.benchmarks.robotwin.adapter import policy_batch
    observation, instruction = decode_observation(payload)
    # select_action owns history, normalization, chunk scheduling and inference.
    action = policy.select_action(policy_batch(observation, instruction))
    actions = np.asarray(action.detach().cpu(), dtype=np.float32)
    if actions.shape != (1, 14) or not np.isfinite(actions).all():
        raise RuntimeError('Long-WAM returned invalid native joint targets')
    return {'actions': actions.tolist(), 'action_type': 'qpos', 'replan_steps': 1}



def inference_model_config(checkpoint_config, out):
    """Supply an unused legacy training field without changing eval transforms."""
    from omegaconf import OmegaConf
    config = OmegaConf.load(checkpoint_config)
    processor = config.data.train.processor
    if "train_transforms" in processor:
        return checkpoint_config
    if "val_transforms" not in processor:
        raise ValueError("released checkpoint has no evaluation transform configuration")
    processor.train_transforms = []
    path = out / "inference_model_config.yaml"
    OmegaConf.save(config, path)
    (out / "config_compatibility.json").write_text(json.dumps({
        "source": str(checkpoint_config), "effective": str(path),
        "added": {"data.train.processor.train_transforms": []},
        "inference_transforms": "unchanged; official processor is placed in eval mode"}, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    from omegaconf import OmegaConf
    from longwam.benchmarks.robotwin.longwam_policy.deploy_policy import get_model
    settings = OmegaConf.load(args.source / 'configs/eval/robotwin2.yaml')
    settings.checkpoint = str(args.checkpoint_dir / 'model.pt')
    settings.model_config = str(args.checkpoint_dir / 'config.yaml')
    settings.stats = str(args.checkpoint_dir / 'dataset_stats.json')
    settings.output_dir = str(args.out)
    settings.seed = args.seed
    settings.inference_mode = 'idm'
    settings.execution = 'sync'
    settings.hardware = 'reference'
    settings.streaming_vae = False
    args.out.mkdir(parents=True, exist_ok=True)
    settings.model_config = str(inference_model_config(args.checkpoint_dir / 'config.yaml', args.out))
    path = args.out / 'official_runtime.yaml'
    OmegaConf.save(settings, path)
    policy = get_model({'runtime_config': str(path)})
    try:
        print(json.dumps({'model': 'longwam-robotwin-idm', 'policy': type(policy).__name__,
                          'upstream_entrypoint': get_model.__module__ + '.get_model',
                          'observations': 'every native control step'}), flush=True)
        def respond(current, payload):
            result = infer_payload(current, payload)
            runtime = current.runtime
            (args.out / 'progress.json').write_text(json.dumps({
                'observation_steps': runtime.step,
                'inference_chunks': runtime.metrics['requests'],
                'source': 'official LongWAMPolicy.runtime'}, indent=1))
            return result
        serve(policy, args.port, model_name='longwam-robotwin-idm', inference=respond)
    finally:
        policy.close()


if __name__ == '__main__':
    main()
