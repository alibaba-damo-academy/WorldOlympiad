"""Loopback FastWAM service; no simulator or GPU imports until CLI startup.

Uses upstream RobotWin preprocessing/normalization unchanged. The wire format
carries raw RGB and native 14-D joint proprioception, NOT LingBot EE features.
Start from a registered episode job, never as untracked background GPU work.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")
MAX_REQUEST_BYTES = 8 * 1024 * 1024


def encode_observation(observation: dict, instruction: str) -> dict:
    cameras = {}
    for name in CAMERAS:
        rgb = np.asarray(observation["observation"][name]["rgb"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"{name}: expected uint8 HWC RGB")
        cameras[name] = {"shape": list(rgb.shape),
                         "data": base64.b64encode(rgb.tobytes()).decode("ascii")}
    return {"cameras": cameras, "joint_qpos": np.asarray(
        observation["joint_action"]["vector"]).tolist(), "instruction": instruction}


def decode_observation(payload: dict) -> tuple[dict, str]:
    instruction = payload["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("nonempty instruction required")
    qpos = np.asarray(payload["joint_qpos"], dtype=np.float32)
    if qpos.shape != (14,) or not np.isfinite(qpos).all():
        raise ValueError("expected finite native 14-D joint qpos")
    cameras = {}
    for name in CAMERAS:
        item = payload["cameras"][name]
        shape = item["shape"]
        if (len(shape) != 3 or any(type(v) is not int for v in shape)
                or not (0 < shape[0] <= 2048 and 0 < shape[1] <= 2048)
                or shape[2] != 3):
            raise ValueError(f"{name}: invalid RGB shape")
        raw = base64.b64decode(item["data"], validate=True)
        if len(raw) != int(np.prod(shape)):
            raise ValueError(f"{name}: RGB byte count mismatch")
        cameras[name] = {"rgb": np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()}
    return {"observation": cameras, "joint_action": {"vector": qpos}}, instruction


def infer_payload(policy, payload: dict) -> dict:
    observation, instruction = decode_observation(payload)
    actions = np.asarray(policy._infer_action_chunk(
        observation=observation, instruction=instruction))
    if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
        raise RuntimeError("FastWAM returned invalid absolute joint action chunk")
    return {"actions": actions.tolist(), "action_type": "qpos",
            "replan_steps": min(int(policy.replan_steps), len(actions))}


def serve(policy, port: int, *, model_name="fastwam", inference=None) -> None:
    # Single-threaded: the policy is stateful and inference cannot overlap.
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404,
                       {"model": model_name, "action_type": "qpos"})

        def do_POST(self):
            if self.path != "/infer":
                return self.reply(404, {"error": "unknown endpoint"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request length")
                payload = json.loads(self.rfile.read(size))
                result = (inference or infer_payload)(policy, payload)
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)})
            except Exception as exc:
                return self.reply(500, {"error": str(exc)})
            self.reply(200, result)

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def local_asset_config(source):
    """Scoped Hydra overlay selecting original upstream Wan assets."""
    from contextlib import contextmanager
    from pathlib import Path
    import tempfile

    @contextmanager
    def config():
        directory = Path(source) / 'configs'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', prefix='wo_assets_',
                                         dir=directory, delete=False) as stream:
            path = Path(stream.name)
            stream.write('# @package _global_\ndefaults:\n  - sim_robotwin\n  - _self_\n'
                         'model:\n  redirect_common_files: false\n')
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)
    return config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--port", type=int, default=29058)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from experiments.robotwin.fastwam_policy.deploy_policy import get_model
    # Use the official Wan-AI .pth assets already downloaded, through the
    # upstream loader's supported non-redirected configuration. Do not rename
    # pickle checkpoints as safetensors or alter checkpoint model parameters.
    with local_asset_config(args.source) as config:
        policy = get_model({"ckpt_setting": args.checkpoint,
                            "sim_cfg_path": str(config),
                            "dataset_stats_path": args.dataset_stats,
                            "action_horizon": 32, "replan_steps": 24,
                            # Released robotwin_uncond_3cam_384 uses the original
                            # shift=5 schedule; upstream's new default is 1.
                            "sigma_shift": 5.0,
                            "seed": args.seed, "device": "cuda"})
    print(json.dumps({"checkpoint": args.checkpoint, "sigma_shift": policy.sigma_shift,
                      "action_horizon": policy.action_horizon, "replan_steps": policy.replan_steps}), flush=True)
    serve(policy, args.port)


if __name__ == "__main__":
    main()
