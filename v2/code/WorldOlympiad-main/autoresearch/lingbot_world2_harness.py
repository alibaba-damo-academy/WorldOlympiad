"""Durable single-case harness for LingBot-World2 14B causal-fast."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any

from .lingbot_world2_adapter import (
    INFER_MODE,
    MODEL_ID,
    PROFILE_ID,
    PROTOCOL,
    TASK,
    runtime_identity,
    validate_input_contract,
)
from .lingbot_world_physics import digest, native_dimensions
from .robotics_resource_lock import exclusive_gpu_job


MANIFEST_SCHEMA = "lingbot_world2_run_manifest.v1"
REQUEST_SCHEMA = "lingbot_world2_run_request.v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_run_id(value: str) -> str:
    if not value or not value[0].isalnum() or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in value):
        raise ValueError("run_id must be a safe path component")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def build_request(
    *, workspace: Path, repository: Path, input_contract: Path, gpu_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not gpu_id.isdigit():
        raise ValueError("gpu_id must identify exactly one GPU")
    contract = validate_input_contract(input_contract)
    runtime = runtime_identity(workspace)
    request = {
        "schema_version": REQUEST_SCHEMA,
        "profile_id": PROFILE_ID,
        "protocol": PROTOCOL,
        "model_id": MODEL_ID,
        "task": TASK,
        "infer_mode": INFER_MODE,
        "input_contract": str(input_contract),
        "input_contract_sha256": digest(input_contract),
        "adapter_sha256": digest(repository / "autoresearch" / "lingbot_world2_adapter.py"),
        "runner_sha256": digest(Path(__file__).resolve()),
        "runtime_identity": runtime,
        "seed": contract["sampling"]["base_seed"],
        "gpu_id": str(int(gpu_id)),
    }
    request["identity_sha256"] = hashlib.sha256(_canonical(request)).hexdigest()
    return request, contract, runtime


def validate_artifacts(directory: Path, request_sha256: str) -> dict[str, Any]:
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("request_identity_sha256") != request_sha256:
        raise ValueError("artifact result does not match this request")
    for name, expected in result.get("artifact_sha256", {}).items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"published artifact digest mismatch: {name}")
    if not result.get("full_decode_verified") or not result.get("finite"):
        raise ValueError("published artifact validation is incomplete")
    return result


def _finalize(run_root: Path, request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "profile_id": PROFILE_ID,
        "protocol": PROTOCOL,
        "request_identity_sha256": request["identity_sha256"],
        "request": "request.json",
        "artifacts": "artifacts",
        "result": result,
    }
    _atomic_json(run_root / "run_manifest.json", manifest)
    return manifest


def _generate(
    *, workspace: Path, contract_path: Path, contract: dict[str, Any],
    runtime: dict[str, Any], request: dict[str, Any], attempt: Path, gpu_id: str,
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import sys

    sys.path.insert(0, runtime["upstream"])
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.cam_utils import get_Ks_transformed
    from wan.utils.utils import save_video

    cfg = WAN_CONFIGS[TASK]
    if float(cfg.sample_shift) != 10.0 or int(cfg.sample_fps) != 16:
        raise ValueError("upstream 14B config no longer matches the frozen adapter")
    base = contract_path.parent
    conditioning = contract["conditioning"]
    image_path = base / conditioning["image"]
    image = Image.open(image_path).convert("RGB")
    expected_wh = native_dimensions(*image.size)
    if expected_wh != image.size or list(expected_wh) != contract["sampling"]["output_wh"]:
        raise ValueError("World2 conditioning must already be rendered at native dimensions")
    frames = contract["sampling"]["frame_num"]
    c2w = np.load(base / conditioning["poses"])
    if not np.array_equal(c2w, np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)):
        raise ValueError("World2 v1 adapter requires an exact fixed-camera pose sequence")
    intrinsics = torch.from_numpy(np.load(base / conditioning["intrinsics"]))
    actual_k = get_Ks_transformed(intrinsics, 480, 832, image.height, image.width, image.height, image.width)[0]
    expected_k = torch.tensor(contract["camera_alignment"]["expected_model_intrinsics"])
    if not torch.allclose(actual_k, expected_k, atol=1e-4, rtol=0):
        raise ValueError("camera intrinsics transformation mismatch")

    started = time.time()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=runtime["checkpoint"], device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
        init_on_cpu=True, convert_model_dtype=True, local_attn_size=18,
        sink_size=6, infer_mode=INFER_MODE,
    )
    torch.cuda.synchronize()
    load_seconds = time.time() - started
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    video = pipe.generate(
        conditioning["prompt"], image, action_path=str(base), chunk_size=4,
        max_area=480 * 832, frame_num=frames,
        timesteps_index=[0, 250, 500, 750], shift=10.0,
        seed=contract["sampling"]["base_seed"], offload_model=False,
    )
    torch.cuda.synchronize()
    generation_seconds = time.time() - started
    expected_shape = (3, frames, expected_wh[1], expected_wh[0])
    if tuple(video.shape) != expected_shape or not torch.isfinite(video).all().item():
        raise ValueError("model returned invalid raw tensor")
    tensor_path = attempt / "prediction_tensor.pt"
    video_path = attempt / "imagination.mp4"
    torch.save(video.detach().cpu(), tensor_path)
    save_video(tensor=video[None], save_file=str(video_path), fps=16, nrow=1,
               normalize=True, value_range=(-1, 1))
    capture = cv2.VideoCapture(str(video_path))
    decoded = 0
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    sample_indices = {0, frames // 4, frames // 2, 3 * frames // 4, frames - 1}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (expected_wh[1], expected_wh[0]):
            raise ValueError("video encoder changed output dimensions")
        if decoded in sample_indices:
            cv2.imwrite(str(attempt / f"frame_{decoded:03d}.png"), frame)
        decoded += 1
    capture.release()
    if decoded != frames or abs(actual_fps - 16) > 1e-6:
        raise ValueError("video decode or timing verification failed")
    artifacts = [tensor_path.name, video_path.name] + [f"frame_{i:03d}.png" for i in sorted(sample_indices)]
    result = {
        "schema_version": "lingbot_world2_result.v1",
        "status": "complete",
        "request_identity_sha256": request["identity_sha256"],
        "frames": decoded,
        "fps": actual_fps,
        "width": expected_wh[0],
        "height": expected_wh[1],
        "model_load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "finite": True,
        "full_decode_verified": True,
        "artifact_sha256": {name: digest(attempt / name) for name in artifacts},
    }
    _atomic_json(attempt / "result.json", result)
    return result


def run(
    *, workspace: str | Path, repository: str | Path, input_contract: str | Path,
    run_id: str, gpu_id: str,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    repository = Path(repository).resolve()
    contract_path = Path(input_contract).resolve()
    run_root = workspace / "outputs" / "robotics_wam" / PROFILE_ID / _safe_run_id(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    request, contract, runtime = build_request(
        workspace=workspace, repository=repository, input_contract=contract_path, gpu_id=gpu_id,
    )
    request_path = run_root / "request.json"
    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != request:
            raise ValueError("run_id already belongs to a different immutable request")
    else:
        _atomic_json(request_path, request)
    artifacts = run_root / "artifacts"
    if artifacts.is_dir():
        result = validate_artifacts(artifacts, request["identity_sha256"])
        return _finalize(run_root, request, result)
    attempt = run_root / f".attempt-{uuid.uuid4().hex}"
    attempt.mkdir()
    with exclusive_gpu_job(workspace / "outputs", str(int(gpu_id)), job_kind=PROFILE_ID):
        result = _generate(
            workspace=workspace, contract_path=contract_path, contract=contract,
            runtime=runtime, request=request, attempt=attempt, gpu_id=gpu_id,
        )
    attempt.replace(artifacts)
    return _finalize(run_root, request, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--required-protocol", required=True, choices=[PROTOCOL])
    args = parser.parse_args()
    print(json.dumps(run(
        workspace=args.workspace, repository=args.repository,
        input_contract=args.input_contract, run_id=args.run_id, gpu_id=args.gpu_id,
    ), indent=2))


if __name__ == "__main__":
    main()
