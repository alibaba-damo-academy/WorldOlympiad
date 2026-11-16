#!/usr/bin/env python3
"""Serve the 3D reward backend over HTTP without world-r1 dependencies."""

from __future__ import annotations

from _bootstrap import setup_paths

setup_paths()

import argparse
import json
import os
import pickle
import signal
import sys
import threading
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model.vlm import resolve_model_name, resolve_vlm_backend

DEFAULT_MODEL_NAME = "worldeval/weights/da3"
DEFAULT_SCORING_MODEL: str | None = None
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8089

reward_3d_manager = None


def resolve_model_arg(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[1]
    project_candidate = repo_root.parent / path
    if project_candidate.exists():
        return str(project_candidate.resolve())
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    return value


def shutdown_server(signum=None, frame=None):
    global reward_3d_manager
    if reward_3d_manager is not None:
        print("Shutting down 3D reward backend...")
        reward_3d_manager.shutdown()
        reward_3d_manager = None


class Reward3DHandler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._write_json(
                200,
                {
                    "status": "ok",
                    "model_name": getattr(self.server, "model_name", None),
                    "scorer": getattr(self.server, "scorer", None),
                    "scoring_model": getattr(self.server, "scoring_model", None),
                    "num_workers": getattr(self.server, "num_workers", None),
                    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                },
            )
            return
        self._write_json(404, {"error": f"Unsupported path: {self.path}"})

    def do_POST(self):
        if self.path == "/score_file":
            self._handle_score_file()
            return
        if self.path == "/extract_trajectory":
            self._handle_extract_trajectory()
            return
        if self.path != "/":
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length)

        try:
            data = pickle.loads(payload)
            batch_videos = data["videos"]
            batch_prompts = data["prompts"]
            batch_camera_trajectories = data.get("camera_trajectories")

            if reward_3d_manager is None:
                raise RuntimeError("3D reward backend is not initialized")

            with self.server.inference_lock:
                outputs = reward_3d_manager.compute_batch_scores(
                    batch_videos,
                    batch_prompts,
                    camera_trajectories=batch_camera_trajectories,
                )
            details = getattr(reward_3d_manager, "last_results", {}).get("per_video_results")
            response = pickle.dumps({"outputs": outputs, "details": details})

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except Exception:
            response = traceback.format_exc().encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length)
        return json.loads(payload.decode("utf-8"))

    def _handle_score_file(self) -> None:
        try:
            data = self._read_json_body()
            video_path = data.get("video_path") or data.get("video")
            if not video_path:
                raise ValueError("Request JSON must contain 'video_path'.")

            if reward_3d_manager is None:
                raise RuntimeError("3D reward backend is not initialized")

            from reward_3d import load_dynamic_masks, load_video_frames_as_jpeg

            resolved_video_path = str(Path(video_path).resolve())
            max_frames = data.get("max_frames")
            frames = load_video_frames_as_jpeg(resolved_video_path, max_frames=max_frames)

            dynamic_masks = None
            dynamic_mask_video_path = data.get("dynamic_mask_video_path")
            resolved_dynamic_mask_video_path = None
            if dynamic_mask_video_path:
                resolved_dynamic_mask_video_path = str(Path(dynamic_mask_video_path).resolve())
                dynamic_masks = load_dynamic_masks(resolved_dynamic_mask_video_path, max_frames=max_frames)
                if len(dynamic_masks) != len(frames):
                    raise ValueError(
                        "Dynamic mask frame count does not match sampled video frames: "
                        f"masks={len(dynamic_masks)}, frames={len(frames)}"
                    )

            with self.server.inference_lock:
                reward_3d_manager.compute_batch_scores(
                    [frames],
                    [str(data.get("prompt") or "")],
                    camera_trajectories=[data.get("camera_trajectory")],
                    dynamic_masks=[dynamic_masks],
                    use_lpips=data.get("use_lpips"),
                )

            if not reward_3d_manager.last_results or not reward_3d_manager.last_results["per_video_results"]:
                raise RuntimeError("3D reward backend did not return any per-video result.")

            result = dict(reward_3d_manager.last_results["per_video_results"][0])
            result["video_path"] = resolved_video_path
            result["reconstruction_model_name"] = getattr(self.server, "model_name", None)
            result["scorer_type"] = reward_3d_manager.scorer_type
            result["scoring_model_name"] = reward_3d_manager.scoring_model_name
            result["num_workers"] = reward_3d_manager.num_workers
            result["max_frames"] = len(frames)
            result["dynamic_mask_video_path"] = resolved_dynamic_mask_video_path
            result["batch_dir"] = reward_3d_manager.last_results["batch_dir"]
            result["reward_3d_server"] = {
                "url": f"http://{self.server.server_address[0]}:{self.server.server_address[1]}",
                "model_name": getattr(self.server, "model_name", None),
            }
            self._write_json(200, {"result": result})
        except Exception as exc:
            self._write_json(
                500,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    def _handle_extract_trajectory(self) -> None:
        try:
            data = self._read_json_body()
            video_path = data.get("video_path") or data.get("video")
            if not video_path:
                raise ValueError("Request JSON must contain 'video_path'.")

            if reward_3d_manager is None:
                raise RuntimeError("3D reward backend is not initialized")

            from reward_3d import load_video_frames_as_jpeg_with_indices

            resolved_video_path = Path(video_path).resolve()
            max_frames = data.get("max_frames")
            process_res = int(data.get("process_res") or 504)
            output_path = Path(data.get("output_path")).resolve() if data.get("output_path") else (
                resolved_video_path.parent / f"{resolved_video_path.stem}_da3_camera_trajectory.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            frames, frame_indices = load_video_frames_as_jpeg_with_indices(
                resolved_video_path,
                max_frames=max_frames,
            )
            with self.server.inference_lock:
                trajectory = reward_3d_manager.extract_camera_trajectory(
                    frames,
                    frame_indices,
                    process_res=process_res,
                )

            output_path.write_text(
                json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._write_json(
                200,
                {
                    "trajectory": trajectory,
                    "path": str(output_path),
                    "frame_indices": frame_indices,
                    "reward_3d_server": {
                        "url": f"http://{self.server.server_address[0]}:{self.server.server_address[1]}",
                        "model_name": getattr(self.server, "model_name", None),
                    },
                },
            )
        except Exception as exc:
            self._write_json(
                500,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, format, *args):
        print(f"[serve_reward_3d] {self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Local 3D reward server")
    parser.add_argument("--host", default=os.getenv("REWARD_3D_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("REWARD_3D_PORT", DEFAULT_PORT)))
    parser.add_argument(
        "--vlm-backend",
        "--scorer",
        dest="scorer",
        default=os.getenv("REWARD_3D_SCORER") or os.getenv("VLM_BACKEND"),
        help="VLM backend used for GS/meta scoring: api/openrouter or local/qwenvl.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("REWARD_3D_MODEL_NAME", DEFAULT_MODEL_NAME),
        help="DA3 reconstruction model name or local path",
    )
    parser.add_argument(
        "--vlm-model",
        "--scoring-model",
        dest="scoring_model",
        default=os.getenv("REWARD_3D_SCORING_MODEL") or DEFAULT_SCORING_MODEL,
        help="VLM model name or local path used to score GS/meta renderings",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.getenv("REWARD_3D_NUM_WORKERS", "1")),
        help="Number of local DA3 worker processes / GPUs to use",
    )
    parser.add_argument(
        "--gpus",
        default=os.getenv("CUDA_VISIBLE_DEVICES"),
        help="Visible GPU ids, e.g. '0' or '0,1'.",
    )
    parser.add_argument("--lpips", dest="lpips", action="store_true")
    parser.add_argument("--no-lpips", dest="lpips", action="store_false")
    parser.set_defaults(
        lpips=os.getenv("REWARD_3D_USE_LPIPS", "1").strip().lower() not in {"0", "false", "no"}
    )
    args = parser.parse_args()
    args.scorer = resolve_vlm_backend(args.scorer)
    args.scoring_model = resolve_model_name(args.scoring_model, args.scorer)
    args.model_name = resolve_model_arg(args.model_name)

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    from reward_3d import MultiGPUReward3DManager

    global reward_3d_manager
    reward_3d_manager = MultiGPUReward3DManager(
        model_name=args.model_name,
        scorer_type=args.scorer,
        scoring_model_name=args.scoring_model,
        use_lpips=args.lpips,
        num_workers=args.num_workers,
    )
    reward_3d_manager.initialize()

    if not reward_3d_manager.processes:
        raise RuntimeError("3D reward backend failed to initialize. CUDA is required.")

    signal.signal(signal.SIGINT, shutdown_server)
    signal.signal(signal.SIGTERM, shutdown_server)

    server = ThreadingHTTPServer((args.host, args.port), Reward3DHandler)
    server.model_name = args.model_name
    server.scorer = args.scorer
    server.scoring_model = args.scoring_model
    server.num_workers = args.num_workers
    server.inference_lock = threading.Lock()
    print(f"Serving 3D reward backend on http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        shutdown_server()


if __name__ == "__main__":
    main()
