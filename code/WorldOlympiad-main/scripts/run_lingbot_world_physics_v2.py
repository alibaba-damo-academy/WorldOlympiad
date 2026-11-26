#!/usr/bin/env python3
"""Run released LingBot-World v2 (1.3B or 14B causal-fast) on native physics inputs.

Same contract as `run_lingbot_world_physics.py`, without the 1.3B pilot
evidence gate: the task id follows from the checkpoint that is passed in, and
the flow-matching shift follows from that task's config instead of a
hard-coded 5.0.
"""
import argparse, hashlib, json, logging, os, sys, time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def task_for_checkpoint(root):
    name = Path(root).name.lower()
    if "1.3b" in name or "1p3b" in name:
        return "i2v-1.3B"
    if "14b" in name or "a14b" in name:
        return "i2v-A14B"
    raise ValueError(f"cannot infer the Wan task from checkpoint directory {root!r}; pass --task")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--official-control", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    w = Path(args.workspace).resolve()
    upstream = Path(args.upstream_root).resolve()
    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    task = args.task or task_for_checkpoint(ckpt)
    plan_path = Path(args.plan).resolve()
    sys.path.insert(0, str(w / "code/WorldOlympiad-main"))
    sys.path.insert(0, str(upstream))
    from autoresearch.robotics_resource_lock import exclusive_gpu_job
    from autoresearch.lingbot_world_physics import native_dimensions, INPUT_VERSION
    plan = json.loads(plan_path.read_text())
    cases = plan["cases"]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    state = {"status": "preparing", "task": task, "checkpoint": str(ckpt),
             "runner_sha256": sha(__file__),
             "input_adapter_sha256": sha(w / "code/WorldOlympiad-main/autoresearch/lingbot_world_physics.py"),
             "plan_sha256": sha(plan_path), "base_seed": plan.get("base_seed", 42),
             "cases": {}, "numeric_physics_score": None}
    write(out / "manifest.json", state)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        with exclusive_gpu_job(w / "outputs", "0"):
            import torch, numpy as np, cv2
            from PIL import Image
            import wan
            from wan.configs import WAN_CONFIGS
            from wan.utils.utils import save_video
            from wan.utils.cam_utils import get_Ks_transformed
            cfg = WAN_CONFIGS[task]
            state["status"] = "loading_model"
            write(out / "manifest.json", state)
            started = time.time()
            pipe = wan.WanI2VCausal(config=cfg, checkpoint_dir=str(ckpt), device_id=0, rank=0,
                                    t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
                                    init_on_cpu=True, convert_model_dtype=True,
                                    local_attn_size=18, sink_size=6, infer_mode="causal_fast")
            torch.cuda.synchronize()
            state["model_load_seconds"] = time.time() - started
            state["status"] = "model_ready"
            write(out / "manifest.json", state)
            jobs = []
            if args.official_control:
                jobs.append({"case_id": "official_lake", "image": upstream / "examples/03/image.jpg",
                             "action_path": upstream / "examples/03",
                             "prompt": "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere.",
                             "frame_num": 77, "kind": "official_installation_control"})
            for item in cases:
                d = Path(item["input_dir"])
                receipt = json.loads((d / "input_contract.json").read_text())
                if receipt["schema_version"] != INPUT_VERSION:
                    raise ValueError("stale physics inputs: " + item["case_id"])
                for field in ("image", "poses", "intrinsics"):
                    if sha(d / receipt["conditioning"][field]) != receipt["conditioning"][field + "_sha256"]:
                        raise ValueError("conditioning file changed: " + field)
                jobs.append({"case_id": item["case_id"], "image": d / "image.png", "action_path": d,
                             "prompt": receipt["conditioning"]["prompt"],
                             "frame_num": receipt["sampling"]["frame_num"],
                             "kind": "native_render_physics", "input_contract": receipt})
            for job in jobs:
                name = job["case_id"]
                case = out / name
                case.mkdir(parents=True, exist_ok=True)
                if (case / "result.json").exists():
                    raise FileExistsError("use a fresh run directory")
                image = Image.open(job["image"]).convert("RGB")
                expected_wh = native_dimensions(*image.size)
                frames = job["frame_num"]
                if job["kind"] == "native_render_physics":
                    assert expected_wh == image.size, "native physics images must not be resized"
                    values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
                    unchanged = torch.nn.functional.interpolate(values[None], size=(image.height, image.width), mode="bicubic")
                    assert torch.equal(values, unchanged[0]), "unexpected same-size interpolation change"
                    c2w = np.load(job["action_path"] / "poses.npy")
                    assert np.array_equal(c2w, np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0))
                    k = torch.from_numpy(np.load(job["action_path"] / "intrinsics.npy"))
                    actual_k = get_Ks_transformed(k, 480, 832, image.height, image.width, image.height, image.width)[0]
                    expected_k = torch.tensor(job["input_contract"]["camera_alignment"]["expected_model_intrinsics"])
                    assert torch.allclose(actual_k, expected_k, atol=1e-4, rtol=0)
                parameters = {"task": task, "infer_mode": "causal_fast", "image": str(job["image"]),
                              "image_sha256": sha(job["image"]), "image_wh": list(image.size),
                              "action_path": str(job["action_path"]), "prompt": job["prompt"],
                              "prompt_sha256": hashlib.sha256(job["prompt"].encode()).hexdigest(),
                              "poses_sha256": sha(job["action_path"] / "poses.npy"),
                              "intrinsics_sha256": sha(job["action_path"] / "intrinsics.npy"),
                              "chunk_size": 4, "frame_num": frames, "max_area": 480 * 832,
                              "sample_shift": float(cfg.sample_shift), "timesteps_index": [0, 250, 500, 750],
                              "local_attn_size": 18, "sink_size": 6, "seed": state["base_seed"],
                              "output_wh": list(expected_wh), "fps": cfg.sample_fps}
                write(case / "request.json", parameters)
                state["status"] = "running"
                state["cases"][name] = {"status": "generating", "kind": job["kind"]}
                write(out / "manifest.json", state)
                logging.info("BEGIN_CASE %s prompt=%s", name, job["prompt"])
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.time()
                video = pipe.generate(job["prompt"], image, action_path=str(job["action_path"]),
                                      chunk_size=4, max_area=480 * 832, frame_num=frames,
                                      timesteps_index=[0, 250, 500, 750],
                                      shift=float(cfg.sample_shift), seed=state["base_seed"],
                                      offload_model=False)
                torch.cuda.synchronize()
                seconds = time.time() - t0
                if tuple(video.shape) != (3, frames, expected_wh[1], expected_wh[0]):
                    raise ValueError("unexpected native video shape")
                if not torch.isfinite(video).all().item():
                    raise ValueError("nonfinite generated video")
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                torch.save(video.detach().cpu(), case / "prediction_tensor.pt")
                save_video(tensor=video[None], save_file=str(case / "imagination.mp4"),
                           fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
                cap = cv2.VideoCapture(str(case / "imagination.mp4"))
                decoded = 0
                first = None
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (expected_wh[1], expected_wh[0]):
                        raise ValueError("encoder resized output")
                    if decoded == 0:
                        first = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if decoded in [0, frames // 4, frames // 2, 3 * frames // 4, frames - 1]:
                        cv2.imwrite(str(case / f"frame_{decoded:03d}.png"), frame)
                    decoded += 1
                cap.release()
                if decoded != frames or abs(actual_fps - cfg.sample_fps) > 1e-6:
                    raise ValueError("video decode/timing mismatch")
                result = {"case_id": name, "kind": job["kind"], "status": "complete", "task": task,
                          "frames": decoded, "fps": actual_fps, "width": expected_wh[0], "height": expected_wh[1],
                          "temporal_span_seconds": (decoded - 1) / actual_fps,
                          "generation_seconds": seconds, "peak_allocated_gpu_gib": peak,
                          "video_sha256": sha(case / "imagination.mp4"),
                          "request_sha256": sha(case / "request.json"),
                          "raw_tensor_sha256": sha(case / "prediction_tensor.pt"),
                          "finite": True, "full_decode_verified": True, "numeric_physics_score": None}
                if job["kind"] == "native_render_physics":
                    result["first_frame_mean_abs_error_0_255"] = float(
                        np.abs(first.astype(float) - np.asarray(image).astype(float)).mean())
                write(case / "result.json", result)
                state["cases"][name] = result
                write(out / "manifest.json", state)
                logging.info("COMPLETE_CASE %s generation_seconds=%.2f peak_gib=%.2f", name, seconds, peak)
                del video
                pipe.clear_text_cache()
                torch.cuda.empty_cache()
            state["status"] = "complete"
            write(out / "manifest.json", state)
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        write(out / "manifest.json", state)
        raise


if __name__ == "__main__":
    main()
