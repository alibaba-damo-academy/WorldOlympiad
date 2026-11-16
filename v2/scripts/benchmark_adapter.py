"""Built-in model adapters for the fixed benchmark protocol."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts.benchmark import ROOT, WORLD, ffmpeg, video_info, run_process, sha

CAMERAS = {"head_camera": "observation.images.cam_high",
           "left_camera": "observation.images.cam_left_wrist",
           "right_camera": "observation.images.cam_right_wrist"}
CROPS = {"head_camera": "crop=320:256:0:128",
         "left_camera": "crop=160:128:0:0",
         "right_camera": "crop=160:128:160:0"}


def video_gpus(profile, inventory):
    """Reserve policy cards without requiring an unused simulator or scoring card."""
    from wo.agent.config import resolve_gpu, _gpu_available
    first = resolve_gpu(profile, inventory=inventory)
    info = next(card for card in inventory if card["index"] == first)
    small = info["total_mib"] <= 40 * 1024
    requested = profile["tested_model"].get("policy_gpus", "auto")
    eligible = {card["index"] for card in inventory
                if _gpu_available(card, 30 * 1024 if small else 48 * 1024)}
    if isinstance(requested, list):
        cards = requested
    elif small:
        others = sorted(eligible - {first})
        if not others:
            raise RuntimeError("LingBot i2av needs two idle policy GPUs on small-memory cards")
        cards = [first, others[0]]
    else:
        cards = [first]
    if (not cards or cards[0] != first or len(set(cards)) != len(cards)
            or not set(cards).issubset(eligible) or (small and len(cards) < 2)):
        raise ValueError("configured video policy GPUs are unavailable or incompatible")
    return list(cards)


def builtin(model, mode, payload, attempt, env, timeout):
    log = attempt / "generation.log"
    env = dict(env, PYTHONPATH=str(WORLD), WO_WORKSPACE=str(ROOT),
               WO_EVAL_CONFIG=env.get("WO_EVAL_CONFIG", str(ROOT / "config/two-interface.json")))
    if model != "lingbot_va":
        if mode != "sim_rollout":
            raise ValueError("registered native adapters require sim_rollout; i2av needs a command adapter")
        from wo.agent.stages import episode_adapter
        from wo.agent.episode_stage import verify_evaluation_view
        reference = Path(payload["execution"]["reference_dir"]).resolve()
        if reference.parent.name != "references":
            raise ValueError("registered native adapter requires a reference in its WOV2 run")
        run_root = reference.parent.parent
        if not (run_root / "tasks" / f"{reference.name}.json").is_file():
            raise ValueError("registered native adapter requires its original WOV2 TaskSpec")
        if payload["output_view"] != "exterior_left":
            raise ValueError("registered native export supports exterior_left evaluation view")
        window = payload["execution"].get("window_s")
        if window is not None and (type(window) not in (int, float) or window <= 0):
            raise ValueError("positive simulator window_s required")
        native = attempt / "native"
        python = str(ROOT / "envs/autoresearch/bin/python")
        command = [python, "-m", "wo.agent.episode_stage", "--run-root", str(run_root),
                   "--ref", str(reference), "--out", str(native), "--render-gpu-id",
                   env["WO_SIMULATOR_GPU"], "--", python, "-m",
                   episode_adapter({"model": model}), "--ref", str(reference), "--out", str(native),
                   "--gpu", env["WO_TESTED_MODEL_GPU"], "--seed", "0",
                   "--port", str(28000 + os.getpid() % 1000)]
        if window is not None:
            command += ["--window-s", str(window)]
        run_process(command, cwd=WORLD, env=env, timeout=timeout, log=log)
        verify_evaluation_view(native, reference)
        return native / "native_replay/exterior_left.mp4", "exterior_left"
    if mode == "sim_rollout":
        reference = Path(payload["execution"]["reference_dir"])
        if payload["output_view"] != "exterior_left":
            raise ValueError("native LingBot export supports exterior_left evaluation view")
        native = attempt / "native"
        command = [str(ROOT / "envs/autoresearch/bin/python"), "-m", "wo.models.lingbot_va",
                   "--ref", str(reference), "--out", str(native),
                   "--gpu", env["WO_TESTED_MODEL_GPU"], "--simulator-gpu", env["WO_SIMULATOR_GPU"],
                   "--policy-gpus", *env["WO_TESTED_MODEL_GPUS"].split(",")]
        window = payload["execution"].get("window_s")
        if window is not None:
            if type(window) not in (int, float) or window <= 0:
                raise ValueError("positive simulator window_s required")
            command += ["--window-s", str(window)]
        # Native replay export uses the evaluation camera contract.
        import json
        certificate = json.loads((reference / "certificate.json").read_text())
        check = [str(ROOT / "envs/autoresearch/bin/python"), "-m", "wo.models.scene_check",
                 "--ref", str(reference), "--out", str(attempt / "scene_check"),
                 "--task", certificate["task"]]
        run_process(check, cwd=WORLD,
                    env=dict(env, WO_TESTED_MODEL_GPU=env["WO_SIMULATOR_GPU"]),
                    timeout=timeout, log=attempt / "scene_check.log")
        readiness = json.loads((attempt / "scene_check/scene_ready.json").read_text())
        if readiness.get("scene_ready") is not True:
            raise RuntimeError("simulator scene check did not pass; refusing model rollout")
        run_process(command, cwd=WORLD, env=env, timeout=timeout, log=log)
        from wo.agent.episode_stage import ensure_evaluation_view
        if payload["output_view"] != "exterior_left":
            raise ValueError("native LingBot export supports exterior_left evaluation view")
        ensure_evaluation_view(native, reference, int(env["WO_SIMULATOR_GPU"]))
        return native / "native_replay/exterior_left.mp4", "exterior_left"
    missing = set(CAMERAS) - payload["conditioning"].keys()
    if missing or payload["output_view"] not in CROPS:
        raise ValueError("LingBot i2av requires actual head, left and right conditioning cameras")
    repo = Path(env["WO_LINGBOT_REPO"])
    checkpoint = Path(env["WO_TESTED_MODEL_CHECKPOINT"])
    if not checkpoint.is_dir() or not (repo / "wan_va/wan_va_server.py").is_file():
        raise ValueError("LingBot runtime or configured checkpoint is missing")
    images = attempt / "initial_images"
    images.mkdir()
    from PIL import Image
    for role, upstream in CAMERAS.items():
        with Image.open(payload["conditioning"][role]) as image:
            image.convert("RGB").save(images / (upstream + ".png"))
    save = attempt / "model_output"
    save.mkdir()
    entry = attempt / "lingbot_i2av_entry.py"
    entry.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo / 'wan_va')!r})\n"
        "from configs import VA_CONFIGS\n"
        "import hashlib\n"
        "import wan_va_server as backend\n"
        "class BenchmarkVA(backend.VA_Server):\n"
        "    @property\n"
        "    def exp_name(self):\n"
        "        return self._benchmark_directory\n"
        "    @exp_name.setter\n"
        "    def exp_name(self, value):\n"
        "        self._benchmark_directory = hashlib.sha256(value.encode()).hexdigest()[:24]\n"
        "backend.VA_Server = BenchmarkVA\n"
        "config = VA_CONFIGS['robotwin_i2av']\n"
        f"config.wan22_pretrained_model_name_or_path = {str(checkpoint)!r}\n"
        f"config.input_img_path = {str(images)!r}\n"
        f"config.prompt = {payload['instruction']!r}\n"
        f"config.num_chunks_to_infer = {payload['chunks']!r}\n"
        "try:\n"
        f"    backend.run(type('Args', (), {{'config_name': 'robotwin_i2av', 'port': None, 'save_root': {str(save)!r}}})())\n"
        "finally:\n"
        "    import torch.distributed as distributed\n"
        "    if distributed.is_initialized():\n"
        "        distributed.destroy_process_group()\n"
    )
    # torchrun owns an ephemeral rendezvous instead of a shared hard-coded port.
    gpus = env["WO_TESTED_MODEL_GPUS"].split(",")
    env = dict(env, CUDA_VISIBLE_DEVICES=",".join(gpus), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    command = [env.get("WO_LINGBOT_PYTHON", str(ROOT / "envs/lingbot/bin/python")),
               "-m", "torch.distributed.run",
               "--standalone", "--nproc_per_node", str(len(gpus)), str(entry)]
    run_process(command, cwd=repo, env=env, timeout=timeout, log=log)
    videos = list(save.rglob("demo.mp4"))
    if len(videos) != 1:
        raise RuntimeError("LingBot did not produce exactly one native demo.mp4")
    info = video_info(videos[0])
    if (info["width"], info["height"]) != (320, 384):
        raise ValueError("unexpected LingBot composite shape; refusing an unverified crop")
    output = attempt / "generated.mp4"
    width, height = payload["output_size_wh"]
    pixel_format = "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p"
    subprocess.run([ffmpeg(), "-v", "error", "-nostdin", "-i", str(videos[0]),
                    "-vf", CROPS[payload["output_view"]] + f",scale={width}:{height}",
                    "-c:v", "libx264", "-pix_fmt", pixel_format, str(output)], check=True)
    return output, payload["output_view"]
