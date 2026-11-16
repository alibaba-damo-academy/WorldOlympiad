"""Versioned fixed-benchmark generation and scoring, independent of model adapters."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "code/WorldOlympiad-main"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORLD))
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
VIDEO_PROTOCOL = {"name": "reference_video_benchmark.v1",
                  "native_trajectory_cae": "not_applicable",
                  "reason": "video_only_protocol_has_no_native_trajectory_measurement"}


def identifier(value):
    if not isinstance(value, str) or not SAFE.fullmatch(value):
        raise ValueError("invalid identifier")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def bound_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"immutable run input changed: {path}")
    else:
        write(path, value)


def file_path(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("file path must be a nonempty string")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return path


def run_process(command, *, env, cwd, log, timeout):
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        write(log.with_suffix(".process.json"), {"pid": process.pid, "started": time.time()})
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if result:
            raise subprocess.CalledProcessError(result, command)


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_info(path):
    import imageio_ffmpeg
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(frames)
    finally:
        frames.close()
    fps = metadata["fps"]
    width, height = metadata["size"]
    decoded = subprocess.run(
        [ffmpeg(), "-v", "error", "-xerror", "-nostdin", "-i", str(path),
         "-map", "0:v:0", "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    counts = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines()
              if line.startswith("frame=")]
    count = counts[-1] if counts else 0
    if not math.isfinite(fps) or fps <= 0 or count < 1 or min(width, height) < 1:
        raise ValueError(f"invalid video: {path}")
    duration = metadata.get("duration") or count / fps
    return {"fps": fps, "frames": count, "width": width, "height": height,
            "duration_s": duration}


def dataset(path, selected=None):
    from PIL import Image
    path = path.resolve()
    data = read(path)
    if data.get("schema_version") != "wov2_benchmark.v1":
        raise ValueError("unsupported dataset schema")
    identifier(data["dataset_id"])
    if not isinstance(data.get("version"), str) or not data["version"]:
        raise ValueError("dataset version is required")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("dataset cases must be nonempty")
    seen, rows = set(), []
    for raw in data["cases"]:
        cid = identifier(raw["case_id"])
        if cid in seen:
            raise ValueError(f"duplicate case: {cid}")
        seen.add(cid)
        if raw.get("domain") not in ("real", "sim"):
            raise ValueError(f"{cid}: domain must be real or sim")
        if not isinstance(raw.get("instruction"), str) or not raw["instruction"].strip():
            raise ValueError(f"{cid}: instruction required")
        identifier(raw["reference_view"])
        if selected and cid not in selected:
            continue
        reference = file_path(path.parent, raw["reference_video"])
        info = video_info(reference)
        conditioning = raw.get("conditioning")
        if not isinstance(conditioning, dict):
            raise ValueError(f"{cid}: conditioning must be a camera/image map")
        images = {}
        for role, value in conditioning.items():
            identifier(role)
            image = file_path(path.parent, value)
            with Image.open(image) as img:
                img.verify()
            images[role] = str(image)
        captions = raw.get("caption_chunks", [{"start_s": 0, "end_s": info["duration_s"],
                                               "caption": raw["instruction"]}])
        if not isinstance(captions, list) or not captions:
            raise ValueError(f"{cid}: caption_chunks required")
        previous_end = 0
        for chunk in captions:
            start, end = chunk["start_s"], chunk["end_s"]
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not previous_end <= start < end <= info["duration_s"] + 0.001
                    or not isinstance(chunk.get("caption"), str) or not chunk["caption"].strip()):
                raise ValueError(f"{cid}: invalid caption interval")
            previous_end = end
        execution = dict(raw.get("execution", {}))
        if "reference_dir" in execution:
            p = Path(execution["reference_dir"])
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if not (p / "certificate.json").is_file():
                raise ValueError(f"{cid}: simulator reference is missing")
            if read(p / "certificate.json").get("success") is not True:
                raise ValueError(f"{cid}: simulator reference must have a passing certificate")
            execution["reference_dir"] = str(p)
            execution["reference_hashes"] = {
                name: sha(p / name) for name in ("certificate.json", "trajectory.json", "scene_inputs.json")
            }
        row = {"case_id": cid, "domain": raw["domain"], "instruction": raw["instruction"],
               "reference_view": raw["reference_view"], "reference_video": str(reference),
               "conditioning": images, "caption_chunks": captions, "video": info,
               "execution": execution, "hashes": {"reference": sha(reference),
                   **{role: sha(value) for role, value in images.items()}}}
        rows.append(row)
    if selected and set(selected) - seen:
        raise ValueError("unknown selected cases")
    return data, rows


def predictions(path, data, rows, model=None):
    path = path.resolve()
    doc = read(path)
    if (doc.get("schema_version") != "wov2_predictions.v1"
            or doc.get("dataset_id") != data["dataset_id"]
            or doc.get("dataset_version") != data["version"]):
        raise ValueError("submission does not match dataset version")
    identifier(doc["model"])
    if model and doc["model"] != model:
        raise ValueError("submission model changed")
    expected = {row["case_id"]: row for row in rows}
    all_cases = {row["case_id"] for row in data["cases"]}
    found, normalized = set(), {}
    for item in doc["predictions"]:
        cid = item["case_id"]
        if cid in found or cid not in all_cases:
            raise ValueError("duplicate or unknown prediction")
        found.add(cid)
        if cid not in expected:
            continue
        if item["view"] != expected[cid]["reference_view"]:
            raise ValueError(f"{cid}: prediction viewpoint mismatch")
        video = file_path(path.parent, item["video"])
        info = video_info(video)
        if (info["width"], info["height"]) != (expected[cid]["video"]["width"], expected[cid]["video"]["height"]):
            raise ValueError(f"{cid}: prediction dimensions must match reference")
        normalized[cid] = {"case_id": cid, "video": str(video), "view": item["view"],
                           "sha256": sha(video), "video_info": info}
    if set(expected) != set(normalized):
        raise ValueError("missing predictions for selected cases")
    return doc["model"], normalized


def prepare_case(row, prediction, case):
    case.mkdir(parents=True, exist_ok=True)
    for source, name in ((row["reference_video"], "reference.mp4"),
                         (prediction["video"], "generated.mp4")):
        dest = case / name
        if dest.is_symlink():
            raise ValueError("unsafe symlinked scoring input; use a fresh run ID")
        if not dest.exists():
            descriptor, temporary = tempfile.mkstemp(dir=case, prefix="." + name + ".copy-")
            os.close(descriptor)
            shutil.copyfile(source, temporary)
            os.replace(temporary, dest)
        if sha(dest) != sha(source):
            raise ValueError("prepared video input changed")
    source_duration = row["video"]["duration_s"]
    generated_duration = prediction["video_info"]["duration_s"]
    prompts, intervals = [], []
    for index, chunk in enumerate(row["caption_chunks"]):
        prompts.append({"chunk_index": index,
                        "interval": f'[{chunk["start_s"]:.4f}, {chunk["end_s"]:.4f})',
                        "caption": chunk["caption"], "action": chunk["caption"]})
        intervals.append({"chunk_index": index,
                          "generated_start_sec": chunk["start_s"] * generated_duration / source_duration,
                          "generated_end_sec": chunk["end_s"] * generated_duration / source_duration,
                          "source_interval": prompts[-1]["interval"]})
    bound_json(case / "prompt.json", prompts)
    bound_json(case / "generated_chunk_timestamps.json", {"chunks": intervals})
    bound_json(case / "worldolympiad_case.json", {
        "case_id": row["case_id"], "generated_video": "generated.mp4",
        "reference_video": "reference.mp4",
        "tracks": {"continuous_action": "not_applicable"},
        "protocol": VIDEO_PROTOCOL["name"], "domain": row["domain"],
        "scoring_applicability": VIDEO_PROTOCOL})


def score_complete(score):
    tracks = score.get("track_results", {})
    if not isinstance(tracks, dict) or not tracks:
        return False
    for name in ("physical", "geometry", "interaction"):
        track = tracks.get(name, {})
        value = track.get("score")
        if (track.get("status") != "scored" or type(value) not in (int, float)
                or not math.isfinite(value)):
            return False
    overall = score.get("worldolympiad2_score")
    return type(overall) in (int, float) and math.isfinite(overall)


def validate_video_evidence(raw, case):
    """Reject silent judge failures, following the recovered paper worker gates."""
    def number(value, low, high):
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    physical, geometry = raw.get("physical") or {}, raw.get("three_d") or {}
    interaction, clip = raw.get("interaction") or {}, raw.get("clip_interaction") or {}
    rules = physical.get("results", [])
    if (physical.get("errored_question_count", 0) != 0 or not rules
            or any(type(rule.get("related")) is not bool or rule.get("error")
                   or (rule["related"] and type(rule.get("compliant")) is not bool) for rule in rules)):
        raise RuntimeError("physical judge evidence is incomplete")
    for key in ("gs_video_path", "meta_view_path", "trajectory_comparison_path"):
        value = geometry.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError("geometry evidence artifact is missing")
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError("geometry evidence artifact is absent or empty")
    if not all(number(geometry.get(key), 0, 1) for key in ("gs_score", "meta_score", "camera_motion_score")):
        raise RuntimeError("geometry judge evidence is incomplete")
    chunks, transitions = interaction.get("chunks", []), interaction.get("transitions", [])
    expected = len(read(case / "prompt.json"))
    evidence = list(chunks) + list(transitions) + [interaction.get("global") or {}]
    if (len(chunks) != expected or len(transitions) != max(0, expected - 1)
            or any(not isinstance(row.get("reason"), str) or not row["reason"].strip()
                   or row.get("error") for row in evidence)):
        raise RuntimeError("interaction judge evidence is incomplete")
    summary = interaction.get("summary") or {}
    for key in ("chunk_mean", "transition_mean", "global_score"):
        if key == "transition_mean" and expected == 1:
            if summary.get(key) is not None:
                raise RuntimeError("single-chunk transition evidence is inconsistent")
        elif not number(summary.get(key), 0, 5):
            raise RuntimeError("interaction summary is incomplete")
    summary = clip.get("summary") or {}
    if (summary.get("num_scored_chunks") != expected or summary.get("num_chunks") != expected
            or not number(summary.get("semantic_adherence"), -math.inf, math.inf)):
        raise RuntimeError("CLIP interaction coverage is incomplete")


def video_score(raw, contract):
    if contract != VIDEO_PROTOCOL:
        raise ValueError("video aggregation requires the predeclared video-only applicability")
    from autoresearch.score_hierarchy import from_judge_summaries
    result = from_judge_summaries(raw.get("physical"), raw.get("interaction"), raw.get("three_d"),
                                 {"status": "not_applicable", "score": None, "reason": contract["reason"]})
    result.update(schema_version="wov2_video_score.v1", protocol=contract)
    return result


def settle_scoring_gpus(indices, *, timeout_s=30):
    """Wait for utilization left by this scoring stage to drain before the next gate."""
    from wo.agent.config import all_gpu_info

    deadline = time.monotonic() + timeout_s
    announced = False
    while True:
        inventory = {card["index"]: card for card in all_gpu_info()}
        cards = [inventory.get(index) for index in indices]
        if any(card is None or not card.get("process_query_ok", True)
               or card.get("compute_process_count", 0) for card in cards):
            return  # The next case's fresh preflight will reject a busy or unqueryable card.
        if all(card.get("utilization", 0) <= 5 for card in cards):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return  # Preserve the strict next-stage preflight after a bounded wait.
        if not announced:
            print("Waiting for previous scoring GPU utilization to settle", flush=True)
            announced = True
        time.sleep(min(1, remaining))


def scoring(rows, items, run, model):
    from scripts.preflight import queue_check
    from wo.agent.config import resolve_score_gpus
    from autoresearch.robotics_resource_lock import exclusive_gpu_job
    from scripts.score_robotics_cases import run as score_cases
    from wo.agent.score_stage import valid_receipt

    result = {"schema_version": "wov2_benchmark_report.v1", "model": model,
              "protocol": "reference_video_benchmark.v1", "complete": False,
              "native_task_success": "not_applicable", "native_trajectory_cae": "not_applicable",
              "cases": []}
    identity = read(run / "inputs.json")
    contract = identity.get("scoring_protocol")
    if contract != VIDEO_PROTOCOL:
        raise ValueError("run must bind video-only applicability before scoring")
    result["scoring_protocol"] = contract
    result.update(dataset_id=identity["dataset_id"], dataset_version=identity["dataset_version"],
                  manifest_sha256=identity.get("manifest_sha256"), execution_mode=identity.get("mode"),
                  implementation_sha256=identity.get("implementation_sha256"))
    for position, row in enumerate(rows):
        selected_gpus = None
        case = run / "cases" / row["case_id"] / "score"
        prepare_case(row, items[row["case_id"]], case)
        status = {"case_id": row["case_id"], "domain": row["domain"], "status": "pending",
                  "reference_sha256": row["hashes"]["reference"],
                  "prediction_sha256": items[row["case_id"]]["sha256"]}
        result["cases"].append(status)
        write(run / "report.json", result)
        try:
            if not valid_receipt(case):
                queue_check(None, False)
                if "score_profile" in identity:
                    profile = identity["score_profile"]
                else:  # Existing immutable runs retain their original configured scorer.
                    from wo.agent.config import load_profile
                    profile = load_profile()
                gpus = resolve_score_gpus(profile)
                gpu, compute = gpus["gpu"], gpus["compute_gpu"]
                selected_gpus = {gpu, compute}
                from contextlib import ExitStack
                with ExitStack() as locks:
                    for index in sorted({gpu, compute}):
                        locks.enter_context(exclusive_gpu_job(ROOT / "outputs", str(index)))
                    score_cases([case], ROOT, gpu, gpus["mode"], compute)
            raw = read(case / "score.json")
            if not valid_receipt(case):
                raise RuntimeError("raw score receipt does not match the scored inputs")
            validate_video_evidence(raw, case)
            value = video_score(raw, contract)
            if not score_complete(value):
                raise RuntimeError("three video-track measurements are incomplete")
            value.update(raw_score=str(case / "score.json"), raw_score_sha256=sha(case / "score.json"),
                         raw_receipt_sha256=sha(case / "three_track_receipt.json"))
            write(case / "benchmark_score.json", value)
            status.update(status="complete", score=str(case / "benchmark_score.json"),
                          raw_score=str(case / "score.json"), tracks=value["track_results"],
                          overall=value.get("worldolympiad2_score"))
            if selected_gpus and position + 1 < len(rows):
                settle_scoring_gpus(selected_gpus)
        except Exception:
            status["status"] = "failed"
            completed_ids = {case["case_id"] for case in result["cases"]}
            result["cases"].extend({"case_id": item["case_id"], "domain": item["domain"],
                                     "status": "pending"} for item in rows
                                    if item["case_id"] not in completed_ids)
            write(run / "report.json", result)
            raise
    groups = {}
    for domain in ("real", "sim"):
        values = [item["overall"] for item in result["cases"]
                  if item["domain"] == domain and item["status"] == "complete"]
        groups[domain] = {"count": len(values),
                          "mean_overall": sum(values) / len(values) if values else None}
    result["summary"] = groups
    result["complete"] = (len(result["cases"]) == len(rows)
                          and all(row["status"] == "complete" for row in result["cases"]))
    write(run / "report.json", result)


def generate(rows, args, run):
    from scripts.preflight import queue_check
    from wo.agent.config import load_profile, apply_environment, all_gpu_info, _gpu_available
    from autoresearch.robotics_resource_lock import exclusive_gpu_job
    from scripts.benchmark_adapter import builtin
    from contextlib import ExitStack

    adapter = read(args.adapter) if args.adapter else None
    if not adapter and (args.model != "lingbot_va" and args.mode != "sim_rollout"):
        raise ValueError("non-LingBot i2av requires a command adapter")
    if not adapter and args.mode == "sim_rollout":
        from wo.agent.config import ADAPTERS
        if args.model not in ADAPTERS:
            raise ValueError("model requires a registered native adapter or a command adapter")
        if any("reference_dir" not in row["execution"] for row in rows):
            raise ValueError("registered native sim_rollout requires execution.reference_dir")
        if any(row["reference_view"] != "exterior_left" for row in rows):
            raise ValueError("registered native sim_rollout requires exterior_left reference view")
    if not adapter and args.mode == "i2av":
        required = {"head_camera", "left_camera", "right_camera"}
        if any(required - row["conditioning"].keys() for row in rows):
            raise ValueError("LingBot i2av requires three actual conditioning camera images")
    model = identifier(adapter["model"] if adapter else args.model)
    if adapter and (adapter.get("schema_version") != "wov2_adapter.v1"
                    or not isinstance(adapter.get("modes"), list)
                    or args.mode not in adapter["modes"]):
        raise ValueError("adapter does not support the requested mode")
    if adapter and (type(adapter.get("minimum_free_mib")) is not int
                    or adapter["minimum_free_mib"] <= 0
                    or type(adapter.get("gpu_count", 1)) is not int
                    or adapter.get("gpu_count", 1) <= 0
                    or not isinstance(adapter.get("command"), list)
                    or not adapter["command"] or not all(isinstance(a, str) for a in adapter["command"])
                    or not {"{input}", "{output}"}.issubset(adapter["command"])):
        raise ValueError("invalid command adapter")
    results = []
    for row in rows:
        if args.mode == "sim_rollout" and row["domain"] != "sim":
            raise ValueError("execution mode does not match reference domain")
        output = run / "cases" / row["case_id"]
        output.mkdir(parents=True, exist_ok=True)
        receipt = output / "generation.json"
        if receipt.exists():
            item = read(receipt)
            if sha(item["video"]) != item["sha256"]:
                raise ValueError("completed generated video changed")
            results.append(item)
            continue
        queue_check(None, False)
        env = dict(os.environ)
        env.setdefault("WO_FFMPEG", ffmpeg())
        if adapter:
            cards = [c for c in all_gpu_info() if _gpu_available(c, adapter["minimum_free_mib"])]
            if len(cards) < adapter.get("gpu_count", 1):
                raise RuntimeError("not enough idle allowlisted GPUs meet adapter memory requirements")
            selected = [c["index"] for c in sorted(cards, key=lambda c: c["index"])
                        [:adapter.get("gpu_count", 1)]]
            gpu = selected[0]
        else:
            profile = load_profile(required=True)
            env["WO_EVAL_CONFIG"] = profile["_config_path"]
            if args.mode == "i2av":
                from scripts.benchmark_adapter import video_gpus
                selected = video_gpus(profile, all_gpu_info())
                apply_environment(env, profile, phase="worker")
                env["WO_TESTED_MODEL_GPU"] = str(selected[0])
                env["WO_TESTED_MODEL_GPUS"] = ",".join(map(str, selected))
            else:
                apply_environment(env, profile, phase="episode")
                if "WO_SIMULATOR_GPU" not in env:
                    raise RuntimeError("registered native rollout needs an available simulator GPU")
                selected = [int(x) for x in env["WO_TESTED_MODEL_GPUS"].split(",")]
                selected.append(int(env["WO_SIMULATOR_GPU"]))
            gpu = selected[0]
        attempt = output / f"attempt_{time.time_ns()}"
        attempt.mkdir()
        payload = {"case_id": row["case_id"], "instruction": row["instruction"],
                   "mode": args.mode, "conditioning": row["conditioning"],
                   "output_view": row["reference_view"],
                   "output_size_wh": [row["video"]["width"], row["video"]["height"]],
                   "chunks": args.chunks}
        if args.mode == "sim_rollout":
            payload["execution"] = row["execution"]
            if not adapter:
                reference = Path(payload["execution"]["reference_dir"])
                if sha(reference / "replay/exterior_left.mp4") != row["hashes"]["reference"]:
                    raise ValueError("sim reference video is not bound to the execution reference")
        write(attempt / "input.json", payload)
        with ExitStack() as locks:
            for index in sorted(set(selected)):
                locks.enter_context(exclusive_gpu_job(ROOT / "outputs", str(index)))
            if adapter:
                replacements = {"{input}": str(attempt / "input.json"), "{output}": str(attempt),
                                "{gpu}": str(gpu), "{gpus}": ",".join(map(str, selected))}
                command = [replacements.get(arg, arg) for arg in adapter["command"]]
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
                run_process(command, env=env, cwd=args.adapter.resolve().parent,
                            log=attempt / "generation.log", timeout=args.timeout)
                generated = read(attempt / "prediction.json")
                video = file_path(attempt, generated["video"])
                view = generated["view"]
            else:
                video, view = builtin(model, args.mode, payload, attempt, env, args.timeout)
        info = video_info(video)
        if (info["width"], info["height"]) != (row["video"]["width"], row["video"]["height"]):
            raise ValueError("generated video dimensions must match the evaluation reference")
        if view != row["reference_view"]:
            raise ValueError("generated viewpoint mismatch")
        item = {"case_id": row["case_id"], "video": str(video), "view": view,
                "sha256": sha(video), "video_info": info}
        write(receipt, item)
        results.append(item)
    return model, results


def implementation_contract():
    return {name: sha(ROOT / name) for name in (
        "scripts/benchmark.py", "scripts/benchmark_adapter.py",
        "code/WorldOlympiad-main/scripts/score_robotics_cases.py",
        "code/WorldOlympiad-main/scripts/score_video_physical_3d.py",
        "code/WorldOlympiad-main/physical/sam_process.py",
        "sources.lock.json", "runtime.lock.json")}


def model_contract(args):
    """Bind model configuration so a new checkpoint cannot reuse old predictions."""
    if args.action != "run" or args.adapter:
        return None
    from wo.agent.config import load_profile
    profile = load_profile(required=True)
    model = dict(profile["tested_model"])
    if model["adapter"] != args.model:
        raise ValueError("configured model adapter differs from requested built-in model")
    for field in ("gpu", "policy_gpus", "simulator_gpu"):
        model.pop(field, None)
    return {"workspace_root": profile["workspace_root"], "tested_model": model,
            "implementation_sha256": implementation_contract()}


def score_profile(args):
    """Bind scorer resources without requiring a tested-model profile for user adapters."""
    settings = {}
    if args.action == "run" and not args.adapter:
        from wo.agent.config import load_profile
        settings = dict(load_profile(required=True).get("scoring", {}))
    for argument, field in (("score_mode", "mode"), ("score_gpu", "gpu"),
                            ("score_compute_gpu", "compute_gpu")):
        value = getattr(args, argument)
        if value is not None:
            settings[field] = value
    return {"scoring": settings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "score"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--run")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--mode", choices=("i2av", "sim_rollout"))
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--score-mode", choices=("auto", "full", "split"))
    parser.add_argument("--score-gpu", type=int)
    parser.add_argument("--score-compute-gpu", type=int)
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args(argv)
    data, rows = dataset(args.manifest, args.case)
    if args.action == "check":
        print(json.dumps({"dataset_id": data["dataset_id"], "version": data["version"],
                          "cases": len(rows), "status": "inputs_valid"}))
        return
    if not args.run or args.timeout < 1 or args.chunks < 1:
        parser.error("run, positive timeout and positive chunks required")
    if args.action == "run" and (not args.mode or bool(args.adapter) == bool(args.model)):
        parser.error("choose mode and exactly one of model or adapter")
    if args.action == "score" and not args.predictions:
        parser.error("predictions required")
    root = ROOT / "outputs/benchmark"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = root / identifier(args.run)
        run.mkdir(exist_ok=True)
        identity = {"manifest_sha256": sha(args.manifest), "rows": rows,
                    "action": args.action, "model": args.model, "mode": args.mode,
                    "dataset_id": data["dataset_id"], "dataset_version": data["version"],
                    "adapter_sha256": sha(args.adapter) if args.adapter else None,
                    "submission_sha256": sha(args.predictions) if args.predictions else None,
                    "chunks": args.chunks, "model_contract": model_contract(args),
                    "score_profile": score_profile(args),
                    "implementation_sha256": implementation_contract(), "scoring_protocol": VIDEO_PROTOCOL}
        bound_json(run / "inputs.json", identity)
        if args.action == "run":
            model, items = generate(rows, args, run)
            submission = {"schema_version": "wov2_predictions.v1", "dataset_id": data["dataset_id"],
                          "dataset_version": data["version"], "model": model,
                          "predictions": items}
            write(run / "predictions.json", submission)
            if args.generation_only:
                print(run / "predictions.json")
                return
            items = {item["case_id"]: item for item in items}
        else:
            model, items = predictions(args.predictions, data, rows)
        scoring(rows, items, run, model)
        print(run / "report.json")


if __name__ == "__main__":
    main()
