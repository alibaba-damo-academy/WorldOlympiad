import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import benchmark as B


def raw_video_measurements(case):
    artifacts = {}
    for key, name in (("gs_video_path", "geometry.mp4"), ("meta_view_path", "meta.png"),
                      ("trajectory_comparison_path", "comparison.png")):
        target = case / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture evidence artifact")
        artifacts[key] = str(target)
    raw = {"physical": {"score": 0.5, "errored_question_count": 0, "results": [
        {"related": True, "compliant": True}, {"related": True, "compliant": False}]},
        "three_d": {"final_score_normalized": 0.5, "gs_score": 0.5, "meta_score": 0.5,
                    "camera_motion_score": 0.5, **artifacts},
        "interaction": {"score": 0.5, "summary": {"chunk_mean": 2.5, "transition_mean": None,
                                                   "global_score": 2.5},
                        "chunks": [{"reason": "fixture chunk evidence"}], "transitions": [],
                        "global": {"reason": "fixture global evidence"}},
        "clip_interaction": {"summary": {"num_chunks": 1, "num_scored_chunks": 1,
                                            "semantic_adherence": 0.4}},
        "continuous_action": {"status": "not_applicable", "reason": "no_cae_manifest", "score": None}}
    from autoresearch.score_hierarchy import from_judge_summaries
    raw.update(from_judge_summaries(raw["physical"], raw["interaction"], raw["three_d"], raw["continuous_action"]))
    return raw


@pytest.fixture
def sample(tmp_path):
    import imageio_ffmpeg
    import subprocess
    image = tmp_path / "initial.png"
    Image.new("RGB", (32, 32), "red").save(image)
    video = tmp_path / "reference.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-loop", "1",
                    "-i", str(image), "-t", "0.4", "-r", "10",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    data = {"schema_version": "wov2_benchmark.v1", "dataset_id": "fixture", "version": "1",
            "cases": [{"case_id": "a", "domain": "real", "instruction": "Move the object.",
                       "reference_video": "reference.mp4", "reference_view": "head_camera",
                       "conditioning": {"head_camera": "initial.png"}}]}
    manifest = tmp_path / "dataset.json"
    B.write(manifest, data)
    return manifest, data, tmp_path


def test_real_input_decodes_and_hashes(sample):
    manifest, _, _ = sample
    data, rows = B.dataset(manifest)
    assert len(rows) == 1
    assert rows[0]["video"]["frames"] == 4
    assert rows[0]["hashes"]["reference"]


def test_file_backed_initial_state_is_bound_and_passed_to_adapter(sample, monkeypatch):
    from contextlib import nullcontext
    from scripts import preflight
    from wo.agent import config
    from autoresearch import robotics_resource_lock

    manifest, data, root = sample
    state_path = root / "state.json"
    state = {"joint_position": [0.1] * 7, "gripper_position": 0.25}
    B.write(state_path, state)
    data["cases"][0]["conditioning_state"] = "state.json"
    B.write(manifest, data)
    _, rows = B.dataset(manifest)
    assert rows[0]["hashes"]["conditioning_state"] == B.sha(state_path)
    assert rows[0]["conditioning_state"] == state

    monkeypatch.setattr(preflight, "queue_check", lambda *_: None)
    monkeypatch.setattr(config, "all_gpu_info", lambda: [
        {"index": 3, "free_mib": 100, "utilization": 0,
         "compute_app_pids": [], "process_query_ok": True}])
    monkeypatch.setattr(config, "_gpu_available", lambda *a: True)
    monkeypatch.setattr(robotics_resource_lock, "exclusive_gpu_job", lambda *a: nullcontext())
    monkeypatch.setattr(B, "ROOT", root)
    adapter = root / "adapter.json"
    B.write(adapter, {"schema_version": "wov2_adapter.v1", "model": "state_model",
                     "modes": ["i2av"], "minimum_free_mib": 1,
                     "command": ["adapter", "{input}", "{output}"]})

    def execute(command, **kwargs):
        payload = B.read(command[1])
        assert payload["conditioning_state"] == state
        assert "reference_video" not in payload
        B.write(Path(command[2]) / "prediction.json",
                {"video": str(root / "reference.mp4"), "view": "head_camera"})

    monkeypatch.setattr(B, "run_process", execute)
    B.generate(rows, SimpleNamespace(adapter=adapter, mode="i2av", chunks=1,
                                     timeout=10), root / "run")
    state["gripper_position"] = 0.5
    B.write(state_path, state)
    _, changed = B.dataset(manifest)
    assert changed[0]["hashes"]["conditioning_state"] != rows[0]["hashes"]["conditioning_state"]
    state_path.write_text('{"joint_position": [NaN]}')
    with pytest.raises(ValueError, match="finite numeric"):
        B.dataset(manifest)


def test_reject_invalid_cohort_and_intervals(sample):
    manifest, data, _ = sample
    data["cases"].append(dict(data["cases"][0]))
    B.write(manifest, data)
    with pytest.raises(ValueError, match="duplicate"):
        B.dataset(manifest)
    data["cases"].pop()
    data["cases"][0]["caption_chunks"] = [{"start_s": 0, "end_s": 100, "caption": "bad"}]
    B.write(manifest, data)
    with pytest.raises(ValueError, match="interval"):
        B.dataset(manifest)


def test_submission_requires_matching_ids_views_and_version(sample):
    manifest, data, root = sample
    _, rows = B.dataset(manifest)
    doc = {"schema_version": "wov2_predictions.v1", "dataset_id": "fixture",
           "dataset_version": "1", "model": "other_model",
           "predictions": [{"case_id": "a", "view": "head_camera", "video": "reference.mp4"}]}
    submission = root / "predictions.json"
    B.write(submission, doc)
    assert B.predictions(submission, data, rows)[0] == "other_model"
    doc["predictions"][0]["view"] = "wrist"
    B.write(submission, doc)
    with pytest.raises(ValueError, match="viewpoint"):
        B.predictions(submission, data, rows)
    doc["predictions"] = []
    B.write(submission, doc)
    with pytest.raises(ValueError, match="missing"):
        B.predictions(submission, data, rows)


def test_changed_completed_inputs_rejected(tmp_path):
    path = tmp_path / "inputs.json"
    B.bound_json(path, {"input": 1})
    B.bound_json(path, {"input": 1})
    with pytest.raises(ValueError, match="changed"):
        B.bound_json(path, {"input": 2})


def test_adapter_gets_conditioning_only_and_reuses_result(sample, monkeypatch):
    from contextlib import nullcontext
    from scripts import preflight
    from wo.agent import config
    from autoresearch import robotics_resource_lock
    manifest, _, root = sample
    _, rows = B.dataset(manifest)
    monkeypatch.setattr(preflight, "queue_check", lambda *_: None)
    monkeypatch.setattr(config, "all_gpu_info",
                        lambda: [{"index": 3, "free_mib": 100, "utilization": 0,
                                  "compute_app_pids": [], "process_query_ok": True}])
    monkeypatch.setattr(config, "_gpu_available", lambda *a: True)
    monkeypatch.setattr(robotics_resource_lock, "exclusive_gpu_job", lambda *a: nullcontext())
    monkeypatch.setattr(B, "ROOT", root)
    adapter = root / "adapter.json"
    B.write(adapter, {"schema_version": "wov2_adapter.v1", "model": "fixture_model",
                     "modes": ["i2av"], "minimum_free_mib": 1,
                     "command": ["test-adapter", "{input}", "{output}"]})
    calls = []
    def execute(command, **kwargs):
        payload = B.read(command[1])
        assert "reference_video" not in payload
        assert "caption_chunks" not in payload
        assert "execution" not in payload
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        B.write(Path(command[2]) / "prediction.json",
                {"video": str(root / "reference.mp4"), "view": "head_camera"})
        calls.append(command)
    monkeypatch.setattr(B, "run_process", execute)
    args = SimpleNamespace(adapter=adapter, mode="i2av", chunks=1, timeout=10)
    run = root / "run"
    B.generate(rows, args, run)
    B.generate(rows, args, run)
    assert len(calls) == 1


def test_external_adapter_reserves_declared_gpu_count(sample, monkeypatch):
    from contextlib import nullcontext
    from scripts import preflight
    from wo.agent import config
    from autoresearch import robotics_resource_lock
    manifest, _, root = sample
    _, rows = B.dataset(manifest)
    monkeypatch.setattr(preflight, "queue_check", lambda *_: None)
    monkeypatch.setattr(config, "all_gpu_info", lambda: [
        {"index": index, "free_mib": 100, "utilization": 0,
         "compute_app_pids": [], "process_query_ok": True} for index in (3, 4)])
    monkeypatch.setattr(config, "_gpu_available", lambda *a: True)
    monkeypatch.setattr(robotics_resource_lock, "exclusive_gpu_job", lambda *a: nullcontext())
    monkeypatch.setattr(B, "ROOT", root)
    adapter = root / "adapter.json"
    B.write(adapter, {"schema_version": "wov2_adapter.v1", "model": "other_model",
                     "modes": ["i2av"], "minimum_free_mib": 1, "gpu_count": 2,
                     "command": ["adapter", "{input}", "{output}", "{gpus}"]})
    def execute(command, **kwargs):
        assert command[3] == kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3,4"
        B.write(Path(command[2]) / "prediction.json",
                {"video": str(root / "reference.mp4"), "view": "head_camera"})
    monkeypatch.setattr(B, "run_process", execute)
    model, results = B.generate(rows, SimpleNamespace(adapter=adapter, mode="i2av",
                                                        chunks=1, timeout=10), root / "run")
    assert model == "other_model" and len(results) == 1


def test_external_model_cli_generates_and_scores_without_lingbot_profile(sample, monkeypatch):
    from contextlib import nullcontext
    from scripts import preflight, score_robotics_cases
    from wo.agent import config
    from autoresearch import robotics_resource_lock
    manifest, _, root = sample
    monkeypatch.setattr(B, "ROOT", root)
    monkeypatch.setattr(B, "implementation_contract", lambda: {})
    monkeypatch.setattr(preflight, "queue_check", lambda *_: None)
    monkeypatch.setattr(config, "all_gpu_info", lambda: [
        {"index": 3, "free_mib": 100, "utilization": 0,
         "compute_app_pids": [], "process_query_ok": True}])
    monkeypatch.setattr(config, "_gpu_available", lambda *a: True)
    monkeypatch.setattr(config, "resolve_score_gpus", lambda profile: (
        {"gpu": 3, "compute_gpu": 3, "mode": "full"}
        if profile == {"scoring": {}} else None))
    monkeypatch.setattr(config, "load_profile", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("external model must not require a LingBot profile")))
    monkeypatch.setattr(robotics_resource_lock, "exclusive_gpu_job", lambda *a: nullcontext())
    generator = root / "generator.py"
    generator.write_text('''import json, subprocess, sys
from pathlib import Path
import imageio_ffmpeg
payload = json.loads(Path(sys.argv[1]).read_text())
assert "reference_video" not in payload and "caption_chunks" not in payload
out = Path(sys.argv[2])
width, height = payload["output_size_wh"]
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-loop", "1",
    "-i", payload["conditioning"]["head_camera"], "-t", "0.4", "-r", "10",
    "-vf", f"scale={width}:{height}", "-pix_fmt", "yuv420p",
    str(out / "generated.mp4")], check=True)
(out / "prediction.json").write_text(json.dumps({"video": "generated.mp4", "view": payload["output_view"]}))
''')
    adapter = root / "adapter.json"
    B.write(adapter, {"schema_version": "wov2_adapter.v1", "model": "other_model",
                     "modes": ["i2av"], "minimum_free_mib": 1,
                     "command": [sys.executable, str(generator), "{input}", "{output}"]})
    def scorer(cases, workspace, gpu, mode, compute):
        from wo.agent.score_stage import valid_receipt
        for case in cases:
            metadata, gen, ref, prompts, chunks, hashes = score_robotics_cases.inputs(case)
            assert gen.is_file() and ref.is_file()
            score_robotics_cases._commit_score(case, raw_video_measurements(case),
                                               hashes, metadata["case_id"])
            assert valid_receipt(case)
    monkeypatch.setattr(score_robotics_cases, "run", scorer)
    B.main(["run", "--manifest", str(manifest), "--adapter", str(adapter),
            "--mode", "i2av", "--run", "other_model_case"])
    report = B.read(root / "outputs/benchmark/other_model_case/report.json")
    assert report["complete"] is True and report["model"] == "other_model"
    B.main(["score", "--manifest", str(manifest),
            "--predictions", str(root / "outputs/benchmark/other_model_case/predictions.json"),
            "--run", "other_model_submitted"])
    assert B.read(root / "outputs/benchmark/other_model_submitted/report.json")["complete"] is True


def test_score_handoff_and_report(sample, monkeypatch):
    from contextlib import nullcontext
    from scripts import preflight, score_robotics_cases
    from wo.agent import config
    from autoresearch import robotics_resource_lock
    manifest, data, root = sample
    _, rows = B.dataset(manifest)
    monkeypatch.setattr(B, "ROOT", root)
    monkeypatch.setattr(preflight, "queue_check", lambda *_: None)
    monkeypatch.setattr(config, "load_profile", lambda: (_ for _ in ()).throw(
        AssertionError("external scoring must not load a tested-model profile")))
    monkeypatch.setattr(config, "resolve_score_gpus",
                        lambda _: {"gpu": 3, "compute_gpu": 3, "mode": "full"})
    monkeypatch.setattr(robotics_resource_lock, "exclusive_gpu_job", lambda *a: nullcontext())
    def scorer(roots, workspace, gpu, mode, compute):
        from wo.agent.score_stage import valid_receipt
        assert gpu == compute == 3 and mode == "full"
        for case in roots:
            metadata, gen, ref, prompts, chunks, hashes = score_robotics_cases.inputs(case)
            assert not gen.is_symlink() and not ref.is_symlink()
            assert gen.resolve().parent == ref.resolve().parent == case.resolve()
            gen.with_name("generated_mask.mp4").write_bytes(b"private cache")
            ref.with_name("reference_mask.mp4").write_bytes(b"private cache")
            assert not (root / "reference_mask.mp4").exists()
            assert metadata["tracks"]["continuous_action"] == "not_applicable"
            score = raw_video_measurements(case)
            assert score["worldolympiad2_score"] is None
            score_robotics_cases._commit_score(case, score, hashes, metadata["case_id"])
            assert valid_receipt(case)
    monkeypatch.setattr(score_robotics_cases, "run", scorer)
    run = root / "run"
    B.write(run / "inputs.json", {"dataset_id": "fixture", "dataset_version": "1",
                                 "scoring_protocol": B.VIDEO_PROTOCOL,
                                 "score_profile": {"scoring": {}}})
    items = {"a": {"video": str(root/"reference.mp4"),
                   "sha256": B.sha(root/"reference.mp4"), "video_info": rows[0]["video"]}}
    B.scoring(rows, items, run, "fixture_model")
    report = B.read(run/"report.json")
    assert report["complete"] is True
    assert report["cases"][0]["status"] == "complete"
    assert report["native_task_success"] == "not_applicable"
    assert B.read(run / "cases/a/score/score.json")["worldolympiad2_score"] is None
    assert B.read(run / "cases/a/score/benchmark_score.json")["worldolympiad2_score"] == 0.5
    B.scoring(rows, items, run, "fixture_model")


def test_incomplete_score_is_not_complete():
    assert not B.score_complete({"track_results": {}})
    assert not B.score_complete({"track_results": {
        name: {"status": "pending", "score": None}
        for name in ("physical", "geometry", "interaction")}})


def test_builtin_lingbot_export_contract(sample, monkeypatch):
    import subprocess
    import imageio_ffmpeg
    from scripts import benchmark_adapter as adapter
    _, _, root = sample
    repo = root / "repo"
    (repo / "wan_va").mkdir(parents=True)
    (repo / "wan_va/wan_va_server.py").touch()
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    attempt = root / "attempt"
    attempt.mkdir()
    seen = []
    import runpy
    from types import ModuleType
    cleaned = []
    configurations = ModuleType("configs")
    configurations.VA_CONFIGS = {"robotwin_i2av": SimpleNamespace()}
    backend = ModuleType("wan_va_server")
    backend.VA_Server = type("VA_Server", (), {})
    def upstream(args):
        server = backend.VA_Server()
        server.exp_name = "long prompt " * 1000
        assert len(server.exp_name) == 24
    backend.run = upstream
    distributed = ModuleType("torch.distributed")
    distributed.is_initialized = lambda: True
    distributed.destroy_process_group = lambda: cleaned.append(True)
    torch = ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "configs", configurations)
    monkeypatch.setitem(sys.modules, "wan_va_server", backend)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    def run_generator(command, **kwargs):
        entry = Path(command[-1]).read_text()
        assert "robotwin_i2av" in entry and "num_chunks_to_infer = 1" in entry
        assert "reference_video" not in entry
        assert command[0] == "/configured/model/python"
        assert kwargs["env"]["WO_EVAL_CONFIG"] == "/configured/profile.json"
        seen.append(command)
        runpy.run_path(command[-1])
        assert cleaned == [True]
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi",
                        "-i", "color=red:size=320x384:rate=10", "-t", "0.4",
                        "-pix_fmt", "yuv420p", str(attempt/"model_output/demo.mp4")], check=True)
    monkeypatch.setattr(adapter, "ROOT", root)
    monkeypatch.setattr(adapter, "run_process", run_generator)
    video, view = adapter.builtin("lingbot_va", "i2av", {
        "conditioning": {k: str(root/"initial.png") for k in adapter.CAMERAS},
        "instruction": "Move the object.", "output_view": "head_camera",
        "output_size_wh": [32, 32], "chunks": 1
    }, attempt, {"WO_LINGBOT_REPO": str(repo), "WO_TESTED_MODEL_CHECKPOINT": str(checkpoint),
                 "WO_TESTED_MODEL_GPUS": "3", "WO_LINGBOT_PYTHON": "/configured/model/python",
                 "WO_EVAL_CONFIG": "/configured/profile.json"}, 10)
    assert len(seen) == 1 and view == "head_camera"
    assert B.video_info(video)["frames"] == 4
    def failed(args):
        raise RuntimeError("upstream failure")
    backend.run = failed
    with pytest.raises(RuntimeError, match="upstream failure"):
        runpy.run_path(str(attempt / "lingbot_i2av_entry.py"))
    assert cleaned == [True, True]


def test_registered_native_adapter_uses_bound_wov2_reference(tmp_path, monkeypatch):
    from scripts import benchmark_adapter as adapter
    from wo.agent import episode_stage, stages
    assert stages.episode_adapter({"model": "fastwam"}) == "wo.models.fastwam"
    run = tmp_path / "original_run"
    ref = run / "references" / "case"
    ref.mkdir(parents=True)
    (run / "tasks").mkdir()
    (run / "tasks/case.json").write_text("{}")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    checked = []
    monkeypatch.setattr(episode_stage, "verify_evaluation_view",
                        lambda native, source: checked.append((native, source)))
    def execute(command, **kwargs):
        assert "wo.agent.episode_stage" in command
        assert "wo.models.fastwam" in command
        assert command[command.index("--run-root") + 1] == str(run)
        assert command[command.index("--window-s") + 1] == "2"
        assert command[command.index("--gpu") + 1] == "3"
        assert command[command.index("--simulator-gpu") + 1] == "4"
        video = attempt / "native/native_replay/exterior_left.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"native video")
    monkeypatch.setattr(adapter, "run_process", execute)
    video, view = adapter.builtin("fastwam", "sim_rollout",
        {"execution": {"reference_dir": str(ref), "window_s": 2},
         "output_view": "exterior_left"}, attempt,
        {"WO_SIMULATOR_GPU": "4", "WO_TESTED_MODEL_GPU": "3"}, 10)
    assert video.is_file() and view == "exterior_left"
    assert checked == [(attempt / "native", ref)]
    with pytest.raises(ValueError, match="original WOV2 TaskSpec"):
        other = run / "references" / "other_case"
        other.mkdir()
        adapter.builtin("fastwam", "sim_rollout",
            {"execution": {"reference_dir": str(other)}, "output_view": "exterior_left"},
            attempt, {"WO_SIMULATOR_GPU": "4", "WO_TESTED_MODEL_GPU": "3"}, 10)


def test_cli_check_uses_actual_video(sample, capsys):
    manifest, _, _ = sample
    B.main(["check", "--manifest", str(manifest)])
    assert json.loads(capsys.readouterr().out)["status"] == "inputs_valid"


def test_i2av_reserves_policy_without_simulator(monkeypatch):
    from scripts.benchmark_adapter import video_gpus
    from wo.agent import config
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "1,2")
    profile = {"tested_model": {"adapter": "lingbot_va", "gpu": "auto", "policy_gpus": "auto"}}
    def card(index, total, free):
        return {"index": index, "total_mib": total, "free_mib": free, "utilization": 0,
                "compute_pids": [], "compute_process_count": 0, "process_query_ok": True}
    assert video_gpus(profile, [card(1, 85000, 84000)]) == [1]
    assert video_gpus(profile, [card(1, 32000, 31000), card(2, 32000, 31000)]) == [1, 2]
    with pytest.raises(RuntimeError, match="two idle"):
        video_gpus(profile, [card(1, 32000, 31000)])


def test_changed_checkpoint_cannot_reuse_completed_run(tmp_path, monkeypatch):
    from wo.agent import config
    profile = {"workspace_root": str(tmp_path), "tested_model": {
        "adapter": "lingbot_va", "checkpoint": "/weights/model-v1", "gpu": 1}}
    monkeypatch.setattr(config, "load_profile", lambda **kwargs: profile)
    monkeypatch.setattr(B, "ROOT", tmp_path)
    for name in ("scripts/benchmark.py", "scripts/benchmark_adapter.py",
                 "code/WorldOlympiad-main/scripts/score_robotics_cases.py",
                 "code/WorldOlympiad-main/scripts/score_video_physical_3d.py",
                 "code/WorldOlympiad-main/physical/sam_process.py",
                 "sources.lock.json", "runtime.lock.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original")
    args = SimpleNamespace(action="run", adapter=None, model="lingbot_va")
    path = tmp_path / "inputs.json"
    B.bound_json(path, B.model_contract(args))
    profile["tested_model"]["gpu"] = 2
    B.bound_json(path, B.model_contract(args))
    profile["tested_model"]["checkpoint"] = "/weights/model-v2"
    with pytest.raises(ValueError, match="changed"):
        B.bound_json(path, B.model_contract(args))

    profile["tested_model"]["adapter"] = "other_model"
    with pytest.raises(ValueError, match="differs"):
        B.model_contract(args)

    profile["tested_model"].update(adapter="lingbot_va", checkpoint="/weights/model-v1")
    (tmp_path / "scripts/benchmark_adapter.py").write_text("revised")
    with pytest.raises(ValueError, match="changed"):
        B.bound_json(path, B.model_contract(args))


def test_simulator_scene_failure_blocks_model_rollout(tmp_path, monkeypatch):
    from scripts import benchmark_adapter as adapter
    reference, attempt = tmp_path / "reference", tmp_path / "attempt"
    reference.mkdir()
    attempt.mkdir()
    B.write(reference / "certificate.json", {"task": "task.json"})
    calls = []
    def check(command, **kwargs):
        assert kwargs["env"]["WO_TESTED_MODEL_GPU"] == "2"
        calls.append(command)
        B.write(attempt / "scene_check/scene_ready.json", {"scene_ready": False})
    monkeypatch.setattr(adapter, "run_process", check)
    with pytest.raises(RuntimeError, match="scene check did not pass"):
        adapter.builtin("lingbot_va", "sim_rollout", {
            "execution": {"reference_dir": str(reference)}, "output_view": "exterior_left"
        }, attempt, {"WO_TESTED_MODEL_GPU": "1", "WO_SIMULATOR_GPU": "2",
                     "WO_TESTED_MODEL_GPUS": "1"}, 10)
    assert len(calls) == 1


def test_reject_existing_scoring_symlink(sample):
    manifest, _, root = sample
    _, rows = B.dataset(manifest)
    case = root / "case"
    case.mkdir()
    (case / "reference.mp4").symlink_to(root / "reference.mp4")
    with pytest.raises(ValueError, match="unsafe symlinked"):
        B.prepare_case(rows[0], {"video": str(root / "reference.mp4"),
                                "video_info": rows[0]["video"]}, case)


def test_submission_run_binds_scoring_implementation(sample, monkeypatch):
    manifest, data, root = sample
    monkeypatch.setattr(B, "ROOT", root)
    for name in ("scripts/benchmark.py", "scripts/benchmark_adapter.py",
                 "code/WorldOlympiad-main/scripts/score_robotics_cases.py",
                 "code/WorldOlympiad-main/scripts/score_video_physical_3d.py",
                 "code/WorldOlympiad-main/physical/sam_process.py",
                 "sources.lock.json", "runtime.lock.json"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original")
    submission = root / "prediction.json"
    B.write(submission, {"schema_version": "wov2_predictions.v1", "dataset_id": "fixture",
                        "dataset_version": "1", "model": "submitted_model",
                        "predictions": [{"case_id": "a", "view": "head_camera", "video": "reference.mp4"}]})
    monkeypatch.setattr(B, "scoring", lambda *args: None)
    command = ["score", "--manifest", str(manifest), "--predictions", str(submission), "--run", "submitted"]
    B.main(command)
    identity = B.read(root / "outputs/benchmark/submitted/inputs.json")
    assert len(identity["implementation_sha256"]) == 7
    assert identity["scoring_protocol"] == B.VIDEO_PROTOCOL
    (root / "scripts/benchmark.py").write_text("revised")
    with pytest.raises(ValueError, match="changed"):
        B.main(command)


def test_video_applicability_does_not_relabel_native_missing_evidence(tmp_path):
    raw = raw_video_measurements(tmp_path)
    assert raw["worldolympiad2_score"] is None
    value = B.video_score(raw, B.VIDEO_PROTOCOL)
    assert B.score_complete(value) and value["worldolympiad2_score"] == 0.5
    assert raw["track_results"]["interaction"]["status"] == "pending"
    raw["physical"] = None
    assert not B.score_complete(B.video_score(raw, B.VIDEO_PROTOCOL))
    with pytest.raises(ValueError, match="predeclared"):
        B.video_score(raw, dict(B.VIDEO_PROTOCOL, native_trajectory_cae="required"))


@pytest.mark.parametrize("failure", ["physical", "geometry", "interaction", "clip"])
def test_incomplete_real_judge_evidence_cannot_be_aggregated(tmp_path, failure):
    raw = raw_video_measurements(tmp_path)
    B.write(tmp_path / "prompt.json", [{"caption": "Move the object"}])
    B.validate_video_evidence(raw, tmp_path)
    if failure == "physical":
        raw["physical"]["errored_question_count"] = 1
    elif failure == "geometry":
        raw["three_d"]["gs_video_path"] = ""
    elif failure == "interaction":
        raw["interaction"]["global"]["error"] = "judge failed"
    else:
        raw["clip_interaction"]["summary"]["num_scored_chunks"] = 0
    with pytest.raises(RuntimeError, match="incomplete|missing"):
        B.validate_video_evidence(raw, tmp_path)


def test_batch_waits_for_own_gpu_utilization_to_settle(monkeypatch):
    from wo.agent import config

    readings = iter((38, 0))
    monkeypatch.setattr(config, "all_gpu_info", lambda: [{
        "index": 3, "process_query_ok": True, "compute_process_count": 0,
        "utilization": next(readings),
    }])
    sleeps = []
    monkeypatch.setattr(B.time, "sleep", sleeps.append)
    B.settle_scoring_gpus({3})
    assert sleeps == [1]


def test_batch_does_not_wait_on_another_compute_process(monkeypatch):
    from wo.agent import config

    monkeypatch.setattr(config, "all_gpu_info", lambda: [{
        "index": 3, "process_query_ok": True, "compute_process_count": 1,
        "utilization": 38,
    }])
    monkeypatch.setattr(B.time, "sleep", lambda seconds: pytest.fail("unexpected wait"))
    B.settle_scoring_gpus({3})
