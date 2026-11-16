import json
from pathlib import Path

import pytest

from wo import render
from wo.agent import episode_stage


def _bound_episode(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir()
    episode = tmp_path / "episode"
    (episode / "native_run/simulation").mkdir(parents=True)
    (episode / "native_run/simulation/trajectory.json").write_text('{"frames":[{},{}]}')
    (episode / "episode.json").write_text(json.dumps({
        "model": "lingbot_va", "seed": 0, "task_success": False,
        "reference": str(ref),
        "capture": {"videos": {"head": "capture/head.mp4", "left_wrist": "capture/wrist.mp4"}},
    }))
    (episode / "episode_reference_binding.json").write_text("{}")
    return ref, episode


def test_render_absolute_output_uses_host_egl_without_cuda_gpu_binding(tmp_path, monkeypatch):
    ref = tmp_path / "reference"
    ref.mkdir()
    output = tmp_path / "episode/native_replay"
    monkeypatch.setattr(render, "BLENDER_PY", Path("/runtime/blender-python"))
    monkeypatch.setattr(render, "SCENECODE", tmp_path)
    monkeypatch.setattr(render, "REPO", tmp_path / "repo")
    monkeypatch.setattr(render, "SETTINGS", tmp_path / "settings.py")
    vendor_json = tmp_path / "nvidia.json"
    vendor_json.write_text('{"ICD":{"library_path":"libEGL_nvidia.so.0"}}')
    monkeypatch.setattr(render, "NVIDIA_EGL", vendor_json)
    monkeypatch.delenv("WOV2_EGL_VENDOR_JSON", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.delenv("EGL_DEVICE_ID", raising=False)
    monkeypatch.setattr(render, "build_plan", lambda *_a, **_k: {
        "views": {"exterior_left": [320, 180, 71.5]}, "frames": [{}, {}]})
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        Path(command[3]).mkdir(parents=True, exist_ok=True)
        (Path(command[3]) / "exterior_left.mp4").write_bytes(b"video")

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    assert render.render(ref, trajectory=tmp_path / "trajectory.json", out=output,
                         views=("exterior_left",), gpu_id=6) == output
    assert "CUDA_VISIBLE_DEVICES" not in observed["env"]
    assert "CUDA_DEVICE_ORDER" not in observed["env"]
    assert (tmp_path / "episode/native_replay_plan.json").is_file()
    assert not (ref / "native_replay_plan.json").exists()
    assert observed["env"]["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(vendor_json)
    assert "EGL_DEVICE_ID" not in observed["env"]


def test_ensure_evaluation_view_replays_native_trajectory_and_preserves_native_cameras(tmp_path, monkeypatch):
    ref, episode = _bound_episode(tmp_path)
    binding = {"view": "exterior_left", "reference": str(ref), "files": {"trajectory.json": "abc"}}
    monkeypatch.setattr(episode_stage, "verify_episode_binding", lambda *_: {"status": "verified"})
    monkeypatch.setattr(episode_stage, "capture", lambda *_: binding)
    calls = {}

    def fake_render(ref_arg, *, trajectory, out, views, gpu_id):
        calls.update(ref=ref_arg, trajectory=trajectory, views=views, gpu_id=gpu_id)
        out.mkdir(parents=True)
        (out / "exterior_left.mp4").write_bytes(b"rendered-video")
        plan = {"views": {"exterior_left": [320, 180, 71.5]}, "frames": [{}, {}]}
        out.with_name(out.name + "_plan.json").write_text(json.dumps(plan))
        return out

    monkeypatch.setattr(render, "render", fake_render)
    receipt = episode_stage.ensure_evaluation_view(episode, ref, 6)
    record = json.loads((episode / "episode.json").read_text())
    assert calls["views"] == ("exterior_left",)
    assert calls["trajectory"] == episode / "native_run/simulation/trajectory.json"
    assert calls["gpu_id"] == 6
    assert receipt["renderer_device_selection"] == "host-configured EGL default"
    assert record["videos"]["exterior_left"] == "native_replay/exterior_left.mp4"
    assert record["videos"]["head"] == "capture/head.mp4"
    assert receipt["frame_count"] == 2
    assert episode_stage.verify_evaluation_view(episode, ref)["video_sha256"] == receipt["video_sha256"]
