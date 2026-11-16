from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wov2_preflight_allowlist", ROOT / "scripts/preflight.py")
preflight = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(preflight)


def test_allowlist_parses_unique_nonnegative_indices():
    assert preflight.parse_gpu_allowlist("8, 2") == [2, 8]


@pytest.mark.parametrize("value", ["", "2,,8", "-1,2", "2,2", "two,3", "²,3"])
def test_allowlist_rejects_missing_or_malformed_indices(value):
    with pytest.raises(ValueError):
        preflight.parse_gpu_allowlist(value)


def test_resource_query_passes_only_user_indices_to_nvidia_smi(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "2,8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--query-gpu=" in command[3]:
            output = (
                "2, NVIDIA A100, GPU-2, 81920, 70000, 1\n"
                "8, NVIDIA L40S, GPU-8, 46080, 40000, 0\n"
            )
        else:
            output = "GPU-8, 123, python, 1024\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    snapshot = preflight.query_gpu_resources()

    assert snapshot["queried_gpu_indices"] == [2, 8]
    assert [card["index"] for card in snapshot["cards"]] == [2, 8]
    assert snapshot["cards"][1]["compute_app_pids"] == [123]
    assert all(command[:3] == ["nvidia-smi", "-i", "2,8"] for command in calls)


def test_missing_allowlist_fails_before_gpu_query(monkeypatch):
    monkeypatch.delenv("WOV2_GPU_ALLOWLIST", raising=False)
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("must not query GPUs without an allowlist"),
    )
    with pytest.raises(SystemExit, match="WOV2_GPU_ALLOWLIST"):
        preflight.query_gpu_resources()


def test_render_preflight_warns_about_missing_vendor_json(monkeypatch, tmp_path):
    vendor = tmp_path / "10_nvidia.json"
    monkeypatch.setenv("WOV2_EGL_VENDOR_JSON", str(vendor))
    reason = preflight.renderer_config_warning("preparation")
    assert reason and "NVIDIA EGL vendor JSON" in reason
    vendor.write_text('{"ICD":{"library_path":"libEGL_nvidia.so.0"}}')
    assert preflight.renderer_config_warning("preparation") is None
    assert preflight.renderer_config_warning("evaluation") is None


def test_non_rendering_preflight_does_not_require_egl_config(monkeypatch, tmp_path):
    monkeypatch.setenv("WOV2_EGL_VENDOR_JSON", str(tmp_path / "missing.json"))
    assert preflight.renderer_config_warning("scoring") is None
