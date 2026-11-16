from pathlib import Path

from wo.agent.config import apply_environment, preflight, scenecode_runtime_status


def _profile(workspace: Path, *, scenecode_source: str | None = None) -> dict:
    tested_model = {
        "adapter": "lingbot_va",
        "checkpoint": "weights/test-model",
        "gpu": 0,
        "scenecode_runtime": "tmp/scenecode_runtime",
    }
    if scenecode_source is not None:
        tested_model["scenecode_source"] = scenecode_source
    return {
        "_config_path": str(workspace / "config/two-interface.json"),
        "workspace_root": str(workspace),
        "agent_api": {
            "base_url": "https://agent.example/v1",
            "model": "test-model",
            "api_key_env": "WO_AGENT_API_KEY",
        },
        "tested_model": tested_model,
        "scoring": {"mode": "full", "gpu": 0},
    }


def test_scenecode_source_profile_path_is_resolved_and_preflighted(tmp_path: Path) -> None:
    source = tmp_path / "shared/SceneCode"
    source.mkdir(parents=True)
    (source / "main.py").write_text("# source fixture\n", encoding="utf-8")
    profile = _profile(tmp_path, scenecode_source="shared/SceneCode")
    env = {"SCENECODE_SOURCE_ROOT": "/stale/override"}

    apply_environment(env, profile, phase="worker")
    assert env["SCENECODE_SOURCE_ROOT"] == str(source.resolve())
    result = preflight(profile, env, phase="worker")
    assert result["tested_model"]["scenecode_source"] == str(source.resolve())
    assert result["tested_model"]["scenecode_source_exists"] is True


def test_scenecode_source_defaults_to_workspace_checkout(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    env: dict[str, str] = {}

    apply_environment(env, profile, phase="worker")
    expected = (tmp_path / "code/SceneCode-main").resolve()
    assert env["SCENECODE_SOURCE_ROOT"] == str(expected)
    assert preflight(profile, env, phase="worker")["tested_model"]["scenecode_source_exists"] is False


def test_private_source_override_is_preserved_without_profile_value(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    env = {"SCENECODE_SOURCE_ROOT": "/shared/SceneCode"}

    apply_environment(env, profile, phase="worker")
    assert env["SCENECODE_SOURCE_ROOT"] == "/shared/SceneCode"


def test_missing_runtime_can_be_staged_but_existing_unmanaged_runtime_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "tmp/scenecode_runtime"
    assert scenecode_runtime_status(runtime) == "not_staged"

    runtime.mkdir(parents=True)
    assert scenecode_runtime_status(runtime) == "unrecognized"
    (runtime / ".worldolympiad_scenecode_runtime_v1").write_text("managed\n", encoding="utf-8")
    assert scenecode_runtime_status(runtime) == "ready"
