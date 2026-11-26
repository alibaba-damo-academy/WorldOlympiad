from pathlib import Path

import pytest

from wo.models.lingbot_robotwin_runtime import resolve_robotwin_repo


_REQUIRED = (
    "env_cfg/task_config/demo_clean.yml",
    "env_cfg/task_config/_embodiment_config.yml",
    "assets/embodiments/aloha-agilex/config.yml",
    "assets/objects/objaverse/list.json",
)


def make_runtime(root: Path, *, missing=()) -> Path:
    root.mkdir(parents=True)
    for relative in _REQUIRED:
        if relative in missing:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return root


def test_complete_configured_runtime_is_used(tmp_path):
    configured = make_runtime(tmp_path / "runtime/RoboTwin")
    assert resolve_robotwin_repo(configured) == configured.resolve()


def test_missing_required_assets_fail_without_searching_sibling_branches(tmp_path):
    configured = make_runtime(
        tmp_path / "runtime/RoboTwin",
        missing={"assets/objects/objaverse/list.json"},
    )
    # A complete runtime in another candidate branch must never be selected.
    make_runtime(tmp_path / "branches/rtx5090/runtime/RoboTwin-5090")
    with pytest.raises(FileNotFoundError, match="assets/objects/objaverse/list.json"):
        resolve_robotwin_repo(configured)


def test_missing_embodiment_config_fails_clearly(tmp_path):
    configured = make_runtime(
        tmp_path / "runtime/RoboTwin",
        missing={"assets/embodiments/aloha-agilex/config.yml"},
    )
    with pytest.raises(FileNotFoundError, match="assets/embodiments/aloha-agilex/config.yml"):
        resolve_robotwin_repo(configured)


def test_nonexistent_runtime_remains_importable(tmp_path):
    configured = tmp_path / "not-mounted/RoboTwin"
    assert resolve_robotwin_repo(configured) == configured.resolve()


def test_inaccessible_runtime_remains_importable(monkeypatch, tmp_path):
    configured = tmp_path / "denied/RoboTwin"
    resolved = configured.resolve()
    original_exists = Path.exists

    def inaccessible(path):
        if path == resolved:
            raise PermissionError("simulated inaccessible workspace")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", inaccessible)
    assert resolve_robotwin_repo(configured) == resolved
