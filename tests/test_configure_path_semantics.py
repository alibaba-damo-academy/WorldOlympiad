from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/configure.py"
spec = importlib.util.spec_from_file_location("wov2_configure", SCRIPT)
configure = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(configure)


def test_absolute_path_keeps_venv_python_symlink(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    link = project / "envs/robotwin/bin/python"
    target = tmp_path / "python-base/bin/python3.10"
    link.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    target.write_text("interpreter")
    link.symlink_to(target)
    monkeypatch.setattr(configure, "ROOT", project)

    assert configure.resolved("envs/robotwin/bin/python") == str(link)
    assert configure.resolved(str(link)) == str(link)


def test_configure_selects_registered_non_lingbot_adapter(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config/two-interface.example.json").write_text(json.dumps({
        "schema_version": 1, "workspace_root": str(project),
        "agent_api": {"base_url": "", "model": "", "api_key_env": "WO_AGENT_API_KEY"},
        "tested_model": {"adapter": "lingbot_va", "checkpoint": ""},
        "scoring": {"mode": "auto"}}))
    (project / "code/SceneCode").mkdir(parents=True)
    (project / "code/SceneCode/main.py").write_text("")
    checkpoint = tmp_path / "other_model.bin"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(configure, "ROOT", project)
    monkeypatch.setattr(sys, "argv", ["configure", "--agent-base-url", "https://example.org/v1",
        "--agent-model", "agent", "--tested-model", "fastwam", "--checkpoint", str(checkpoint)])
    assert configure.main() == 0
    configured = json.loads((project / "config/two-interface.json").read_text())
    assert configured["tested_model"]["adapter"] == "fastwam"
    assert configured["tested_model"]["checkpoint"] == str(checkpoint)
