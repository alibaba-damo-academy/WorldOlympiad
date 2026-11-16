import os

import pytest

from autoresearch.runtime_paths import scenecode_python, scenecode_runtime


def test_scenecode_runtime_comes_from_workspace_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("SCENECODE_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("SCENECODE_PYTHON", raising=False)
    source = tmp_path / "code/SceneCode-main/scenecode"
    source.mkdir(parents=True)
    python = tmp_path / "envs/scenecode/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    assert scenecode_runtime(tmp_path) == source.parent.resolve()
    assert scenecode_python(tmp_path) == python.resolve()
    monkeypatch.setenv("SCENECODE_RUNTIME_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="runtime is unavailable"):
        scenecode_runtime(tmp_path)
