"""Resolve installed runtime profiles from the selected workspace."""

from __future__ import annotations

import os
from pathlib import Path


def scenecode_runtime(workspace: str | Path) -> Path:
    workspace = Path(workspace).resolve()
    path = Path(os.environ.get("SCENECODE_RUNTIME_ROOT",
                               os.environ.get("SCENECODE_RUNTIME",
                                              workspace / "code/SceneCode-main"))).resolve()
    if not (path / "scenecode").is_dir():
        raise FileNotFoundError(f"selected SceneCode runtime is unavailable: {path}")
    return path


def scenecode_python(workspace: str | Path) -> Path:
    workspace = Path(workspace).resolve()
    path = Path(os.environ.get("SCENECODE_PYTHON",
                               os.environ.get("SCENECODE_RUNTIME_PYTHON",
                                              workspace / "envs/scenecode/bin/python"))).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"selected SceneCode Python is unavailable: {path}")
    return path
