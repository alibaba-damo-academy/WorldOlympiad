"""Workspace roots shared by the runner and all registered stages."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WORKSPACE = Path(__file__).resolve().parents[3]
WORKSPACE = Path(
    os.environ.get("WO_WORKSPACE", str(DEFAULT_WORKSPACE))
).expanduser().resolve()
