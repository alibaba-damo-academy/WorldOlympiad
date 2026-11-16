"""Resolve this checkout's evaluation profile for isolated test runners."""

import os

os.environ.setdefault("WOV2_GPU_ALLOWLIST", "0,1,2,3,4,5,6")
from pathlib import Path


def _readable_profile(value: str | None) -> bool:
    if not value:
        return False
    try:
        return Path(value).expanduser().is_file()
    except OSError:
        return False


for parent in Path(__file__).resolve().parents:
    candidate = parent / "config" / "two-interface.json"
    if candidate.is_file():
        # Some isolated runners inherit inaccessible defaults from another
        # checkout. Resolve profile and source paths against this test checkout.
        if not _readable_profile(os.environ.get("WO_EVAL_CONFIG")):
            os.environ["WO_EVAL_CONFIG"] = str(candidate)
        os.environ["WO_WORKSPACE"] = str(parent)
        break
