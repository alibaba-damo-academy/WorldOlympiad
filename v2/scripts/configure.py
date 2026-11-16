#!/usr/bin/env python3
"""Write a non-secret two-interface profile from explicit user inputs."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def resolved(value: str) -> str:
    """Make paths absolute without dereferencing runtime/venv symlinks."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return os.path.abspath(os.fspath(path))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent-base-url", required=True)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--tested-model", choices=("cosmos3", "lingbot_va", "gr00t", "fastwam", "longwam"),
                   default="lingbot_va", help="registered native episode adapter")
    p.add_argument("--checkpoint", required=True,
                   help="path to the already-downloaded tested-model checkpoint")
    p.add_argument("--robotwin-repo", default="runtime/RoboTwin")
    p.add_argument("--lingbot-repo", default="runtime/lingbot-va")
    p.add_argument("--robotwin-python", default="envs/robotwin/bin/python")
    p.add_argument("--lingbot-python", default="envs/lingbot/bin/python")
    p.add_argument("--scenecode-source", default="code/SceneCode")
    p.add_argument("--scenecode-runtime", default="runtime/scenecode")
    p.add_argument("--score-mode", choices=("auto", "full", "split"), default="auto")
    p.add_argument("--replace", action="store_true")
    a = p.parse_args()

    u = urlsplit(a.agent_base_url)
    if u.scheme != "https" or not u.hostname or u.username or u.password or u.query or u.fragment:
        p.error("Agent API base URL must be HTTPS without embedded credentials, query, or fragment")
    base_url = a.agent_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        p.error("Agent API base URL must end in /v1")
    if not a.agent_model.strip() or "your_" in a.agent_model.lower():
        p.error("provide the exact Agent API model identifier")

    checkpoint = Path(resolved(a.checkpoint))
    if not (checkpoint.is_dir() if a.tested_model == "lingbot_va" else checkpoint.exists()):
        p.error("configured checkpoint path does not exist; no download was attempted")
    source = Path(resolved(a.scenecode_source))
    if not (source / "main.py").is_file():
        p.error("SceneCode source main.py is missing; complete source setup first")

    data = json.loads((ROOT / "config/two-interface.example.json").read_text(encoding="utf-8"))
    data["workspace_root"] = str(ROOT)
    data["agent_api"]["base_url"] = base_url
    data["agent_api"]["model"] = a.agent_model.strip()
    data["tested_model"]["adapter"] = a.tested_model
    data["tested_model"]["checkpoint"] = str(checkpoint)
    for key in ("robotwin_repo", "lingbot_repo", "robotwin_python", "lingbot_python",
                "scenecode_runtime"):
        data["tested_model"][key] = resolved(getattr(a, key))
    data["tested_model"]["scenecode_source"] = str(source)
    data["scoring"]["mode"] = a.score_mode

    out = ROOT / "config/two-interface.json"
    if out.exists() and not a.replace:
        p.error("config/two-interface.json already exists; pass --replace to rewrite it")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".two-interface.", dir=out.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, out)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print("Wrote config/two-interface.json; no API key was read or stored.")
    print("Set WO_AGENT_API_KEY in .secrets/scenecode.env using scripts/set_api_key.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
