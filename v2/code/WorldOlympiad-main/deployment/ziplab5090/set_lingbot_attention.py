#!/usr/bin/env python3
"""Set LingBot checkpoint attention mode for inference, preserving upstream config."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--mode", choices=("torch", "flashattn"), default="torch")
    args = parser.parse_args()
    config_path = Path(args.model_path).resolve() / "transformer/config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    backup_path = config_path.with_name("config.upstream.json")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    previous = config.get("attn_mode")
    config["attn_mode"] = args.mode
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(config_path), "previous": previous, "current": args.mode}))


if __name__ == "__main__":
    main()
