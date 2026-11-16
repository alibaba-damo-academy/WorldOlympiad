#!/usr/bin/env python3
"""Validate a LingBot checkpoint and select the official inference attention mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root")
    parser.add_argument("--source-model", required=True)
    args = parser.parse_args()

    model_root = Path(args.model_root).resolve()
    config_path = model_root / "transformer" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LingBot transformer config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.source_model == "Robbyant/lingbot-va-base" and config.get("action_dim") != 30:
        raise ValueError("the released LingBot-VA base checkpoint must expose action_dim=30")

    original_mode = config.get("attn_mode")
    if original_mode not in {"flex", "torch"}:
        raise ValueError(f"unsupported LingBot attention mode: {original_mode!r}")
    repairs: list[str] = []
    if original_mode == "flex":
        upstream_path = config_path.with_name("config.upstream.json")
        if not upstream_path.exists():
            _write_json_atomic(upstream_path, config)
        config["attn_mode"] = "torch"
        _write_json_atomic(config_path, config)
        repairs.append("attn_mode:flex->torch")

    _write_json_atomic(
        model_root / ".worldolympiad_runtime.json",
        {
            "schema_version": "worldolympiad_lingbot_runtime.v1",
            "source_model": args.source_model,
            "attention_mode": "torch",
            "repairs": repairs,
        },
    )
    print(f"LingBot checkpoint ready: {args.source_model} (attention_mode=torch)")


if __name__ == "__main__":
    main()
