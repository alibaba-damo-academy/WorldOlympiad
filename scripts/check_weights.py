#!/usr/bin/env python3
"""Validate cached LingBot-VA weights without downloading or loading them."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/two-interface.json"
CHECKER = ROOT / "code/WorldOlympiad-main/scripts/lingbot_checkpoint_contract.py"
SOURCE_MODEL_ID = "robbyant/lingbot-va-posttrain-robotwin"


def main() -> int:
    if not CONFIG.is_file():
        raise SystemExit("missing config/two-interface.json; configure the existing checkpoint path first")
    if not CHECKER.is_file():
        raise SystemExit("pinned WorldOlympiad source is not installed; run setup.sh after completing sources.lock")
    profile = json.loads(CONFIG.read_text())
    model = profile.get("tested_model", {})
    if model.get("adapter") != "lingbot_va":
        raise SystemExit("this evaluator currently locks the LingBot-VA checkpoint contract")
    checkpoint = Path(model.get("checkpoint", ""))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory missing: {checkpoint}; no download was attempted")
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--checkpoint", str(checkpoint),
         "--config", "robotwin", "--source-model-id", SOURCE_MODEL_ID],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise SystemExit("checkpoint contract failed; no download was attempted:\n" + result.stderr.strip())
    summary = json.loads(result.stdout)
    components = summary.get("components", {})
    print("PASS: existing tested-model checkpoint satisfies the pinned RoboTwin contract.")
    print("Weight acquisition skipped; no network download or checkpoint load was performed.")
    for name, item in components.items():
        print(f"{name}: {item.get('shard_count')} shards, {item.get('total_size')} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
