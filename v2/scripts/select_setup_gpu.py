#!/usr/bin/env python3
"""Select an idle allowlisted GPU for small runtime installation smoke checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import _available, query_gpu_resources  # noqa: E402

MIN_FREE_MIB = 8192

def select_gpu(cards: list[dict], min_free_mib: int = MIN_FREE_MIB) -> dict:
    eligible = [card for card in cards if _available(card, min_free_mib)]
    if not eligible:
        raise RuntimeError(
            f"setup CUDA smoke requires an idle allowlisted NVIDIA GPU with at least {min_free_mib} MiB free"
        )
    return min(eligible, key=lambda card: (-card["free_mib"], card["index"]))

def main() -> int:
    card = select_gpu(query_gpu_resources()["cards"])
    print("{}	{}	{}	{}".format(card["index"], card["uuid"], card["free_mib"], card["utilization"]))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"setup GPU selection blocked: {exc}")
