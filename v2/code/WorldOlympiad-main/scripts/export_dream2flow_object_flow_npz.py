#!/usr/bin/env python3
"""Export a Dream2Flow object-flow checkpoint as a portable NumPy archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def export_object_flow(source: str | Path, output: str | Path) -> Path:
    """Load the upstream pickle in its native runtime and remove Python objects."""

    import torch

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise FileNotFoundError(source_path)
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source_path, map_location="cpu")
    if not isinstance(payload, dict) or "flow" not in payload:
        raise ValueError("Dream2Flow object-flow checkpoint must contain a 'flow' object")

    flow = payload["flow"]
    position = getattr(flow, "position", None)
    valid_mask = getattr(flow, "valid_mask", None)
    if position is None or valid_mask is None:
        raise ValueError("Dream2Flow flow object must contain position and valid_mask")

    arrays = {
        "position": _as_numpy(position),
        "valid_mask": _as_numpy(valid_mask),
    }
    for key in ("initial_pixel_coords", "region_mask"):
        value = payload.get(key)
        if value is not None:
            arrays[key] = _as_numpy(value)

    if arrays["position"].ndim != 3 or arrays["position"].shape[-1] != 3:
        raise ValueError(f"invalid Dream2Flow position shape: {arrays['position'].shape}")
    if arrays["valid_mask"].shape != arrays["position"].shape[:2]:
        raise ValueError(
            "Dream2Flow valid_mask must match position's first two dimensions"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary_path, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    print(export_object_flow(args.source, args.output))


if __name__ == "__main__":
    main()
