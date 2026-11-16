"""Strict checkpoint/config pairing for LingBot-VA robotics inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROBOTWIN_MODEL_ID = "robbyant/lingbot-va-posttrain-robotwin"
LIBERO_MODEL_ID = "robbyant/lingbot-va-posttrain-libero-long"
ROBOTWIN_CONFIGS = {"robotwin", "robotwin_i2av"}
LIBERO_CONFIGS = {"libero", "libero_i2av"}
INFERENCE_ATTN_MODES = {"torch", "flashattn"}


def _indexed_component(model_path: Path, component: str, index_name: str) -> dict[str, int]:
    root = model_path / component
    index_path = root / index_name
    if not index_path.is_file():
        raise FileNotFoundError(f"LingBot {component} index is missing: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    names = sorted(set(payload.get("weight_map", {}).values()))
    if not names:
        raise ValueError(f"LingBot {component} index contains no weight shards")
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"LingBot {component} checkpoint is incomplete; missing shards: {missing}"
        )
    actual_size = sum((root / name).stat().st_size for name in names)
    expected_size = int(payload.get("metadata", {}).get("total_size", actual_size))
    # ``metadata.total_size`` is the tensor payload total. Safetensors file
    # headers make complete shard files slightly larger, so it is a lower bound
    # rather than an exact on-disk byte count.
    if actual_size < expected_size:
        raise ValueError(
            f"LingBot {component} shard bytes are below the indexed tensor payload: "
            f"expected at least {expected_size}, got {actual_size}"
        )
    return {"shard_count": len(names), "total_size": actual_size}


def validate_checkpoint_files(model_path: Path) -> dict[str, object]:
    transformer = _indexed_component(
        model_path, "transformer", "diffusion_pytorch_model.safetensors.index.json"
    )
    text_encoder = _indexed_component(
        model_path, "text_encoder", "model.safetensors.index.json"
    )
    required = [
        model_path / "vae/config.json",
        model_path / "vae/diffusion_pytorch_model.safetensors",
        model_path / "tokenizer/tokenizer.json",
        model_path / "tokenizer/tokenizer_config.json",
        model_path / "tokenizer/special_tokens_map.json",
        model_path / "tokenizer/spiece.model",
    ]
    missing = [str(path.relative_to(model_path)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LingBot checkpoint is incomplete; missing files: {missing}")
    return {"transformer": transformer, "text_encoder": text_encoder}


def validate_lingbot_checkpoint(
    model_path: Path,
    config_name: str,
    source_model_id: str | None = None,
) -> dict[str, object]:
    model_path = model_path.resolve()
    transformer_config_path = model_path / "transformer/config.json"
    if not transformer_config_path.is_file():
        raise FileNotFoundError(f"invalid LingBot checkpoint: {model_path}")
    transformer_config = json.loads(transformer_config_path.read_text(encoding="utf-8"))
    attn_mode = str(transformer_config.get("attn_mode", ""))
    if attn_mode not in INFERENCE_ATTN_MODES:
        raise ValueError(
            f"LingBot inference requires attn_mode in {sorted(INFERENCE_ATTN_MODES)}, "
            f"got {attn_mode!r}"
        )
    component_summary = validate_checkpoint_files(model_path)

    local_id = model_path.name.lower()
    declared_id = (source_model_id or "").strip().lower()
    if config_name in ROBOTWIN_CONFIGS:
        if "posttrain-robotwin" not in local_id:
            raise ValueError(
                "RoboTwin LingBot inference requires the post-trained RoboTwin checkpoint; "
                f"refusing {model_path.name!r}"
            )
        if source_model_id and declared_id != ROBOTWIN_MODEL_ID.lower():
            raise ValueError(
                f"RoboTwin source model must be {ROBOTWIN_MODEL_ID!r}, got {source_model_id!r}"
            )
        resolved_model_id = ROBOTWIN_MODEL_ID
    elif config_name in LIBERO_CONFIGS:
        if "posttrain-libero-long" not in local_id:
            raise ValueError(
                "LIBERO LingBot inference requires the post-trained LIBERO-Long checkpoint"
            )
        if source_model_id and declared_id != LIBERO_MODEL_ID.lower():
            raise ValueError(
                f"LIBERO source model must be {LIBERO_MODEL_ID!r}, got {source_model_id!r}"
            )
        resolved_model_id = LIBERO_MODEL_ID
    else:
        if "posttrain-robotwin" in local_id or declared_id == ROBOTWIN_MODEL_ID.lower():
            raise ValueError(
                f"RoboTwin post-trained checkpoint cannot use non-RoboTwin config {config_name!r}"
            )
        resolved_model_id = source_model_id or model_path.name

    return {
        "model_path": str(model_path),
        "source_model_id": resolved_model_id,
        "config_name": config_name,
        "attn_mode": attn_mode,
        "components": component_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", required=True, dest="config_name")
    parser.add_argument("--source-model-id")
    args = parser.parse_args()
    contract = validate_lingbot_checkpoint(
        args.checkpoint,
        args.config_name,
        args.source_model_id,
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
