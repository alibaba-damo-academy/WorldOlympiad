from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/lingbot_checkpoint_contract.py"
SPEC = importlib.util.spec_from_file_location("lingbot_checkpoint_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _checkpoint(root: Path, name: str, attn_mode: str = "torch") -> Path:
    checkpoint = root / name
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps({"attn_mode": attn_mode}), encoding="utf-8"
    )
    (transformer / "weights.safetensors").write_bytes(b"transformer")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(b"transformer")},
                "weight_map": {"x": "weights.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    text_encoder = checkpoint / "text_encoder"
    text_encoder.mkdir()
    (text_encoder / "weights.safetensors").write_bytes(b"text")
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(b"text")},
                "weight_map": {"x": "weights.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    vae = checkpoint / "vae"
    vae.mkdir()
    (vae / "config.json").write_text("{}", encoding="utf-8")
    (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    tokenizer = checkpoint / "tokenizer"
    tokenizer.mkdir()
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
    ):
        (tokenizer / filename).write_bytes(b"tokenizer")
    return checkpoint


def test_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "lingbot-va-posttrain-robotwin")
    (checkpoint / "transformer/weights.safetensors").unlink()
    try:
        MODULE.validate_lingbot_checkpoint(checkpoint, "robotwin_i2av")
    except FileNotFoundError as error:
        assert "missing shards" in str(error)
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def test_robotwin_rejects_base_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "lingbot-va-base")
    try:
        MODULE.validate_lingbot_checkpoint(checkpoint, "robotwin_i2av")
    except ValueError as error:
        assert "post-trained RoboTwin" in str(error)
    else:
        raise AssertionError("base checkpoint was accepted for RoboTwin")


def test_robotwin_accepts_exact_posttrain_pair(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "lingbot-va-posttrain-robotwin")
    result = MODULE.validate_lingbot_checkpoint(
        checkpoint,
        "robotwin_i2av",
        MODULE.ROBOTWIN_MODEL_ID,
    )
    assert result["source_model_id"] == MODULE.ROBOTWIN_MODEL_ID


def test_inference_rejects_flex_attention(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "lingbot-va-posttrain-robotwin", "flex")
    try:
        MODULE.validate_lingbot_checkpoint(checkpoint, "robotwin")
    except ValueError as error:
        assert "attn_mode" in str(error)
    else:
        raise AssertionError("training attention mode was accepted for inference")
