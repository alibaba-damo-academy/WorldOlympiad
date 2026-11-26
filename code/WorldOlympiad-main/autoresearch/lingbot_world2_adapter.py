"""Typed standalone adapter for LingBot-World2 14B causal-fast imagination."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .lingbot_world_physics import INPUT_VERSION, digest, prepare_case


PROFILE_ID = "lingbot_world2_14b_causal_fast"
PROTOCOL = "open_loop_first_frame_text_camera_video"
MODEL_ID = "Robbyant/lingbot-world-v2-14b-causal-fast"
TASK = "i2v-A14B"
INFER_MODE = "causal_fast"
UPSTREAM_COMMIT = "1895d300d8ac936401689b26389f51cbd36530eb"
HF_REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"
DOWNLOAD_MANIFEST_SHA256 = "ea9c2bbd284b1369023076d2bba4a15a89fec15eab47d6cc78a4cfc143f0486e"
CONFIG_SHA256 = "1b73e66e5d82858ed9cfb65b93793e3221c2e3d3fe6ecb8dca29c106b2f20930"
PROMPT_STANDARD = "lingbot_world2_event_prompt.v1"
ADAPTER_CONTRACT = "lingbot_world2_adapter_contract.v1"
FORBIDDEN_STATIC_PHRASES = (
    "camera remains stationary",
    "background remains stationary",
    "camera and background remain stationary",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def compile_event_prompt(subject: str, trigger: str, ordered_events: Iterable[str]) -> str:
    """Compile one visible-subject, immediate-trigger, ordered-event prompt."""

    parts = [subject.strip(), trigger.strip(), *(str(item).strip() for item in ordered_events)]
    if any(not part for part in parts) or len(parts) < 3:
        raise ValueError("prompt needs a visible subject, immediate trigger, and event chain")
    prompt = " ".join(part.rstrip(". ") + "." for part in parts)
    validate_event_prompt(prompt)
    return prompt


def validate_event_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip() or "\n" in prompt:
        raise ValueError("prompt must be one nonempty paragraph")
    normalized = " ".join(prompt.split())
    lowered = normalized.casefold()
    forbidden = [phrase for phrase in FORBIDDEN_STATIC_PHRASES if phrase in lowered]
    if forbidden:
        raise ValueError(f"prompt contains static-scene suppression phrase(s): {forbidden}")
    if len(normalized.split()) < 8:
        raise ValueError("prompt is too short to encode subject, trigger, and ordered event")
    return normalized


def prepare_world2_case(source_case: str | Path, output: str | Path, prompt: str) -> dict[str, Any]:
    normalized = validate_event_prompt(prompt)
    receipt = prepare_case(source_case, output, normalized, MODEL_ID)
    validate_input_contract(Path(output) / "input_contract.json")
    return receipt


def validate_input_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != INPUT_VERSION:
        raise ValueError("unsupported LingBot-World2 input contract schema")
    if contract.get("model_id") != MODEL_ID:
        raise ValueError("input contract is not pinned to the 14B causal-fast model")
    sampling = contract.get("sampling", {})
    expected = {
        "task": TASK,
        "infer_mode": INFER_MODE,
        "sample_shift": 10.0,
        "fps": 16,
        "output_wh": [832, 464],
        "chunk_size": 4,
    }
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(f"input contract has invalid 14B sampling field {key!r}")
    frame_num = sampling.get("frame_num")
    if (not isinstance(frame_num, int) or frame_num < 13
            or (frame_num - 1) % 4
            or ((frame_num - 1) // 4 + 1) % 4):
        raise ValueError("frame_num must contain complete causal chunks")
    conditioning = contract.get("conditioning", {})
    prompt = validate_event_prompt(conditioning.get("prompt"))
    if conditioning.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError("prompt digest mismatch")
    base = contract_path.parent
    for field in ("image", "poses", "intrinsics"):
        relative = conditioning.get(field)
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"invalid conditioning path: {field}")
        artifact = (base / relative).resolve()
        if artifact.parent != base:
            raise ValueError(f"conditioning path escapes input bundle: {field}")
        if digest(artifact) != conditioning.get(f"{field}_sha256"):
            raise ValueError(f"conditioning digest mismatch: {field}")
    return contract


def runtime_identity(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    upstream = root / "code" / "lingbot-world-v2"
    checkpoint = root / "models" / "lingbot-world-v2-14b-causal-fast"
    python = root / "envs" / "lingbot-world-v2" / "bin" / "python"
    if not python.is_file():
        raise ValueError("LingBot-World2 environment is missing")
    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    if commit != UPSTREAM_COMMIT or dirty:
        raise ValueError("LingBot-World2 upstream checkout is not the pinned clean commit")
    manifest_path = checkpoint / "download_manifest.json"
    if digest(manifest_path) != DOWNLOAD_MANIFEST_SHA256:
        raise ValueError("checkpoint download manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("model_id") != MODEL_ID:
        raise ValueError("checkpoint download manifest is not the pinned complete model")
    if manifest.get("hf_revision") != HF_REVISION:
        raise ValueError("checkpoint Hugging Face revision mismatch")
    verified = manifest.get("verified_sha256", {})
    for relative, record in verified.items():
        artifact = checkpoint / relative
        if not artifact.is_file() or artifact.stat().st_size != record.get("size"):
            raise ValueError(f"checkpoint artifact missing or wrong size: {relative}")
    if digest(checkpoint / "config.json") != CONFIG_SHA256:
        raise ValueError("checkpoint config digest mismatch")
    identity = {
        "schema_version": "lingbot_world2_runtime_identity.v1",
        "profile_id": PROFILE_ID,
        "protocol": PROTOCOL,
        "model_id": MODEL_ID,
        "task": TASK,
        "infer_mode": INFER_MODE,
        "upstream_commit": commit,
        "checkpoint_hf_revision": HF_REVISION,
        "download_manifest_sha256": DOWNLOAD_MANIFEST_SHA256,
        "config_sha256": CONFIG_SHA256,
        "checkpoint": str(checkpoint),
        "upstream": str(upstream),
        "python": str(python),
    }
    identity["identity_sha256"] = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-case", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--prompt", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--input-contract", required=True)
    args = parser.parse_args()
    result = (
        prepare_world2_case(args.source_case, args.output, args.prompt)
        if args.command == "prepare"
        else validate_input_contract(args.input_contract)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
