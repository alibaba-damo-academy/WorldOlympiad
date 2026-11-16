"""Predeclared shared physical-time windows; native success is diagnostic."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

POLICY = "fixed_common_physical_window.v1"


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def from_reference(reference_path: Path, reference, view_id: str) -> dict:
    view = reference.view(view_id)
    frames, fps = int(view.frame_count), float(view.fps)
    if frames < 2 or fps != 15.:
        raise ValueError("matched DROID window requires a certified 15 Hz reference")
    from .io import sha256_file
    return {"policy": POLICY, "control_hz": fps, "max_steps": frames-1,
            "frame_count": frames, "physical_duration_s": (frames-1)/fps,
            "reference_manifest_sha256": sha256_file(reference_path),
            "reference_view_id": view_id, "reference_video_sha256": view.sha256,
            "success_ends_episode": False,
            "model_time_rescaled": False, "reference_time_rescaled": False}


def validate(contract: dict) -> None:
    if (contract.get("policy") != POLICY or contract.get("control_hz") != 15
            or contract.get("success_ends_episode") is not False
            or contract.get("model_time_rescaled") is not False
            or contract.get("reference_time_rescaled") is not False
            or not isinstance(contract.get("max_steps"), int)
            or isinstance(contract.get("max_steps"), bool) or contract["max_steps"] < 1
            or contract.get("frame_count") != contract["max_steps"]+1
            or abs(contract.get("physical_duration_s", -1)-contract["max_steps"]/15) > 1e-8):
        raise ValueError("invalid shared physical-time evaluation window")


def validate_capture(metadata: dict, contract: dict) -> None:
    validate(contract)
    if (metadata.get("evaluation_window_sha256") != digest(contract)
            or metadata.get("termination_reason") != "evaluation_window_complete"
            or metadata.get("frame_count") != contract["frame_count"]
            or metadata.get("action_count") != contract["max_steps"]
            or not isinstance(metadata.get("task_success"), bool)):
        raise ValueError("capture does not complete its bound evaluation window")
    first = metadata.get("completion_frame")
    if first is not None and (not isinstance(first, int) or isinstance(first, bool)
                              or not 0 <= first <= contract["max_steps"]):
        raise ValueError("diagnostic first-success frame is outside the window")


def validate_binding(metadata: dict, simulator_state: dict) -> dict | None:
    """Reject both dropped and injected contracts while accepting old captures."""
    bound = simulator_state.get("evaluation_window")
    captured = metadata.get("evaluation_window")
    if bound != captured:
        raise ValueError("native evaluation window differs from the certified input")
    if bound is not None:
        validate_capture(metadata, bound)
        if metadata.get("official_task_max_steps") != bound["max_steps"]:
            raise ValueError("runtime episode budget differs from its shared window")
        for record in metadata.get("windows", []):
            if record.get("executed_action_count") != record.get("planned_action_count"):
                raise ValueError("shared evaluation window stopped an action chunk early")
    return bound
