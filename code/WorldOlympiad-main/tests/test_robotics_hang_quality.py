import json
from pathlib import Path

import cv2
import numpy as np

from autoresearch.robotics_quality import certify_reference_physical_render


def _mask(path: Path) -> Path:
    mask = np.zeros((120, 180), dtype=np.uint8)
    cv2.rectangle(mask, (80, 50), (100, 70), 1, -1)
    cv2.rectangle(mask, (72, 52), (77, 68), 2, -1)
    cv2.rectangle(mask, (103, 52), (108, 68), 3, -1)
    assert cv2.imwrite(str(path), mask)
    return path


def _render(tmp_path: Path, frames: list[dict]) -> dict:
    views = {}
    for view in ("top", "exterior_left", "exterior_right"):
        views[view] = {
            "semantic_masks": [
                {"frame": row["frame"], "image": str(_mask(tmp_path / f"{view}_{row['frame']}.png"))}
                for row in frames
            ]
        }
    return {
        "views": views,
        "robot_registration": {"maximum_finger_translation_residual_m": 0.0},
    }


def _simulation() -> dict:
    checks = {
        "lifted_by_robot": True,
        "body_grasp": True,
        "dynamic_subject": True,
        "no_object_attachment": True,
        "released": True,
        "unsupported_by_table": True,
        "handle_supported": True,
        "hook_through_handle": True,
        "only_handle_support": True,
        "above_table_plane": True,
        "stable": True,
    }
    return {
        "success": True,
        "operator": "hang",
        "success_predicate": "hung",
        "grasp_contact_left_ever": True,
        "grasp_contact_right_ever": True,
        "object_attachment_constraint_count": 0,
        "expected_gripper_coupling_constraint_count": 0,
        "gripper_coupling_constraint_count": 0,
        "dual_finger_supported_carry_fraction": 1.0,
        "opposed_external_carry_fraction": 1.0,
        "hung_physics_certificate": {
            "schema_version": "robotics_hung_physics_certificate.v2",
            "passed": True,
            "checks": checks,
            "stability_definition": "sustained handle-hook capture",
        },
    }


def test_hang_gate_requires_the_physical_hung_certificate(tmp_path: Path) -> None:
    frames = [{"frame": i, "phase": phase} for i, phase in enumerate(("lift", "transfer", "lower"))]
    simulation = _simulation()
    simulation["hung_physics_certificate"]["checks"]["hook_through_handle"] = False
    path = certify_reference_physical_render(
        {"frames": frames}, simulation, _render(tmp_path, frames), tmp_path / "physical.json"
    )
    report = json.loads(path.read_text())
    assert report["passed"] is False
    assert report["simulation_checks"]["all_hang_checks_pass"] is False


def test_hang_gate_accepts_registered_grasp_and_hung_evidence(tmp_path: Path) -> None:
    frames = [{"frame": i, "phase": phase} for i, phase in enumerate(("lift", "transfer", "lower"))]
    path = certify_reference_physical_render(
        {"frames": frames}, _simulation(), _render(tmp_path, frames), tmp_path / "physical.json"
    )
    report = json.loads(path.read_text())
    assert report["passed"] is True
    assert report["simulation_checks"]["handle_hook_engagement"] is True
