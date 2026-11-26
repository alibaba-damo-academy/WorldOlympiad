from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.package_robotics_reference import _operator_contact_render
from scripts.package_robotics_reference import _role_visible
from scripts.package_robotics_reference import _certified_same_asset_aliases
from autoresearch.wam_adapter_contracts import ReferenceRoleRequirement
from autoresearch.robotics_views import sha256_file


def test_same_asset_alias_requires_inventory_identity_and_asset_digest(tmp_path: Path) -> None:
    table = tmp_path / "table.sdf"
    bottle = tmp_path / "bottle.sdf"
    table.write_text("<sdf>table</sdf>")
    bottle.write_text("<sdf>bottle</sdf>")
    rows = [
        {"task_object_id": name, "scene_object_id": scene,
         "sdf_path": str(path), "sdf_sha256": sha256_file(path)}
        for name, scene, path in (
            ("table", "table_0", table),
            ("table_surface", "table_0", table),
            ("bottle", "bottle_0", bottle),
        )
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"passed": True, "case_id": "case",
                                     "matches": rows}))
    generation = tmp_path / "generation.json"
    generation.write_text(json.dumps({
        "task_inventory_certificate": str(inventory),
        "task_inventory_certificate_sha256": sha256_file(inventory),
    }))
    task = SimpleNamespace(
        case_id="case",
        objects=[SimpleNamespace(object_id=row["task_object_id"]) for row in rows],
    )
    aliases = _certified_same_asset_aliases(generation, task)
    assert len(aliases) == 1
    assert aliases[0]["object_ids"] == ["table", "table_surface"]
    bottle.write_text("changed")
    with pytest.raises(ValueError, match="asset changed"):
        _certified_same_asset_aliases(generation, task)


def test_native_visibility_allows_cropped_planar_destination_only() -> None:
    cropped = {"linear_fraction": 0.87, "fully_in_frame": False}
    destination = ReferenceRoleRequirement(
        role="destination", minimum_linear_fraction=0.02,
        require_complete_frame=False,
    )
    manipuland = ReferenceRoleRequirement(
        role="manipuland", minimum_linear_fraction=0.02,
        require_complete_frame=True,
    )
    assert _role_visible(cropped, destination)
    assert not _role_visible(cropped, manipuland)
    assert not _role_visible(
        {"linear_fraction": 0.01, "fully_in_frame": True}, destination
    )


def test_articulated_release_masks_are_excluded_with_source_bound_receipt(tmp_path: Path) -> None:
    trajectory = {"frames": [
        {"frame": 1, "phase": "contact"},
        {"frame": 2, "phase": "actuate"},
        {"frame": 3, "phase": "release"},
    ]}
    render = {"views": {
        name: {"semantic_masks": [{"frame": frame} for frame in (1, 2, 3)]}
        for name in ("top", "exterior_left", "exterior_right")
    }}
    trajectory_path = tmp_path / "trajectory.json"
    render_path = tmp_path / "render.json"
    trajectory_path.write_text(json.dumps(trajectory))
    render_path.write_text(json.dumps(render))
    selected = _operator_contact_render(
        trajectory, render, operator="pull", trajectory_path=trajectory_path,
        render_path=render_path, output=tmp_path,
    )
    assert [item["frame"] for item in selected["views"]["top"]["semantic_masks"]] == [1, 2]
    assert [item["frame"] for item in render["views"]["top"]["semantic_masks"]] == [1, 2, 3]
    receipt = json.loads((tmp_path / "semantic_evidence_selection.json").read_text())
    assert receipt["excluded_frames"] == [{"frame": 3, "phase": "release"}]
    assert receipt["retained_frame_count"] == 2


def test_articulated_mask_selection_rejects_view_disagreement(tmp_path: Path) -> None:
    trajectory = {"frames": [{"frame": 1, "phase": "contact"}]}
    render = {"views": {
        "top": {"semantic_masks": [{"frame": 1}]},
        "exterior_left": {"semantic_masks": []},
        "exterior_right": {"semantic_masks": [{"frame": 1}]},
    }}
    with pytest.raises(ValueError, match="disagree"):
        _operator_contact_render(
            trajectory, render, operator="pull", trajectory_path=tmp_path / "t",
            render_path=tmp_path / "r", output=tmp_path,
        )
