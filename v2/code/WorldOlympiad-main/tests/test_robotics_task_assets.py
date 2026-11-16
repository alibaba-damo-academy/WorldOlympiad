import pytest

from autoresearch.robotics_task_assets import resolve_task_assets
from autoresearch.io import write_json as atomic_json, sha256_file as sha


def test_candidate_asset_api_resolves_any_declared_object_ids(tmp_path):
    outputs = tmp_path / "outputs"
    generation = outputs / "robotics_scenecode_generations" / "batch" / "case" / "generation_manifest.json"
    generation.parent.mkdir(parents=True)
    objects = {}
    matches = []
    for object_id, role in (("odd_source", "manipuland"), ("fixture_7", "support")):
        asset = generation.parent / "assets" / object_id / "model.sdf"
        asset.parent.mkdir(parents=True)
        asset.write_text(f"<sdf><model name='{object_id}'/></sdf>")
        objects[object_id] = asset
        matches.append({"task_object_id": object_id, "role": role,
                        "sdf_path": str(asset), "sdf_sha256": sha(asset)})
    inventory = generation.parent / "task_inventory_certificate.json"
    atomic_json(inventory, {"passed": True, "matches": matches})
    atomic_json(generation, {"passed": True,
        "task_inventory_certificate_sha256": sha(inventory)})
    urdf = outputs / "robot_assets" / "registered" / "robot.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot name='registered'/>")
    atomic_json(urdf.parent / "robot_asset_manifest.json", {
        "robot_embodiment": "robolab_droid", "generated_urdf": str(urdf),
        "generated_urdf_sha256": sha(urdf),
    })

    resolved = resolve_task_assets(workspace=tmp_path, generation=generation)
    assert set(resolved.sdf_by_object_id) == {"odd_source", "fixture_7"}
    assert resolved.robot_urdf == str(urdf)
    assert resolved.inventory_sha256 == sha(inventory)


def test_candidate_asset_api_fails_closed_when_selected_sdf_changes(tmp_path):
    outputs = tmp_path / "outputs"
    generation = outputs / "robotics_scenecode_generations" / "batch" / "case" / "generation_manifest.json"
    generation.parent.mkdir(parents=True)
    asset = generation.parent / "assets" / "object_x" / "model.sdf"
    asset.parent.mkdir(parents=True)
    asset.write_text("<sdf/>")
    inventory = generation.parent / "task_inventory_certificate.json"
    atomic_json(inventory, {"passed": True, "matches": [{
        "task_object_id": "object_x", "sdf_path": str(asset), "sdf_sha256": sha(asset),
    }]})
    atomic_json(generation, {"passed": True,
        "task_inventory_certificate_sha256": sha(inventory)})
    asset.write_text("<sdf>changed</sdf>")
    with pytest.raises(ValueError, match="asset changed"):
        resolve_task_assets(workspace=tmp_path, generation=generation)
