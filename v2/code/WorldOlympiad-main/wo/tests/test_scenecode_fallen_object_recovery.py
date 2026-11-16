import json
from pathlib import Path
from types import SimpleNamespace

from autoresearch.robotics_scenecode_case_generator import (
    restore_validated_fallen_task_objects,
)


TASK_DESCRIPTION = (
    "An ordinary rigid ceramic countertop fruit bowl with a shallow rotational "
    "profile, stable thick flat base, continuous side wall, rounded rim, and a "
    "real concave cavity in visual and collision geometry. Empty initially and "
    "visibly contains the apple after stable release."
)
ASSET_DESCRIPTION = (
    "Ordinary rigid ceramic countertop fruit bowl, shallow rotational profile, "
    "0.24 m wide and deep and 0.07 m tall, stable thick flat base, continuous "
    "gently curved side wall and rounded rim, visibly open real concave cavity; "
    "empty, matte neutral warm white ceramic."
)


def _fallen_bowl_scene(tmp_path: Path, *, log_text: str, asset_name: str = "fruit_bowl"):
    scene_root = tmp_path / "scene_run" / "scene_000"
    room_root = scene_root / "room_robotics_workspace"
    snapshot_dir = room_root / "scene_states" / "before_drop"
    snapshot_dir.mkdir(parents=True)
    sdf_path = room_root / "generated_assets" / "bowl.sdf"
    sdf_path.parent.mkdir(parents=True)
    sdf_path.write_text("<sdf version='1.7'/>", encoding="utf-8")
    (room_root / "room.log").write_text(log_text, encoding="utf-8")

    support = {
        "object_type": "furniture",
        "support_surfaces": [
            {
                "surface_id": "S_0",
                "transform": {"translation": [0.0, 0.0, 0.9]},
            }
        ],
    }
    record = {
        "name": asset_name,
        "description": ASSET_DESCRIPTION,
        "bbox_min": [-0.12, -0.12, -0.035],
        "transform": {"translation": [0.24, -0.06, 1.4]},
        "sdf_path": "generated_assets/bowl.sdf",
    }
    (snapshot_dir / "scene_state.json").write_text(
        json.dumps({"objects": {"fruit_bowl_0": record}}), encoding="utf-8"
    )
    house_state = scene_root / "combined_house" / "house_state.json"
    house_state.parent.mkdir(parents=True)
    house_state.write_text(
        json.dumps(
            {
                "rooms": {
                    "robotics_workspace": {
                        "objects": {"work_counter_0": support}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    task = SimpleNamespace(
        case_id="test_case",
        objects=[
            SimpleNamespace(
                object_id="fruit_bowl",
                category="bowl",
                visual_description=TASK_DESCRIPTION,
            )
        ],
    )
    inventory = {
        "matches": [{"task_object_id": "fruit_bowl", "status": "missing"}]
    }
    return task, house_state, inventory


def test_restore_uses_room_validation_log_and_physical_identity(tmp_path):
    log_text = (
        f"Code-generated validation passed for '{ASSET_DESCRIPTION}': visual check passed.\n"
        f"Successfully converted asset: '{ASSET_DESCRIPTION}'.\n"
        "Removing fallen manipuland fruit_bowl_0: z=-122.78m\n"
    )
    task, house_state, inventory = _fallen_bowl_scene(tmp_path, log_text=log_text)

    report = restore_validated_fallen_task_objects(
        task=task, house_state=house_state, inventory=inventory
    )

    assert report["changed"] is True
    assert report["restored"][0]["scene_object_id"] == "fruit_bowl_0"
    published = json.loads(house_state.read_text(encoding="utf-8"))
    restored = published["rooms"]["robotics_workspace"]["objects"]["fruit_bowl_0"]
    assert restored["transform"]["translation"][2] == 0.937
    assert restored["metadata"]["worldolympiad_recovery"]["policy"] == (
        "validated_fallen_task_object_support_reprojection"
    )


def test_restore_still_requires_explicit_fall_and_conversion_evidence(tmp_path):
    log_text = f"validation passed for '{ASSET_DESCRIPTION}'\n"
    task, house_state, inventory = _fallen_bowl_scene(tmp_path, log_text=log_text)

    report = restore_validated_fallen_task_objects(
        task=task, house_state=house_state, inventory=inventory
    )

    assert report["changed"] is False
    assert report["restored"] == []
    assert report["skipped"][0]["reason"] == (
        "no previously-live validated fallen asset is recoverable"
    )


def test_restore_does_not_match_same_category_without_task_identity(tmp_path):
    generic_description = "Ordinary ceramic bowl, shallow profile, empty."
    log_text = (
        f"validation passed for '{generic_description}'\n"
        f"Successfully converted asset: '{generic_description}'\n"
        "Removing fallen manipuland bowl_0\n"
    )
    task, house_state, inventory = _fallen_bowl_scene(
        tmp_path, log_text=log_text, asset_name="bowl"
    )
    # The generated asset description itself does not name fruit, so it cannot
    # stand in for this specifically identified task object.
    snapshot = (
        house_state.parent.parent
        / "room_robotics_workspace/scene_states/before_drop/scene_state.json"
    )
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    candidate = data["objects"].pop("fruit_bowl_0")
    candidate["description"] = generic_description
    data["objects"]["bowl_0"] = candidate
    snapshot.write_text(json.dumps(data), encoding="utf-8")

    report = restore_validated_fallen_task_objects(
        task=task, house_state=house_state, inventory=inventory
    )

    assert report["changed"] is False
    assert report["restored"] == []
