from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess

import pytest
import autoresearch.robotics_scenecode_case_generator as generator

from autoresearch.robotics_contracts import TaskInstructionSpec
from autoresearch.robotics_scenecode_case_generator import (
    ROBOTICS_SCENECODE_OVERRIDES,
    _declared_contents_outside,
    _tokens,
    _preserve_completed_generation,
    _preserve_partial_generation,
    _partial_resume_stage,
    _validate_recovery_candidate,
    _validate_partial_recovery_candidate,
    can_retry_inventory_at_manipuland_stage,
    certify_task_inventory,
    classify_generation_exception,
    completed_despite_broken_stdout,
    collect_failure_log_tails,
    CODE_GENERATED_RETRIES,
    find_unvalidated_task_assets,
    generation_recovery_path,
    generate_case_with_bounded_inventory_resume,
    partial_generation_recovery_path,
    prune_unreferenced_generated_assets,
    remove_staging_latest_run_link,
    restore_validated_fallen_task_objects,
    rewrite_published_references,
    runtime_executable,
    separate_prefilled_container_task_objects,
    write_final_inventory_binding,
)


def test_robotics_generation_disables_optional_context_furniture_vlm() -> None:
    assert ROBOTICS_SCENECODE_OVERRIDES == (
        "floor_plan_agent.materials.generator.enabled=false",
        "manipuland_agent.context_furniture.enabled=false",
        "furniture_agent.context_image_generation.enabled=false",
        "manipuland_agent.context_image_generation.enabled=false",
        "furniture_agent.asset_manager.router.strategies.code_generated"
        f".max_retries={CODE_GENERATED_RETRIES}",
        "manipuland_agent.asset_manager.router.strategies.code_generated"
        f".max_retries={CODE_GENERATED_RETRIES}",
        "furniture_agent.max_critique_rounds=0",
        "wall_agent.max_critique_rounds=0",
        "ceiling_agent.max_critique_rounds=0",
        "manipuland_agent.max_critique_rounds=0",
        "wall_agent.asset_manager.router.strategies.code_generated.max_retries=1",
        "wall_agent.asset_manager.code_object.max_attempts=1",
        "ceiling_agent.asset_manager.router.strategies.code_generated.max_retries=1",
        "ceiling_agent.asset_manager.code_object.max_attempts=1",
    )


def test_signal_interruption_is_not_misclassified_as_prompt_semantics() -> None:
    classification, owner = classify_generation_exception(
        stage="scenecode_generation",
        exception=subprocess.CalledProcessError(-signal.SIGTERM, ["scenecode"]),
    )
    assert classification == "runtime_interrupted"
    assert owner == "scenecode_runtime"


def test_python_broken_pipe_exit_is_runtime_interruption() -> None:
    classification, owner = classify_generation_exception(
        stage="scenecode_generation",
        exception=subprocess.CalledProcessError(120, ["scenecode"]),
    )
    assert classification == "runtime_interrupted"
    assert owner == "scenecode_runtime"


def test_inventory_normalizes_single_and_one_as_the_same_quantity() -> None:
    assert _tokens("white block with one vertical hole") == _tokens(
        "white block with a single vertical hole"
    )


def test_broken_stdout_is_accepted_only_after_terminal_scene_state(tmp_path: Path) -> None:
    scene_run = tmp_path / "scene_run"
    (scene_run / "scene_000" / "combined_house").mkdir(parents=True)
    exception = subprocess.CalledProcessError(120, ["scenecode"])
    assert not completed_despite_broken_stdout(
        scene_run=scene_run, exception=exception
    )
    (scene_run / "scene_000" / "combined_house" / "house_state.json").write_text(
        "{}\n", encoding="utf-8"
    )
    assert not completed_despite_broken_stdout(
        scene_run=scene_run, exception=exception
    )
    (scene_run / "main.log").write_text(
        "ALL SCENES COMPLETED!\n", encoding="utf-8"
    )
    assert completed_despite_broken_stdout(
        scene_run=scene_run, exception=exception
    )
    assert not completed_despite_broken_stdout(
        scene_run=scene_run,
        exception=subprocess.CalledProcessError(1, ["scenecode"]),
    )


def test_provider_503_is_not_misclassified_as_prompt_semantics() -> None:
    classification, owner = classify_generation_exception(
        stage="scenecode_generation",
        exception=subprocess.CalledProcessError(1, ["scenecode"]),
        log_tails={
            "scene_run/scene.log": (
                "openai.InternalServerError: Error code: 503 - "
                "Service temporarily unavailable"
            )
        },
    )
    assert classification == "provider_unavailable"
    assert owner == "model_provider"


def test_failed_staging_preserves_bounded_sanitized_log_tails(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".case.partial.123"
    log = staging / "scene_run" / "scene.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join([f"line {index}" for index in range(60)])
        + f"\nBroken pipe at {staging}\n",
        encoding="utf-8",
    )

    tails = collect_failure_log_tails(
        staging=staging, max_lines_per_file=4, max_total_chars=500
    )

    assert list(tails) == ["scene_run/scene.log"]
    assert "line 58" in tails["scene_run/scene.log"]
    assert "Broken pipe" in tails["scene_run/scene.log"]
    assert str(staging) not in tails["scene_run/scene.log"]
    assert "<discarded-staging>" in tails["scene_run/scene.log"]


def test_partial_generation_recovery_binds_task_prompt_and_checkpoint(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".case.partial.123"
    checkpoint = (
        staging
        / "scene_run"
        / "scene_000"
        / "room_workspace"
        / "scene_states"
        / "scene_after_ceiling_objects"
        / "scene_state.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{"checkpoint":true}\n')
    recovery = partial_generation_recovery_path(
        root=tmp_path, batch_id="batch", case_id="case"
    )

    result = _preserve_partial_generation(
        staging=staging,
        recovery=recovery,
        task_sha256="task-sha",
        prompt_sha256="prompt-sha",
    )
    assert result == recovery
    assert not staging.exists()
    assert _validate_partial_recovery_candidate(
        recovery=recovery,
        task_sha256="task-sha",
        prompt_sha256="prompt-sha",
    ) == "manipuland"
    with pytest.raises(ValueError, match="another selected task"):
        _validate_partial_recovery_candidate(
            recovery=recovery,
            task_sha256="other-task",
            prompt_sha256="prompt-sha",
        )


def test_furniture_checkpoint_recovers_at_wall_stage(tmp_path: Path) -> None:
    staging = tmp_path / ".case.partial.456"
    checkpoint = (
        staging
        / "scene_run"
        / "scene_000"
        / "room_workspace"
        / "scene_states"
        / "scene_after_furniture"
        / "scene_state.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{"checkpoint":true}\n')
    assert _partial_resume_stage(staging / "scene_run") == "wall_mounted"
    recovery = partial_generation_recovery_path(
        root=tmp_path, batch_id="batch", case_id="furniture_case"
    )
    _preserve_partial_generation(
        staging=staging,
        recovery=recovery,
        task_sha256="task-sha",
        prompt_sha256="prompt-sha",
    )
    assert _validate_partial_recovery_candidate(
        recovery=recovery,
        task_sha256="task-sha",
        prompt_sha256="prompt-sha",
    ) == "wall_mounted"


def test_complete_floor_plan_can_resume_before_furniture_without_regenerating_room(tmp_path):
    scene=tmp_path/'scene_000';scene.mkdir()
    (scene/'house_layout.json').write_text(json.dumps({'rooms':[{'id':'workspace'}]}))
    for directory in ['room_geometry','floor_plans']:
        path=scene/directory;path.mkdir();(path/'native_artifact.json').write_text('{}')
    assert _partial_resume_stage(tmp_path)=='furniture'
    later=scene/'room_workspace/scene_states/scene_after_furniture/scene_state.json'
    later.parent.mkdir(parents=True);later.write_text('{}')
    assert _partial_resume_stage(tmp_path)=='wall_mounted'


def test_layout_json_alone_is_not_a_resumable_floor_plan(tmp_path):
    scene=tmp_path/'scene_000';scene.mkdir()
    (scene/'house_layout.json').write_text(json.dumps({'rooms':[{'id':'workspace'}]}))
    assert _partial_resume_stage(tmp_path) is None


def test_final_attempt_without_validation_blocks_only_matching_task_assets(
    tmp_path: Path,
) -> None:
    scene_run = tmp_path / "scene_run"
    room = scene_run / "scene_000" / "room_workspace"
    room.mkdir(parents=True)
    (room / "room.log").write_text(
        "Code-generated final attempt accepted without validation for "
        "'Minimal round wall clock'\n"
        "Code-generated final attempt accepted without validation for "
        "'matte red block'\n",
        encoding="utf-8",
    )

    findings = find_unvalidated_task_assets(task=_task(), scene_run=scene_run)

    assert findings == [
        {
            "task_object_id": "red_block",
            "category": "block",
            "visual_description": "matte red block",
            "unvalidated_asset_description": "matte red block",
        },
    ]


def _task() -> TaskInstructionSpec:
    return TaskInstructionSpec(
        case_id="sort_blocks",
        instruction="Place each colored block in its matching bin.",
        scene_brief="A red block, blue block, matching bins, and worktable.",
        initial_state="Both blocks start outside the bins.",
        objects=[
            {
                "object_id": "red_block",
                "category": "block",
                "role": "manipuland",
                "visual_description": "matte red block",
            },
            {
                "object_id": "blue_block",
                "category": "block",
                "role": "manipuland",
                "visual_description": "matte blue block",
            },
            {
                "object_id": "red_bin",
                "category": "bin",
                "role": "destination",
                "visual_description": "low red bin",
            },
            {
                "object_id": "blue_bin",
                "category": "bin",
                "role": "destination",
                "visual_description": "low blue bin",
            },
            {
                "object_id": "table",
                "category": "worktable",
                "role": "support",
                "visual_description": "solid oak worktable",
            },
        ],
        action_program=[
            {"operation": "pick", "subjects": ["red_block"]},
            {"operation": "place", "subjects": ["red_block"], "target": "red_bin"},
            {"operation": "pick", "subjects": ["blue_block"]},
            {"operation": "place", "subjects": ["blue_block"], "target": "blue_bin"},
        ],
        success_predicates=[
            {"kind": "inside", "subject": "red_block", "object": "red_bin"},
            {"kind": "inside", "subject": "blue_block", "object": "blue_bin"},
        ],
    )


def _write_scene(tmp_path: Path, *, omit_blue_bin_sdf: bool = False) -> Path:
    scene_root = tmp_path / "scene_000"
    room_root = scene_root / "room_lab"
    assets = room_root / "generated_assets"
    assets.mkdir(parents=True)
    objects = {}
    for name in ["red_block", "blue_block", "red_bin", "blue_bin", "worktable"]:
        sdf = assets / f"{name}.sdf"
        if not (omit_blue_bin_sdf and name == "blue_bin"):
            sdf.write_text(f"<sdf version='1.9'><model name='{name}'/></sdf>\n")
        objects[f"asset_{name}"] = {
            "name": name.replace("_", " "),
            "description": f"generated {name.replace('_', ' ')}",
            "object_type": "manipuland" if "block" in name or "bin" in name else "furniture",
            "sdf_path": f"generated_assets/{name}.sdf",
            "metadata": {},
        }
    combined = scene_root / "combined_house"
    combined.mkdir()
    house_state = combined / "house_state.json"
    house_state.write_text(
        json.dumps({"rooms": {"lab": {"objects": objects}}}), encoding="utf-8"
    )
    return house_state


def test_inventory_matches_repeated_categories_by_color_and_never_reuses_asset(
    tmp_path: Path,
) -> None:
    report = certify_task_inventory(task=_task(), house_state=_write_scene(tmp_path))
    assert report["passed"] is True
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["red_block"]["scene_object_id"] == "asset_red_block"
    assert matches["blue_block"]["scene_object_id"] == "asset_blue_block"
    assert matches["red_bin"]["scene_object_id"] == "asset_red_bin"
    assert matches["blue_bin"]["scene_object_id"] == "asset_blue_bin"
    assert len({item["scene_object_id"] for item in report["matches"]}) == 5
    assert all("room_lab/generated_assets" in item["sdf_path"] for item in report["matches"])


def test_inventory_never_uses_color_overlap_as_category_compatibility(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    del payload["rooms"]["lab"]["objects"]["asset_red_bin"]
    house_state.write_text(json.dumps(payload), encoding="utf-8")
    base = _task()
    task = base.model_copy(
        update={
            "objects": [
                item.model_copy(
                    update={"category": "sorting_bin"}
                    if item.object_id.endswith("_bin")
                    else {}
                )
                for item in base.objects
            ]
        }
    )

    report = certify_task_inventory(task=task, house_state=house_state)

    assert report["passed"] is False
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["blue_block"]["scene_object_id"] == "asset_blue_block"
    assert matches["blue_bin"]["scene_object_id"] == "asset_blue_bin"
    assert matches["red_bin"]["status"] == "missing"
    assert matches["red_bin"]["required_category_tokens"] == ["bin", "sorting"]


def _write_articulated_scene(tmp_path: Path, *, jointless_only: bool = False) -> Path:
    """Scene with a token-matching static cabinet and a hinged cupboard."""

    scene_root = tmp_path / "scene_000"
    room_root = scene_root / "room_lab"
    assets = room_root / "generated_assets"
    assets.mkdir(parents=True)
    (assets / "cabinet.sdf").write_text(
        "<sdf version='1.9'><model name='cabinet'>"
        "<link name='body'><collision name='body_collision'><geometry>"
        "<box><size>0.5 0.5 0.8</size></box></geometry></collision></link>"
        "</model></sdf>\n"
    )
    cupboard = assets / "cupboard.sdf"
    cupboard.write_text(
        "<sdf version='1.9'><model name='white storage cupboard'>"
        "<link name='body'><collision name='body_collision'><geometry>"
        "<box><size>0.5 0.5 0.8</size></box></geometry></collision></link>"
        "<link name='front_door'><collision name='door_collision'><geometry>"
        "<box><size>0.4 0.03 0.6</size></box></geometry></collision></link>"
        "<joint name='front_door_hinge_joint' type='revolute'>"
        "<parent>body</parent><child>front_door</child>"
        "<axis><xyz>0 0 1</xyz><limit><lower>-1.5708</lower>"
        "<upper>0</upper></limit></axis></joint>"
        "</model></sdf>\n"
    )
    objects: dict[str, object] = {}
    if not jointless_only:
        objects["asset_cupboard"] = {
            "name": "white storage cupboard",
            "description": "white storage cupboard with a hinged front door",
            "object_type": "furniture",
            "sdf_path": "generated_assets/cupboard.sdf",
            "metadata": {},
        }
    objects["asset_cabinet"] = {
        "name": "cabinet",
        "description": "generated cabinet",
        "object_type": "furniture",
        "sdf_path": "generated_assets/cabinet.sdf",
        "metadata": {},
    }
    objects["asset_worktable"] = {
        "name": "worktable",
        "description": "generated worktable",
        "object_type": "furniture",
        "support_surfaces": [{"name": "top"}],
        "sdf_path": "generated_assets/cabinet.sdf",
        "metadata": {},
    }
    combined = scene_root / "combined_house"
    combined.mkdir()
    house_state = combined / "house_state.json"
    house_state.write_text(
        json.dumps({"rooms": {"lab": {"objects": objects}}}), encoding="utf-8"
    )
    return house_state


def _articulated_task() -> TaskInstructionSpec:
    return TaskInstructionSpec.model_validate(
        {
            "case_id": "open_cabinet",
            "instruction": "Open the cupboard door.",
            "scene_brief": "A white storage cupboard and a worktable.",
            "initial_state": "The cupboard door is closed.",
            "objects": [
                {
                    "object_id": "cupboard",
                    "category": "cabinet",
                    "role": "manipuland",
                    "visual_description": (
                        "white storage cupboard with a hinged front door"
                    ),
                },
                {
                    "object_id": "table",
                    "category": "worktable",
                    "role": "support",
                    "visual_description": "solid worktable",
                },
            ],
            "articulation_components": [
                {
                    "component_id": "cupboard_door",
                    "host_object_id": "cupboard",
                    "semantic_kind": "hinged_door",
                    "joint_type": "revolute",
                    "visual_description": "flat white cupboard door",
                }
            ],
            "action_program": [
                {"operation": "pull", "subjects": ["cupboard_door"]}
            ],
            "success_predicates": [
                {
                    "kind": "articulated_open",
                    "subject": "cupboard_door",
                    "threshold": 0.9,
                }
            ],
        }
    )


def test_articulation_host_prefers_joint_qualified_asset_over_token_lookalike(
    tmp_path: Path,
) -> None:
    house_state = _write_articulated_scene(tmp_path)

    report = certify_task_inventory(task=_articulated_task(), house_state=house_state)

    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["cupboard"]["scene_object_id"] == "asset_cupboard"
    assert matches["cupboard"]["articulation_joint_qualified"] is True
    assert matches["cupboard"]["articulation_host_joint_types"] == ["revolute"]
    assert matches["cupboard"]["articulation_required_joint_type"] == "revolute"
    # Category-token overlap stays truthful even though the joint qualified it.
    assert matches["cupboard"]["category_token_overlap"] == []


def test_articulation_host_without_declared_joint_stays_unqualified(
    tmp_path: Path,
) -> None:
    house_state = _write_articulated_scene(tmp_path, jointless_only=True)

    report = certify_task_inventory(task=_articulated_task(), house_state=house_state)

    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["cupboard"]["scene_object_id"] == "asset_cabinet"
    assert matches["cupboard"]["articulation_joint_qualified"] is False
    assert matches["cupboard"]["articulation_host_joint_types"] == []


def test_primary_support_can_match_unique_load_bearing_furniture_structure(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    support = objects.pop("asset_worktable")
    objects["asset_surface"] = support
    support["name"] = "oak dining surface"
    support["description"] = "four leg oak dining surface"
    support["support_surfaces"] = [{"surface_id": "top"}]
    house_state.write_text(json.dumps(payload), encoding="utf-8")

    report = certify_task_inventory(task=_task(), house_state=house_state)

    assert report["passed"] is True
    match = next(
        item for item in report["matches"] if item["task_object_id"] == "table"
    )
    assert match["structural_support_fallback"] is True


def test_validated_fallen_task_asset_is_recovered_from_live_snapshot(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    fallen = objects.pop("asset_red_bin")
    fallen.update(
        {
            "description": "low red bin",
            "bbox_min": [-0.1, -0.1, -0.05],
            "bbox_max": [0.1, 0.1, 0.05],
            "transform": {
                "translation": [-0.25, 0.0, 0.9],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        }
    )
    objects["asset_worktable"]["support_surfaces"] = [
        {
            "surface_id": "S_0",
            "transform": {
                "translation": [0.0, 0.0, 1.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        }
    ]
    house_state.write_text(json.dumps(payload), encoding="utf-8")
    scene_root = house_state.parent.parent
    room_root = scene_root / "room_lab"
    snapshot = room_root / "scene_renders" / "pass" / "renders_001" / "scene_state.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        json.dumps({"objects": {"asset_red_bin": fallen}}), encoding="utf-8"
    )
    (scene_root / "scene.log").write_text(
        "Code-generated validation passed for 'low red bin'\n"
        "Successfully converted asset: 'low red bin'\n",
        encoding="utf-8",
    )
    (room_root / "room.log").write_text(
        "Removing fallen manipuland asset_red_bin: bottom_z=-0.1m\n",
        encoding="utf-8",
    )
    task = _task()
    before = certify_task_inventory(task=task, house_state=house_state)
    assert before["passed"] is False

    recovery = restore_validated_fallen_task_objects(
        task=task,
        house_state=house_state,
        inventory=before,
    )

    assert recovery["changed"] is True
    assert recovery["restored"][0]["task_object_id"] == "red_bin"
    repaired = json.loads(house_state.read_text(encoding="utf-8"))
    restored = repaired["rooms"]["lab"]["objects"]["asset_red_bin"]
    assert restored["transform"]["translation"][2] == pytest.approx(1.052)
    assert certify_task_inventory(task=task, house_state=house_state)["passed"] is True


def test_declared_contents_inside_survives_unrelated_container_beside_clause() -> None:
    """A container-level "beside" clause must not move declared-inside contents.

    SceneCode composite expansion uses this predicate to decide whether member
    assets are expanded in place (contents stay inside the container) or
    re-solved separately onto the support. An unrelated statement that the
    *container* sits beside some other object must not override an explicit
    declaration that the contents are inside it.
    """

    state = (
        "The small open container sits beside the large bin on the table, "
        "and the three cubes are fully inside the small container."
    )

    assert _declared_contents_outside(state) is False


def test_declared_contents_inside_keeps_transfer_and_outside_states() -> None:
    """The heuristic still separates genuinely outside or negated states."""

    transfer_state = (
        "A small open red plastic container holding three rigid blue wooden cubes "
        "sits separately from a large open gray plastic bin on a light wooden table. "
        "The cubes are fully inside the small container, and the large bin is empty."
    )
    pick_place_state = (
        "The red bottle stands upright on the tabletop beside the basket and is "
        "not inside it; the basket rests open and empty on the same tabletop; "
        "the task is not solved."
    )

    assert _declared_contents_outside(transfer_state) is False
    assert _declared_contents_outside(pick_place_state) is True
    assert _declared_contents_outside("Both blocks start outside the bins.") is True


@pytest.mark.parametrize("invalid_asset", [None, "same_sdf", "same_sdf_hardlink", "duplicate_task_member"])
def test_prefilled_container_composite_is_separated_into_unsolved_assets(
    tmp_path: Path, invalid_asset: str | None,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    objects.pop("asset_red_block")
    objects.pop("asset_red_bin")
    objects["asset_worktable"]["support_surfaces"] = [
        {
            "surface_id": "S_0",
            "transform": {
                "translation": [0.0, 0.0, 1.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "bounding_box_min": [-1.0, -0.5, 0.0],
            "bounding_box_max": [1.0, 0.5, 0.2],
        }
    ]
    registry_root = house_state.parent.parent / "room_lab" / "generated_assets" / "manipuland"
    registry_root.mkdir(parents=True)
    bin_sdf = registry_root / "red_bin.sdf"
    bin_sdf.write_text("<sdf version='1.9'><model name='red_bin'/></sdf>\n")
    block_sdf = registry_root / "red_block.sdf"
    block_sdf.write_text("<sdf version='1.9'><model name='red_block'/></sdf>\n")
    (registry_root / "asset_registry.json").write_text(
        json.dumps(
            {
                "red_bin": {
                    "object_id": "red_bin",
                    "object_type": "manipuland",
                    "name": "red bin",
                    "description": "low red bin",
                    "sdf_path": str(bin_sdf),
                    "geometry_path": str(bin_sdf),
                    "scale_factor": 1.0,
                    "metadata": {"asset_source": "code_generated"},
                },
                "red_block": {
                    "object_id": "red_block",
                    "object_type": "manipuland",
                    "name": "red block",
                    "description": "matte red block",
                    "sdf_path": str(block_sdf),
                    "geometry_path": str(block_sdf),
                    "scale_factor": 1.0,
                    "metadata": {"asset_source": "code_generated"},
                },
            }
        ),
        encoding="utf-8",
    )
    (house_state.parent.parent / "scene.log").write_text(
        "Successfully converted asset: 'low red bin'\n"
        "Successfully converted asset: 'matte red block'\n",
        encoding="utf-8",
    )
    objects["filled_container_0"] = {
        "object_id": "filled_container_0",
        "object_type": "manipuland",
        "name": "filled_bin",
        "description": "bin filled with block",
        "geometry_path": None,
        "sdf_path": None,
        "transform": {
            "translation": [-0.2, 0.0, 1.01],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "placement_info": {"parent_surface_id": "S_0", "position_2d": [-0.2, 0.0]},
        "bbox_min": [-0.12, -0.12, -0.01],
        "bbox_max": [0.12, 0.12, 0.11],
        "metadata": {
            "composite_type": "filled_container",
            "container_asset": {
                "asset_id": "red_bin",
                "name": "red bin",
                "sdf_path": str(bin_sdf),
                "geometry_path": str(bin_sdf),
                "transform": {
                    "translation": [-0.2, 0.0, 1.01],
                    "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            },
            "fill_assets": [
                {
                    "asset_id": "red_block",
                    "name": "red block",
                    "sdf_path": str(block_sdf),
                    "geometry_path": str(block_sdf),
                    "transform": {
                        "translation": [-0.2, 0.0, 1.05],
                        "rotation_wxyz": [0.7071067811865476, 0.0, 0.0, 0.7071067811865475],
                    },
                }
            ],
        },
    }
    if invalid_asset == "duplicate_task_member":
        duplicate = dict(objects["filled_container_0"]["metadata"]["fill_assets"][0])
        objects["filled_container_0"]["metadata"]["fill_assets"].append(duplicate)
    elif invalid_asset in {"same_sdf", "same_sdf_hardlink"}:
        if invalid_asset == "same_sdf_hardlink":
            block_sdf.unlink()
            block_sdf.hardlink_to(bin_sdf)
        else:
            objects["filled_container_0"]["metadata"]["fill_assets"][0]["sdf_path"] = str(bin_sdf)
            registry_path = registry_root / "asset_registry.json"
            registry_data = json.loads(registry_path.read_text())
            registry_data["red_block"]["sdf_path"] = str(bin_sdf)
            registry_path.write_text(json.dumps(registry_data))
    house_state.write_text(json.dumps(payload), encoding="utf-8")
    base = _task()
    task = base.model_copy(
        update={
            "objects": [base.objects[0], base.objects[2], base.objects[4]],
            "action_program": [
                {"operation": "pick", "subjects": ["red_block"]},
                {"operation": "place", "subjects": ["red_block"], "target": "red_bin"},
            ],
            "success_predicates": [
                {"kind": "inside", "subject": "red_block", "object": "red_bin"}
            ],
        }
    )
    before = certify_task_inventory(task=task, house_state=house_state)
    assert before["passed"] is False

    report = separate_prefilled_container_task_objects(
        task=task, house_state=house_state, inventory=before
    )

    if invalid_asset is not None:
        assert report["changed"] is False
        assert report["skipped"]
        assert json.loads(house_state.read_text()) == payload
        return
    assert report["changed"] is True
    repaired = json.loads(house_state.read_text(encoding="utf-8"))["rooms"]["lab"]["objects"]
    assert "filled_container_0" not in repaired
    assert repaired["red_bin"]["sdf_path"] == str(bin_sdf)
    assert repaired["red_block"]["sdf_path"] == str(block_sdf)
    block_x = repaired["red_block"]["transform"]["translation"][0]
    assert block_x - (-0.2) > 0.25
    assert repaired["red_block"]["transform"]["translation"][2] == pytest.approx(1.01)
    after = certify_task_inventory(task=task, house_state=house_state)
    assert after["passed"] is True
    assert {value["scene_object_id"] for value in after["matches"]} >= {
        "red_bin",
        "red_block",
    }


def test_prefilled_separation_never_relabels_a_geometry_less_composite(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    objects.pop("asset_red_block")
    objects.pop("asset_red_bin")
    objects["filled_container_0"] = {
        "object_id": "filled_container_0",
        "object_type": "manipuland",
        "name": "filled_bin",
        "description": "bin filled with block",
        "geometry_path": None,
        "sdf_path": None,
        "transform": {
            "translation": [-0.2, 0.0, 1.01],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "metadata": {
            "composite_type": "filled_container",
            "container_asset": {"asset_id": "red_bin", "name": "red bin"},
            "fill_assets": [{"asset_id": "red_block", "name": "red block"}],
        },
    }
    house_state.write_text(json.dumps(payload), encoding="utf-8")
    base = _task()
    task = base.model_copy(
        update={
            "objects": [base.objects[0], base.objects[2], base.objects[4]],
            "action_program": [
                {"operation": "pick", "subjects": ["red_block"]},
                {"operation": "place", "subjects": ["red_block"], "target": "red_bin"},
            ],
            "success_predicates": [
                {"kind": "inside", "subject": "red_block", "object": "red_bin"}
            ],
        }
    )
    before = certify_task_inventory(task=task, house_state=house_state)
    assert before["passed"] is False

    report = separate_prefilled_container_task_objects(
        task=task, house_state=house_state, inventory=before
    )

    assert report["changed"] is False
    assert report["skipped"]
    repaired = json.loads(house_state.read_text(encoding="utf-8"))["rooms"]["lab"]["objects"]
    assert "filled_container_0" in repaired


def test_declared_support_region_destination_binds_to_matched_support(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    objects["asset_worktable"]["support_surfaces"] = [
        {"surface_id": "S_0", "transform": {"translation": [0.0, 0.0, 1.0]}}
    ]
    house_state.write_text(json.dumps(payload), encoding="utf-8")
    base = _task()
    task_payload = base.model_dump()
    task_payload.update(
        {
            "objects": [
                {
                    "object_id": "red_block",
                    "category": "block",
                    "role": "manipuland",
                    "visual_description": "matte red block",
                },
                {
                    "object_id": "wooden_table",
                    "category": "table",
                    "role": "support",
                    "visual_description": "a broad matte worktable surface",
                },
                {
                    "object_id": "table_surface",
                    "category": "table_region",
                    "role": "destination",
                    "visual_description": "the broad worktable surface where the block stands",
                },
            ],
            "action_program": [
                {"operation": "pick", "subjects": ["red_block"]},
                {"operation": "place", "subjects": ["red_block"], "target": "wooden_table"},
            ],
            "success_predicates": [
                {"kind": "on_region", "subject": "red_block", "object": "wooden_table"}
            ],
        }
    )
    task = TaskInstructionSpec.model_validate(task_payload)

    report = certify_task_inventory(task=task, house_state=house_state)

    assert report["passed"] is True
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["wooden_table"]["status"] == "matched"
    region = matches["table_surface"]
    assert region["status"] == "matched"
    assert region["asset_resolution_source"] == "declared_support_region"
    assert region["region_of_scene_object_id"] == "asset_worktable"
    assert region["sdf_path"] == matches["wooden_table"]["sdf_path"]
    assert report["scene_inventory_count"] == len(payload["rooms"]["lab"]["objects"])


@pytest.mark.parametrize("description", ["low red bin", "broad open-top red bin on the table surface"])
def test_non_region_destination_without_its_own_asset_stays_missing(
    tmp_path: Path, description: str,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    objects["asset_worktable"]["support_surfaces"] = [
        {"surface_id": "S_0", "transform": {"translation": [0.0, 0.0, 1.0]}}
    ]
    objects.pop("asset_red_bin")
    house_state.write_text(json.dumps(payload), encoding="utf-8")

    payload_task = _task().model_dump()
    next(item for item in payload_task["objects"] if item["object_id"] == "red_bin")["visual_description"] = description
    report = certify_task_inventory(task=TaskInstructionSpec.model_validate(payload_task), house_state=house_state)

    assert report["passed"] is False
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["red_bin"]["status"] == "missing"
    assert "region_of_scene_object_id" not in matches["red_bin"]


def test_inventory_treats_workbench_as_worktable_support_alias(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    support = objects.pop("asset_worktable")
    support["name"] = "workbench"
    support["description"] = "compact industrial workbench with broad top"
    objects["workbench_0"] = support
    house_state.write_text(json.dumps(payload), encoding="utf-8")

    report = certify_task_inventory(task=_task(), house_state=house_state)

    assert report["passed"] is True
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["table"]["scene_object_id"] == "workbench_0"


def test_inventory_treats_plain_table_as_worktable_support_alias(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path)
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    objects = payload["rooms"]["lab"]["objects"]
    support = objects.pop("asset_worktable")
    support["name"] = "table"
    support["description"] = "complete stable wooden table"
    objects["table_0"] = support
    house_state.write_text(json.dumps(payload), encoding="utf-8")

    report = certify_task_inventory(task=_task(), house_state=house_state)

    assert report["passed"] is True
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["table"]["scene_object_id"] == "table_0"


def test_completed_generation_moves_to_reusable_recovery_and_rewrites_paths(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".case.partial.123"
    house_state = staging / "scene_run" / "scene_000" / "combined_house" / "house_state.json"
    house_state.parent.mkdir(parents=True)
    house_state.write_text(
        json.dumps({"absolute_candidate": str(staging / "asset.sdf")}),
        encoding="utf-8",
    )
    (staging / "asset.sdf").write_text("<sdf/>", encoding="utf-8")
    recovery = generation_recovery_path(
        root=tmp_path, batch_id="batch", case_id="case"
    )

    preserved = _preserve_completed_generation(
        staging=staging,
        recovery=recovery,
        task_sha256="task-digest",
        prompt_sha256="prompt-digest",
        failed_stage="task_asset_inventory",
    )

    assert preserved == recovery
    assert not staging.exists()
    assert recovery.is_dir()
    rewritten = (recovery / house_state.relative_to(staging)).read_text(
        encoding="utf-8"
    )
    assert str(recovery / "asset.sdf") in rewritten
    assert str(staging) not in rewritten
    _validate_recovery_candidate(
        recovery=recovery,
        task_sha256="task-digest",
        prompt_sha256="prompt-digest",
    )


def test_inventory_fails_closed_when_a_task_asset_has_no_sdf(tmp_path: Path) -> None:
    report = certify_task_inventory(
        task=_task(), house_state=_write_scene(tmp_path, omit_blue_bin_sdf=True)
    )
    assert report["passed"] is False
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["blue_bin"]["status"] == "matched_without_sdf"


def test_inventory_retry_is_narrowed_only_for_late_stage_assets(tmp_path: Path) -> None:
    task = _task()
    destination_failure = certify_task_inventory(
        task=task, house_state=_write_scene(tmp_path, omit_blue_bin_sdf=True)
    )
    assert can_retry_inventory_at_manipuland_stage(
        task=task, inventory=destination_failure
    ) is True

    support_failure = json.loads(json.dumps(destination_failure))
    support_failure["matches"] = [
        {
            "task_object_id": "table",
            "status": "missing",
            "sdf_path": None,
        }
    ]
    assert can_retry_inventory_at_manipuland_stage(
        task=task, inventory=support_failure
    ) is False


def test_inventory_retry_treats_interactive_source_container_as_late_stage() -> None:
    task = TaskInstructionSpec(
        case_id="pour", instruction="Pour cubes into the bin.",
        scene_brief="A source container and bin sit on a table.",
        initial_state="Cubes begin in the source container.",
        objects=[
            {"object_id":"source","category":"open_container","role":"support",
             "visual_description":"small open red container"},
            {"object_id":"contents","category":"rigid_cubes","role":"manipuland",
             "visual_description":"three blue cubes"},
            {"object_id":"bin","category":"open_bin","role":"destination",
             "visual_description":"large gray bin"},
            {"object_id":"table","category":"table","role":"support",
             "visual_description":"wooden table"}],
        action_program=[{"operation":"pick","subjects":["contents"],"target":"source"},
                        {"operation":"place","subjects":["contents"],"target":"bin"}],
        success_predicates=[{"kind":"inside","subject":"contents","object":"bin"}],
    )
    inventory={"matches":[
        {"task_object_id":"source","status":"missing"},
        {"task_object_id":"contents","status":"missing"},
        {"task_object_id":"bin","status":"missing"},
        {"task_object_id":"table","status":"matched","sdf_path":"table.sdf"}]}
    assert can_retry_inventory_at_manipuland_stage(task=task,inventory=inventory) is True


def test_managed_generation_retries_once_from_new_partial_recovery(tmp_path: Path, monkeypatch) -> None:
    batch_id,case_id="batch","case"
    partial=partial_generation_recovery_path(root=tmp_path,batch_id=batch_id,case_id=case_id)
    failure=tmp_path/"robotics_scenecode_failures"/batch_id/f"{case_id}.json"
    calls=[]
    def fake_generate_case(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            partial.mkdir(parents=True)
            failure.parent.mkdir(parents=True)
            failure.write_text(json.dumps({"classification":"asset_geometry",
                "failed_stage":"task_asset_inventory","recoverable_candidate":str(partial)}))
            raise ValueError("inventory")
        return tmp_path/"generation_manifest.json"
    monkeypatch.setattr(generator,"generate_case",fake_generate_case)
    result=generate_case_with_bounded_inventory_resume(output_root=str(tmp_path),
        batch_id=batch_id,case_id=case_id,scenecode_root="runtime",scenecode_python="python",
        supersede_existing=False)
    assert result==tmp_path/"generation_manifest.json"
    assert len(calls)==2


def test_inventory_uses_registry_only_for_an_object_live_in_house_state(
    tmp_path: Path,
) -> None:
    house_state = _write_scene(tmp_path, omit_blue_bin_sdf=True)
    scene_root = house_state.parent.parent
    room_root = scene_root / "room_lab"
    registry_root = room_root / "generated_assets" / "manipuland"
    registry_root.mkdir()
    live_sdf = room_root / "generated_assets" / "blue_bin_from_registry.sdf"
    live_sdf.write_text("<sdf version='1.9'><model name='blue_bin'/></sdf>\n")
    rejected_sdf = room_root / "generated_assets" / "rejected_blue_bin.sdf"
    rejected_sdf.write_text("<sdf/>\n")
    (registry_root / "asset_registry.json").write_text(
        json.dumps(
            {
                "asset_blue_bin": {
                    "object_id": "asset_blue_bin",
                    "sdf_path": str(live_sdf),
                },
                "blue_bin_rejected": {
                    "object_id": "blue_bin_rejected",
                    "sdf_path": str(rejected_sdf),
                },
            }
        )
    )

    report = certify_task_inventory(task=_task(), house_state=house_state)

    assert report["passed"] is True
    matches = {item["task_object_id"]: item for item in report["matches"]}
    assert matches["blue_bin"]["asset_resolution_source"] == "live_object_registry"
    assert matches["blue_bin"]["sdf_path"] == str(live_sdf.resolve())


def test_runtime_executable_preserves_virtual_environment_entrypoint(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base_python"
    base.write_text("binary placeholder", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)
    assert runtime_executable(venv_python) == venv_python.absolute()
    assert runtime_executable(venv_python) != base.resolve()


def test_publication_rewrites_staging_paths_in_runtime_text_only(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".case.partial.123"
    destination = tmp_path / "case"
    staging.mkdir()
    json_path = staging / "house_state.json"
    sdf_path = staging / "asset.sdf"
    log_path = staging / "runtime.log"
    binary_path = staging / "mesh.glb"
    json_path.write_text(json.dumps({"asset": str(staging / "asset.sdf")}))
    sdf_path.write_text(f"<uri>{staging}/mesh.glb</uri>\n")
    log_path.write_text(f"hydra path: {staging.name}/scene_run\n")
    binary_path.write_bytes(b"\x00" + str(staging).encode())

    assert rewrite_published_references(
        root=staging, staging=staging, destination=destination
    ) == 3
    assert str(staging) not in json_path.read_text()
    assert str(destination) in json_path.read_text()
    assert str(staging) not in sdf_path.read_text()
    assert staging.name not in log_path.read_text()
    assert destination.name in log_path.read_text()
    assert binary_path.read_bytes() == b"\x00" + str(staging).encode()


def test_inventory_digest_is_refreshed_after_publication_rewrite(
    tmp_path: Path,
) -> None:
    house_state = tmp_path / "house_state.json"
    house_state.write_text(json.dumps({"asset": "/published/asset.sdf"}))
    inventory_path = tmp_path / "task_inventory_certificate.json"
    inventory = {
        "house_state": "/published/house_state.json",
        "house_state_sha256": "0" * 64,
        "passed": True,
    }

    write_final_inventory_binding(
        inventory=inventory,
        inventory_path=inventory_path,
        house_state=house_state,
    )

    written = json.loads(inventory_path.read_text())
    assert written["house_state_sha256"] != "0" * 64
    assert written["house_state_sha256"] == inventory["house_state_sha256"]
    assert written["house_state"] == "/published/house_state.json"


def test_publication_removes_only_latest_run_link_to_staging(tmp_path: Path) -> None:
    staging = tmp_path / ".case.partial.123"
    destination = tmp_path / "case"
    staging.mkdir()
    latest = tmp_path / "latest-run"
    latest.symlink_to(staging / "scene_run")

    assert remove_staging_latest_run_link(
        staging=staging, destination=destination
    ) is True
    assert not latest.is_symlink()

    unrelated = tmp_path / "other"
    unrelated.mkdir()
    latest.symlink_to(unrelated)
    assert remove_staging_latest_run_link(
        staging=staging, destination=destination
    ) is False
    assert latest.is_symlink()


def test_cleanup_keeps_final_dependency_closure_and_removes_rejected_candidates(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene_000"
    room = scene / "room_lab"
    generated = room / "generated_assets" / "manipuland"
    live_code = generated / "code_object" / "cup_live"
    dead_code = generated / "code_object" / "cup_rejected"
    live_sdf = generated / "sdf" / "cup_live"
    dead_sdf = generated / "sdf" / "cup_rejected"
    live_code.mkdir(parents=True)
    dead_code.mkdir(parents=True)
    live_sdf.mkdir(parents=True)
    dead_sdf.mkdir(parents=True)
    (live_code / "ObjectPlan.json").write_text("{}\n")
    (dead_code / "ObjectPlan.json").write_text("{}\n")
    (live_sdf / "cup.glb").write_bytes(b"live")
    (live_sdf / "cup.sdf").write_text("<uri>cup.glb</uri>\n")
    (dead_sdf / "cup.sdf").write_text("<sdf/>\n")
    registry = generated / "asset_registry.json"
    registry.write_text(
        json.dumps(
            {
                "cup_live": {
                    "sdf_path": str(live_sdf / "cup.sdf"),
                    "code_dir": str(live_code),
                },
                "cup_rejected": {
                    "sdf_path": str(dead_sdf / "cup.sdf"),
                    "code_dir": str(dead_code),
                },
            }
        )
    )
    combined = scene / "combined_house"
    combined.mkdir()
    house_state = combined / "house_state.json"
    house_state.write_text(
        json.dumps(
            {
                "rooms": {
                    "lab": {
                        "objects": {
                            "cup_0": {
                                "sdf_path": "generated_assets/manipuland/sdf/cup_live/cup.sdf",
                                "description": "complete generated asset " + "x" * 300,
                                "metadata": {"code_object_output_dir": str(live_code)},
                            }
                        }
                    }
                }
            }
        )
    )

    report = prune_unreferenced_generated_assets(house_state=house_state)

    assert live_code.is_dir()
    assert live_sdf.is_dir()
    assert (live_sdf / "cup.glb").is_file()
    assert not dead_code.exists()
    assert not dead_sdf.exists()
    assert set(json.loads(registry.read_text())) == {"cup_live"}
    assert report["registry_entries_removed"] == 1
    assert len(report["removed_candidate_roots"]) == 2


@pytest.mark.parametrize("category", ["kitchen_counter", "dining_table", "work_table", "storage_cabinet"])
def test_inventory_retry_keeps_compound_structural_supports_in_early_stage(category):
    task = _task()
    task = task.model_copy(update={"objects": [
        item.model_copy(update={"category": category}) if item.object_id == "table" else item
        for item in task.objects
    ]})
    assert not can_retry_inventory_at_manipuland_stage(task=task, inventory={"matches": [
        {"task_object_id": "table", "status": "missing", "sdf_present": False}
    ]})


def test_inventory_retry_reads_diagnostic_sdf_presence(tmp_path):
    task = _task()
    inventory = certify_task_inventory(task=task, house_state=_write_scene(tmp_path, omit_blue_bin_sdf=True))
    for match in inventory["matches"]:
        match["sdf_present"] = bool(match.pop("sdf_path", None))
    assert can_retry_inventory_at_manipuland_stage(task=task, inventory=inventory)


def test_recovery_arbitration_archives_missing_structural_support_checkpoints(tmp_path):
    task = _task()
    task = task.model_copy(update={"objects": [
        item.model_copy(update={"category": "kitchen_counter"}) if item.object_id == "table" else item
        for item in task.objects
    ]})
    recovery = tmp_path / "complete" / "case"
    partial = tmp_path / "partial" / "case"
    for folder in (recovery, partial):
        folder.mkdir(parents=True)
        (folder / "evidence.txt").write_bytes(b"original failed checkpoint")
    failure = tmp_path / "failure.json"
    failure.write_text(json.dumps({"failed_stage": "task_asset_inventory",
        "recoverable_candidate": str(partial), "matches": [
            {"task_object_id": "table", "status": "missing", "sdf_present": False}]}))
    assert not generator._arbitrate_inventory_recovery(task=task, failure_path=failure,
        recovery=recovery, partial_recovery=partial)
    for folder in (recovery, partial):
        assert not folder.exists()
        archived = list(folder.parent.glob("case.superseded_*"))
        assert len(archived) == 1
        assert (archived[0] / "evidence.txt").read_bytes() == b"original failed checkpoint"


def test_recovery_arbitration_keeps_valid_late_stage_resume(tmp_path):
    recovery, partial = tmp_path / "complete", tmp_path / "partial"
    recovery.mkdir(); partial.mkdir()
    failure = tmp_path / "failure.json"
    failure.write_text(json.dumps({"failed_stage": "task_asset_inventory",
        "recoverable_candidate": str(partial), "matches": [
            {"task_object_id": "blue_bin", "status": "missing", "sdf_present": False},
            {"task_object_id": "table", "status": "matched", "sdf_present": True}]}))
    assert generator._arbitrate_inventory_recovery(task=_task(), failure_path=failure,
        recovery=recovery, partial_recovery=partial)
    assert recovery.is_dir() and partial.is_dir()


@pytest.mark.parametrize("with_case_named_path", [False, True])
def test_articulation_joint_type_cannot_turn_cupboard_into_oven(tmp_path, with_case_named_path):
    house = _write_articulated_scene(tmp_path)
    payload = json.loads(house.read_text())
    if with_case_named_path:
        payload["rooms"]["lab"]["objects"]["asset_cupboard"]["metadata"] = {
            "code_object_output_dir": "/outputs/household_toaster_oven_case/white_cupboard",
            "source_task": "toaster_oven",
        }
        house.write_text(json.dumps(payload))
    task = _articulated_task()
    task = task.model_copy(update={"objects": [
        item.model_copy(update={"category": "toaster_oven", "visual_description": "compact toaster oven"})
        if item.object_id == "cupboard" else item for item in task.objects
    ]})
    result = certify_task_inventory(task=task, house_state=house)
    matches = {row["task_object_id"]: row for row in result["matches"]}
    assert matches["cupboard"]["status"] == "missing"
    assert matches["table"]["scene_object_id"] == "asset_worktable"
    assert not result["passed"]


def test_case_named_metadata_cannot_supply_bin_identity(tmp_path):
    house = _write_scene(tmp_path)
    data = json.loads(house.read_text())
    data["rooms"]["lab"]["objects"]["asset_red_block"]["metadata"] = {
        "code_object_output_dir": "/outputs/blue_bin_case/red_block",
        "source_task": "blue_bin",
    }
    house.write_text(json.dumps(data))
    inventory = generator._scene_inventory(house)
    red = next(row for row in inventory if row["object_id"] == "asset_red_block")
    assert "red" in red["tokens"] and "block" in red["tokens"]
    assert "blue" not in red["tokens"] and "bin" not in red["tokens"]


def test_recovery_routing_rechecks_current_assets_instead_of_stale_missing_support(tmp_path):
    recovery = tmp_path / "complete"
    house = _write_scene(recovery, omit_blue_bin_sdf=True)
    target = recovery / "scene_run/scene_000/combined_house/house_state.json"
    target.parent.mkdir(parents=True)
    # Move the complete test scene to the ordinary recovery layout.
    import shutil
    shutil.rmtree(target.parent.parent)
    shutil.move(str(recovery / "scene_000"), str(recovery / "scene_run/scene_000"))
    partial = tmp_path / "partial"; partial.mkdir()
    prior = tmp_path / "failure.json"
    prior.write_text(json.dumps({"failed_stage": "task_asset_inventory",
        "recoverable_candidate": str(recovery), "matches": [
            {"task_object_id": "table", "status": "missing", "sdf_present": False}]}))
    assert not generator._arbitrate_inventory_recovery(task=_task(), failure_path=prior,
        recovery=recovery, partial_recovery=partial)
    assert recovery.is_dir() and partial.is_dir()
    assert not list(tmp_path.glob("complete.superseded_*"))
