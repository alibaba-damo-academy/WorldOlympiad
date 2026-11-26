from __future__ import annotations

import math
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from autoresearch.robotics_native_cameras import (
    native_fixed_camera_table_poses,
    pinhole_intrinsics,
)

from autoresearch.robotics_reference_materializer import (
    COSMOS_NATIVE_SIZE_WH_BY_VIEW,
    DETERMINISTIC_LABEL_VISUAL_SUFFIXES,
    DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET,
    PYBULLET_COLLISION_MARGIN_PER_SIDE_M,
    UNIFORM_SCALE_STEP,
    _bounded_log_tails,
    _preserve_failure_evidence,
    _cli_float,
    _closed_section_aperture,
    _object_orientations,
    _oriented_box_extent,
    _primary_support_id,
    _run,
    _sdf_mesh_uris,
    _stabilized_grasp_band_plan,
    _terminal_alignment_axes,
    _validate_completed_cache,
    _write_completed_cache,
    compiled_execution_steps,
    droid_destination_fit_scale,
    droid_uniform_grasp_scale,
    droid_destination_native_view_scale,
    label_render_visual_sdfs,
    normalized_object_layout,
    normalized_two_object_layout,
    pick_place_execution_geometry,
    planar_reorientation_profile,
    project_visual_meshes_for_label_renderer,
    scale_grasp_offset_xy,
    stack_destination_visibility_scale,
)


from autoresearch.robotics_contracts import (
    SuccessPredicate,
    TaskInstructionSpec,
    TaskObject,
)


def test_planar_reorientation_profile_uses_measured_grasp_and_nearby_support() -> None:
    profile = planar_reorientation_profile(
        (0.0738161616, -0.1972466508), (0.0000244971, -0.0484306621), 0.65
    )
    assert profile["profile"] == "droid_planar_reorientation_v1"
    assert profile["landing_xy_m"] == pytest.approx([0.0738161616, -0.2772466508])
    assert profile["grasp_offset_xy_m"] == pytest.approx([0.0, -0.006])
    assert profile["measured_grasp_offset_xy_m"] == [0.0000244971, -0.0484306621]


def test_planar_reorientation_profile_reverses_toward_robot_base() -> None:
    profile = planar_reorientation_profile((0.0, -0.55), (0.0, 0.002), 1.0)
    assert profile["landing_xy_m"] == pytest.approx([0.0, -0.47])
    assert profile["grasp_offset_xy_m"] == [0.0, 0.002]


def test_failure_measurements_survive_cache_cleanup(tmp_path: Path) -> None:
    import shutil
    from autoresearch.io import sha256_file
    work = tmp_path / "cache"
    simulation = work / "simulation"
    simulation.mkdir(parents=True)
    result = {"success": False, "planning_evidence": {"failure": "reachability_or_ik_continuity"}}
    (simulation / "result.json").write_text(json.dumps(result))
    (simulation / "trajectory.json").write_text('{"frames": []}')
    (work / "execution_request.json").write_text('{"input_digests": {"source": "frozen"}}')
    retained = _preserve_failure_evidence(work, tmp_path / "evidence")
    shutil.rmtree(work)
    root = Path(retained["directory"])
    assert json.loads((root / "simulation/result.json").read_text()) == result
    assert all(sha256_file(root / path) == digest for path, digest in retained["files"].items())
    assert json.loads((root / "evidence_manifest.json").read_text())["passed"] is False


def test_droid_scale_is_derived_from_aperture_not_case_identity() -> None:
    scale, report = droid_uniform_grasp_scale(0.09)
    assert scale == pytest.approx(0.8)
    assert report["scaled_horizontal_extent_m"] == pytest.approx(0.072)
    assert report["usable_extent_m"] == pytest.approx(0.072)
    assert report["passed"] is True


def test_float32_bound_noise_does_not_drop_an_exact_scale_step() -> None:
    scale, report = droid_uniform_grasp_scale(0.09000000357627869)
    assert scale == pytest.approx(0.8)
    assert report["scaled_horizontal_extent_m"] - report["usable_extent_m"] < 1e-6


def test_materially_wider_object_still_quantizes_down() -> None:
    scale, _ = droid_uniform_grasp_scale(0.091)
    assert scale == pytest.approx(0.75)


def test_small_object_keeps_native_scale() -> None:
    scale, report = droid_uniform_grasp_scale(0.05)
    assert scale == 1.0
    assert report["passed"] is True


def test_stack_scale_fits_subject_inside_destination_support() -> None:
    scale, report = droid_uniform_grasp_scale(
        0.05,
        placement_horizontal_limit_m=0.0225 * 0.80,
    )
    assert scale == pytest.approx(0.35)
    assert report["scaled_horizontal_extent_m"] == pytest.approx(0.0175)
    assert report["placement_horizontal_limit_m"] == pytest.approx(0.018)
    assert report["grasp_passed"] is True
    assert report["placement_passed"] is True
    assert report["passed"] is True


def test_stack_collision_support_keeps_one_scale_step_margin() -> None:
    subject_extent = 0.05
    physical_target_support = 0.03
    one_scale_step = subject_extent * 0.05
    scale, report = droid_uniform_grasp_scale(
        subject_extent,
        placement_horizontal_limit_m=physical_target_support - one_scale_step,
    )
    assert scale == pytest.approx(0.55)
    assert report["scaled_horizontal_extent_m"] == pytest.approx(0.0275)
    assert report["placement_passed"] is True


def test_stack_limit_includes_pybullet_collision_margin() -> None:
    subject_extent = 0.05
    sdf_target_support = 0.022174389
    runtime_support = (
        sdf_target_support + 2.0 * PYBULLET_COLLISION_MARGIN_PER_SIDE_M
    )
    placement_limit = runtime_support - subject_extent * UNIFORM_SCALE_STEP
    scale, report = droid_uniform_grasp_scale(
        subject_extent,
        placement_horizontal_limit_m=placement_limit,
    )
    assert runtime_support == pytest.approx(0.030174389)
    assert scale == pytest.approx(0.55)
    assert report["placement_passed"] is True


def test_insert_fit_fails_when_only_fitting_scale_cannot_be_grasped() -> None:
    with pytest.raises(ValueError, match="geometry scale policy"):
        droid_uniform_grasp_scale(
            0.0283,
            placement_horizontal_limit_m=0.01055,
            minimum_horizontal_extent_m=0.020,
        )


def test_layout_preserves_direction_and_caps_separation() -> None:
    subject, destination, report = normalized_two_object_layout(
        (-0.5167, 0.3488), (0.3850, 0.3528)
    )
    assert subject[0] < destination[0]
    assert math.dist(subject, destination) == pytest.approx(0.44)
    assert report["direction_preserved"] is True
    source = report["source_destination_xy_m"][0] - report["source_subject_xy_m"][0]
    compiled = (
        report["compiled_destination_xy_m"][0]
        - report["compiled_subject_xy_m"][0]
    )
    assert source * compiled > 0


def test_layout_rejects_non_distinct_objects() -> None:
    with pytest.raises(ValueError, match="not spatially distinct"):
        normalized_two_object_layout((0.0, 0.0), (0.01, 0.01))


def test_layout_widens_separation_to_clear_both_objects() -> None:
    """A wide destination and a tall subject must both clear the reset pose.

    The pick_place basket is 0.96 m wide and the bottle is 0.254 m tall; a
    destination-only shift pushed the bottle under the reset gripper, so the
    layout must widen the separation until both objects clear.
    """

    subject, destination, report = normalized_two_object_layout(
        (-0.0789, 0.2054),
        (0.5461, 0.0634),
        subject_half_extents_m=(0.045, 0.045),
        destination_half_extents_m=(0.31, 0.31),
        home_xy=(0.0, -0.06544),
        home_clearance_m=0.11,
    )
    clearance = report["workcell_clearance"]
    assert clearance["clearance_satisfied"] is True
    assert clearance["subject_gap_m"] >= 0.11 - 1.0e-6
    assert clearance["destination_gap_m"] >= 0.11 - 1.0e-6
    assert clearance["separation_after_m"] >= clearance["separation_before_m"]
    assert clearance["separation_after_m"] <= report["maximum_reachable_separation_m"]
    # Direction is preserved by the rigid layout.
    compiled_vector = (
        report["compiled_destination_xy_m"][0] - report["compiled_subject_xy_m"][0],
        report["compiled_destination_xy_m"][1] - report["compiled_subject_xy_m"][1],
    )
    source_vector = (
        report["source_destination_xy_m"][0] - report["source_subject_xy_m"][0],
        report["source_destination_xy_m"][1] - report["source_subject_xy_m"][1],
    )
    assert compiled_vector[0] * source_vector[0] > 0
    assert compiled_vector[1] * source_vector[1] > 0


def test_layout_keeps_small_destination_untouched() -> None:
    subject, destination, report = normalized_two_object_layout(
        (-0.2, 0.2),
        (0.2, -0.1),
        destination_half_extents_m=(0.05, 0.05),
        home_xy=(0.0, -0.06544),
        home_clearance_m=0.11,
    )
    assert report["workcell_clearance"]["clearance_satisfied"] is True
    assert report["workcell_clearance"]["applied_shift_m"] == [0.0, 0.0]


def test_oversized_destination_is_scaled_into_the_workcell() -> None:
    """A ~1 m SceneCode destination must be scaled down to fit the workcell.

    The pick_place basket is realized at 0.96 m but the DROID two-object
    workcell is capped at a 0.44 m separation, so it cannot be laid out beside
    the manipuland without a uniform embodiment-compatibility scale.
    """

    scale, report = droid_destination_fit_scale(
        destination_half_extents_m=(0.48, 0.48),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        layout_direction_xy=(1.0, 0.0),
    )
    assert report["passed"] is True
    assert scale < 1.0
    assert report["scaled_oriented_support_m"] <= (
        report["available_clearance_envelope_m"] + 1.0e-6
    )
    # A destination already inside the envelope is left alone.
    scale, report = droid_destination_fit_scale(
        destination_half_extents_m=(0.15, 0.15),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        layout_direction_xy=(1.0, 0.0),
    )
    assert scale == 1.0
    assert report["uniform_scale"] == 1.0


def test_destination_fit_rejects_invalid_extent() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        droid_destination_fit_scale(
            destination_half_extents_m=(0.0, 0.1),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            layout_direction_xy=(1.0, 0.0),
        )


def test_carry_height_clears_a_tall_destination() -> None:
    """The carried subject must clear the destination top face.

    A 0.254 m bottle carried at the default lift height collides with a 0.30 m
    basket wall, pushes the basket and topples instead of being placed. The
    compiled lift waypoint must rise above the destination top.
    """

    geometry = pick_place_execution_geometry(
        subject_height_m=0.254,
        destination_height_m=0.30,
        terminal_operator="place",
        grasp_band_center_offset_m=0.127,
    )
    assert geometry["lift_base_offset_z_m"] >= 0.30 + 0.5 * 0.254 + 0.05 - 1.0e-9
    assert (
        geometry["grasp_base_offset_z_m"]
        < geometry["hover_base_offset_z_m"]
        <= geometry["lift_base_offset_z_m"]
        <= 0.55
    )


def test_compiled_offsets_never_leave_the_adapter_vertical_envelope() -> None:
    """The executor rejects any offset above its ceiling.

    A tall subject whose grasp band sits high in its body previously compiled a
    lift offset of 0.5786 m, which the shared DROID expert refused before it
    loaded a single asset.  The compiler must clamp into the same envelope.
    """

    geometry = pick_place_execution_geometry(
        subject_height_m=0.55,
        destination_height_m=0.30,
        terminal_operator="place",
        grasp_band_center_offset_m=0.30,
    )
    assert (
        geometry["grasp_base_offset_z_m"]
        < geometry["hover_base_offset_z_m"]
        <= geometry["lift_base_offset_z_m"]
        == DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET
    )


def test_grasp_band_above_the_adapter_envelope_is_rejected() -> None:
    """A band that leaves no vertical room must fail as geometry, not physics."""

    with pytest.raises(ValueError, match="vertical envelope"):
        pick_place_execution_geometry(
            subject_height_m=1.0,
            destination_height_m=0.20,
            terminal_operator="place",
            grasp_band_center_offset_m=0.47,
        )


def test_materializer_ceiling_matches_the_bridge_executor() -> None:
    """Compiler and executor share one adapter-owned vertical envelope."""

    bridge = pytest.importorskip("scenecode_pybullet_bridge.droid_pick_place")
    assert DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET == bridge.MAXIMUM_BASE_OFFSET_Z_M
    assert bridge.MINIMUM_BASE_OFFSET_Z_M < bridge.MAXIMUM_BASE_OFFSET_Z_M


def test_grasp_lateral_offset_scales_with_the_physical_variant() -> None:
    """The measured grasp feature must follow the derived physical variant.

    The grasp band is measured on the registered native asset while the
    executor loads the uniformly scaled variant; the lateral offset therefore
    has to carry the same factor as the vertical band, otherwise a shrunk
    subject is grasped outside its loaded collision footprint (the measured
    transfer container offset 0.251 m becomes 0.063 m at scale 0.25).
    """

    assert scale_grasp_offset_xy(
        (-0.2509060197733993, 0.008212818368696848), 0.25
    ) == pytest.approx([-0.06272650494334983, 0.002053204592174212])
    assert scale_grasp_offset_xy(None, 0.5) == [0.0, 0.0]
    assert scale_grasp_offset_xy((0.02, -0.04), 1.0) == pytest.approx([0.02, -0.04])
    with pytest.raises(ValueError, match="finite pair"):
        scale_grasp_offset_xy((float("nan"), 0.0), 1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        scale_grasp_offset_xy((0.0, 0.0), 0.0)


def test_multi_object_layout_preserves_relative_geometry_without_case_rules() -> None:
    compiled, report = normalized_object_layout(
        {
            "alpha": (-0.8, -0.2),
            "beta": (-0.2, -0.1),
            "destination_one": (0.5, 0.2),
            "destination_two": (0.7, -0.2),
        }
    )
    assert max(
        math.dist(compiled[left], compiled[right])
        for index, left in enumerate(sorted(compiled))
        for right in sorted(compiled)[index + 1 :]
    ) == pytest.approx(0.52)
    assert compiled["alpha"][0] < compiled["beta"][0]
    assert compiled["destination_one"][0] < compiled["destination_two"][0]
    assert report["relative_geometry_preserved"] is True


def test_primary_support_comes_from_declared_scene_surface_not_object_name(
    tmp_path: Path,
) -> None:
    task = TaskInstructionSpec(
        case_id="press_any_control",
        instruction="Press the button.",
        objects=[
            TaskObject(
                object_id="control_base",
                category="panel",
                role="support",
                visual_description="control base",
            ),
            TaskObject(
                object_id="button",
                category="button",
                role="manipuland",
                visual_description="raised button",
            ),
            TaskObject(
                object_id="bench_surface",
                category="bench",
                role="support",
                visual_description="stable bench",
            ),
        ],
        initial_state="The button is raised on the control base.",
        success_predicates=[SuccessPredicate(kind="pressed", subject="button")],
    )
    house = tmp_path / "house_state.json"
    house.write_text(
        json.dumps(
            {
                "rooms": {
                    "room": {
                        "objects": {
                            "scene_panel": {"support_surfaces": []},
                            "scene_button": {"support_surfaces": []},
                            "scene_bench": {
                                "support_surfaces": [{"surface_id": "surface"}]
                            },
                        }
                    }
                }
            }
        )
        + "\n"
    )
    matches = {
        "control_base": {"scene_object_id": "scene_panel"},
        "button": {"scene_object_id": "scene_button"},
        "bench_surface": {"scene_object_id": "scene_bench"},
    }
    assert _primary_support_id(task=task, matches=matches, house_state=house) == (
        "bench_surface"
    )


def test_scenecode_wxyz_orientation_is_normalized_and_preserved_as_xyzw(
    tmp_path: Path,
) -> None:
    house = tmp_path / "house_state.json"
    house.write_text(
        json.dumps(
            {
                "rooms": {
                    "room": {
                        "objects": {
                            "peg": {
                                "transform": {
                                    "rotation_wxyz": [0.0, 2.0, 0.0, 0.0]
                                }
                            }
                        }
                    }
                }
            }
        )
        + "\n"
    )
    assert _object_orientations(house) == {"peg": (1.0, 0.0, 0.0, 0.0)}


def test_oriented_bounds_turn_a_long_horizontal_peg_upright() -> None:
    half_turn_y = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    extent = _oriented_box_extent(
        (-0.06, -0.01, -0.01),
        (0.06, 0.01, 0.01),
        half_turn_y,
    )
    assert extent == pytest.approx((0.02, 0.02, 0.12))


def test_compiler_collapses_arbitrary_repeated_pick_terminal_pairs() -> None:
    plan = {
        "controller_program": [
            {"operator": "pick", "subjects": ["alpha"], "target": None},
            {"operator": "place", "subjects": ["alpha"], "target": "zone_one"},
            {"operator": "pick", "subjects": ["beta"], "target": None},
            {"operator": "place", "subjects": ["beta"], "target": "zone_two"},
        ],
        "success_program": [
            {"subject": "alpha", "object": "zone_one", "predicate": "in_container"},
            {"subject": "beta", "object": "zone_two", "predicate": "in_container"},
        ],
    }
    assert compiled_execution_steps(plan) == [
        {
            "subject_id": "alpha",
            "destination_id": "zone_one",
            "terminal_operator": "place",
            "terminal_predicate": "in_container",
        },
        {
            "subject_id": "beta",
            "destination_id": "zone_two",
            "terminal_operator": "place",
            "terminal_predicate": "in_container",
        },
    ]


def test_compiler_rejects_predicate_for_the_wrong_pair() -> None:
    plan = {
        "controller_program": [
            {"operator": "pick", "subjects": ["alpha"], "target": None},
            {"operator": "place", "subjects": ["alpha"], "target": "zone_one"},
        ],
        "success_program": [
            {"subject": "beta", "object": "zone_one", "predicate": "in_container"},
        ],
    }
    with pytest.raises(ValueError, match="matching physical predicate"):
        compiled_execution_steps(plan)


def test_reference_cannot_certify_an_unverified_executor() -> None:
    plan = {
        "controller_program": [
            {"operator": "pick", "subjects": ["source"]},
            {"operator": "pour", "subjects": ["source"], "target": "bin",
             "capability_status": "native_registered"},
        ],
        "success_program": [
            {"subject": "contents", "object": "bin", "predicate": "in_container"},
        ],
    }
    with pytest.raises(ValueError, match="lacks verified operations: pour"):
        compiled_execution_steps(plan)


def test_compiler_maps_pick_place_to_planar_region_predicate() -> None:
    plan = {
        "controller_program": [
            {"operator": "pick", "subjects": ["cylinder"], "target": None},
            {"operator": "place", "subjects": ["cylinder"], "target": "target_mat"},
        ],
        "success_program": [
            {"subject": "cylinder", "object": "target_mat", "predicate": "on_region"},
        ],
    }
    assert compiled_execution_steps(plan) == [
        {
            "subject_id": "cylinder",
            "destination_id": "target_mat",
            "terminal_operator": "place",
            "terminal_predicate": "on_region",
        }
    ]


def test_compiler_maps_pull_to_articulated_open_predicate() -> None:
    plan = {
        "controller_program": [
            {"operator": "pull", "subjects": ["drawer"], "target": "cabinet"},
        ],
        "success_program": [
            {"subject": "drawer", "object": "cabinet", "predicate": "drawer_open"},
        ],
    }
    assert compiled_execution_steps(plan) == [
        {
            "subject_id": "drawer",
            "destination_id": "cabinet",
            "terminal_operator": "pull",
            "terminal_predicate": "drawer_open",
        }
    ]


def test_materializer_source_exposes_planar_destination_extent() -> None:
    source = (
        Path(__file__).parents[1]
        / "autoresearch"
        / "robotics_reference_materializer.py"
    ).read_text(encoding="utf-8")
    assert '"destination_local_extent_xyz": first_step[' in source
    assert 'size_xyz_m=tuple(source["destination_local_extent_xyz"])' in source
    assert "simulation.mkdir(parents=True)" in source
    assert "render.mkdir(parents=True)" in source


def test_stack_waypoint_accounts_for_destination_height() -> None:
    geometry = pick_place_execution_geometry(
        subject_height_m=0.05,
        destination_height_m=0.06,
        terminal_operator="stack",
    )
    assert geometry["grasp_base_offset_z_m"] == pytest.approx(
        0.025 + 0.1036321380
    )
    assert geometry["place_target_offset_z_m"] == pytest.approx(0.06)


def test_short_object_respects_native_droid_base_height_floor() -> None:
    geometry = pick_place_execution_geometry(
        subject_height_m=0.02,
        destination_height_m=0.06,
        terminal_operator="stack",
    )
    assert geometry["grasp_base_offset_z_m"] == pytest.approx(0.12)
    assert geometry["hover_base_offset_z_m"] == pytest.approx(0.31)


def test_tiny_stack_base_is_enlarged_before_subject_is_shrunk() -> None:
    assert stack_destination_visibility_scale(
        subject_horizontal_extent_m=0.05,
        destination_support_extent_m=0.022174389,
    ) == pytest.approx(2.0)


def test_container_place_keeps_destination_base_height() -> None:
    geometry = pick_place_execution_geometry(
        subject_height_m=0.08,
        destination_height_m=0.04,
        terminal_operator="place",
    )
    assert geometry["place_target_offset_z_m"] == 0.0


def test_insert_waypoint_enters_hole_block() -> None:
    geometry = pick_place_execution_geometry(
        subject_height_m=0.10,
        destination_height_m=0.08,
        terminal_operator="insert",
    )
    assert geometry["place_target_offset_z_m"] == pytest.approx(0.028)


def _synthetic_grasp_profile() -> dict:
    bands = []
    for index in range(80):
        center = 0.0025 + 0.005 * index
        narrow = center >= 0.36
        extent = 0.06 if narrow else 0.20
        bands.append(
            {
                "band_index": index,
                "base_offset_low_m": center - 0.0025,
                "base_offset_high_m": center + 0.0025,
                "base_offset_center_m": center,
                "x_low_m": -0.5 * extent,
                "x_high_m": 0.5 * extent,
                "y_low_m": -0.5 * extent,
                "y_high_m": 0.5 * extent,
                "x_extent_m": extent,
                "y_extent_m": extent,
            }
        )
    return {
        "object_height_m": 0.4,
        "band_height_m": 0.06,
        "aabb_low_m": [-0.1, -0.1, 0.0],
        "aabb_high_m": [0.1, 0.1, 0.4],
        "bands": bands,
    }


def test_fallback_grasp_band_is_recentred_on_the_collision_centroid() -> None:
    """A top-taper fallback must not hang the subject above its centre of mass."""

    payload = {
        "profile": _synthetic_grasp_profile(),
        "plan": {
            "feature_grasp_feasible": False,
            "band_center_offset_m": 0.385,
            "grip_extent_m": 0.06,
        },
    }
    stabilized = _stabilized_grasp_band_plan(
        payload,
        usable_aperture_m=0.072,
        minimum_extent_m=0.02,
        maximum_lateral_offset_m=0.10,
        minimum_straddle_margin_m=0.004,
    )
    assert stabilized is not payload
    centroid = stabilized["stabilized_centroid_offset_m"]
    assert 0.10 < centroid < 0.30
    assert stabilized["plan"]["band_center_offset_m"] == pytest.approx(
        centroid, abs=0.01
    )
    assert stabilized["plan"]["policy"] == "centroid_stabilized_uniform_scale"
    assert stabilized["stabilized_from_band_center_offset_m"] == pytest.approx(0.385)


def test_natively_feasible_grasp_band_is_left_untouched() -> None:
    payload = {
        "profile": _synthetic_grasp_profile(),
        "plan": {
            "feature_grasp_feasible": True,
            "band_center_offset_m": 0.385,
            "grip_extent_m": 0.06,
        },
    }
    assert (
        _stabilized_grasp_band_plan(
            payload,
            usable_aperture_m=0.072,
            minimum_extent_m=0.02,
            maximum_lateral_offset_m=0.10,
            minimum_straddle_margin_m=0.004,
        )
        is payload
    )


def test_failure_log_tails_are_bounded_and_relative(tmp_path) -> None:
    nested = tmp_path / "render"
    nested.mkdir()
    (nested / "parallel_top.log").write_text(
        "\n".join(f"line {index}" for index in range(100)), encoding="utf-8"
    )
    tails = _bounded_log_tails(tmp_path)
    assert set(tails) == {"render/parallel_top.log"}
    assert "line 99" in tails["render/parallel_top.log"]
    assert "line 0" not in tails["render/parallel_top.log"]


def test_subprocess_failure_keeps_bounded_stderr_evidence(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="specific-package-error"):
        _run(
            [
                sys.executable,
                "-c",
                "import sys; print('specific-package-error', file=sys.stderr); raise SystemExit(7)",
            ],
            cwd=tmp_path,
            env=dict(os.environ),
        )


def test_materializer_binds_scenecode_runtime_before_repository() -> None:
    source = (
        Path(__file__).parents[1]
        / "autoresearch"
        / "robotics_reference_materializer.py"
    ).read_text(encoding="utf-8")
    assert 'selected_scenecode_runtime(workspace)' in source
    assert 'str(scenecode_runtime),' in source
    assert 'selected_scenecode_python(workspace)' in source


def test_reference_planes_are_direct_cosmos_native_panel_sizes() -> None:
    assert COSMOS_NATIVE_SIZE_WH_BY_VIEW == {
        "top": (640, 360),
        "wrist": (640, 360),
        "exterior_left": (320, 180),
        "exterior_right": (320, 180),
    }


def test_materializer_does_not_pass_a_user_or_case_duration() -> None:
    source = (
        Path(__file__).parents[1]
        / "autoresearch"
        / "robotics_reference_materializer.py"
    ).read_text(encoding="utf-8")
    assert '"--target-frame-count"' not in source
    assert 'stage = "camera_observability"' in source
    assert '"owner_stage": stage' in source


def test_reference_cache_identity_covers_every_execution_decision_owner() -> None:
    source = (
        Path(__file__).parents[1]
        / "autoresearch"
        / "robotics_reference_materializer.py"
    ).read_text(encoding="utf-8")
    assert '"materializer": Path(__file__).resolve()' in source
    assert '"sdf_bridge": bridge / "scenecode_pybullet_bridge" / "sdf.py"' in source
    assert '"resource_lock": repository' in source
    assert '"agentic_operator_graph_droid_v12"' in source


def test_terminal_alignment_is_compiled_from_geometry_not_case_identity() -> None:
    assert _terminal_alignment_axes(
        terminal_operator="insert", primary_axis=0, thin_axis=2
    ) == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert _terminal_alignment_axes(
        terminal_operator="cover", primary_axis=0, thin_axis=2
    ) == ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    assert _terminal_alignment_axes(
        terminal_operator="place", primary_axis=0, thin_axis=2
    ) == (None, None)


def test_upright_place_requests_alignment_and_carries_certification() -> None:
    """An explicit upright placement must request and certify terminal alignment."""

    # The dominant local axis is aligned with world up.
    assert _terminal_alignment_axes(
        terminal_operator="place", primary_axis=1, thin_axis=0, upright=True
    ) == ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # stack and non-upright place keep their existing behaviour.
    assert _terminal_alignment_axes(
        terminal_operator="stack", primary_axis=2, thin_axis=0, upright=False
    ) == (None, None)

    steps = compiled_execution_steps(
        {
            "controller_program": [
                {"operator": "pick", "subjects": ["bottle"]},
                {
                    "operator": "place",
                    "subjects": ["bottle"],
                    "target": "basket",
                    "constraints": {
                        "terminal_upright": True,
                        "terminal_upright_axis_world": [0.0, 0.0, 1.0],
                        "terminal_upright_min_cosine": 0.99,
                    },
                },
            ],
            "success_program": [
                {"predicate": "in_container", "subject": "bottle", "object": "basket"}
            ],
        }
    )
    assert steps[0]["terminal_upright"] is True
    assert steps[0]["terminal_upright_min_cosine"] == pytest.approx(0.99)


def test_cli_float_keeps_tiny_negative_nargs_values_positional() -> None:
    rendered = _cli_float(-7.39155765346975e-05)
    assert rendered == "-0.0000739155765347"
    assert "e" not in rendered.lower()
    assert _cli_float(-0.0) == "0"


def test_closed_section_aperture_measures_inner_loop_not_outer_body() -> None:
    import numpy as np

    outer = np.asarray(
        [(-0.04, -0.03), (0.04, -0.03), (0.04, 0.03), (-0.04, 0.03), (-0.04, -0.03)]
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 33)
    inner = np.column_stack(
        (0.002 + 0.01 * np.cos(angles), -0.003 + 0.01 * np.sin(angles))
    )
    vertices = np.vstack((outer, inner))

    class Entity:
        closed = True

        def __init__(self, points: np.ndarray) -> None:
            self.points = points

    class Section:
        pass

    section = Section()
    section.vertices = np.column_stack((vertices, np.zeros(len(vertices))))
    section.entities = (
        Entity(np.arange(len(outer))),
        Entity(np.arange(len(outer), len(vertices))),
    )
    center, radius = _closed_section_aperture(section)
    np.testing.assert_allclose(center, (0.002, -0.003), atol=5e-4)
    assert radius == pytest.approx(0.01, rel=0.02)


def test_materializer_binds_compiled_physical_success_predicate() -> None:
    source = (
        Path(__file__).parents[1]
        / "autoresearch"
        / "robotics_reference_materializer.py"
    ).read_text(encoding="utf-8")
    assert '"--success-predicate"' in source
    assert 'source["terminal_predicate"]' in source


def test_completed_render_cache_is_content_addressed(tmp_path) -> None:
    simulation = tmp_path / "simulation"
    render = tmp_path / "render"
    simulation.mkdir()
    render.mkdir()
    (simulation / "trajectory.json").write_text(
        '{"frames":[{"frame":0}]}\n', encoding="utf-8"
    )
    (simulation / "result.json").write_text(
        '{"success":true}\n', encoding="utf-8"
    )
    views = {}
    for view in ("top", "wrist", "exterior_left", "exterior_right"):
        frame = render / view / "frames" / "00000.png"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(view.encode("utf-8"))
        views[view] = {"frames": [{"frame": 0, "image": str(frame)}]}
    (render / "render_manifest.json").write_text(
        json.dumps({"views": views}), encoding="utf-8"
    )
    inputs = {"task": "abc", "execution_adapter_profile": "v2"}
    _write_completed_cache(tmp_path, inputs)
    assert _validate_completed_cache(tmp_path, inputs) is True
    assert _validate_completed_cache(tmp_path, {**inputs, "materializer":"packaging-only-change"}) is True
    assert _validate_completed_cache(tmp_path, {**inputs, "execution_adapter_profile":"v3"}) is False
    (render / "top" / "frames" / "00000.png").write_bytes(b"changed")
    assert _validate_completed_cache(tmp_path, inputs) is False


def test_cache_recovery_selects_the_same_variant_as_fresh_execution(tmp_path):
    from autoresearch.robotics_reference_materializer import _cached_physical_manifest
    for name in ("aaa_target", "zzz_subject"):
        p=tmp_path/"physical_variant"/name/"physical_variant_manifest.json"
        p.parent.mkdir(parents=True);p.write_text("{}")
    selected=_cached_physical_manifest(tmp_path,{"subject_id":"zzz_subject","destination_id":"aaa_target"})
    assert selected.parent.name == "zzz_subject"
    assert _cached_physical_manifest(tmp_path,{"destination_id":"aaa_target"}).parent.name == "aaa_target"


def test_v6_packager_archives_every_prepared_source_asset() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "package_robotics_reference.py"
    ).read_text(encoding="utf-8")
    assert 'output / "simulation_assets" / "source_sdf"' in source
    assert '"every prepared asset must declare one source SDF"' in source


# Drake's deterministic VTK label renderer -- the authority for model-native
# reference admission -- silently ignores visual meshes that are not .obj/.gltf,
# so a repaired .glb asset would carry no label pixels and be rejected as
# "too small, cropped, or occluded".  The projection below must therefore be
# the only thing the deterministic renderer is pointed at, and it must not
# change the declared geometry or the frozen asset.
_LABEL_RENDER_SDF = """<?xml version='1.0'?>
<sdf xmlns:ns0="drake.mit.edu" version="1.7">
  <model name="asset">
    <link name="base_link">
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>{visual_uri}</uri>
            <scale>{scale}</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name="collision_0">
        <geometry>
          <mesh>
            <uri>{collision_uri}</uri>
          </mesh>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def _obj_text(mesh) -> str:
    payload = mesh.export(file_type="obj")
    return payload if isinstance(payload, str) else payload.decode()


def _write_probe_meshes(directory: Path):
    trimesh = pytest.importorskip("trimesh")
    import numpy as np

    body = trimesh.creation.box(extents=(0.20, 0.10, 0.05))
    rim = trimesh.creation.box(extents=(0.05, 0.05, 0.15))
    transform = np.eye(4)
    transform[:3, 3] = (0.12, 0.0, 0.08)
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body")
    scene.add_geometry(rim, node_name="rim", transform=transform)
    (directory / "asset.glb").write_bytes(scene.export(file_type="glb"))
    (directory / "collision_0.obj").write_text(
        _obj_text(trimesh.creation.box(extents=(0.20, 0.10, 0.05)))
    )
    return trimesh


def test_label_renderer_projects_a_glb_visual_without_touching_the_asset(
    tmp_path: Path,
) -> None:
    _write_probe_meshes(tmp_path)
    sdf = tmp_path / "asset.sdf"
    sdf.write_text(
        _LABEL_RENDER_SDF.format(
            visual_uri="asset.glb", scale="0.5 0.5 0.5", collision_uri="collision_0.obj"
        )
    )
    before = sdf.read_bytes()
    projected, evidence = project_visual_meshes_for_label_renderer(
        sdf_path=sdf, output_dir=tmp_path / "projection"
    )
    assert DETERMINISTIC_LABEL_VISUAL_SUFFIXES == {".obj", ".gltf"}
    assert evidence["projection"] == "declared_visual_scene_graph_to_gltf"
    assert evidence["passed"] is True
    assert projected == (tmp_path / "projection" / "asset.sdf")
    assert _sdf_mesh_uris(projected, visual_only=True) == ["asset.gltf"]
    assert _sdf_mesh_uris(projected, visual_only=False) == [
        "asset.gltf",
        "collision_0.obj",
    ]
    # The declared SDF scale and every collision dependency survive verbatim.
    projected_text = projected.read_text()
    assert "<scale>0.5 0.5 0.5</scale>" in projected_text
    assert (projected.parent / "collision_0.obj").is_file()
    assert (projected.parent / "asset.gltf").is_file()
    # The registered asset is never rewritten.
    assert sdf.read_bytes() == before
    fidelity = evidence["projected_visuals"][0]
    assert fidelity["passed"] is True
    assert fidelity["declared_triangle_corner_count"] == (
        fidelity["projected_triangle_corner_count"]
    )
    assert fidelity["max_triangle_corner_error_m"] == 0.0
    receipt = json.loads(
        (projected.parent / "label_render_projection.json").read_text()
    )
    assert receipt["projected_sdf"] == str(projected)
    assert receipt["passed"] is True


def test_label_renderer_projection_leaves_supported_visuals_untouched(
    tmp_path: Path,
) -> None:
    trimesh = pytest.importorskip("trimesh")

    (tmp_path / "collision_0.obj").write_text(
        _obj_text(trimesh.creation.box(extents=(0.2, 0.1, 0.05)))
    )
    sdf = tmp_path / "asset.sdf"
    sdf.write_text(
        _LABEL_RENDER_SDF.format(
            visual_uri="collision_0.obj", scale="1 1 1", collision_uri="collision_0.obj"
        )
    )
    projected, evidence = project_visual_meshes_for_label_renderer(
        sdf_path=sdf, output_dir=tmp_path / "projection"
    )
    assert projected == sdf
    assert evidence["projection"] == "not_required"
    assert evidence["projected_visuals"] == []
    assert not (tmp_path / "projection").exists()


def test_label_render_visual_sdfs_projects_only_unsupported_assets(
    tmp_path: Path,
) -> None:
    trimesh = _write_probe_meshes(tmp_path)
    glb_sdf = tmp_path / "asset.sdf"
    glb_sdf.write_text(
        _LABEL_RENDER_SDF.format(
            visual_uri="asset.glb", scale="1 1 1", collision_uri="collision_0.obj"
        )
    )
    obj_sdf = tmp_path / "support.sdf"
    obj_sdf.write_text(
        _LABEL_RENDER_SDF.format(
            visual_uri="collision_0.obj", scale="1 1 1", collision_uri="collision_0.obj"
        )
    )
    projected = label_render_visual_sdfs(
        work=tmp_path / "cache",
        assets={"subject": glb_sdf, "support": obj_sdf},
    )
    assert projected["support"] == obj_sdf
    assert projected["subject"] == tmp_path / "cache/label_render_projection/subject/asset.sdf"
    assert _sdf_mesh_uris(projected["subject"], visual_only=True) == ["asset.gltf"]
    assert trimesh is not None


def _native_destination_views() -> dict[str, dict[str, float]]:
    """Required native destination views of the cosmos droid adapter."""

    return {
        "exterior_left": {
            "size_wh": [320, 180],
            "minimum_linear_fraction": 0.045,
        },
        "exterior_right": {
            "size_wh": [320, 180],
            "minimum_linear_fraction": 0.045,
        },
    }


def test_native_fixed_camera_matches_the_registered_reference_run() -> None:
    """The predicted camera pose must equal the pose the renderer published.

    The published pick_place reference rendered ``exterior_left`` from a fixed
    world transform whose translation and rotation are recorded in the render
    manifest and in the CAE camera calibration.  The materializer composes the
    same RoboLab preset on the registered DROID mount, so predicting the framing
    before rendering is only sound if this composition reproduces it.
    """

    poses = native_fixed_camera_table_poses()
    assert set(poses) == {"exterior_left", "exterior_right"}
    table_top_m = 1.40257153
    measured_world = np.asarray([-0.57, -0.38000001, 2.06457153])
    for name, pose in poses.items():
        assert pose.shape == (4, 4)
        assert np.allclose(np.linalg.det(pose[:3, :3]), 1.0)
    assert np.allclose(
        poses["exterior_left"][:3, 3] + np.asarray([0.0, 0.0, table_top_m]),
        measured_world,
        atol=1.0e-6,
    )
    registered_rotation = np.asarray(
        [
            [0.78855766, -0.37251007, 0.48926678],
            [-0.61493648, -0.47726318, 0.62775233],
            [-0.00033610, -0.79589877, -0.60542963],
        ]
    )
    assert np.allclose(
        poses["exterior_left"][:3, :3], registered_rotation, atol=1.0e-3
    )


def test_native_view_intrinsics_match_the_registered_native_size() -> None:
    """The 320x180 native exterior view renders with the registered focal."""

    intrinsics = pinhole_intrinsics((320, 180), 71.5077745089)
    assert intrinsics[0][0] == pytest.approx(125.0, abs=1.0e-6)
    assert intrinsics[0][2] == 160.0
    assert intrinsics[1][1] == pytest.approx(125.0, abs=1.0e-6)
    assert intrinsics[1][2] == 90.0
    with pytest.raises(ValueError):
        pinhole_intrinsics((320, 0), 60.0)
    with pytest.raises(ValueError):
        pinhole_intrinsics((320, 180), 200.0)


def test_oversized_destination_survives_the_native_view_envelope() -> None:
    """A one-metre SceneCode destination must be shrunk until it is framed.

    The workcell fit alone leaves the pick_place basket at 0.6, where the native
    over-shoulder camera sits inside the basket footprint and the rendered
    destination crops at the frame border.  The framing derivative must keep
    descending the quantised ladder until every required native view frames the
    destination box.
    """

    views = _native_destination_views()
    scale, report = droid_destination_native_view_scale(
        candidate_scale=0.6,
        subject_xy=(0.19596536165921752, 0.46063809155425306),
        destination_xy=(-0.19666926560668807, 0.45662183048677907),
        subject_half_extents_m=(0.04500001110136509, 0.04500000923871994),
        subject_orientation_xyzw=(
            0.0004162730482132605,
            -0.00019954885372646498,
            -0.029788113643004124,
            0.9995561290806013,
        ),
        destination_half_extents_m=(0.47999998927116394, 0.47999998927116394),
        destination_orientation_xyzw=(
            0.0004162730482132605,
            -0.00019954885372646498,
            -0.029788113643004124,
            0.9995561290806013,
        ),
        destination_height_m=0.41999998688697815,
        required_views=views,
    )
    assert report["passed"] is True
    assert report["candidate_scale"] == 0.6
    assert scale == 0.35
    assert [item["scale"] for item in report["candidates"]][0] == 0.6
    first = report["candidates"][0]
    assert first["passed"] is False
    assert any(
        measurement["fully_in_frame"] is False
        for seat in first["views"]["exterior_left"].values()
        for measurement in [seat]
    )
    selected = report["candidates"][-1]
    assert selected["scale"] == scale
    assert selected["passed"] is True
    for view in views:
        for seat in selected["views"][view].values():
            assert seat["fully_in_frame"] is True
            assert seat["linear_fraction"] >= seat["minimum_linear_fraction"]
    # A destination that already fits the workcell must be left alone.
    untouched, untouched_report = droid_destination_native_view_scale(
        candidate_scale=1.0,
        subject_xy=(0.24, -0.08),
        destination_xy=(-0.24, 0.10),
        subject_half_extents_m=(0.0477, 0.0477),
        subject_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        destination_half_extents_m=(0.09, 0.09),
        destination_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        destination_height_m=0.05,
        required_views=views,
    )
    assert untouched == 1.0
    assert untouched_report["passed"] is True
    assert len(untouched_report["candidates"]) == 1


def test_native_view_fit_fails_closed_when_no_scale_is_framed() -> None:
    """A destination too large for every ladder step must not be materialized."""

    with pytest.raises(ValueError, match="cannot be framed"):
        droid_destination_native_view_scale(
            candidate_scale=2.0,
            subject_xy=(0.4173665339698609, 0.01426924383343932),
            destination_xy=(-0.4173665339698609, 0.00573075616656068),
            subject_half_extents_m=(0.0477, 0.0477),
            subject_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            destination_half_extents_m=(2.0, 2.0),
            destination_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            destination_height_m=3.0,
            required_views=_native_destination_views(),
        )


def test_native_view_fit_reports_unmodelled_native_views() -> None:
    """A native view without a registered preset is reported, never assumed."""

    views = _native_destination_views()
    views["top"] = {"size_wh": [320, 180], "minimum_linear_fraction": 0.045}
    scale, report = droid_destination_native_view_scale(
        candidate_scale=1.0,
        subject_xy=(0.24, -0.08),
        destination_xy=(-0.24, 0.10),
        subject_half_extents_m=(0.0477, 0.0477),
        subject_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        destination_half_extents_m=(0.09, 0.09),
        destination_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        destination_height_m=0.05,
        required_views=views,
    )
    assert report["unmodelled_views"] == ["top"]
    assert "top" not in report["candidates"][0]["views"]
    assert scale == 1.0
