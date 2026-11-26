"""Objective media gates for SceneCode robotics references and WAM outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .io import sha256_file, write_json


def _semantic_contact_measurement(
    mask: np.ndarray, *, required_finger_count: int = 2
) -> dict:
    """Measure visible two-finger proximity to the subject in one registered view.

    Semantic masks are deliberately a coarse registration check layered on top
    of the exact PyBullet contact certificate.  The rendered finger labels do
    not necessarily include light-coloured contact pads and very small subjects
    have only a few raster pixels, so a fixed pixel threshold is not invariant
    to object or camera scale.  Allow three pixels of rasterisation slack plus
    half the subject's equivalent side length, capped at twelve pixels.  This
    still rejects a visibly detached finger while remaining meaningful for
    arbitrary manipulands rather than only the original cup sample.
    """

    if mask.ndim == 3:
        mask = mask[..., 0]
    subject = mask == 1
    left = mask == 2
    right = mask == 3
    subject_pixels = int(subject.sum())
    left_pixels = int(left.sum())
    right_pixels = int(right.sum())
    subject_scale_px = float(np.sqrt(subject_pixels)) if subject_pixels else 0.0
    maximum_gap_px = max(3.0, min(12.0, 3.0 + 0.5 * subject_scale_px))

    def gap(finger: np.ndarray) -> float | None:
        if not subject.any() or not finger.any():
            return None
        distance = cv2.distanceTransform((~subject).astype(np.uint8), cv2.DIST_L2, 3)
        return float(distance[finger].min())

    left_gap = gap(left)
    right_gap = gap(right)
    if required_finger_count not in {1, 2}:
        raise ValueError("semantic contact gate supports one or two required fingers")
    base_checks = {
        "subject_visible": subject_pixels >= 50,
        "left_finger_visible": left_pixels >= 8,
        "right_finger_visible": right_pixels >= 8,
        "left_finger_near_subject": (
            left_gap is not None and left_gap <= maximum_gap_px
        ),
        "right_finger_near_subject": (
            right_gap is not None and right_gap <= maximum_gap_px
        ),
    }
    if required_finger_count == 2:
        passed = all(base_checks.values())
    else:
        passed = bool(
            base_checks["subject_visible"]
            and (
                (
                    base_checks["left_finger_visible"]
                    and base_checks["left_finger_near_subject"]
                )
                or (
                    base_checks["right_finger_visible"]
                    and base_checks["right_finger_near_subject"]
                )
            )
        )
    return {
        "passed": passed,
        "checks": base_checks,
        "required_finger_count": required_finger_count,
        "subject_pixels": subject_pixels,
        "subject_equivalent_side_px": subject_scale_px,
        "left_finger_pixels": left_pixels,
        "right_finger_pixels": right_pixels,
        "left_gap_px": left_gap,
        "right_gap_px": right_gap,
        "maximum_gap_px": maximum_gap_px,
    }


def certify_reference_physical_render(
    trajectory: dict,
    simulation_result: dict,
    render_manifest: dict,
    output: str | Path,
) -> Path:
    """Bind successful PyBullet contact to operator-aware render evidence.

    Collision contacts alone cannot certify a reference video: a mismatched
    visual mesh can still show one finger floating beside the object.  This
    gate consumes registered Drake label renders from the exact SceneCode
    replay and requires every sampled contact instant to show the subject and
    the operator-required fingers in scale-aware proximity in a fixed camera.
    """

    left_contact = simulation_result.get("grasp_contact_left")
    if left_contact is None:
        left_contact = simulation_result.get("grasp_contact_left_ever")
    right_contact = simulation_result.get("grasp_contact_right")
    if right_contact is None:
        right_contact = simulation_result.get("grasp_contact_right_ever")
    expected_coupling_count = int(
        simulation_result.get("expected_gripper_coupling_constraint_count", 1)
    )
    actual_coupling_count = int(
        simulation_result.get("gripper_coupling_constraint_count", -1)
    )
    operator = str(simulation_result.get("operator") or "pick_place")
    common_simulation_checks = {
        "simulation_success": simulation_result.get("success") is True,
        "no_object_attachment": simulation_result.get(
            "object_attachment_constraint_count"
        )
        == 0,
        "gripper_coupling_matches_reviewed_contract": (
            actual_coupling_count == expected_coupling_count
        ),
    }
    simulation_evidence: dict[str, object] = {}
    if operator == "hang":
        hung = simulation_result.get("hung_physics_certificate") or {}
        checks = hung.get("checks") or {}
        required = {
            "lifted_by_robot",
            "body_grasp",
            "dynamic_subject",
            "no_object_attachment",
            "released",
            "unsupported_by_table",
            "handle_supported",
            "hook_through_handle",
            "only_handle_support",
            "above_table_plane",
            "stable",
        }
        simulation_checks = {
            **common_simulation_checks,
            "left_body_grasp_contact": left_contact is True,
            "right_body_grasp_contact": right_contact is True,
            "dual_finger_carry": float(
                simulation_result.get("dual_finger_supported_carry_fraction") or 0.0
            )
            >= 0.95,
            "opposed_external_finger_carry": float(
                simulation_result.get("opposed_external_carry_fraction") or 0.0
            )
            >= 0.95,
            "hang_predicate": (
                simulation_result.get("success_predicate") == "hung"
                and hung.get("schema_version") == "robotics_hung_physics_certificate.v2"
                and hung.get("passed") is True
            ),
            "complete_hang_evidence": required.issubset(checks),
            "all_hang_checks_pass": (
                required.issubset(checks)
                and all(checks[name] is True for name in required)
            ),
            "handle_hook_engagement": (
                checks.get("hook_through_handle") is True
                and checks.get("handle_supported") is True
                and checks.get("only_handle_support") is True
            ),
            "released_above_table": (
                checks.get("released") is True
                and checks.get("unsupported_by_table") is True
                and checks.get("above_table_plane") is True
            ),
        }
        simulation_evidence = {
            "hung_schema_version": hung.get("schema_version"),
            "hung_checks": checks,
            "stability_definition": hung.get("stability_definition"),
            "max_final_speed": hung.get("max_final_speed"),
        }
    elif operator in {"pull", "press", "close"}:
        predicate_evidence = simulation_result.get("predicate_evidence") or {}
        simulation_checks = {
            **common_simulation_checks,
            "unassisted_robot_contact": (
                simulation_result.get("execution_mode") == "robot_contact"
                and simulation_result.get("physical_success") is True
                and simulation_result.get("external_assist_step_count") == 0
                and simulation_result.get("contact_pull_force_n") == 0.0
            ),
            "stable_contact_execution": (
                (simulation_result.get("contact_execution_evidence") or {}).get("passed") is True
                and (simulation_result.get("contact_execution_evidence") or {}).get("stable_grasp") is True
                and (simulation_result.get("contact_execution_evidence") or {}).get("released") is True
            ),
            "articulation_contact": int(
                simulation_result.get("contact_frame_count") or 0
            )
            > 0,
            "articulation_predicate": predicate_evidence.get("joint_at_target") is True,
            "articulation_success_predicate": simulation_result.get(
                "success_predicate"
            )
            in {"joint_at_open_limit", "joint_at_closed_limit"},
        }
    elif operator == "push":
        simulation_checks = {
            **common_simulation_checks,
            "push_contact": int(
                simulation_result.get("push_contact_frame_count") or 0
            )
            > 0,
            "horizontal_displacement": float(
                simulation_result.get("subject_horizontal_displacement_m") or 0.0
            )
            > 0.02,
            "on_region": (
                simulation_result.get("success_predicate") == "on_region"
                and (simulation_result.get("predicate_evidence") or {}).get(
                    "predicate_passed"
                )
                is True
            ),
        }
    elif operator == "pour":
        verdict = simulation_result.get("independent_verdict") or {}
        verdict_path_value = simulation_result.get("independent_verdict_path")
        verdict_path = None
        if isinstance(verdict_path_value, str):
            declared_path = Path(verdict_path_value)
            verdict_path = (
                declared_path if declared_path.is_absolute()
                else Path(output).resolve().parent / declared_path
            ).resolve()
        verdict_file_bound = bool(
            verdict_path is not None and verdict_path.is_file()
            and sha256_file(verdict_path) == simulation_result.get("independent_verdict_sha256")
            and json.loads(verdict_path.read_text()) == verdict
        )
        checks = verdict.get("checks") or {}
        required = {
            "initial_contents_in_source", "contents_remain_distinct",
            "source_lifted", "source_tilted", "source_over_target",
            "robot_contact_during_tilt", "all_contents_inside_target",
            "all_contents_stable", "all_contents_clear_of_source",
            "target_remained_stable", "robot_released_source",
            "no_object_attachment", "all_content_motion_plausible",
        }
        simulation_checks = {
            **common_simulation_checks,
            "independent_verdict_file_bound": verdict_file_bound,
            "independent_pour_verdict": (
                verdict.get("schema_version") ==
                "worldolympiad_independent_pour_verdict.v1"
                and verdict.get("passed") is True
                and required.issubset(checks)
                and all(checks[key] is True for key in required)
                and len(verdict.get("content_instances") or []) >= 2
            ),
        }
        simulation_evidence = {
            "independent_verdict_sha256": simulation_result.get(
                "independent_verdict_sha256"),
            "pour_checks": checks,
        }
    else:
        simulation_checks = {
            **common_simulation_checks,
            "left_contact": left_contact is True,
            "right_contact": right_contact is True,
            "dual_finger_carry": float(
                simulation_result.get("dual_finger_supported_carry_fraction") or 0.0
            )
            == 1.0,
        }
        if simulation_result.get("robot_embodiment") == "robolab_droid":
            # 2026-09-11 household-scale repair: the projection margin uses the
            # subject root link as its reference, so a tall subject carried by a
            # handle can show a negative projection while its contacts are
            # provably force-closed.  Prefer the reference-frame-independent
            # contact-normal opposition when the bridge reports it, and keep the
            # projection margin as recorded evidence.
            opposed_fraction = float(
                simulation_result.get("opposed_external_carry_fraction") or 0.0
            )
            normal_margin = simulation_result.get("minimum_opposed_normal_margin")
            normal_threshold = simulation_result.get("opposed_normal_threshold")
            if normal_margin is None or normal_threshold is None:
                opposed_ok = (
                    opposed_fraction == 1.0
                    and float(
                        simulation_result.get("minimum_opposed_contact_margin_m")
                        or 0.0
                    )
                    >= 0.004
                )
            else:
                opposed_ok = opposed_fraction == 1.0 and float(
                    normal_margin
                ) >= float(normal_threshold)
            simulation_checks["opposed_external_finger_carry"] = opposed_ok
        simulation_evidence = {
            "opposed_external_carry_fraction": simulation_result.get(
                "opposed_external_carry_fraction"
            ),
            "opposed_external_geometric_fraction": simulation_result.get(
                "opposed_external_geometric_fraction"
            ),
            "minimum_opposed_contact_margin_m": simulation_result.get(
                "minimum_opposed_contact_margin_m"
            ),
            "minimum_opposed_normal_margin": simulation_result.get(
                "minimum_opposed_normal_margin"
            ),
            "opposed_normal_threshold": simulation_result.get(
                "opposed_normal_threshold"
            ),
        }
    views = render_manifest.get("views") or {}
    fixed_views = ("top", "exterior_left", "exterior_right")
    if any(name not in views for name in fixed_views):
        raise ValueError("physical render gate requires all three fixed cameras")

    expected_frames = {
        int(item["frame"])
        for item in views[fixed_views[0]].get("semantic_masks", [])
    }
    if not expected_frames:
        raise ValueError("render manifest has no registered operator-contact masks")
    trajectory_frames = {int(item["frame"]): item for item in trajectory.get("frames", [])}
    if operator in {"pull", "press", "close"}:
        contact_phases = ["contact", "actuate"]
    elif operator == "push":
        contact_phases = ["push"]
    elif operator == "pour":
        # The protected replay selects visible dual-contact frames from actual
        # robot contacts. Candidate-authored phase names cannot create evidence.
        contact_phases = ["candidate_action"]
    else:
        contact_phases = ["lift"]
        if any(item.get("phase") == "orient" for item in trajectory_frames.values()):
            contact_phases.append("orient")
        contact_phases.extend(["transfer", "lower"])
    expected_phases = set(contact_phases)
    expected_step_ids = {
        int(item.get("operator_step_index", 0))
        for item in trajectory_frames.values()
        if item.get("phase") in expected_phases
    }
    expected_step_phases = {
        (int(item.get("operator_step_index", 0)), str(item.get("phase")))
        for item in trajectory_frames.values()
        if item.get("phase") in expected_phases
    }
    evidence_step_ids: set[int] = set()
    passing_step_phases: set[tuple[int, str]] = set()
    reports: dict[str, dict] = {}
    for frame_id in sorted(expected_frames):
        source = trajectory_frames.get(frame_id)
        if source is None or source.get("phase") not in expected_phases:
            raise ValueError(
                f"semantic evidence frame {frame_id} is not an operator-contact frame"
            )
        step_index = int(source.get("operator_step_index", 0))
        evidence_step_ids.add(step_index)
        per_view: dict[str, dict] = {}
        for view_id in fixed_views:
            records = {
                int(item["frame"]): item
                for item in views[view_id].get("semantic_masks", [])
            }
            if set(records) != expected_frames:
                raise ValueError("fixed cameras do not share the same semantic evidence frames")
            active_subject_id = source.get("active_subject_id")
            if active_subject_id is not None and records[frame_id].get(
                "active_subject_id"
            ) != active_subject_id:
                raise ValueError(
                    "semantic evidence mask names another active subject"
                )
            mask_path = Path(records[frame_id]["image"]).resolve()
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"unable to decode registered semantic mask: {mask_path}")
            per_view[view_id] = _semantic_contact_measurement(
                mask, required_finger_count=1 if operator == "push" else 2
            )
        passing_views = [name for name, report in per_view.items() if report["passed"]]
        if passing_views:
            passing_step_phases.add((step_index, str(source["phase"])))
        reports[str(frame_id)] = {
            "phase": source["phase"],
            "operator_step_index": step_index,
            "active_subject_id": source.get("active_subject_id"),
            "passed": bool(passing_views),
            "passing_views": passing_views,
            "views": per_view,
        }

    render_checks = {
        "every_operator_phase_has_visible_registered_sample": (
            passing_step_phases == expected_step_phases
        ),
        "multiple_contact_samples": len(reports) >= 3,
        "all_operator_steps_sampled": evidence_step_ids == expected_step_ids,
        "robot_replay_link_registered": float(
            (render_manifest.get("robot_registration") or {}).get(
                "maximum_finger_translation_residual_m", float("inf")
            )
        )
        <= 1.0e-4,
    }
    passed = all(simulation_checks.values()) and all(render_checks.values())
    return write_json(
        output,
        {
            "schema_version": "robotics_reference_physical_render.v6",
            "passed": passed,
            "simulation_checks": simulation_checks,
            "simulation_contract": {
                "operator": operator,
                "robot_embodiment": simulation_result.get(
                    "robot_embodiment", "legacy_franka_panda"
                ),
                "expected_gripper_coupling_constraint_count": (
                    expected_coupling_count
                ),
                "actual_gripper_coupling_constraint_count": actual_coupling_count,
            },
            "render_checks": render_checks,
            "simulation_evidence": simulation_evidence,
            "evidence_frame_count": len(reports),
            "frames": reports,
            "policy": {
                "contact_phases": contact_phases,
                "required_visible_entities": (
                    ["subject", "at_least_one_pushing_finger"]
                    if operator == "push"
                    else ["subject", "left_finger", "right_finger"]
                ),
                "frame_rule": (
                    "one pushing finger must be visibly near the subject in at least one fixed camera"
                    if operator == "push"
                    else "both fingers must be visibly near the subject in at least one fixed camera"
                ),
                "semantic_gap_rule": (
                    "min(12 px, 3 px + 0.5 * sqrt(subject mask area))"
                ),
                "bundle_rule": (
                    "every operator step and carry phase must have at least one "
                    "visible registered sample; additional target-occluded samples "
                    "remain recorded but do not invalidate physical registration"
                ),
                "maximum_robot_replay_link_residual_m": 1.0e-4,
                "minimum_opposed_contact_margin_m": 0.004,
                "opposed_normal_opposition_threshold": 0.5,
                "opposed_carry_rule": (
                    "dual opposed finger contacts must persist through every "
                    "carry frame; the reference-frame-independent contact-normal "
                    "opposition is required when the bridge reports it, and the "
                    "root-link projection margin is recorded as evidence"
                ),
                "reference_failure_attribution": "reference_invalid_not_wam_score",
            },
        },
    )


def inspect_video(
    video: str | Path,
    *,
    min_width: int,
    min_height: int,
    min_duration_sec: float,
    min_mean_frame_delta: float = 0.05,
    min_p90_frame_delta: float = 0.15,
    min_first_last_delta: float = 0.25,
) -> dict:
    path = Path(video).resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = 0
    deltas: list[float] = []
    previous: np.ndarray | None = None
    first: np.ndarray | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if first is None:
            first = gray.copy()
        if previous is not None:
            deltas.append(float(np.mean(np.abs(gray.astype(np.float32) - previous))))
        previous = gray
        frame_count += 1
    capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    p90_delta = float(np.percentile(deltas, 90)) if deltas else 0.0
    first_last_delta = (
        float(np.mean(np.abs(first.astype(np.float32) - previous.astype(np.float32))))
        if first is not None and previous is not None
        else 0.0
    )
    checks = {
        "decodable": fps > 0 and frame_count >= 2,
        "resolution": width >= min_width and height >= min_height,
        "duration": duration >= min_duration_sec,
        "non_static": (
            mean_delta >= min_mean_frame_delta
            and p90_delta >= min_p90_frame_delta
            and first_last_delta >= min_first_last_delta
        ),
    }
    return {
        "path": str(path),
        "passed": all(checks.values()),
        "checks": checks,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "mean_adjacent_frame_delta_8bit": mean_delta,
        "p90_adjacent_frame_delta_8bit": p90_delta,
        "first_last_frame_delta_8bit": first_last_delta,
    }


def certify_reference_views(
    view_videos: dict[str, str | Path],
    output: str | Path,
    *,
    expected_sizes: dict[str, tuple[int, int]] | None = None,
    source_frames: dict[str, list[Path]] | None = None,
    expected_fps: float | None = None,
) -> Path:
    required = {"top", "wrist", "exterior_left", "exterior_right"}
    if set(view_videos) != required:
        raise ValueError(f"reference quality gate requires exact views: {sorted(required)}")
    if expected_sizes is not None and set(expected_sizes) != required:
        raise ValueError("native reference sizes must cover every physical view")
    if source_frames is not None and (set(source_frames)!=required or expected_fps is None):
        raise ValueError("source-aware quality needs every physical view and its declared clock")
    reports = {}
    for view_id, path in view_videos.items():
        expected = expected_sizes[view_id] if expected_sizes is not None else (1280, 720)
        report = inspect_video(
            path,
            min_width=expected[0],
            min_height=expected[1],
            min_duration_sec=8.0 if source_frames is None else 0.0,
        )
        report["checks"]["exact_native_resolution"] = (
            report["width"], report["height"]
        ) == expected
        if source_frames is not None:
            from .reference_video_fidelity import inspect_source_fidelity
            fidelity=inspect_source_fidelity(path,source_frames[view_id],fps=expected_fps)
            report["motion_diagnostics"]={"legacy_non_static":report["checks"].pop("non_static")}
            report["source_fidelity"]=fidelity
            report["checks"]["renderer_frame_fidelity"]=fidelity["passed"]
        report["passed"] = all(report["checks"].values())
        reports[view_id] = report
    passed = all(report["passed"] for report in reports.values())
    episode_motion=None
    if source_frames is not None:
        episode_motion=any(reports[name]["source_fidelity"]["source_has_local_motion"]
                           for name in ("top","exterior_left","exterior_right"))
        passed=passed and episode_motion
    return write_json(
        output,
        {
            "schema_version": "robotics_reference_quality.v2" if source_frames is None else "robotics_reference_quality.v3",
            "passed": passed,
            **({"episode_checks":{"fixed_view_motion_evidence":episode_motion}} if source_frames is not None else {}),
            "views": reports,
            "policy": {
                "exact_native_resolution_by_view": {
                    view: list(expected_sizes[view])
                    for view in sorted(expected_sizes)
                }
                if expected_sizes is not None
                else {view: [1280, 720] for view in sorted(required)},
                "minimum_duration_sec": 8.0 if source_frames is None else None,
                "motion_scope": "each_view_legacy" if source_frames is None else "episode_any_fixed_view",
                "adjacent_frame_deltas_are_diagnostic_only":source_frames is not None,
                "physical_success_and_observability_require_separate_certificates":True,
                "minimum_mean_adjacent_frame_delta_8bit": 0.05,
                "minimum_p90_adjacent_frame_delta_8bit": 0.15,
                "minimum_first_last_frame_delta_8bit": 0.25,
            },
        },
    )


def certify_wam_output(
    video: str | Path,
    output: str | Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
    expected_fps: float,
) -> Path:
    video_path = Path(video).resolve()
    output_path = Path(output).resolve()
    report = inspect_video(
        video_path,
        min_width=expected_width,
        min_height=expected_height,
        min_duration_sec=0.0,
    )
    report["motion_diagnostics"]={"legacy_non_static":report["checks"].pop("non_static")}
    report["checks"].update(
        exact_expected_resolution=(report["width"],report["height"])==(expected_width,expected_height),
        expected_frame_count=report["frame_count"]==expected_frame_count,
        expected_fps=abs(report["fps"]-expected_fps)<1e-6)
    report["passed"]=all(report["checks"].values())
    try:
        relative_video = video_path.relative_to(output_path.parent).as_posix()
    except ValueError as exc:
        raise ValueError("formal WAM video must share its immutable bundle") from exc
    report["path"] = relative_video
    report["sha256"] = sha256_file(video_path)
    return write_json(
        output_path,
        {
            "schema_version": "robotics_wam_video_quality.v4",
            "passed": report["passed"],
            "video": report,
            "policy": {
                "exact_expected_size": [expected_width, expected_height],
                "expected_frame_count":expected_frame_count,"expected_fps":expected_fps,
                "task_success_certified":False,
                "stationary_model_output_is_a_measurable_model_outcome":True,
                "adjacent_frame_deltas_are_diagnostic_only":True,
                "minimum_mean_adjacent_frame_delta_8bit": 0.05,
                "minimum_p90_adjacent_frame_delta_8bit": 0.15,
                "minimum_first_last_frame_delta_8bit": 0.25,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", required=True)
    parser.add_argument("--wrist", required=True)
    parser.add_argument("--exterior-left", required=True)
    parser.add_argument("--exterior-right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = certify_reference_views(
        {
            "top": args.top,
            "wrist": args.wrist,
            "exterior_left": args.exterior_left,
            "exterior_right": args.exterior_right,
        },
        args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
