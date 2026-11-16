"""Cosmos3 Nano DROID adapter using its official three-camera layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cosmos3_droid_window import restore_decoded_droid_panels
from .io import write_json
from .media import write_h264_rgb
from .robotics_contracts import WAMInputManifest, WAMOutputManifest
from .robotics_prompts import (
    copy_reference_prompt_snapshot,
    load_bound_robotics_prompt,
    render_reference_robotics_prompt,
)
from .robotics_quality import certify_wam_output
from .robotics_wam_bundle import (
    build_bundle_artifact_digests,
    discover_preflight_artifacts,
    write_raw_video_certificate,
)
from .robotics_views import (
    compose_cosmos_droid_view,
    load_reference_manifest,
    read_reference_video_frame,
    resolve_artifact,
    resize_with_pad,
)
from .wam_adapter_contracts import (
    COSMOS_SOURCE_MODEL,
    build_cosmos_adapter_contract,
    load_adapter_contract,
    sha256_file,
)
from .wam_lingbot import _read_video


SOURCE_MODEL = COSMOS_SOURCE_MODEL
PROTOCOL = "cosmos3_policy_droid_physical_multiview_v5"
MODEL_SIZE = (640, 540)
FROZEN_FRAMEWORK_REVISION = "ee58e41467f33c49ddde08b4d0ef4923876a95ac"
FROZEN_CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
FROZEN_MODEL_CONFIG_SHA256 = (
    "11577909f49e95d62456b6a497cf7b883f8987a3186d720d838ea5de0e493366"
)
FROZEN_WAN_VAE_SHA256 = (
    "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
)


def _metadata_artifact(
    input_root: Path,
    metadata_file: Path,
    metadata: dict[str, Any],
    field: str,
) -> Path:
    value = metadata.get(field)
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Cosmos metadata {field} must be a relative artifact path")
    path = (metadata_file.parent / value).resolve()
    try:
        path.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"Cosmos metadata {field} escapes its formal bundle") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Cosmos metadata {field} is missing: {path}")
    return path


def _validate_native_execution_provenance(
    *,
    input_root: Path,
    metadata_file: Path,
    metadata: dict[str, Any],
    official_runtime_task: str,
    runtime_controller: str,
    native_fps: float,
    prompt: str,
) -> int:
    if metadata.get("schema_version") != "cosmos_worldir_isaac_closed_loop.v3":
        raise ValueError("Cosmos formal run must come from the native WorldIR/Isaac loop")
    if metadata.get("status") != "complete" or metadata.get("terminated_early") is not False:
        raise ValueError("Cosmos native rollout is incomplete or terminated early")
    if metadata.get("physics_stability_pass") is not True:
        raise ValueError("Cosmos native rollout failed its physics-stability gate")
    if metadata.get("action_source") != "cosmos3_nano_policy_droid":
        raise ValueError("Cosmos formal output must come from live policy inference")
    runtime_path = _metadata_artifact(
        input_root, metadata_file, metadata, "runtime_identity"
    )
    runtime_identity = json.loads(runtime_path.read_text(encoding="utf-8"))
    identity = runtime_identity.get("identity")
    if (
        runtime_identity.get("schema_version") != "wam_runtime_identity.v1"
        or not isinstance(identity, dict)
        or identity.get("profile") != "cosmos3_nano_droid"
        or identity.get("source_model") != SOURCE_MODEL
        or runtime_identity.get("identity_sha256")
        != metadata.get("runtime_identity_sha256")
        or identity.get("checkpoint", {}).get("resolved_revision")
        != metadata.get("checkpoint_revision")
    ):
        raise ValueError("Cosmos native runtime identity changed")
    world_hash = metadata.get("worldir_hash")
    if not isinstance(world_hash, str) or len(world_hash) != 64:
        raise ValueError("Cosmos native rollout is missing its WorldIR identity")
    plan_path = _metadata_artifact(
        input_root, metadata_file, metadata, "worldir_plan"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("world_hash") != world_hash:
        raise ValueError("Cosmos native rollout WorldIR hash differs from its compile plan")
    if metadata.get("prompt") != prompt:
        raise ValueError("Cosmos native rollout changes the selected task prompt")
    actual_frames = int(metadata.get("frame_count") or 0)
    if actual_frames < 2:
        raise ValueError("Cosmos native rollout has no executable timeline")
    if metadata.get("action_count") != actual_frames - 1:
        raise ValueError("Cosmos native action count differs from its actual video timeline")
    if metadata.get("actual_future_steps") != actual_frames - 1:
        raise ValueError("Cosmos native future steps differ from its actual timeline")
    graph_runtime = runtime_controller == "task_graph_runtime"
    legacy_graph = graph_runtime and official_runtime_task == "worldolympiad/task_graph_fixed_horizon"
    expected_controller = ("task_graph_fixed_horizon" if legacy_graph else
                           "task_graph_native_episode" if graph_runtime else "official_robolab_task")
    if metadata.get("rollout_controller") != expected_controller:
        raise ValueError("Cosmos rollout controller differs from its bound runtime mode")
    if metadata.get("official_runtime_task") != official_runtime_task:
        raise ValueError("Cosmos runtime used a different official RoboLab task")
    official_max_steps = int(metadata.get("official_task_max_steps") or 0)
    if official_max_steps < actual_frames - 1:
        raise ValueError("Cosmos timeline exceeds the official task episode")
    actual_duration = float(metadata.get("actual_duration_sec") or 0)
    if abs(actual_duration - actual_frames / native_fps) > 1e-6:
        raise ValueError("Cosmos native duration does not describe its actual timeline")
    reason = metadata.get("termination_reason")
    success = metadata.get("task_success")
    from .cosmos_evaluation_window import validate_binding
    input_spec = WAMInputManifest.model_validate_json((input_root / "input_manifest.json").read_text())
    fixed_window = validate_binding(metadata, input_spec.simulator_state)
    allowed_reasons = (
        {"evaluation_window_complete"} if fixed_window else {"reference_horizon_complete"} if legacy_graph
        else {"task_completed", "native_task_timeout"} if graph_runtime
        else {"task_completed", "official_task_timeout"}
    )
    if reason not in allowed_reasons:
        raise ValueError("Cosmos native rollout has no formal completion reason")
    if legacy_graph and success is not False:
        raise ValueError("fixed-horizon task graph may not claim task success")
    if not fixed_window and not legacy_graph and success is not (reason == "task_completed"):
        raise ValueError("Cosmos task success disagrees with its completion reason")
    if legacy_graph and actual_frames != official_max_steps + 1:
        raise ValueError("task-graph runtime did not complete the reference horizon")
    if reason in {"official_task_timeout", "native_task_timeout"} and actual_frames != official_max_steps + 1:
        raise ValueError("Cosmos timeout did not come from the official task episode")
    if reason == "task_completed":
        evidence = metadata.get("completion_evidence")
        if not isinstance(evidence, dict) or metadata.get("completion_frame") != actual_frames - 1:
            raise ValueError("Cosmos task completion lacks terminal-frame evidence")
        if evidence.get("official_predicate_passed") is not True:
            raise ValueError("Cosmos terminal state failed the official task predicate")
    windows = metadata.get("windows")
    expected_window_count = (actual_frames - 1 + 31) // 32
    if not isinstance(windows, list) or len(windows) != expected_window_count:
        raise ValueError("Cosmos native rollout is missing a 32-step execution window")
    if metadata.get("window_count") != expected_window_count:
        raise ValueError("Cosmos native window count differs from its capture metadata")
    executed_total = 0
    capture_root = metadata_file.parent
    for index, record in enumerate(windows):
        if not isinstance(record, dict) or record.get("window_index") != index:
            raise ValueError(f"Cosmos native window {index} has invalid provenance")
        planned_actions = min(32, official_max_steps - executed_total)
        executed_actions = int(record.get("executed_action_count") or 0)
        if (
            record.get("conditioning_frame") != executed_total
            or record.get("planned_action_count") != planned_actions
            or not 1 <= executed_actions <= planned_actions
        ):
            raise ValueError(f"Cosmos native window {index} has invalid action bounds")
        if index < expected_window_count - 1 and executed_actions != planned_actions:
            raise ValueError("only the final completion window may stop before its plan ends")
        model_metadata = record.get("model_metadata")
        if (
            not isinstance(model_metadata, dict)
            or model_metadata.get("schema_version") != "cosmos3_droid_server_window.v1"
            or model_metadata.get("action_source")
            != "official_robolab_websocket_server"
            or model_metadata.get("request_index") != index
        ):
            raise ValueError(f"Cosmos native window {index} is not official server output")
        if metadata.get('prompt_format_evidence') == 'per_window_server_transform_attestation':
            if (not isinstance(model_metadata.get('format_prompt_as_json'), bool)
                    or model_metadata['format_prompt_as_json'] != metadata.get('format_prompt_as_json')
                    or model_metadata.get('prompt_format_evidence') != 'server_transform_attestation'):
                raise ValueError(f'Cosmos native window {index} prompt format is not attested consistently')
        window_root = capture_root / "windows" / f"window_{index:06d}"
        actions_path = window_root / "executed_actions.npy"
        video_path = window_root / "model" / "droid_concat_rollout.npy"
        if not actions_path.is_file() or not video_path.is_file():
            raise FileNotFoundError(f"Cosmos native window {index} is incomplete")
        actions = np.load(actions_path, allow_pickle=False)
        video = np.load(video_path, mmap_mode="r", allow_pickle=False)
        if actions.shape != (executed_actions, 8) or not np.isfinite(actions).all():
            raise ValueError(f"Cosmos native window {index} has invalid executed actions")
        if video.ndim != 4 or video.shape[0] < executed_actions or video.shape[-1] != 3:
            raise ValueError(f"Cosmos native window {index} has invalid decoded video")
        executed_total += executed_actions
    if executed_total != actual_frames - 1:
        raise ValueError("Cosmos native actions do not cover its actual rollout timeline")
    return actual_frames


COSMOS_RUNTIME_PROFILES = {
    "in_container_single": {
        "official_runtime_task": (
            "robolab/tasks/benchmark/banana_in_bowl_task.py:BananaInBowlTask"
        ),
        "official_runtime_task_sha256": (
            "ceaba5a9944cf6a9cd1ba6809a73251c01786059c0d4bd02604bf0c9c824ab4a"
        ),
        "predicate_family": "in_container",
    },
    "stacked": {
        "official_runtime_task": (
            "robolab/tasks/benchmark/stack_yellow_on_red.py:StackYellowOnRedTask"
        ),
        "official_runtime_task_sha256": (
            "68a73fd9f0a0ccb95e190d45e9b0eea25d9d24e856469b01e671489c56e5f695"
        ),
        "predicate_family": "stacked",
    },
    "on_region": {
        "official_runtime_task": (
            "robolab/tasks/benchmark/white_mug_in_center_of_table.py:"
            "WhiteMugInCenterOfTableTask"
        ),
        "official_runtime_task_sha256": (
            "748afe94f9ff85255beb65a12e22ad661c93559cbe05d9bd509b483f5486a1aa"
        ),
        "predicate_family": "on_region",
    },
    "inserted": {
        # RoboLab does not publish a predicate named ``inserted``.  Its
        # insertion-like marker-in-mug benchmark uses the official
        # object_in_container predicate with gripper detachment.  The WorldIR
        # compiler separately certifies the stricter insertion depth,
        # alignment, contact, and release geometry for the reference.
        "official_runtime_task": (
            "robolab/tasks/benchmark/marker_in_mug_task.py:MarkerInMugTask"
        ),
        "official_runtime_task_sha256": (
            "e6dd8c9f96a1f5a3dbf5792dd72e4246940429c520b39a5c2f8fbe05781f70eb"
        ),
        "predicate_family": "inserted",
    },
    "covering": {
        # RoboLab has no predicate literally named ``covering``.  Its
        # single-object on-top benchmark places one object on another and
        # terminates with the official object_on_top predicate plus gripper
        # detachment.  The WorldIR compiler separately certifies the stricter
        # lid-footprint/opening alignment, contact, and release geometry for
        # the reference.
        "official_runtime_task": (
            "robolab/tasks/benchmark/banana_on_plate_task.py:BananaOnPlateTask"
        ),
        "official_runtime_task_sha256": (
            "c07cd538e616ae5fbefa4a8a7b342ce5a0c0da95d12e2329c2bd9d972ce025fc"
        ),
        "predicate_family": "covering",
    },
}

# Runtime profiles whose terminal relation the native WorldIR/RoboLab executor
# can actually evaluate.  ``scripts/run_cosmos3_worldir_isaac_closed_loop.py``
# whitelists the official RoboLab success functions ``object_in_container``,
# ``stacked``, ``object_on_center`` and ``object_on_top`` and binds their
# parameters to the WorldIR subject/destination roles, which are exactly the
# families these profiles select (``inserted`` and ``covering`` reuse
# ``object_in_container``/``object_on_top`` with stricter reference-side
# geometry certificates).  The gate therefore follows the executable adapter
# instead of the historical single in-container sample; any other family stays
# fail-closed.
COSMOS_EXECUTABLE_PROFILE_KEYS = frozenset(
    {
        "in_container_single",
        "stacked",
        "on_region",
        "inserted",
        "covering",
        "task_graph_fixed_horizon",
        "task_graph_native_episode",
    }
)

#: Official RoboLab success functions the WorldIR executor implements.
COSMOS_EXECUTABLE_SUCCESS_FUNCTIONS = frozenset(
    {
        "object_in_container",
        "stacked",
        "object_on_center",
        "object_on_top",
    }
)


def require_executable_runtime_profile(runtime: dict[str, Any]) -> None:
    """Fail closed unless the selected Cosmos runtime profile is executable.

    Profiles are adapter capabilities selected from typed predicates; a profile
    that names no implemented RoboLab success function must never reach the
    Isaac/PhysX backend, because the model would then run without an official
    termination relation.
    """

    key = runtime.get("profile_key")
    if key not in COSMOS_EXECUTABLE_PROFILE_KEYS:
        raise ValueError(
            "native Cosmos WorldIR execution does not yet implement this typed "
            f"runtime profile: {key}"
        )


COSMOS_NATIVE_ROBOT_USD = (
    "cache/official_assets/robolab/f72d39445ab699c9cfd9bf551e8d21b6075293b9/"
    "franka_robotiq_2f_85_flattened.usd"
)
COSMOS_NATIVE_ROBOT_USD_SHA256 = (
    "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41"
)


def _runtime_profile_for_reference(
    reference_path: Path,
    reference: Any,
) -> dict[str, Any]:
    """Select one Cosmos/RoboLab runtime profile from typed predicates.

    Profiles are adapter capabilities, not sample identities. Case ids, object
    labels, colors, and historical recipe rows never participate in routing.
    """

    if reference.schema_version != "robotics_reference_bundle.v6":
        raise ValueError("Cosmos agentic input requires recipe-free reference v6")
    if reference.recipe_id is not None or not reference.task_graph:
        raise ValueError("Cosmos agentic input must not name a sample recipe")
    graph_path = resolve_artifact(reference_path, reference.task_graph)
    if not graph_path.is_file() or sha256_file(graph_path) != reference.task_graph_sha256:
        raise ValueError("Cosmos task graph is missing or changed")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    predicates = [
        record
        for record in graph.get("success_program", [])
        if isinstance(record, dict)
        and isinstance(record.get("predicate"), str)
    ]
    if not predicates:
        raise ValueError("Cosmos task graph has no typed terminal predicates")
    operators = [
        record.get("operator") for record in graph.get("controller_program", [])
        if isinstance(record, dict) and isinstance(record.get("operator"), str)
    ]
    native_operator_families = {"pick", "place", "insert", "flip", "open", "close"}
    graph_runtime = (
        any(record["predicate"] not in {
            "in_container", "inside", "stacked", "on_region", "inserted", "covering"
        } for record in predicates)
        or any(operator not in native_operator_families for operator in operators)
    )
    families = {
        "in_container" if record["predicate"] in {"in_container", "inside"}
        else str(record["predicate"]) for record in predicates
    }
    graph_runtime |= len(families) != 1 or len(predicates) != 1
    # The released center-of-table sample neither covers a whole support
    # region nor checks a declared upright goal. Bind these task semantics
    # to the shared measured-goal executor instead of changing their meaning.
    graph_runtime |= any(record["predicate"] == "on_region" or record.get("terminal_upright")
                         or record.get("reorientation_required") for record in predicates)
    if graph_runtime:
        from .cosmos_task_graph_predicates import SUPPORTED, protocol
        unknown = {record["predicate"] for record in predicates} - SUPPORTED
        if unknown:
            raise ValueError(f"missing native terminal measurement: {sorted(unknown)}")
        native = protocol()
        return {
            "official_runtime_task": "worldolympiad/task_graph_native_episode",
            "official_runtime_task_sha256": sha256_file(graph_path),
            "predicate_family": "task_graph", "profile_key": "task_graph_native_episode",
            "schema_version": "cosmos3_droid_runtime_selection.v2",
            "selection_policy": "typed_task_graph_native_predicates",
            "task_graph": str(graph_path), "task_graph_sha256": sha256_file(graph_path),
            "terminal_predicates": predicates, "controller_operators": operators,
            "termination_policy": "native_predicates_or_independent_episode_timeout",
            "native_episode_length_s": native["episode_length_s"],
            "native_protocol_sha256": native["sha256"],
        }
    family = next(iter(families))
    profile_key = "in_container_single" if family == "in_container" else family
    profile = dict(COSMOS_RUNTIME_PROFILES[profile_key])
    profile.update(
        {
            "schema_version": "cosmos3_droid_runtime_selection.v1",
            "selection_policy": "typed_terminal_predicate_family",
            "profile_key": profile_key,
            "task_graph": str(graph_path),
            "task_graph_sha256": reference.task_graph_sha256,
            "terminal_predicates": predicates,
        }
    )
    return profile


def _arm_state(trajectory: Path, frame_index: int) -> tuple[list[str], list[float], float]:
    with np.load(trajectory, allow_pickle=False) as payload:
        names = [str(value) for value in payload["robot_joint_names"].tolist()]
        positions = np.asarray(payload["robot_joint_positions"], dtype=np.float64)
        droid_gripper = (
            np.asarray(payload["droid_gripper_position"], dtype=np.float64)
            if "droid_gripper_position" in payload
            else None
        )
    required = [f"panda_joint{index}" for index in range(1, 8)]
    missing = [name for name in required if name not in names]
    if missing or positions.ndim != 2 or not 0 <= frame_index < positions.shape[0]:
        raise ValueError(f"reference robot state is incomplete: missing={missing}")
    row = positions[frame_index]
    arm = [float(row[names.index(name)]) for name in required]
    if droid_gripper is not None:
        if droid_gripper.ndim != 1 or frame_index >= len(droid_gripper):
            raise ValueError("reference DROID gripper timeline is incomplete")
        gripper = float(droid_gripper[frame_index])
    else:
        fingers = [row[names.index(name)] for name in names if "finger_joint" in name]
        if len(fingers) != 2:
            raise ValueError(
                "reference needs a DROID gripper channel or exactly two Panda fingers"
            )
        # Legacy Panda references encode two opening distances in metres.
        panda_max_width_m = 0.08
        opening_width_m = float(np.clip(np.sum(fingers), 0.0, panda_max_width_m))
        gripper = 1.0 - opening_width_m / panda_max_width_m
    if not 0.0 <= gripper <= 1.0:
        raise ValueError("reference DROID gripper value lies outside [0,1]")
    if not np.isfinite(np.asarray([*arm, gripper])).all():
        raise ValueError("reference robot state contains non-finite values")
    return required, arm, gripper


def prepare_input(
    reference_manifest: str | Path,
    output_root: str | Path,
    *,
    source_model: str = SOURCE_MODEL,
    start_frame_index: int | None = None,
) -> Path:
    if source_model != SOURCE_MODEL:
        raise ValueError(
            f"Cosmos DROID adapter is frozen to {SOURCE_MODEL!r}, got {source_model!r}"
        )
    if start_frame_index not in (None, 0):
        raise ValueError("formal Cosmos adaptation must start at reference frame zero")
    reference_path, reference = load_reference_manifest(reference_manifest)
    prompt_contract = render_reference_robotics_prompt(
        reference_path, "cosmos3_nano_droid"
    )
    runtime_profile = _runtime_profile_for_reference(reference_path, reference)
    official_runtime_task = runtime_profile["official_runtime_task"]
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Cosmos input bundle: {output}")
    output.mkdir(parents=True)
    reference_snapshot, reference_task_snapshot, reference_task_sha256 = (
        copy_reference_prompt_snapshot(reference_path, output)
    )
    start_frame = 0 if start_frame_index is None else start_frame_index
    formal_view_id = "exterior_left"
    formal_view_selection = {
        "policy": "registered_droid_fixed_exterior_left_for_calibrated_cae_v2",
        "generated_output_consulted": False,
        "selected_view": formal_view_id,
        "reason": (
            "Dream2Flow formal 3-D object flow requires a fixed calibrated camera; "
            "the generated wrist plane has no per-frame extrinsics"
        ),
    }
    formal_view = reference.view(formal_view_id)
    from .cosmos_evaluation_window import from_reference
    evaluation_window = from_reference(reference_path, reference, formal_view_id)
    reference_frame_count = formal_view.frame_count - start_frame
    if reference_frame_count < 2:
        raise ValueError("Cosmos start frame leaves no formal reference rollout")
    adapter = build_cosmos_adapter_contract(
        case_id=reference.case_id,
        reference_fps=formal_view.fps,
        reference_frame_count=reference_frame_count,
    )
    source_views = adapter.source_views
    camera_by_view = {camera.source_view: camera for camera in adapter.camera_inputs}
    for view_id in source_views.values():
        view = reference.view(view_id)  # type: ignore[arg-type]
        if view.frame_count != formal_view.frame_count or abs(view.fps - formal_view.fps) > 1e-6:
            raise ValueError("Cosmos physical views must share one complete reference timeline")
        expected_size = camera_by_view[view_id].size_wh
        if (view.width, view.height) != expected_size:
            raise ValueError(
                "Cosmos requires SceneCode to render each reference view at its "
                f"native panel size: {view_id}={(view.width, view.height)} "
                f"expected={expected_size}"
            )
    adapter_path = adapter.write_json(output / "adapter_contract.json")
    prompt_path = prompt_contract.write_json(output / "prompt_manifest.json")
    frames = {
        view_id: read_reference_video_frame(
            reference_path, reference, view_id, start_frame
        )
        for view_id in source_views.values()
    }
    images: dict[str, str] = {}
    for view_id, frame in frames.items():
        path = output / f"conditioning_{view_id}.png"
        expected_size = camera_by_view[view_id].size_wh
        if frame.shape[:2] != (expected_size[1], expected_size[0]):
            raise ValueError(
                "Cosmos native reference changed size after decoding: "
                f"{view_id}={frame.shape[1]}x{frame.shape[0]}"
            )
        prepared = np.ascontiguousarray(frame)
        if not cv2.imwrite(str(path), prepared):
            raise RuntimeError(f"unable to write Cosmos view: {path}")
        images[view_id] = str(path.relative_to(output))
    concat = compose_cosmos_droid_view(
        frames["wrist"], frames["exterior_left"], frames["exterior_right"]
    )
    concat_path = output / "conditioning_droid_concat.png"
    if not cv2.imwrite(str(concat_path), concat):
        raise RuntimeError(f"unable to write Cosmos concat view: {concat_path}")
    joint_names, joints, gripper = _arm_state(
        resolve_artifact(reference_path, reference.trajectory), start_frame
    )
    manifest = WAMInputManifest(
        case_id=reference.case_id,
        wam="cosmos3_nano_droid",
        source_model=source_model,
        task_instruction=reference.task_instruction,
        model_conditioning_prompt=prompt_contract.rendered_prompt,
        prompt_manifest=prompt_path.name,
        prompt_manifest_sha256=sha256_file(prompt_path),
        reference_manifest=str(reference_path),
        reference_manifest_sha256=sha256_file(reference_path),
        reference_snapshot=reference_snapshot,
        reference_task_snapshot=reference_task_snapshot,
        reference_task_sha256=reference_task_sha256,
        adapter_contract=adapter_path.name,
        adapter_contract_sha256=sha256_file(adapter_path),
        input_images=images,
        source_views=source_views,  # type: ignore[arg-type]
        formal_reference_view=formal_view_id,
        reference_start_frame_index=start_frame,
        reference_frame_count=reference_frame_count,
        runtime_controller=(
            "task_graph_runtime"
            if runtime_profile["profile_key"] == "task_graph_native_episode"
            else "official_robolab_task"
        ),
        official_runtime_task=official_runtime_task,
        formal_reference_transform="identity_exact",
        model_input_size_wh=MODEL_SIZE,
        simulator_state={
            "protocol": PROTOCOL,
            "evaluation_window": evaluation_window,
            "adapter_profile": adapter.profile_id,
            "concat_image": str(concat_path.relative_to(output)),
            "action_space": "joint_pos",
            "joint_names": joint_names,
            "initial_joint_position": joints,
            "initial_gripper_position": gripper,
            "gripper_position_convention": "droid_raw_0_open_1_closed",
            "conditioning_prompt_policy": "typed_robotics_prompt_manifest_v2",
            "runtime_profile_key": runtime_profile["profile_key"],
            "runtime_profile_selection_policy": runtime_profile["selection_policy"],
            "runtime_predicate_family": runtime_profile["predicate_family"],
            "official_runtime_task_sha256": runtime_profile[
                "official_runtime_task_sha256"
            ],
            "task_graph_sha256": runtime_profile["task_graph_sha256"],
            **({"native_episode_length_s": runtime_profile["native_episode_length_s"],
                "native_protocol_sha256": runtime_profile["native_protocol_sha256"]}
               if runtime_profile["profile_key"] == "task_graph_native_episode" else {}),
            "terminal_predicates": runtime_profile["terminal_predicates"],
            "formal_view_selection": formal_view_selection,
            "timeline_policy": "conditioning_frame_zero_plus_future_frames_v1",
            "conditioning_frame_selection": (
                "full_reference_rollout_from_frame_zero_v1"
                if start_frame_index is None
                else "explicit_simulator_candidate_v1"
            ),
        },
    )
    return manifest.write_json(output / "input_manifest.json")


def finalize_output(
    input_manifest: str | Path,
    raw_video: str | Path,
    output_video: str | Path,
    metadata_path: str | Path,
    *,
    raw_view: str | None = None,
) -> Path:
    input_path = Path(input_manifest).resolve()
    spec = WAMInputManifest.model_validate_json(input_path.read_text(encoding="utf-8"))
    if spec.wam != "cosmos3_nano_droid":
        raise ValueError("input manifest is not for Cosmos3 Nano DROID")
    _prompt_path, prompt_contract = load_bound_robotics_prompt(
        input_path,
        schema_version=spec.schema_version,
        reference_manifest=spec.reference_manifest,
        reference_manifest_sha256=spec.reference_manifest_sha256,
        reference_snapshot=spec.reference_snapshot,
        reference_task_snapshot=spec.reference_task_snapshot,
        reference_task_sha256=spec.reference_task_sha256,
        case_id=spec.case_id,
        task_instruction=spec.task_instruction,
        wam=spec.wam,
        prompt_manifest=spec.prompt_manifest,
        prompt_manifest_sha256=spec.prompt_manifest_sha256,
        rendered_prompt=spec.model_conditioning_prompt,
    )
    _adapter_path, adapter = load_adapter_contract(
        input_path, spec.adapter_contract, spec.adapter_contract_sha256
    )
    if spec.runtime_controller not in {"official_robolab_task", "task_graph_runtime"}:
        raise ValueError("Cosmos rollout needs a supported native simulator controller")
    adapter.assert_input_binding(
        wam=spec.wam,
        source_model=spec.source_model,
        source_views=spec.source_views,
        formal_reference_view=spec.formal_reference_view,
        reference_start_frame_index=spec.reference_start_frame_index,
        reference_frame_count=spec.reference_frame_count,
        model_input_size_wh=spec.model_input_size_wh,
    )
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    metadata_file = Path(metadata_path).resolve()
    if metadata.get("model") != spec.source_model:
        raise ValueError("loaded Cosmos model does not match the input manifest")
    frozen_identity = {
        "checkpoint_revision": FROZEN_CHECKPOINT_REVISION,
        "framework_revision": FROZEN_FRAMEWORK_REVISION,
        "model_config_sha256": FROZEN_MODEL_CONFIG_SHA256,
        "wan_vae_sha256": FROZEN_WAN_VAE_SHA256,
    }
    changed_identity = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in frozen_identity.items()
        if metadata.get(key) != expected
    }
    if changed_identity:
        raise ValueError(f"Cosmos frozen upstream identity changed: {changed_identity}")
    if metadata.get("action_space") != "joint_pos":
        raise ValueError("Cosmos3-Nano-Policy-DROID requires the joint_pos action space")
    if metadata.get("action_postprocessing") != "official_robolab_gripper_gt_0.5_binary_v1":
        raise ValueError("Cosmos DROID actions must use the official RoboLab gripper threshold")
    if metadata.get("observation_feedback") != "simulator_closed_loop":
        raise ValueError("Cosmos formal continuation must use fresh simulator feedback")
    if metadata.get("closed_loop_horizon_steps") != 32:
        raise ValueError("Cosmos simulator refresh must use the released 32-step horizon")
    if metadata.get("benchmark_target") != "imagination_video_only" or metadata.get("actions_scored") is not False:
        raise ValueError("Cosmos metadata must preserve video-only benchmark semantics")
    expected_prompt = prompt_contract.rendered_prompt
    if metadata.get("prompt") != expected_prompt:
        raise ValueError("Cosmos inference prompt differs from the frozen model prompt")
    if not isinstance(metadata.get("format_prompt_as_json"), bool):
        raise ValueError("Cosmos DROID inference must record its effective prompt format")
    native_fps = float(metadata.get("native_action_hz") or 0.0)
    if abs(native_fps - adapter.temporal.native_output_fps) > 1e-6:
        raise ValueError("Cosmos metadata does not use the adapter's native 15 fps")
    if metadata.get("conditioning_frame_included") is not True:
        raise ValueError("Cosmos output must include its conditioning frame exactly once")
    if metadata.get("video_stitching_policy") != "fresh_condition_once_per_32_step_window_v1":
        raise ValueError("Cosmos closed-loop video uses an unsupported overlap policy")
    if metadata.get("refresh_horizon_steps") != 32:
        raise ValueError("Cosmos interaction provenance must preserve 32-step refreshes")
    actual_frames = _validate_native_execution_provenance(
        input_root=input_path.parent,
        metadata_file=metadata_file,
        metadata=metadata,
        official_runtime_task=spec.official_runtime_task,
        runtime_controller=spec.runtime_controller,
        native_fps=adapter.temporal.native_output_fps,
        prompt=expected_prompt,
    )
    raw = Path(raw_video).resolve()
    if raw_view != adapter.output.formal_view:
        raise ValueError(
            "Cosmos raw video must be explicitly bound to the adapter's formal view"
        )
    frames, fps = _read_video(raw)
    height, width = frames[0].shape[:2]
    if (width, height) not in {(320, 180), (640, 360), (640, 528), (640, 540)}:
        raise ValueError(f"unexpected Cosmos DROID output shape: {(width, height)}")
    if abs(fps - adapter.temporal.native_output_fps) > 1e-6:
        raise ValueError("Cosmos output FPS differs from its frozen adapter profile")
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("Cosmos output changes spatial size")
    if adapter.output.formal_view != "exterior_left":
        raise ValueError("Cosmos formal output is not the registered fixed exterior view")
    if (width, height) == (320, 180):
        crop_xywh = (0, 0, 320, 180)
        selected = frames
    elif height == 360:
        crop_xywh = (0, 0, 640, 360)
        selected = frames
    else:
        _wrist, left, _right, _panel_restoration = restore_decoded_droid_panels(
            np.stack(frames)
        )
        crop_xywh = (0, 360, 320, height - 360)
        selected = list(left)
    formal_width, formal_height = adapter.output.formal_size_wh
    if (selected[0].shape[1], selected[0].shape[0]) != (
        formal_width,
        formal_height,
    ):
        selected = [
            resize_with_pad(frame, formal_width, formal_height) for frame in selected
        ]
    if len(selected) != actual_frames:
        raise ValueError(
            "Cosmos decoded video differs from its task-terminated simulator timeline"
        )
    destination = Path(output_video).resolve()
    write_h264_rgb(
        destination,
        (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in selected),
        fps,
        crf=16,
    )
    conditioning = cv2.imread(
        str(resolve_artifact(input_path, spec.input_images[adapter.output.formal_view])),
        cv2.IMREAD_COLOR,
    )
    if conditioning is None:
        raise RuntimeError("Cosmos formal conditioning image cannot be decoded")
    conditioning = resize_with_pad(conditioning, formal_width, formal_height)
    mae = float(np.mean(np.abs(selected[0].astype(np.float32) - conditioning.astype(np.float32))))
    quality_path = certify_wam_output(
        destination,
        input_path.parent / "quality_certificate.json",
        expected_width=formal_width,
        expected_height=formal_height,
        expected_frame_count=len(selected),
        expected_fps=fps,
    )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("passed") is not True:
        raise ValueError(f"Cosmos imagination quality gate failed: {quality_path}")
    raw_certificate = write_raw_video_certificate(
        bundle_root=input_path.parent,
        wam="cosmos3_nano_droid",
        raw_video=raw,
        model_metadata=metadata_file,
        selected_view=adapter.output.formal_view,
        crop_xywh=crop_xywh,
        extraction_policy=adapter.output.extraction_policy,
    )
    result = WAMOutputManifest(
        case_id=spec.case_id,
        wam="cosmos3_nano_droid",
        source_model=spec.source_model,
        checkpoint_revision=str(metadata.get("checkpoint_revision") or "unknown"),
        input_manifest=input_path.name,
        adapter_contract_sha256=spec.adapter_contract_sha256,
        model_metadata=str(metadata_file.relative_to(input_path.parent)),
        raw_video=str(raw.relative_to(input_path.parent)),
        raw_video_certificate=raw_certificate.name,
        selected_view=adapter.output.formal_view,
        crop_xywh=crop_xywh,
        aligned_video=str(destination.relative_to(input_path.parent)),
        width=formal_width,
        height=formal_height,
        fps=fps,
        frame_count=len(selected),
        duration_sec=len(selected) / fps,
        termination_reason=metadata["termination_reason"],
        task_success=bool(metadata["task_success"]),
        conditioning_frame_mae_8bit=mae,
        quality_certificate=quality_path.name,
        quality_passed=True,
        **discover_preflight_artifacts(input_path.parent),
        bundle_artifacts_sha256=build_bundle_artifact_digests(input_path.parent),
    )
    return result.write_json(input_path.parent / "output_manifest.json")


def finalize_run(run_root: str | Path) -> Path:
    root = Path(run_root).resolve()
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/output_manifest.json")):
        manifest = WAMOutputManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.wam == "cosmos3_nano_droid":
            records.append({"case_id": manifest.case_id, "manifest": str(path)})
    if not records:
        raise FileNotFoundError(f"no completed Cosmos cases below {root}")
    return write_json(
        root / "wam_run_manifest.json",
        {
            "schema_version": "wam_run.v2",
            "wam": "cosmos3_nano_droid",
            "source_model": SOURCE_MODEL,
            "case_count": len(records),
            "cases": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("reference_manifest")
    prepare.add_argument("output_root")
    prepare.add_argument("--source-model", default=SOURCE_MODEL)
    prepare.add_argument("--start-frame-index", type=int)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("input_manifest")
    finalize.add_argument("raw_video")
    finalize.add_argument("output_video")
    finalize.add_argument("metadata")
    finalize.add_argument(
        "--raw-view",
        choices=("wrist", "exterior_left", "exterior_right"),
        required=True,
    )
    run = sub.add_parser("finalize-run")
    run.add_argument("run_root")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_input(
            args.reference_manifest,
            args.output_root,
            source_model=args.source_model,
            start_frame_index=args.start_frame_index,
        )
    elif args.command == "finalize":
        result = finalize_output(
            args.input_manifest,
            args.raw_video,
            args.output_video,
            args.metadata,
            raw_view=args.raw_view,
        )
    else:
        result = finalize_run(args.run_root)
    print(result)


if __name__ == "__main__":
    main()
