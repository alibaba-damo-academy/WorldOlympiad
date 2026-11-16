from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from autoresearch.worldolympiad_case import prepare_case
from autoresearch.robotics_case import EVALUATOR_PREFIX
from autoresearch.wm_cae import finalize_manifest
from batch_test.make_manifest import build_item
from continuous_action.metrics import (
    FORMAL_INDEPENDENT_METRIC_VERSION,
    FORMAL_METRIC_PROFILE_ID,
    FORMAL_METRIC_SCHEMA_VERSION,
    formal_metric_config,
    formal_metric_config_sha256,
)
from continuous_action.workspaces import robot_workspace_contract_for_model


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    for value in (0, 64, 128, 255):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()


def test_robotics_wam_names_map_to_official_evaluator_prefixes() -> None:
    assert EVALUATOR_PREFIX == {
        "lingbot_va_base": "lingbot_world",
        "lingbot_va_robotwin": "lingbot_world",
        "cosmos3_nano_droid": "cosmos",
    }


def test_prepare_case_materializes_original_worldolympiad_inputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "episode_spec.json").write_text(
        json.dumps({"episode_id": "block_sort_tray"}), encoding="utf-8"
    )
    (reference / "benchmark_certificate.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    _write_video(reference / "rgb.mp4")
    (reference / "camera_calibration_info.json").write_text("{}", encoding="utf-8")
    (reference / "initial_depth.pt").write_bytes(b"test-depth")
    (reference / "scene_data.yaml").write_text(
        "object_name: red rubber ball\n", encoding="utf-8"
    )
    wm_video = tmp_path / "wm.mp4"
    _write_video(wm_video)

    manifest_path = prepare_case(
        reference, wm_video, tmp_path / "cases", model_id="cosmos"
    )
    case_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    assert (case_dir / "ref_block_sort_tray.mp4").is_file()
    assert (case_dir / "cosmos_gen_block_sort_tray.mp4").is_file()
    assert (case_dir / "prompt.json").is_file()
    assert (case_dir / "cosmos_gen_block_sort_tray_chunk_timestamps.json").is_file()
    assert (case_dir / "cae_pair_spec.json").is_file()
    assert (case_dir / "cae_base_scene" / "initial_depth.pt").is_file()
    assert manifest["tracks"]["physical"] == "required"
    assert manifest["tracks"]["geometry"] == "required"
    assert manifest["tracks"]["interaction"] == "required"
    assert manifest["tracks"]["continuous_action"].startswith("pending_")
    assert manifest["benchmark_status"] == "pipeline_validation"
    assert manifest["leaderboard_eligible"] is False
    cae_spec = json.loads((case_dir / "cae_pair_spec.json").read_text())
    assert cae_spec["benchmark_status"] == "pipeline_validation"


def test_prepare_case_rejects_uncertified_reference(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "episode_spec.json").write_text(
        json.dumps({"episode_id": "block_sort_tray"}), encoding="utf-8"
    )
    (reference / "benchmark_certificate.json").write_text(
        json.dumps({"status": "quarantined"}), encoding="utf-8"
    )
    _write_video(reference / "rgb.mp4")
    wm_video = tmp_path / "wm.mp4"
    _write_video(wm_video)
    try:
        prepare_case(reference, wm_video, tmp_path / "cases", model_id="cosmos")
    except ValueError as error:
        assert "not certified" in str(error)
    else:
        raise AssertionError("uncertified reference was accepted")


def _write_official_plan(directory, role, *, fps, frame_count):
    from continuous_action.official_plan import PROTOCOL,TRAJECTORY_SOURCE,UPSTREAM_COMMIT,sha256
    lock=json.loads((Path(__file__).parents[1]/'scripts/official_dream2flow_source_lock.json').read_text())
    plan=directory/'trajectory_optimization_plan.npz';video=directory.parent/'rgb.mp4'
    video.write_bytes(b'fixture video')
    step=max(1,frame_count//lock['optimizer_numerical_defaults']['max_num_timesteps_for_optimization'])
    indices=np.arange(0,frame_count,step)
    poses=np.zeros((len(indices),7));poses[:,0]=indices/fps*.1;poses[:,6]=1.
    np.savez_compressed(plan,ee_trajectory=poses,timestamps=indices/fps,
                        source_frame_indices=indices,observed_frame_mask=np.ones(len(indices),dtype=bool))
    metadata=dict(evaluation_protocol=PROTOCOL,trajectory_source=TRAJECTORY_SOURCE,upstream_commit=UPSTREAM_COMMIT,
      video_role=role,source_video_sha256=sha256(video),plan_sha256=sha256(plan),
      source_sha256=lock['source_sha256'],optimizer_config=lock['optimizer_numerical_defaults'],
      initialization_kind='frame_zero_kinematic_grasp',initial_pose_source='registered_grasp_target',
      object_to_robot_ik_used=True,action_head_used=False,coordinate_frame='registered_world',
      object_name='red rubber ball',source_video_fps=fps,source_video_frame_count=frame_count)
    for name in ['flow','registration','registered_urdf']:
        path=directory/name;path.write_bytes(name.encode())
        metadata[name+'_path']=str(path);metadata[name+'_sha256']=sha256(path)
    plan.with_suffix('.json').write_text(json.dumps(metadata))


def test_finalize_cae_manifest_requires_symmetric_official_plans(tmp_path: Path) -> None:
    from continuous_action.current_metric import CURRENT_METRIC_VERSION,current_metric_config,current_metric_config_sha256
    case = tmp_path / "case"
    for role in ("reference", "imagination"):
        flow = case / "cae_pair" / role / "dream2flow" / "object_flow_result.pt"
        flow.parent.mkdir(parents=True)
        flow.write_bytes(b"flow")
        flow.with_suffix(".npz").write_bytes(b"portable-flow")
        _write_official_plan(flow.parent, role, fps=15.0 if role=="reference" else 10.0,
                             frame_count=60 if role=="reference" else 70)
    original_pair_metadata = json.dumps(
        {
            "timing_policy": "independent_native_v1",
            "role_videos": {
                "reference": {"duration_sec": 4.0, "fps": 15.0},
                "imagination": {"duration_sec": 7.0, "fps": 10.0},
            },
        }
    )
    (case / "cae_pair" / "pair_metadata.json").write_text(
        original_pair_metadata, encoding="utf-8"
    )
    (case / "cae_pair_spec.json").write_text(
        json.dumps(
            {
                "output_dir": "cae_pair",
                "source_model": "cosmos3_nano_droid",
                "object_name": "red rubber ball",
                "robot_workspace_contract": robot_workspace_contract_for_model(
                    "cosmos3_nano_droid"
                ),
            }
        ),
        encoding="utf-8",
    )
    (case / "worldolympiad_case.json").write_text(
        json.dumps({"tracks": {"continuous_action": "pending"}}), encoding="utf-8"
    )
    manifest = json.loads(finalize_manifest(case).read_text())
    assert manifest["benchmark_status"] == "formal"
    assert (
        manifest["evaluation_protocol"]
        == "video_imagination_dream2flow_official_action_trajectory_v2"
    )
    assert manifest["metric_version"] == CURRENT_METRIC_VERSION
    assert (
        manifest["metrics"]["metric_version"]
        == CURRENT_METRIC_VERSION
    )
    assert manifest["metrics"] == current_metric_config()
    assert manifest["metric_config_sha256"] == current_metric_config_sha256()
    assert set(manifest["rollout"]["predictions"]) == {"cosmos3_nano_droid"}
    assert manifest["rollout"]["reference"]["path"].endswith(".npz")
    assert manifest["rollout"]["predictions"]["cosmos3_nano_droid"]["path"].endswith(".npz")
    assert manifest["robot_workspace_contract"] == robot_workspace_contract_for_model(
        "cosmos3_nano_droid"
    )
    assert manifest["rollout"]["end_sec"] == 7.0
    assert (
        (case / "cae_pair" / "pair_metadata.json").read_text()
        == original_pair_metadata
    )
    pair_metadata = json.loads(
        (case / "cae_pair" / "pair_metadata_cae.json").read_text()
    )
    assert (
        pair_metadata["evaluation_protocol"]
        == "video_imagination_dream2flow_official_action_trajectory_v2"
    )
    assert pair_metadata["metric_version"] == CURRENT_METRIC_VERSION
    assert pair_metadata["metric_config_sha256"] == current_metric_config_sha256()
    assert pair_metadata["robot_workspace_contract"] == manifest[
        "robot_workspace_contract"
    ]
    assert pair_metadata["object_to_robot_ik_used"] is True
    case_manifest = json.loads((case / "worldolympiad_case.json").read_text())
    assert case_manifest["tracks"]["continuous_action"] == "required"
    assert (
        case_manifest["continuous_action_manifest"]
        == "continuous_action_manifest.json"
    )
    assert (
        case_manifest["continuous_action_manifests"][CURRENT_METRIC_VERSION]
        == "continuous_action_manifest.json"
    )


    # Raw flow cannot stand in for the second role's optimizer output.
    import pytest
    (case/'cae_pair/imagination/dream2flow/trajectory_optimization_plan.npz').unlink()
    with pytest.raises(FileNotFoundError,match='optimizer output'):
        finalize_manifest(case)






def test_official_batch_builder_accepts_prepared_case(tmp_path: Path) -> None:
    case = tmp_path / "block_sort_tray"
    _write_video(case / "ref_block_sort_tray.mp4")
    _write_video(case / "cosmos_gen_block_sort_tray.mp4")
    (case / "prompt.json").write_text("[]", encoding="utf-8")
    (case / "cosmos_gen_block_sort_tray_chunk_timestamps.json").write_text(
        "{}", encoding="utf-8"
    )
    (case / "continuous_action_manifest.json").write_text("{}", encoding="utf-8")
    item, error = build_item(
        case,
        output_name=None,
        output_name_template="{prefix}_judge_{id}.json",
        output_prefix="cosmos",
        gen_pattern="cosmos_gen_*.mp4",
        ref_pattern="ref_*.mp4",
        chunk_pattern="cosmos_gen_*_chunk_timestamps.json",
        cae_pattern="continuous_action_manifest.json",
        cae_model=None,
        pipeline="cosmos-predict",
    )
    assert error is None
    assert item is not None
    assert item["pipeline"] == "cosmos-predict"
    assert item["cae_manifest"].endswith("continuous_action_manifest.json")
    assert item["output"].endswith("cosmos_judge_block_sort_tray.json")


def test_active_batch_builder_versions_default_judge_output(tmp_path: Path) -> None:
    case = tmp_path / "cup_to_tray"
    _write_video(case / "ref_cup_to_tray.mp4")
    _write_video(case / "cosmos_gen_cup_to_tray.mp4")
    (case / "prompt.json").write_text("[]", encoding="utf-8")
    (case / "cosmos_gen_cup_to_tray_chunk_timestamps.json").write_text(
        "{}", encoding="utf-8"
    )
    (case / "continuous_action_manifest.json").write_text(
        json.dumps({"metric_version": FORMAL_INDEPENDENT_METRIC_VERSION}),
        encoding="utf-8",
    )
    item, error = build_item(
        case,
        output_name=None,
        output_name_template="{prefix}_judge_{id}.json",
        output_prefix="cosmos",
        gen_pattern="cosmos_gen_*.mp4",
        ref_pattern="ref_*.mp4",
        chunk_pattern="cosmos_gen_*_chunk_timestamps.json",
        cae_pattern="continuous_action_manifest.json",
        cae_model="cosmos3_nano_droid",
        pipeline="cosmos-predict",
        version_cae_output=True,
    )
    assert error is None
    assert item is not None
    assert item["output"].endswith("cosmos_judge_cup_to_tray_cae.json")


def test_active_batch_builder_rejects_a_legacy_case_pointer(tmp_path: Path) -> None:
    case = tmp_path / "cup_to_tray"
    _write_video(case / "ref_cup_to_tray.mp4")
    _write_video(case / "cosmos_gen_cup_to_tray.mp4")
    (case / "prompt.json").write_text("[]", encoding="utf-8")
    (case / "cosmos_gen_cup_to_tray_chunk_timestamps.json").write_text(
        "{}", encoding="utf-8"
    )
    (case / "continuous_action_manifest.json").write_text("{}", encoding="utf-8")
    (case / "continuous_action_manifest.json").write_text("{}", encoding="utf-8")
    (case / "worldolympiad_case.json").write_text(
        json.dumps({"continuous_action_manifest": "continuous_action_manifest_v13.json"}),
        encoding="utf-8",
    )

    item, error = build_item(
        case,
        output_name=None,
        output_name_template="{prefix}_judge_{id}.json",
        output_prefix="cosmos",
        gen_pattern="cosmos_gen_*.mp4",
        ref_pattern="ref_*.mp4",
        chunk_pattern="cosmos_gen_*_chunk_timestamps.json",
        cae_pattern="continuous_action_manifest.json",
        cae_model="cosmos3_nano_droid",
        pipeline="cosmos-predict",
    )
    assert item is None
    assert "retired or invalid CAE pointer" in str(error)




def test_batch_builder_rejects_retired_v10_even_when_selected_explicitly(
    tmp_path: Path,
) -> None:
    case = tmp_path / "cup_to_tray"
    _write_video(case / "ref_cup_to_tray.mp4")
    _write_video(case / "cosmos_gen_cup_to_tray.mp4")
    (case / "prompt.json").write_text("[]", encoding="utf-8")
    (case / "cosmos_gen_cup_to_tray_chunk_timestamps.json").write_text(
        "{}", encoding="utf-8"
    )
    (case / "continuous_action_manifest_v10.json").write_text(
        "{}", encoding="utf-8"
    )

    item, error = build_item(
        case,
        output_name="retired.json",
        output_name_template="{prefix}_judge_{id}.json",
        output_prefix="cosmos",
        gen_pattern="cosmos_gen_*.mp4",
        ref_pattern="ref_*.mp4",
        chunk_pattern="cosmos_gen_*_chunk_timestamps.json",
        cae_pattern="continuous_action_manifest_v10.json",
        cae_model="cosmos3_nano_droid",
        pipeline="cosmos-predict",
    )
    assert item is None
    assert "only the current" in str(error)
