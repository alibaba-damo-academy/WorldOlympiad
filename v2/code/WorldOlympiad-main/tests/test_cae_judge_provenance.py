from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from continuous_action.metrics import (
    FORMAL_INDEPENDENT_METRIC_VERSION,
    FORMAL_METRIC_PROFILE_ID,
    FORMAL_METRIC_SCHEMA_VERSION,
    formal_metric_config,
    formal_metric_config_sha256,
)
from continuous_action.schema import ActionTrajectory
from continuous_action.workspaces import robot_workspace_contract_for_model


@pytest.mark.parametrize(
    ("metric_version", "expected_schema", "expected_profile", "expected_components"),
    [
        (
            FORMAL_INDEPENDENT_METRIC_VERSION,
            FORMAL_METRIC_SCHEMA_VERSION,
            FORMAL_METRIC_PROFILE_ID,
            ["trajectory_shape", "speed_curve"],
        ),
    ],
)
def test_judge_preserves_exact_cae_metric_provenance(
    tmp_path: Path,
    metric_version: str,
    expected_schema: str,
    expected_profile: str,
    expected_components: list[str],
) -> None:
    if metric_version == FORMAL_INDEPENDENT_METRIC_VERSION:
        workspace_contract = robot_workspace_contract_for_model(
            "cosmos3_nano_droid"
        )
        anchors = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        translation = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
        for role in ("reference", "imagination"):
            (tmp_path / f"{role}.mp4").write_bytes(role.encode())
            np.savez(
                tmp_path / f"{role}_flow.npz",
                position=anchors[None, :, :] + translation[:, None, :],
                valid_mask=np.ones((3, 2), dtype=bool),
            )
        manifest = {
            "evaluation_protocol": "video_imagination_dream2flow_official_action_trajectory_v2",
            "metric_version": metric_version,
            "metrics": formal_metric_config(),
            "robot_workspace_contract": workspace_contract,
            "rollout": {
                "start_sec": 0.0,
                "end_sec": 1.0,
                "reference": {
                    "type": "dream2flow_video_plan",
                    "path": "reference_flow.npz",
                    "video_path": "reference.mp4",
                    "video_role": "reference",
                    "object_name": "cup",
                    "sample_rate": 2.0,
                    "benchmark_status": "formal",
                },
                "predictions": {
                    "model": {
                        "type": "dream2flow_video_plan",
                        "path": "imagination_flow.npz",
                        "video_path": "imagination.mp4",
                        "video_role": "imagination",
                        "object_name": "cup",
                        "sample_rate": 2.0,
                        "benchmark_status": "formal",
                    }
                },
            },
        }
        manifest["metric_profile"] = FORMAL_METRIC_PROFILE_ID
        manifest["metric_config_sha256"] = formal_metric_config_sha256()
    manifest_path = (
        tmp_path / "continuous_action_manifest.json"
        if metric_version == FORMAL_INDEPENDENT_METRIC_VERSION
        else tmp_path / f"manifest_{metric_version}.json"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    from test_continuous_action import promote_test_flows_to_plan_fixtures
    promote_test_flows_to_plan_fixtures(manifest_path)
    output_path = tmp_path / f"judge_{metric_version}.json"
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "score_worldolympiad.py"),
            "--video",
            str(tmp_path / "unused.mp4"),
            "--skip-physical",
            "--skip-interaction",
            "--skip-3d",
            "--cae-manifest",
            str(manifest_path),
            "--cae-model",
            "model",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    judge = json.loads(output_path.read_text(encoding="utf-8"))
    cae = judge["continuous_action"]
    assert cae["schema_version"] == expected_schema
    assert cae["metric_version"] == metric_version
    assert cae["metric_profile"] == expected_profile
    assert cae["metric_config_sha256"] == (
        formal_metric_config_sha256()
        if metric_version == FORMAL_INDEPENDENT_METRIC_VERSION
        else None
    )
    assert cae["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert cae["scored_components"] == expected_components
    assert cae["robot_workspace_contract"] == (
        workspace_contract
        if metric_version == FORMAL_INDEPENDENT_METRIC_VERSION
        else None
    )
