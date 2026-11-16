import json
import numpy as np
import pytest
from continuous_action.schema import ActionTrajectory
from continuous_action.current_metric import *
from continuous_action.current_evaluator import evaluate_current_manifest,PROTOCOL
from continuous_action.workspaces import robot_workspace_contract_for_model

def fixture(tmp_path):
    m={"evaluation_protocol":PROTOCOL,"metric_version":CURRENT_METRIC_VERSION,
        "metric_profile":CURRENT_METRIC_PROFILE,"metric_config_sha256":current_metric_config_sha256(),
        "metrics":current_metric_config(),"robot_workspace_contract":robot_workspace_contract_for_model("cosmos3_nano_droid"),
        "rollout":{"end_sec":1.,"reference":{"type":"dream2flow_video_plan","video_role":"reference"},
            "predictions":{"test":{"type":"dream2flow_video_plan","video_role":"imagination"}}}}
    path=tmp_path/CURRENT_MANIFEST_NAME;path.write_text(json.dumps(m))
    return path,m

def loader(spec,**kwargs):
    t=np.linspace(0,1,21);scale=1 if spec["video_role"]=="reference" else .5
    return ActionTrajectory(np.column_stack([t*scale,t*0,t*0]),t,
        metadata={"observed_frame_mask":[True]*21,"registration_sha256":"same",
            "registered_urdf_sha256":"robot","tracked_object":"handle"})

def test_manifest_equals_direct_scorer_and_has_current_identity(tmp_path):
    path,m=fixture(tmp_path);r=evaluate_current_manifest(path,m,loader)
    assert r["models"]["test"]["aggregate"]["continuous_score"]==pytest.approx(np.sqrt((2/3)*0.8))
    assert r["models"]["test"]["full_rollout"]["action_episode_score"] is None
    assert r["metric_config_sha256"]==current_metric_config_sha256()

@pytest.mark.parametrize("defect",["profile","hash","action_head","wrong_role","wrong_registration"])
def test_current_manifest_fails_closed(tmp_path,defect):
    path,m=fixture(tmp_path)
    if defect=="profile":m["metrics"]["smoothing"]="gaussian"
    elif defect=="hash":m["metric_config_sha256"]="bad"
    elif defect=="action_head":m["rollout"]["predictions"]["test"]["type"]="canonical"
    elif defect=="wrong_role":m["rollout"]["predictions"]["test"]["video_role"]="reference"
    def load(spec,**kwargs):
        t=loader(spec,**kwargs)
        meta=dict(t.metadata)
        if defect=="wrong_registration" and spec["video_role"]=="imagination":meta["registration_sha256"]="wrong"
        if defect=="missing_coverage":meta["observed_frame_mask"][10]=False
        return ActionTrajectory(t.positions,t.timestamps,metadata=meta)
    with pytest.raises(ValueError):evaluate_current_manifest(path,m,load)


def test_current_manifest_accepts_partial_coverage(tmp_path):
    path, manifest = fixture(tmp_path)
    original_loader = loader

    def partial_loader(spec, **kwargs):
        t = original_loader(spec, **kwargs)
        metadata = dict(t.metadata)
        metadata["observed_frame_mask"] = [True, True, False] + [False] * 18
        return ActionTrajectory(t.positions, t.timestamps, metadata=metadata)

    result = evaluate_current_manifest(path, manifest, partial_loader)
    details = result["models"]["test"]["full_rollout"]["scores"]
    assert details["observation_coverage"] == "partial"
    assert result["models"]["test"]["aggregate"]["continuous_score"] > 0.0


def test_retired_manifest_is_rejected(tmp_path):
    path, manifest = fixture(tmp_path)
    manifest["metric_version"] = "independent_metric_shape_extent_geomean_partial_v18"
    with pytest.raises(ValueError):
        evaluate_current_manifest(path, manifest, loader)
