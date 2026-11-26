"""Manifest validation for the single complete-trajectory CAE metric."""
from pathlib import Path
import hashlib
import numpy as np
from .current_metric import *
from .metrics import score_action_trajectories
from .workspaces import validate_robot_workspace_contract

PROTOCOL = "video_imagination_dream2flow_official_action_trajectory_v2"

def evaluate_current_manifest(path, manifest, load_trajectory):
    metric_version, profile = CURRENT_METRIC_VERSION, CURRENT_METRIC_PROFILE
    schema, filename = CURRENT_SCHEMA_VERSION, CURRENT_MANIFEST_NAME
    config, config_hash, scorer = current_metric_config, current_metric_config_sha256, score_action_trajectories
    path=Path(path)
    if path.name != filename:
        raise ValueError("CAE requires the single current manifest filename")
    if manifest.get("evaluation_protocol") != PROTOCOL:
        raise ValueError("current CAE requires symmetric official Dream2Flow video trajectories")
    if manifest.get("metric_version") != metric_version or manifest.get("metrics") != config():
        raise ValueError("current CAE requires the exact frozen metric profile")
    if manifest.get("metric_profile") != profile or manifest.get("metric_config_sha256") != config_hash():
        raise ValueError("current CAE metric profile/hash mismatch")
    workspace=validate_robot_workspace_contract(manifest.get("robot_workspace_contract"))
    rollout=manifest["rollout"]
    duration=float(rollout["end_sec"])-float(rollout.get("start_sec",0.))
    if not np.isfinite(duration) or duration<=0:raise ValueError("invalid rollout duration")
    ref_spec=rollout["reference"];pred_specs=rollout["predictions"]
    if not isinstance(pred_specs,dict) or not pred_specs:raise ValueError("missing predictions")
    def load(spec,role):
        if spec.get("type")!="dream2flow_video_plan" or spec.get("video_role")!=role:
            raise ValueError("formal CAE requires official optimizer plans with correct video roles")
        t=load_trajectory(spec,manifest_dir=path.parent,duration=duration)
        observed=np.asarray(t.metadata.get("observed_frame_mask"))
        if observed.shape!=(len(t.positions),):
            raise ValueError("evaluator_coverage_failure: observed_frame_mask must align")
        return t
    reference=load(ref_spec,"reference");models={}
    for model,spec in sorted(pred_specs.items()):
        pred=load(spec,"imagination")
        for key in ["registration_sha256","registered_urdf_sha256","tracked_object"]:
            if not reference.metadata.get(key) or reference.metadata[key]!=pred.metadata.get(key):
                raise ValueError("reference/imagination provenance mismatch: "+key)
        score=scorer(reference,pred);s=score.to_dict()
        models[model]={
            "aggregate":{"path_score":score.path_score,"velocity_score":score.velocity_score,
                "continuous_score":score.continuous_score,"aggregation":s["aggregation"],
                "metric_version":metric_version,"metric_profile":profile,
                "metric_config_sha256":config_hash(),
                "robot_workspace_diameter_m":workspace["diameter_m"],
                "robot_workspace_contract_sha256":workspace["contract_sha256"],
                "scored_components":s["scored_components"],
                "benchmark_status":manifest.get("benchmark_status","formal"),
                "visibility_affects_score":False,"num_diagnostic_chunks":0},
            "full_rollout":{"interval_sec":[rollout.get("start_sec",0.),rollout["end_sec"]],
                "reference":dict(reference.metadata),"prediction":dict(pred.metadata),
                "scores":s,"action_episode_score":None},
            "diagnostic_chunks":[],
        }
    return {"schema_version":schema,"metric_version":metric_version,
        "metric_profile":profile,"metric_config_sha256":config_hash(),
        "evaluation_protocol":PROTOCOL,"evaluation_object":"symmetric official Dream2Flow optimized EE trajectories",
        "action_head_used":False,"manifest":str(path),"manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "metric_config":config(),"robot_workspace_contract":workspace,
        "metric_description":{"path_formula":config()["path_score"],"velocity_formula":config()["velocity_score"],
            "continuous_formula":config()["aggregation"]},
        "rollout_protocol":{"primary_unit":"entire supplied trajectory with all leading, internal and trailing pauses",
            "diagnostic_chunks":"posthoc_only","semantic_chunks_control_inference":False,
            "semantic_chunks_reset_model_state":False,"semantic_chunks_inject_reference_gt":False},
        "models":models}
