"""Deterministic input checks for project-owned articulated Cosmos tasks."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .articulated_view_initialization import target_corners

PROJECT_EPISODE_SECONDS = 50.
DROID_CONTROL_HZ = 15
DROID_EXECUTION_HORIZON = 32
INPUT_CONTRACT_VERSION = "cosmos_articulated_inputs.v2"

def project_episode_steps(seconds):
    if isinstance(seconds,bool) or not np.isfinite(seconds) or seconds<=0:
        raise ValueError("project episode duration must be finite and positive")
    return int(np.ceil(seconds*DROID_CONTROL_HZ))

def validate_task_instruction(instruction):
    if not isinstance(instruction,str) or instruction != instruction.strip() or not instruction:
        raise ValueError("instruction must be a nonempty trimmed string")
    if "\n" in instruction or len(instruction.split())>24 or any(c in instruction for c in "{}[]"):
        raise ValueError("use one short natural-language goal sentence (at most 24 words)")
    if re.search(r"\b(camera|frames?|fps|seconds?|timesteps?|joint|controller|quaternion|degrees?|radians?)\b",instruction,re.I):
        raise ValueError("task prompt contains runtime or camera instructions")
    if len(re.findall(r"[.!?](?:\s|$)",instruction))>1:
        raise ValueError("use one task-goal sentence")
    return instruction

def project_target(camera,world_points):
    q=camera["quaternion_opengl_wxyz"]
    rot=Rotation.from_quat([*q[1:],q[0]])
    points=rot.inv().apply(np.asarray(world_points)-camera["position_world"])
    depth=-points[:,2]
    if not np.isfinite(points).all() or np.any(depth<=.02):
        return {"in_front":False,"center_in_frame":False,"bbox_visible_fraction":0.}
    k=np.asarray(camera["intrinsics"]);h,w=camera["image_shape"]
    uv=np.column_stack([k[0,0]*points[:,0]/depth+k[0,2],k[1,2]-k[1,1]*points[:,1]/depth])/[w,h]
    lo,hi=uv[1:].min(0),uv[1:].max(0)
    area=float(np.prod(np.maximum(hi-lo,0.)))
    visible=float(np.prod(np.maximum(np.minimum(hi,1)-np.maximum(lo,0),0.))/max(area,1e-12))
    return {"in_front":True,"center_uv":uv[0].tolist(),
        "center_in_frame":bool(np.all((uv[0]>=.05)&(uv[0]<=.95))),
        "handle_bbox_uv":[*lo.tolist(),*hi.tolist()],"bbox_visible_fraction":visible,
        "whole_handle_in_frame":bool(np.all(lo>=0)&np.all(hi<=1)),"depth_m":float(depth[0])}

def audit_initial_view(config,state,calibration,binding,*,enforce=True):
    corners=target_corners(config["source_sdf"],binding)
    local=np.vstack([binding["contact_target_local_xyz"],corners])
    world=Rotation.from_quat(state["child_link_xyzw"]).apply(local)+state["child_link_position"]
    views={name:project_target(cam,world) for name,cam in calibration["cameras"].items()}
    # The registered Cosmos DROID adapter makes the two exterior cameras the
    # semantic authority; its wrist plane is a native close-up and does not
    # require the target in frame zero. Keep its projection as diagnostic
    # evidence while requiring a complete handle in exterior context.
    wrist=views["wrist"]
    wrist_visible=bool(wrist["center_in_frame"] and wrist["bbox_visible_fraction"]>=.45)
    camera_rig=config.get("camera_rig")
    wrist_required=False
    if camera_rig:
        rig_file=Path(camera_rig)
        if hashlib.sha256(rig_file.read_bytes()).hexdigest()!=config.get("camera_rig_sha256"):
            raise ValueError("initial view camera rig changed")
        wrist_required=bool((json.loads(rig_file.read_text()).get("wrist_mount") or {}).get(
            "require_handle_visible"))
    context=any(v["center_in_frame"] and v.get("whole_handle_in_frame",False)
        for k,v in views.items() if k.startswith("exterior_"))
    passed=bool(context and state["physical_valid"]
        and not any(state["finger_contacts"]) and state["robot"]["gripper"]<=.05
        and (not wrist_required or wrist_visible))
    result={"schema_version":INPUT_CONTRACT_VERSION,"status":"passed" if passed else "rejected",
        "views":views,"context_contains_whole_handle":context,
        "wrist_handle_visible":wrist_visible,
        "wrist_semantic_policy":("candidate_wrist_aim_requires_handle_v1" if wrist_required
            else "native_closeup_not_required_to_contain_target_v1"),
        "validation_scope":"registered exterior semantic frustum plus measured initial physics; wrist target projection is diagnostic; occlusion still requires rendered-view review",
        "native_wrist_mount_changed":False,"frame":state["frame"],
        "contract_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selector_source_sha256":hashlib.sha256((Path(__file__).parent/"articulated_view_initialization.py").read_bytes()).hexdigest()}
    if enforce and not passed:raise ValueError("initial manipulation target is not observable or initial state is contacted/invalid")
    return result
