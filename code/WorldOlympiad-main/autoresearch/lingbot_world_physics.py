"""Inputs for fixed-camera physics diagnostics with released LingBot-World v2.

Only the first RGB, a prompt, and camera conditioning go to the model.
Physics state/trigger records are provenance, not a numeric control interface.
"""
from pathlib import Path
import hashlib
import os
import json
import math
import shutil
import numpy as np
from PIL import Image

INPUT_VERSION = "lingbot_world_v2_native_render_physics.v2"
MODEL_ID = os.environ.get("LINGBOT_WORLD_MODEL_ID", "Robbyant/lingbot-world-v2-1.3b-causal-fast")
CANONICAL_WH = (832, 480)
NATIVE_FPS = 16
CHUNK_SIZE = 4
MAX_AREA = 480*832
NATIVE_RENDER_WH = (832, 464)  # Fixed point of the released 480P dimension calculation.
MODEL_PROFILES = {
    "Robbyant/lingbot-world-v2-1.3b-causal-fast": {
        "task": "i2v-1.3B", "sample_shift": 5.0,
    },
    "Robbyant/lingbot-world-v2-14b-causal-fast": {
        "task": "i2v-A14B", "sample_shift": 10.0,
    },
}

def profile_for_model_id(model_id):
    try:return MODEL_PROFILES[model_id]
    except KeyError as exc:raise ValueError(f"unsupported LingBot-World v2 model: {model_id}") from exc

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def native_dimensions(width, height, max_area=MAX_AREA):
    if width<=0 or height<=0 or max_area<=0:raise ValueError("positive image dimensions required")
    aspect=height/width
    h=int(math.sqrt(max_area*aspect)//8//2*2)*8
    w=int(math.sqrt(max_area/aspect)//8//2*2)*8
    if min(h,w)<16:raise ValueError("image aspect ratio is not supported")
    return w,h

def complete_chunk_frame_count(duration_seconds, fps=NATIVE_FPS, chunk_size=CHUNK_SIZE):
    if not math.isfinite(duration_seconds) or duration_seconds<=0:raise ValueError("positive finite duration required")
    if fps<=0 or chunk_size<=0:raise ValueError("positive fps/chunk size required")
    latent=math.ceil(duration_seconds*fps/4)+1
    latent=math.ceil(latent/chunk_size)*chunk_size
    return 4*(latent-1)+1

def camera_inputs(calibration, image_wh, frame_count):
    width,height=image_wh
    k=np.asarray(calibration["intrinsics"],dtype=float)
    c2w=np.asarray(calibration["camera_to_world_opencv"],dtype=float)
    if k.shape!=(3,3) or c2w.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(c2w).all():
        raise ValueError("invalid camera matrices")
    if not np.allclose(k[2],[0,0,1]) or abs(k[0,1])+abs(k[1,0])>1e-9 or min(k[0,0],k[1,1])<=0:
        raise ValueError("only zero-skew pinhole intrinsics are supported")
    rot=c2w[:3,:3]
    if not np.allclose(rot.T@rot,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rot),1,atol=1e-5):
        raise ValueError("camera rotation is not a proper orthonormal basis")
    if not np.allclose(c2w[3],[0,0,0,1]):raise ValueError("invalid camera homogeneous row")
    if list(image_wh)!=[calibration["width"],calibration["height"]]:
        raise ValueError("image size and calibration disagree")
    if frame_count<13 or ((frame_count-1)//4+1)%CHUNK_SIZE or (frame_count-1)%4:
        raise ValueError("frame count must contain complete native causal chunks")
    # Upstream always interprets intrinsics.npy in 832x480 pixel units.
    canonical=np.array([k[0,0]*832/width,k[1,1]*480/height,k[0,2]*832/width,k[1,2]*480/height],dtype=np.float32)
    intrinsics=np.repeat(canonical[None],frame_count,axis=0)
    # Camera zero is the coordinate origin; exact identity avoids normalization
    # of tiny roundoff translations when upstream computes relative poses.
    poses=np.repeat(np.eye(4,dtype=np.float32)[None],frame_count,axis=0)
    out_w,out_h=native_dimensions(width,height)
    expected=canonical*np.array([out_w/832,out_h/480,out_w/832,out_h/480],dtype=np.float32)
    return poses,intrinsics,expected

def prepare_case(source_case, output, prompt=None, model_id=None):
    source=Path(source_case).resolve();out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError("use a new input bundle directory")
    ref=source/"reference"
    sim=json.loads((source/"simulation.json").read_text())
    calibration=json.loads((ref/"calibration.json").read_text())
    scene=json.loads((ref/"scene_data.json").read_text())
    image=ref/"camera_rgb.png"
    with Image.open(image) as im:
        image_wh=list(im.size)
        if im.mode!="RGB":raise ValueError("input first frame must be RGB")
    if native_dimensions(*image_wh)!=tuple(image_wh):
        raise ValueError("render the source at native model dimensions; adapter image resizing is forbidden")
    frame_count=complete_chunk_frame_count(float(sim["duration_s"]))
    poses,intrinsics,expected=camera_inputs(calibration,image_wh,frame_count)
    text=prompt or ("A fixed camera observes a wooden table in a room. "+sim["instruction"]+" The camera remains stationary.")
    if not isinstance(text,str) or not text.strip() or "\n" in text:raise ValueError("one nonempty natural-language paragraph required")
    out.mkdir(parents=True,exist_ok=True)
    selected_model=model_id or MODEL_ID
    profile=profile_for_model_id(selected_model)
    shutil.copy2(image,out/"image.png")
    np.save(out/"poses.npy",poses);np.save(out/"intrinsics.npy",intrinsics)
    (out/"prompt.txt").write_text(text+"\n")
    native_wh=native_dimensions(*image_wh)
    state=np.load(ref/"state_trajectory.npz",allow_pickle=False)
    names=state["body_names"].tolist()
    initial={name:{"position_m":state["positions"][0,i].tolist(),"velocity_m_s":state["linear_velocities"][0,i].tolist()}
             for i,name in enumerate(names) if not name.startswith("background_") and name!="table"}
    apparatus=json.loads((source/"apparatus.json").read_text()) if (source/"apparatus.json").exists() else None
    bbox=np.asarray(scene["initial_component_region"]["bbox_xyxy"],float)
    bbox*=np.array([native_wh[0]/image_wh[0],native_wh[1]/image_wh[1]]*2)
    if min(bbox)<0 or bbox[2]>native_wh[0] or bbox[3]>native_wh[1]:raise ValueError("initial target is outside the model view")
    receipt={
        "schema_version":INPUT_VERSION,"model_id":selected_model,"case_id":sim["task_id"],
        "source_case":str(source),"source_kind":"existing_scenecode_assets_rebound_to_pybullet",
        "apparatus_provenance":apparatus,
        "reference":{"video":str(ref/"rgb.mp4"),"video_sha256":digest(ref/"rgb.mp4"),
            "fps":sim["fps"],"frames":sim["frames"],"temporal_span_seconds":sim["duration_s"]},
        "conditioning":{"image":"image.png","image_sha256":digest(out/"image.png"),"source_image_wh":image_wh,
            "prompt":text,"prompt_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "poses":"poses.npy","poses_sha256":digest(out/"poses.npy"),
            "intrinsics":"intrinsics.npy","intrinsics_sha256":digest(out/"intrinsics.npy"),
            "pose_convention":"OpenCV c2w relative to first camera; every pose is exact identity",
            "intrinsics_pixel_units_wh":[832,480],"intrinsics_row":intrinsics[0].tolist(),
            "reference_future_rgb_or_states_passed_to_model":False,
            "force_mass_or_velocity_tensor_passed_to_model":False},
        "sampling":{"task":profile["task"],"infer_mode":"causal_fast","max_area":MAX_AREA,
            "frame_num":frame_count,"expected_frames":frame_count,"fps":NATIVE_FPS,
            "temporal_span_seconds":(frame_count-1)/NATIVE_FPS,"output_wh":list(native_wh),
            "chunk_size":CHUNK_SIZE,"sample_shift":profile["sample_shift"],"timesteps_index":[0,250,500,750],
            "local_attn_size":18,"sink_size":6,"base_seed":42},
        "camera_alignment":{"source_c2w":calibration["camera_to_world_opencv"],
            "source_intrinsics":calibration["intrinsics"],"expected_model_intrinsics":expected.tolist(),
            "target_bbox_model_xyxy":bbox.tolist(),"camera_pose_fixed":True,
            "image_transform":"none; directly rendered native pixels; upstream interpolation has identical input/output dimensions"},
        "initial_physics_provenance_only":initial,
        "limitations":["numeric simulator parameters are not a model control interface",
            "reference and model retain their independent native clocks",
            "no robotics EE-trajectory CAE score is inferred for these passive physics tasks"],
    }
    (out/"input_contract.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--model-id")
    args=parser.parse_args()
    print(json.dumps(prepare_case(args.source_case,args.output,args.prompt,args.model_id),indent=2))
