#!/usr/bin/env python3
"""Native DROID closed-loop policy trial for a project-authored articulated task.

This reuses the released Cosmos service and the placement adapter's native
cameras/controller. It does not certify a formal SceneCode/CAE reference.
"""
from __future__ import annotations
import argparse,json,os,sys,traceback
from pathlib import Path
import cv2
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy-model', choices=['cosmos3', 'gr00t'], default='cosmos3')
parser.add_argument("--task",type=Path,required=True)
parser.add_argument("--world-usd",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--policy-server-host",default="127.0.0.1")
parser.add_argument("--policy-server-port",type=int,default=28000)
parser.add_argument("--execute-actions-per-window",type=int,default=32,
                    help="Execute this many of the 32 predicted actions before observing again; default is the upstream 32")
parser.add_argument('--robolab-sim-config',type=Path,default=Path(__file__).resolve().parents[1]/'deployment/seetacloud/robolab_simulation_profile.json',
                    help='Use the exact sim configuration exported by an original RoboLab run, including physics device')
parser.add_argument('--legacy-cpu-physics',action='store_true',help='Reproduce historical CPU physics artifacts only')
parser.add_argument('--contact-diagnostics',action='store_true',help='Record measured contact centroids; does not change policy or control')
parser.add_argument('--camera-diagnostics',action='store_true',help='Record actual camera poses and intrinsics with each observation')
parser.add_argument("--replay-policy-run",type=Path,help="Physics-only diagnostic using immutable previously executed actions; no model inference")
parser.add_argument("--replay-reference-trajectory",type=Path,help="Physics-only diagnostic of a recorded reference joint path; no model inference")
parser.add_argument("--capture-reference-video",action="store_true",help="Render every measured reference replay state at 15 Hz; no interpolation or time scaling")
parser.add_argument("--native-reference-program",type=Path,help="Native measured-state reference controller; no model inference or object actuation")
parser.add_argument("--diagnostic-observation-steps",type=int,nargs='+',default=[],help="Save native input snapshots at these replay steps, not an execution video")
parser.add_argument("--preflight",action="store_true")
parser.add_argument("--physics-only-preflight",action="store_true")
parser.add_argument("--initial-view-only",action="store_true",help="Validate the closed, contact-free initial state and cameras without moving the object through the robot")
AppLauncher.add_app_launcher_args(parser)
ARGS=parser.parse_args()
if ARGS.legacy_cpu_physics: ARGS.robolab_sim_config=None
ARGS.headless=True; ARGS.enable_cameras=not (ARGS.physics_only_preflight or ARGS.replay_policy_run or ARGS.replay_reference_trajectory or ARGS.native_reference_program)
ARGS.device=json.loads(ARGS.robolab_sim_config.read_text())['sim']['device'] if ARGS.robolab_sim_config else 'cpu'
if ARGS.diagnostic_observation_steps: ARGS.enable_cameras=True
if ARGS.capture_reference_video:
    if not ARGS.replay_reference_trajectory: parser.error("reference video capture requires reference trajectory replay")
    ARGS.enable_cameras=True
if ARGS.replay_policy_run and ARGS.replay_reference_trajectory: parser.error('choose one replay source')
if ARGS.native_reference_program and (ARGS.replay_policy_run or ARGS.replay_reference_trajectory or ARGS.preflight):
    parser.error('native reference control cannot be combined with replay or pose-reset preflight')
if ARGS.physics_only_preflight and not ARGS.preflight: parser.error("physics-only requires preflight")
if ARGS.initial_view_only and (not ARGS.preflight or ARGS.physics_only_preflight): parser.error("initial-view-only requires a camera-enabled preflight")
if not 1 <= ARGS.execute_actions_per_window <= 32: parser.error('execute-actions-per-window must be in [1,32]')
APP=AppLauncher(ARGS).app
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaacsim.core.utils.stage import add_reference_to_stage,get_current_stage
from pxr import PhysxSchema,UsdPhysics,Usd
from robolab.robots.droid import DroidCfg
from robolab.variations.camera import OverShoulderLeftCameraCfg,OverShoulderRightCameraCfg
from autoresearch.cosmos_droid_native import _exterior_camera,_wrist_camera,_render_observation,_contact_active
from autoresearch.articulated_native_task import JointTask,JointTermination
from autoresearch.robotics_droid_control import executable_robolab_gripper_target,normalize_robolab_gripper_position
from autoresearch.cosmos3_droid_window import run_server_window,write_json_atomic
from autoresearch.io import sha256_file
from autoresearch.cosmos_simulation_profile import build_simulation_config,read_simulation_profile
from autoresearch.media import write_h264_rgb

def tensor_list(value): return value.detach().cpu().tolist()
def xyzw(quat):
    w,x,y,z=tensor_list(quat)
    return [x,y,z,w]

def main():
    if ARGS.output.exists(): raise FileExistsError(ARGS.output)
    root=ARGS.task.resolve().parent
    config=json.loads(ARGS.task.read_text())
    if config["schema_version"]!="cosmos_articulated_project_task.v1": raise ValueError("unsupported project task")
    if config.get("reference_manifest"):
        receipt_path=root/config["runtime_physics_rebase"]
        receipt=json.loads(receipt_path.read_text())
        if (sha256_file(receipt_path)!=config.get("runtime_physics_rebase_sha256")
            or receipt.get("passed") is not True
            or receipt.get("alignment_certificate_sha256")!=config.get("runtime_alignment_certificate_sha256")):
            raise ValueError("native articulated runtime collision alignment changed")
    from autoresearch.articulated_input_contract import validate_task_instruction, audit_initial_view
    # wo_articulated_inputs.v1: tasks derived by wo.models.cosmos_articulated from a
    # wo expert reference. The prompt is the benchmark instruction (or a prompt
    # ablation) and there is no handle binding, so the legacy instruction lint and
    # handle-visibility audit are not applied; the cameras are the released DROID rig.
    wo_contract = config.get("input_contract_version") == "wo_articulated_inputs.v1"
    visibility_binding=None
    if not wo_contract:
        validate_task_instruction(config["instruction"])
        if config.get("input_contract_version") != "cosmos_articulated_inputs.v2":
            raise ValueError("reprepare the task with the current binding and initial-view contract")
        binding_file=root/config["binding_file"]
        if sha256_file(binding_file) != config["binding_sha256"]:
            raise ValueError("task manipulation binding changed")
        visibility_binding=json.loads(binding_file.read_text())
    task=JointTask.from_dict(config["termination"])
    evaluation_window = config.get("evaluation_window")
    max_steps = task.max_steps
    if evaluation_window is not None:
        from autoresearch.cosmos_evaluation_window import validate, digest
        validate(evaluation_window)
        if evaluation_window["reference_manifest_sha256"] != config.get("reference_manifest_sha256"):
            raise ValueError("articulated shared window is not bound to its certified reference")
        max_steps = evaluation_window["max_steps"]
    plan_file=root/config["worldir_plan"]
    plan=json.loads(plan_file.read_text())
    if config["worldir_hash"]!=plan["world_hash"]: raise ValueError("world identity mismatch")
    for name in ("source_sdf","visual_sdf","robot_usd","runtime_identity"):
        if sha256_file(Path(config[name]))!=config[name+"_sha256"]: raise ValueError(name+" changed")
    camera_overrides = {}
    wrist_mount = None
    if config.get("camera_rig"):
        rig_file=Path(config["camera_rig"])
        if sha256_file(rig_file)!=config["camera_rig_sha256"]: raise ValueError("camera rig changed")
        rig=json.loads(rig_file.read_text())
        if rig["schema_version"]!="worldolympiad_camera_rig.v1": raise ValueError("unknown camera rig")
        camera_overrides=rig["cameras"]
        wrist_mount=rig.get("wrist_mount")
        if not {"exterior_left","exterior_right"} <= set(camera_overrides) or set(camera_overrides)-{"exterior_left","exterior_right","top"}:
            raise ValueError("camera rig lacks exterior model views or adds an unknown view")
    initial_path=root/config["initial_state"]
    if sha256_file(initial_path)!=config["initial_state_sha256"]: raise ValueError("initial state changed")
    initial=json.loads(initial_path.read_text())
    robot_plan=next(b for b in plan["bodies"] if b["name"]=="robot")
    ARGS.output.mkdir(parents=True)
    output=ARGS.output.resolve()
    write_json_atomic(output/"task.json",config)
    runtime={"task_sha256":sha256_file(ARGS.task),"world_usd_sha256":sha256_file(ARGS.world_usd),
      "worldir_plan_sha256":sha256_file(plan_file),
      "runner_sha256":sha256_file(Path(__file__)),
      "window_contract_sha256":sha256_file(Path(__file__).resolve().parents[1]/"autoresearch/cosmos3_droid_window.py"),
      "joint_measurement_sha256":sha256_file(Path(__file__).resolve().parents[1]/"autoresearch/articulated_native_task.py"),
      "camera_helpers_sha256":sha256_file(Path(__file__).resolve().parents[1]/"autoresearch/cosmos_droid_native.py"),
      "input_contract_sha256":sha256_file(Path(__file__).resolve().parents[1]/"autoresearch/articulated_input_contract.py"),
      "view_selector_sha256":sha256_file(Path(__file__).resolve().parents[1]/"autoresearch/articulated_view_initialization.py"),
      "action_source":"runtime_validation_pose_reset" if ARGS.preflight else ('gr00t_n1.6_droid' if ARGS.policy_model == 'gr00t' else "cosmos3_nano_policy_droid"),
      "formal_cae_certified":False}
    runtime['predicted_actions_per_window']=8 if ARGS.policy_model == 'gr00t' else 32
    if ARGS.policy_model == 'gr00t' and ARGS.execute_actions_per_window != 8:
        raise ValueError('GR00T uses its checked 8-action native horizon')
    runtime['executed_actions_per_window']=ARGS.execute_actions_per_window
    runtime['feedback_profile']='upstream_32' if ARGS.execute_actions_per_window==32 else 'project_receding_horizon'
    runtime['finger_midpoint_semantics']='midpoint of rigid-link frame origins, not fingertip or contact positions'
    runtime['contact_centroid_semantics']='per finger/filter mean contact position at the last physics substep; null means no contact' if ARGS.contact_diagnostics else None
    write_json_atomic(output/"runtime.json",runtime)
    physics_profile = config.get("physics_profile", {})
    allowed_physics_fields = {'physics_hz','max_position_iterations','min_velocity_iterations',
                              'max_velocity_iterations','external_forces_every_iteration'}
    if set(physics_profile)-allowed_physics_fields:
        raise ValueError('unsupported/retired physics profile fields; experimental gripper overrides are archived')
    physics_hz = int(physics_profile.get("physics_hz", 120))
    if physics_hz < 120 or physics_hz % 15:
        raise ValueError("physics rate must be >= 120 Hz and an integer multiple of native 15 Hz control")
    control_substeps = physics_hz // 15
    # Do not cap the released DROID articulation's 64 solver iterations at 32.
    max_position_iterations = int(physics_profile.get("max_position_iterations", 255))
    min_velocity_iterations = int(physics_profile.get("min_velocity_iterations", 0))
    max_velocity_iterations = int(physics_profile.get("max_velocity_iterations", 1))
    runtime["physics_profile"] = dict(physics_hz=physics_hz,control_hz=15,
      max_position_iterations=max_position_iterations,min_velocity_iterations=min_velocity_iterations,
      max_velocity_iterations=max_velocity_iterations)
    if ARGS.replay_policy_run or ARGS.replay_reference_trajectory:
        runtime["action_source"] = "reference_joint_path_replay_diagnostic" if ARGS.replay_reference_trajectory else "recorded_policy_physics_replay_diagnostic"
        runtime["inference_launched"] = False
    if ARGS.native_reference_program:
        runtime["action_source"]="native_measured_joint_reference_controller"
        runtime["inference_launched"]=False
        runtime["reference_program_sha256"]=sha256_file(ARGS.native_reference_program)
        runtime["reference_controller_sha256"]=sha256_file(Path(__file__).resolve().parents[1]/"autoresearch/native_joint_reference.py")
    write_json_atomic(output/"runtime.json",runtime)
    sim_cfg=sim_utils.SimulationCfg(device="cpu",dt=1/physics_hz,render_interval=control_substeps,
      gravity=(0,0,-9.81),use_fabric=True,
      physx=sim_utils.PhysxCfg(solver_type=1,min_position_iteration_count=1,max_position_iteration_count=max_position_iterations,
        min_velocity_iteration_count=min_velocity_iterations,max_velocity_iteration_count=max_velocity_iterations,enable_ccd=True,
        enable_external_forces_every_iteration=bool(physics_profile.get("external_forces_every_iteration", False))))
    if ARGS.robolab_sim_config:
        original=read_simulation_profile(ARGS.robolab_sim_config)
        sim_cfg=build_simulation_config(sim_utils,ARGS.robolab_sim_config)
        physics_hz=120;control_substeps=8
        runtime['physics_profile']=original
        runtime['simulation_config_source']=str(ARGS.robolab_sim_config.resolve())
        runtime['simulation_config_source_sha256']=sha256_file(ARGS.robolab_sim_config)
        write_json_atomic(output/'runtime.json',runtime)
    sim=SimulationContext(sim_cfg)
    add_reference_to_stage(str(ARGS.world_usd.resolve()),"/World/WorldIR")
    stage=get_current_stage()
    from wo.models.reference_environment import add_reference_ground
    write_json_atomic(output/'reference_environment.json',add_reference_ground(stage))
    nested=stage.GetPrimAtPath("/World/WorldIR/PhysicsScene")
    if nested.IsValid(): nested.SetActive(False)
    for name,cfg in (
        ("DomeLight",sim_utils.DomeLightCfg(intensity=350,color=(.85,.9,1))),
        ("KeyLight",sim_utils.DistantLightCfg(intensity=750,color=(1,.93,.82)))):
        cfg.func("/World/"+name,cfg)
    robot_cfg=DroidCfg().robot.replace(prim_path="/World/Robot")
    robot_cfg.spawn.usd_path=config["robot_usd"]
    robot_cfg.init_state.pos=tuple(robot_plan["pose"]["xyz"])
    x,y,z,w=robot_plan["pose"]["quat_xyzw"]
    robot_cfg.init_state.rot=(w,x,y,z)
    initial_joint_map = dict(zip(initial['robot']['joint_names'], initial['robot']['joint_positions']))
    if not all(f'panda_joint{i}' in initial_joint_map for i in range(1,8)):
        raise ValueError('task initial state must name all seven native DROID arm joints')
    if not np.isfinite(list(initial_joint_map.values())).all():
        raise ValueError('task initial joints must be finite')
    # Cabinet views can require a different starting wrist orientation than
    # tabletop placement. Keep the same native robot and controller.
    robot_cfg.init_state.joint_pos = initial_joint_map
    robot=Articulation(robot_cfg)
    if ARGS.enable_cameras:
        left=_exterior_camera(OverShoulderLeftCameraCfg().over_shoulder_left_camera,"left_camera",
                              robot_cfg.init_state.pos,robot_cfg.init_state.rot,world_override=camera_overrides.get("exterior_left"))
        right=_exterior_camera(OverShoulderRightCameraCfg().over_shoulder_right_camera,"right_camera",
                               robot_cfg.init_state.pos,robot_cfg.init_state.rot,world_override=camera_overrides.get("exterior_right"))
        wrist=_wrist_camera(wrist_mount)
        if ARGS.camera_diagnostics:
            # Isaac otherwise reports the pose at sensor initialization even
            # after the wrist has moved. This updates telemetry, not rendering.
            for cam in (wrist,left,right):
                cam.cfg.update_latest_camera_pose=True
    child_prim=stage.GetPrimAtPath(config["child_prim"])
    if not child_prim.IsValid() or not child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("moving link is not a native rigid body")
    host_prim=child_prim.GetParent()
    native_roots=[prim for prim in Usd.PrimRange(host_prim) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    runtime['mechanism_representation']='reduced_coordinate_articulation' if native_roots else 'jointed_rigid_bodies'
    write_json_atomic(output/'runtime.json',runtime)
    mechanism=Articulation(ArticulationCfg(prim_path=str(host_prim.GetPath()),spawn=None,actuators={})) if native_roots and ARGS.preflight else None
    PhysxSchema.PhysxContactReportAPI.Apply(child_prim).CreateThresholdAttr().Set(0.0)
    moving=RigidObject(RigidObjectCfg(prim_path=config["child_prim"],spawn=None))
    sensors=[]
    for finger in ("left_inner_finger","right_inner_finger"):
        sensors.append(ContactSensor(ContactSensorCfg(
          prim_path="/World/Robot/Gripper/Robotiq_2F_85/"+finger,
          track_contact_points=ARGS.contact_diagnostics,
          max_contact_data_count_per_prim=64 if ARGS.contact_diagnostics else 4,
          filter_prim_paths_expr=[config["child_prim"]])))
    body_sensors={}
    if ARGS.contact_diagnostics:
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
            if prim.GetName() in {'base_link','panda_link6','panda_link7'} and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_sensors[str(prim.GetName())]=ContactSensor(ContactSensorCfg(
                    prim_path=str(prim.GetPath()),filter_prim_paths_expr=[config['child_prim']]))
    other_body_contact_peaks={name:0. for name in body_sensors}
    joint_properties = {}
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Robot') and prim.IsA(UsdPhysics.Joint):
            joint_properties[str(prim.GetPath())] = {str(a.GetName()):str(a.Get()) for a in prim.GetAttributes()
                if any(word in str(a.GetName()).lower() for word in ['drive','limit','stiffness','damping','friction'])}
    write_json_atomic(output/'robot_joint_properties.json',joint_properties)
    sim.reset()
    arm_ids,arm_names=robot.find_joints([f"panda_joint{i}" for i in range(1,8)],preserve_order=True)
    ee_ids,_=robot.find_bodies('base_link')
    finger_ids,_=robot.find_bodies(['left_inner_finger','right_inner_finger'],preserve_order=True)
    gripper_ids,_=robot.find_joints(["finger_joint"],preserve_order=True)
    if arm_names != [f"panda_joint{i}" for i in range(1,8)]: raise ValueError("DROID arm order changed")
    target=robot.data.default_joint_pos.clone()
    expected=np.array([initial_joint_map[f'panda_joint{i}'] for i in range(1,8)])
    bounds=robot.data.joint_pos_limits[0,arm_ids].detach().cpu().numpy()
    if np.any(expected < bounds[:,0]-1e-6) or np.any(expected > bounds[:,1]+1e-6):
        raise ValueError('task initial arm configuration exceeds native joint limits')
    robot.write_joint_state_to_sim(target,torch.zeros_like(target))
    robot.set_joint_position_target(target)
    def advance(count=control_substeps,render=False):
        nonlocal other_body_contact_peaks
        other_body_contact_peaks={name:0. for name in body_sensors}
        contacts=[False,False]
        for _ in range(count):
            robot.write_data_to_sim()
            sim.step(render=render)
            robot.update(sim_cfg.dt); moving.update(sim_cfg.dt)
            if mechanism is not None: mechanism.update(sim_cfg.dt)
            for i,sensor in enumerate(sensors):
                sensor.update(sim_cfg.dt,force_recompute=True)
                contacts[i] |= _contact_active(sensor)
            for name,sensor in body_sensors.items():
                sensor.update(sim_cfg.dt,force_recompute=True)
                force=sensor.data.force_matrix_w
                if force is not None:
                    other_body_contact_peaks[name]=max(other_body_contact_peaks[name],float(torch.linalg.vector_norm(force)))
        return contacts
    def measured(step,contacts):
        centroids=[]
        if ARGS.contact_diagnostics:
            for sensor in sensors:
                points=sensor.data.contact_pos_w[0,0]
                centroids.append([tensor_list(point) if bool(torch.isfinite(point).all()) else None for point in points])
        pos=tensor_list(moving.data.root_link_pos_w[0])
        quat=xyzw(moving.data.root_link_quat_w[0])
        linear=tensor_list(moving.data.root_link_lin_vel_w[0])
        angular=tensor_list(moving.data.root_link_ang_vel_w[0])
        state=task.measure(pos,quat,linear,angular)
        gripper=normalize_robolab_gripper_position(robot.data.joint_pos[0,gripper_ids[0]].item())
        joints=tensor_list(robot.data.joint_pos[0,arm_ids])
        if not np.isfinite(joints).all(): raise RuntimeError("non-finite robot state")
        controlled = arm_ids + gripper_ids
        raw_q = robot.data.joint_pos[0,controlled]
        bounds = robot.data.joint_pos_limits[0,controlled]
        limit_error = torch.maximum(bounds[:,0]-raw_q,raw_q-bounds[:,1]).clamp(min=0)
        healthy = bool(torch.isfinite(raw_q).all() and float(limit_error.max()) <= .05 and
                       state['within_limits'] and state['translation_residual_m'] <= .01 and state['rotation_residual_rad'] <= .03)
        return {"frame":step,"time_s":step/15,"articulated":state,
          "physical_valid":healthy,"controlled_joint_limit_violation_rad":tensor_list(limit_error),
          "child_link_position":pos,"child_link_xyzw":quat,
          "child_linear_velocity":linear,"child_angular_velocity":angular,
          "robot":{"arm_joints":joints,"joint_names":list(robot.data.joint_names),
             "arm_target_error_rad":tensor_list(target[0,arm_ids]-robot.data.joint_pos[0,arm_ids]),
             "ee_position":tensor_list(robot.data.body_pos_w[0,ee_ids[0]]),
             "ee_quaternion_wxyz":tensor_list(robot.data.body_quat_w[0,ee_ids[0]]),
             "finger_midpoint":tensor_list(robot.data.body_pos_w[0,finger_ids].mean(dim=0)),
             **({"native_finger_actor_positions":tensor_list(robot.data.body_link_pos_w[0,finger_ids]),
                 "native_finger_actor_orientations_wxyz":tensor_list(robot.data.body_link_quat_w[0,finger_ids]),
                 "native_body_names":list(robot.data.body_names),
                 "finger_total_contact_forces_w":[tensor_list(sensor.data.net_forces_w[0]) for sensor in sensors],
                 "finger_filtered_contact_forces_w":[tensor_list(sensor.data.force_matrix_w[0]) if sensor.data.force_matrix_w is not None else None for sensor in sensors]}
                if ARGS.contact_diagnostics else {}),
             "joint_positions":tensor_list(robot.data.joint_pos[0]),
             "joint_position_target":tensor_list(target[0]),"gripper":gripper,
             "base_position":tensor_list(robot.data.root_link_pos_w[0]),
             "base_xyzw":xyzw(robot.data.root_link_quat_w[0])},
          "finger_contacts":contacts,"other_body_contact_peak_normal_force_n":dict(other_body_contact_peaks),
          "last_substep_contact_centroids_world":centroids if ARGS.contact_diagnostics else None}
    def observe(step,folder):
        folder.mkdir(parents=True,exist_ok=True)
        path=_render_observation(frame_index=step,window_root=folder,
          observation_root=output/"observations",sim=sim,wrist=wrist,left=left,right=right)
        if ARGS.camera_diagnostics or step == 0:
            cameras={}
            for name,cam in [('wrist',wrist),('exterior_left',left),('exterior_right',right)]:
                cameras[name]={'position_world':tensor_list(cam.data.pos_w[0]),
                    'quaternion_opengl_wxyz':tensor_list(cam.data.quat_w_opengl[0]),
                    'intrinsics':tensor_list(cam.data.intrinsic_matrices[0]),
                    'image_shape':list(cam.data.output['rgb'].shape[1:3])}
            calibration={'frame':step,'pose_source':'latest_physics_camera_pose','cameras':cameras,
                'ee_position':tensor_list(robot.data.body_pos_w[0,ee_ids[0]]),
                'ee_quaternion_wxyz':tensor_list(robot.data.body_quat_w[0,ee_ids[0]])}
            write_json_atomic(folder/'camera_calibration.json',calibration)
            if step == 0 and visibility_binding is None:
                write_json_atomic(folder/'input_validation.json',{'status':'not_audited',
                    'reason':'wo_articulated_inputs.v1 has no handle binding'})
            elif step == 0:
                receipt=audit_initial_view(config,measured(step,contacts),calibration,visibility_binding,enforce=False)
                write_json_atomic(folder/'input_validation.json',receipt)
                if receipt['status'] != 'passed':
                    raise ValueError('initial view rejected; see input_validation.json; run automatic observable-start preparation')
        return path
    moving.update(0.0)
    write_json_atomic(output/"initial_reset_state.json",measured(0,[False,False]))
    contacts=advance(2*control_substeps,ARGS.enable_cameras)
    first=measured(0,contacts)
    write_json_atomic(output/"initial_measured_state.json",first)
    if abs(first["articulated"]["position"]-task.closed_position)>.01:
        raise RuntimeError("native initial joint drift")
    if ARGS.preflight:
        if not ARGS.physics_only_preflight: observe(0,output/"initial_view")
        if ARGS.initial_view_only:
            if not first["physical_valid"] or any(first["finger_contacts"]) or first["robot"]["gripper"] > .05:
                raise RuntimeError("invalid or contacted initial observation state")
            write_json_atomic(output/"initial_view_validation.json",{
              "schema_version":"cosmos_articulated_initial_view.v1","status":"passed",
              "scope":"closed-state initial pose and cameras only; full motion requires native reference replay",
              "inference_launched":False,"runtime":runtime,"measured":first})
            print("ARTICULATED_INITIAL_VIEW_PASSED",flush=True)
            return

        probes=[]
        for coordinate in (task.closed_position,(task.closed_position+task.target_position)/2,task.target_position):
            mechanism_before=None
            if mechanism is not None:
                joint_ids,_=mechanism.find_joints(config['joint_name'],preserve_order=True)
                mechanism_before=tensor_list(mechanism.data.joint_pos[0,joint_ids])
                positions=mechanism.data.joint_pos.clone();positions[0,joint_ids]=coordinate
                mechanism.write_joint_state_to_sim(positions,torch.zeros_like(positions))
                mechanism_after_write=tensor_list(mechanism.data.joint_pos[0,joint_ids])
            else:
                pos,quat=task.pose_at(coordinate); x,y,z,w=quat
                moving.write_root_link_pose_to_sim(torch.tensor([[*pos,w,x,y,z]],dtype=torch.float32,device=robot.device))
                moving.write_root_velocity_to_sim(torch.zeros((1,6),dtype=torch.float32,device=robot.device))
            contacts=advance(1)
            one_substep=measured(0,contacts)
            mechanism_one_substep=(
                tensor_list(mechanism.data.joint_pos[0,joint_ids])
                if mechanism is not None else None)
            contacts=advance(59)
            item=measured(0,contacts); item["commanded_probe_coordinate"]=coordinate
            item["probe_after_one_substep"]={
                "measured_joint_position":one_substep["articulated"]["position"],
                "mechanism_joint_position":mechanism_one_substep,
            }
            if mechanism is not None:
                item["mechanism_joint_before"]=mechanism_before
                item["mechanism_joint_after_write"]=mechanism_after_write
                item["mechanism_joint_after"]=tensor_list(mechanism.data.joint_pos[0,joint_ids])
            probes.append(item)
            state=item["articulated"]
            if abs(state["position"]-coordinate)>.01 or state["translation_residual_m"]>.005 or state["rotation_residual_rad"]>.02:
                write_json_atomic(output/"joint_probe_failed.json",probes)
                raise RuntimeError("native passive joint probe failed")
        write_json_atomic(output/"preflight.json",{
          "schema_version":"cosmos_articulated_native_preflight.v1","status":"passed",
          "inference_launched":False,"joint_probes":probes,"runtime":runtime,
          "camera_validated":not ARGS.physics_only_preflight})
        print("ARTICULATED_PREFLIGHT_PASSED",flush=True)
        return
    if ARGS.native_reference_program:
        from autoresearch.native_joint_reference import run as run_native_reference
        run_native_reference(ARGS.native_reference_program,config=config,task=task,robot=robot,
            target=target,arm_ids=arm_ids,gripper_ids=gripper_ids,ee_ids=ee_ids,
            advance=advance,measured=measured,output=output,observe=observe if ARGS.enable_cameras else None)
        return
    if ARGS.replay_policy_run or ARGS.replay_reference_trajectory:
        if ARGS.replay_reference_trajectory:
            from autoresearch.articulated_native_task import reference_actions_for_native_replay
            source=ARGS.replay_reference_trajectory.resolve()
            reference=json.loads(source.read_text())
            from autoresearch.articulated_reference_replay_binding import validate_reference_binding
            binding_receipt=validate_reference_binding(reference,config)
            write_json_atomic(output/'reference_replay_binding.json',binding_receipt)
            all_actions=reference_actions_for_native_replay(reference,config['joint_name'])
            if not np.allclose(all_actions[0,:7],first['robot']['arm_joints'],atol=1e-4):
                raise ValueError('reference/native initial joints differ')
            if not np.allclose(reference['frames'][0]['robot']['base_position'],first['robot']['base_position'],atol=.005):
                raise ValueError('reference/native robot roots differ')
            actions=all_actions[1:];action_files=[source]
        else:
            source = ARGS.replay_policy_run.resolve()
            receipt = json.loads((source/"runtime.json").read_text())
            if receipt['world_usd_sha256'] != runtime['world_usd_sha256']:
                raise ValueError('physics replay must use the exact original world USD')
            action_files = sorted(source.glob('windows/window_*/executed_actions.npy'))
            actions = np.concatenate([np.load(f,allow_pickle=False) for f in action_files])
        if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
            raise ValueError('invalid recorded native action array')
        controlled = arm_ids + gripper_ids
        limits = robot.data.joint_pos_limits[0, controlled].clone()
        frames = [first]
        if ARGS.capture_reference_video or 0 in ARGS.diagnostic_observation_steps:
            observe(0,output/'diagnostic_views/step_000000')
        checker = JointTermination(task)
        first_invalid = None
        result = {'task_success':False,'done':False,'reason':None}
        for step, action in enumerate(actions,1):
            target[0,arm_ids]=torch.from_numpy(action[:7]).to(target.device)
            target[0,gripper_ids]=executable_robolab_gripper_target(action[7])
            robot.set_joint_position_target(target)
            contacts=advance()
            frame=measured(step,contacts)
            q=robot.data.joint_pos[0,controlled]
            violations=torch.maximum(limits[:,0]-q,q-limits[:,1]).clamp(min=0)
            frame['controlled_joint_limit_violation_rad']=tensor_list(violations)
            frame['raw_controlled_joint_velocities']=tensor_list(robot.data.joint_vel[0,controlled])
            frame['physical_valid']=bool(torch.isfinite(q).all() and float(violations.max()) <= .05 and
              frame['articulated']['within_limits'] and frame['articulated']['translation_residual_m'] <= .01 and frame['articulated']['rotation_residual_rad'] <= .03)
            result=checker.step(frame['articulated'],contacts=any(contacts),gripper=frame['robot']['gripper'],step=step)
            frame['terminal']=result
            frames.append(frame)
            if ARGS.capture_reference_video or step in ARGS.diagnostic_observation_steps:
                write_json_atomic(output/'diagnostic_views'/f'step_{step:06d}'/'conditioning_state.json',frame)
                observe(step,output/'diagnostic_views'/f'step_{step:06d}')
            if not frame['physical_valid'] and first_invalid is None:
                first_invalid=frame
            if step % 32 == 0:
                print(f"PHYSICS_REPLAY step={step} joint={frame['articulated']['position']:.6f} max_limit_error={float(violations.max()):.6f}",flush=True)
            if first_invalid is not None: break
        write_json_atomic(output/'trajectory.json',{'schema_version':'cosmos_recorded_policy_physics_replay.v1','fps':15,'frames':frames})
        write_json_atomic(output/'result.json',{'schema_version':'cosmos_recorded_policy_physics_replay.v1',
          'status':'runtime_invalid' if first_invalid else 'physics_replay_valid','inference_launched':False,
          'source_run':str(source),'source_actions':[{'path':str(f),'sha256':sha256_file(f)} for f in action_files],
          'replayed_action_count':len(frames)-1,'original_action_count':len(actions),'first_invalid':first_invalid,
          'native_controlled_joint_limits':tensor_list(limits),'runtime':runtime,'terminal':result,
          'maximum_joint_progress':max(abs(f['articulated']['position']-task.closed_position) for f in frames)})
        if ARGS.capture_reference_video:
            if first_invalid is not None or len(frames) != len(actions)+1:
                raise ValueError('cannot publish video of an incomplete or invalid reference replay')
            from autoresearch.native_reference_capture import encode_views
            encode_views(output,frame_count=len(frames),trajectory=output/'trajectory.json')
        print('PHYSICS_REPLAY_COMPLETE '+json.dumps({'valid':first_invalid is None,'steps':len(frames)-1}),flush=True)
        return
    checker=JointTermination(task, evaluation_max_steps=max_steps if evaluation_window else None)
    frames=[first]; windows=[]; step=0
    horizon=ARGS.execute_actions_per_window
    seeds=np.random.default_rng(config["seed_schedule_seed"]).integers(0,2**31,size=(max_steps+horizon-1)//horizon).tolist()
    write_json_atomic(output/"seed_schedule.json",seeds)
    result={"task_success":False,"done":False,"reason":None}
    while step<max_steps and not result["done"]:
        index=len(windows)
        folder=output/"windows"/f"window_{index:06d}"
        observation=observe(step,folder)
        current=measured(step,contacts)
        write_json_atomic(folder/"conditioning_state.json",current)
        print(f"POLICY_REQUEST index={index} step={step} joint={current['articulated']['position']:.6f}",flush=True)
        if ARGS.policy_model == 'gr00t':
            from wo.models.gr00t_window import run_native_window
            actions,raw,metadata=run_native_window(window_root=folder / 'model',
              observation_root=output / 'observations',frame_index=step,
              joints=np.asarray(current['robot']['arm_joints'],dtype=np.float32),
              gripper=current['robot']['gripper'],prompt=config['instruction'],
              host=ARGS.policy_server_host,port=ARGS.policy_server_port)
        else:
            actions,raw,metadata=run_server_window(window_root=folder,input_image=observation,
              joints=np.asarray(current["robot"]["arm_joints"],dtype=np.float32),
              gripper=current["robot"]["gripper"],prompt=config["instruction"],
              host=ARGS.policy_server_host,port=ARGS.policy_server_port,request_index=index,
              seed_schedule_seed=config["seed_schedule_seed"],expected_request_seed=seeds[index])
            if metadata["decoded_video_shape"] is None: raise RuntimeError("policy server must decode imagination")
            observed_format=metadata['format_prompt_as_json']
            if index and runtime['format_prompt_as_json'] != observed_format:
                raise RuntimeError('prompt format changed during the episode')
            runtime['format_prompt_as_json']=observed_format
            runtime['prompt_format_evidence']='server_transform_attestation'
        write_json_atomic(output/'runtime.json',runtime)
        executed=[]
        for action in actions[1:1+horizon]:
            if step>=max_steps: break
            target[0,arm_ids]=torch.from_numpy(action[:7]).to(target.device)
            target[0,gripper_ids]=executable_robolab_gripper_target(action[7])
            robot.set_joint_position_target(target)
            contacts=advance()
            step+=1
            frame=measured(step,contacts)
            if ARGS.policy_model == 'gr00t':
                observe(step, output / 'native_views' / f'step_{step:06d}')
            if not frame['physical_valid']:
                frame['source_window']=index
                np.save(folder/'executed_actions.npy',np.asarray(executed+[action],dtype=np.float32),allow_pickle=False)
                write_json_atomic(output/'failed_physics_state.json',frame)
                write_json_atomic(output/'trajectory.json',{'schema_version':'cosmos_articulated_native_trajectory.v1','fps':15,'frames':frames+[frame]})
                raise RuntimeError('native robot/articulation constraint violation; episode is runtime_invalid')
            result=checker.step(frame["articulated"],contacts=any(contacts),
                               gripper=frame["robot"]["gripper"],step=step)
            frame["terminal"]=result;frame["source_window"]=index
            frames.append(frame); executed.append(action)
            if result["done"]: break
        np.save(folder/"executed_actions.npy",np.asarray(executed,dtype=np.float32),allow_pickle=False)
        entry={"index":index,"conditioning_frame":step-len(executed),
          "executed_count":len(executed),"request_seed":seeds[index],
          "conditioning_image_sha256":sha256_file(observation),
          "conditioning_state_sha256":sha256_file(folder/"conditioning_state.json"),
          "actions_sha256":sha256_file(folder / ('model/policy_actions.npy' if ARGS.policy_model == 'gr00t' else 'model/actions_raw.npy')),
          "decoded_sha256":None if ARGS.policy_model == 'gr00t' else sha256_file(folder/"model/droid_concat_rollout.npy"),
          "executed_actions_sha256":sha256_file(folder/"executed_actions.npy")}
        windows.append(entry)
        write_json_atomic(output/"trajectory.json",{"schema_version":"cosmos_articulated_native_trajectory.v1",
           "physics":f"Isaac Sim 5.1 PhysX on {ARGS.device}","fps":15,"frames":frames})
        write_json_atomic(output/"progress.json",{"step":step,"windows":windows,"terminal":result})
        print(f"POLICY_WINDOW_DONE index={index} step={step} joint={frames[-1]['articulated']['position']:.6f}",flush=True)
    observe(step,output/"terminal_view")
    if ARGS.policy_model == 'cosmos3':
        published=[]
        for entry in windows:
            video=np.load(output/"windows"/f"window_{entry['index']:06d}"/"model/droid_concat_rollout.npy",allow_pickle=False)
            if published: published.pop()
            published.extend(video[:entry["executed_count"]+1])
        write_h264_rgb(output/"imagination_raw_concat.mp4",np.asarray(published,dtype=np.uint8),fps=15)
    write_json_atomic(output/"result.json",{"schema_version":"cosmos_articulated_policy_trial.v1",
      "status":"complete","inference_launched":True,"formal_cae_certified":False,
      "task_success":result["task_success"],"termination_reason":result["reason"],
      "evaluation_window":evaluation_window,
      "evaluation_window_sha256":digest(evaluation_window) if evaluation_window else None,
      "completion_frame":checker.completion_frame,
      "frame_count":len(frames),"official_task_max_steps":max_steps,
      "action_count":step,"window_count":len(windows),"terminal":frames[-1],
      "maximum_progress":max(abs(f["articulated"]["position"]-task.closed_position) for f in frames),
      "contact_frames":sum(any(f["finger_contacts"]) for f in frames),"runtime":runtime,
      "raw_imagination":"imagination_raw_concat.mp4" if ARGS.policy_model == 'cosmos3' else None,"seed_schedule_seed":config["seed_schedule_seed"]})
    print("ARTICULATED_POLICY_EPISODE_COMPLETE "+json.dumps(result),flush=True)

if __name__=="__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        if ARGS.output.is_dir():
            write_json_atomic(ARGS.output/"runtime_error.json",{"status":"runtime_invalid","error":traceback.format_exc(),"model_score":None})
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
