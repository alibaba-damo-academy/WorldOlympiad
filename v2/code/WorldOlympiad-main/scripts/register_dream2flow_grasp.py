#!/usr/bin/env python3
"""Register a frame-zero geometric grasp using the pinned upstream PyRoki IK example.

This initializes offline flow following; it does not execute or condition a policy.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='0'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from continuous_action.official_plan import sha256
from run_official_dream2flow_plan import make_registered_urdf


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-registration',type=Path,required=True)
    p.add_argument('--grasp-pose',type=Path,required=True)
    p.add_argument('--pyroki-repo',type=Path,required=True)
    p.add_argument('--dream2flow-repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interaction-kind',choices=['grasp','contact'],default='grasp')
    a=p.parse_args()
    import numpy as np
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(a.dream2flow_repo.resolve()))
    from dream2flow.planner.trajectory_optimizer import TrajectoryOptimizer,TrajectoryOptimizerConfig
    native=json.loads(a.native_registration.read_text())
    grasp=json.loads(a.grasp_pose.read_text())
    if grasp['source_frame'] != 0:raise ValueError('grasp initialization must use frame-zero geometry')
    target=np.asarray(grasp['pose_xyzw'],dtype=float)
    urdf=make_registered_urdf(native,a.native_registration.with_suffix('.urdf'))
    planner=TrajectoryOptimizer(TrajectoryOptimizerConfig(urdf_path=str(urdf),target_link_name=native['target_link_name']))
    source=a.pyroki_repo/'examples/pyroki_snippets/_solve_ik.py'
    lock=json.loads(Path(__file__).with_name('official_dream2flow_source_lock.json').read_text())
    if sha256(source) != lock['pyroki_ik_sha256']:raise ValueError('upstream PyRoki IK example changed')
    spec=importlib.util.spec_from_file_location('pinned_pyroki_ik_example',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    candidates=[grasp['pose_xyzw'], *grasp.get('equivalent_grasp_poses_xyzw', [])]
    failures=[]
    for candidate_index, candidate in enumerate(candidates):
        target=np.asarray(candidate,dtype=float)
        wxyz=target[[6,3,4,5]]
        joints=module.solve_ik(planner.robot,native['target_link_name'],wxyz,target[:3])
        actual=planner.get_end_effector_pose(joints)
        pe=float(np.linalg.norm(actual[:3]-target[:3]))
        re=float((Rotation.from_quat(actual[3:]).inv()*Rotation.from_quat(target[3:])).magnitude())
        if np.isfinite(joints).all() and pe<=.001 and re<=.01:
            break
        failures.append({'candidate':candidate_index,'position_error_m':pe,'rotation_error_rad':re})
    else:
        raise ValueError(f'no geometrically valid grasp IK candidate: {failures}')
    result={**native,'initialization_kind':'frame_zero_kinematic_'+a.interaction_kind,
        'native_frame_zero_initial_joints':native['initial_joint_positions'],
        'initial_joint_positions':dict(zip(planner.robot.joints.actuated_names,joints.tolist())),
        'initial_pose_validation_position':target[:3].tolist(),
        'grasp_target_pose_xyzw':target.tolist(),'selected_grasp_candidate':candidate_index,'rejected_ik_candidates':failures,'grasp_fk_position_error_m':pe,'grasp_fk_rotation_error_rad':re,
        'native_registration_path':str(a.native_registration.resolve()),'native_registration_sha256':sha256(a.native_registration),
        'grasp_input_path':str(a.grasp_pose.resolve()),'grasp_input_sha256':sha256(a.grasp_pose),
        'grasp_ik_source_path':str(source.resolve()),'grasp_ik_source_sha256':sha256(source),
        'grasp_ik_upstream_commit':'388e43e1fc0d0ee382968d3dd72970fd62a0450c',
        'grasp_source':'project frame-zero task geometry, initialized with upstream PyRoki IK; not AnyGrasp/HaMeR',
        'policy_execution_used':False}
    if a.interaction_kind == 'contact':
        result['contact_target_pose_xyzw']=result.pop('grasp_target_pose_xyzw')
        result['grasp_source']='independently registered frame-zero contact target; no rigid grasp claimed'
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'output':str(a.output),'position_error_m':pe,'orientation_error_rad':re}))

if __name__=='__main__':main()
