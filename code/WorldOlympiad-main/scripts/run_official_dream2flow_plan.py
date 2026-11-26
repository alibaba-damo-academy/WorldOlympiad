#!/usr/bin/env python3
"""Run pinned upstream Dream2Flow on full cached flow, without changing its solver."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import copy

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_action.official_plan import PROTOCOL, TRAJECTORY_SOURCE, UPSTREAM_COMMIT, sha256, validate_plan


def make_registered_urdf(registration, output):
    """Preserve source chain kinematics, apply registered native limits, and attach its world root."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    src = Path(registration['source_urdf'])
    if sha256(src) != registration['source_urdf_sha256']:
        raise ValueError('source robot URDF changed')
    root = ET.parse(src).getroot()
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    current = registration['target_link_name']
    chain = []
    while current in joints:
        j = joints[current]
        chain.append(j)
        current = j.find('parent').get('link')
    chain.reverse()
    names = [j.get('name') for j in chain if j.get('type') != 'fixed']
    if set(names) != set(registration['initial_joint_positions']):
        raise ValueError('initial joint names do not match the selected robot chain')
    out = ET.Element('robot', name='registered_official_dream2flow_chain')
    ET.SubElement(out, 'link', name='registered_world')
    link_names = [current] + [j.find('child').get('link') for j in chain]
    for name in link_names:
        ET.SubElement(out, 'link', name=name)
    mount = ET.SubElement(out, 'joint', name='registered_world_mount', type='fixed')
    ET.SubElement(mount, 'parent', link='registered_world')
    ET.SubElement(mount, 'child', link=current)
    pose = np.asarray(registration['root_pose_xyzw'])
    rpy = Rotation.from_quat(pose[3:]).as_euler('xyz')
    ET.SubElement(mount, 'origin', xyz=' '.join(map(str, pose[:3])), rpy=' '.join(map(str, rpy)))
    native_limits = registration.get('native_joint_limits', {})
    evidence = registration.get('native_joint_limits_evidence')
    if native_limits and evidence and sha256(evidence['path']) != evidence['sha256']:
        raise ValueError('native joint-limit evidence changed')
    for j in chain:
        joint = copy.deepcopy(j)
        if joint.get('name') in native_limits:
            lower, upper = native_limits[joint.get('name')]
            if not np.isfinite([lower,upper]).all() or lower >= upper:
                raise ValueError('invalid native joint limit')
            joint.find('limit').set('lower', str(lower))
            joint.find('limit').set('upper', str(upper))
        out.append(joint)
    data = ET.tostring(out, encoding='utf-8', xml_declaration=True)
    output = Path(output)
    if output.exists() and output.read_bytes() != data:
        raise FileExistsError('registered URDF is immutable; create a new case')
    output.write_bytes(data)
    return output


def run(args):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    repo = Path(args.dream2flow_repo).resolve()
    lock = json.loads((Path(__file__).with_name('official_dream2flow_source_lock.json')).read_text())
    for relative, digest in lock['source_sha256'].items():
        if sha256(repo / relative) != digest:
            raise ValueError(f'upstream source differs from pinned commit: {relative}')
    sys.path.insert(0, str(repo))
    from dream2flow.flow.object_flow import ObjectFlow
    from dream2flow.flow.object_flow_result import ObjectFlowResult
    from dream2flow.planner.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerConfig
    registration_path = Path(args.registration).resolve()
    registration = json.loads(registration_path.read_text())
    if registration.get('initialization_kind') not in {'frame_zero_kinematic_grasp','frame_zero_kinematic_contact','native_grasped_state_verified'}:
        raise ValueError('official rigid-grasp planning requires a registered grasp initialization; use register_dream2flow_grasp.py with frame-zero geometry')
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = validate_plan(out, args.video, args.role)
        if old['registration_sha256'] != sha256(registration_path) or old['flow_sha256'] != sha256(args.flow):
            raise FileExistsError('cached official plan binds different inputs')
        print(json.dumps({'reused': str(out)})); return
    urdf = make_registered_urdf(registration, registration_path.with_suffix('.urdf'))
    import yaml
    settings = yaml.safe_load((repo / 'dream2flow/configs/trajectory_optimizer_default.yaml').read_text())
    settings.update(urdf_path=str(urdf), target_link_name=registration['target_link_name'], visualize=False)
    config = TrajectoryOptimizerConfig(**settings)
    planner = TrajectoryOptimizer(config)
    names = list(planner.robot.joints.actuated_names)
    initial = np.array([registration['initial_joint_positions'][n] for n in names], dtype=np.float64)
    initial_pose = planner.get_end_effector_pose(initial)
    position_error = float(np.linalg.norm(initial_pose[:3] - np.array(registration.get('initial_pose_validation_position',registration['native_target_position']))))
    if position_error > 0.001:
        raise ValueError(f'robot registration disagrees with native initial FK by {position_error} m')
    # The public entry supports a separately measured/registered initial_pose.
    # Use that pose, not FK transformed back onto itself (zero-norm derivatives
    # can become NaN in the upstream particle-distance cost at frame zero).
    if 'contact_target_pose_xyzw' in registration:
        initial_pose = np.asarray(registration['contact_target_pose_xyzw'], dtype=np.float64)
    elif 'grasp_target_pose_xyzw' in registration:
        initial_pose = np.asarray(registration['grasp_target_pose_xyzw'], dtype=np.float64)
    else:
        initial_pose = initial_pose.copy()
        initial_pose[:3] = registration['native_target_position']

    with np.load(args.flow, allow_pickle=False) as f:
        position = f['position'].copy()
        valid = f['valid_mask'].astype(bool)
        pixels = f['initial_pixel_coords'].copy()
    if position.ndim != 3 or position.shape[-1] != 3 or valid.shape != position.shape[:2]:
        raise ValueError('invalid full object-flow tensor shape')
    if not np.isfinite(position).all() or not valid[0].any():
        raise ValueError('official optimizer requires finite positions and observable initial particles')
    if len(position) != args.frame_count or args.fps <= 0:
        raise ValueError('flow/native video timing mismatch')
    flow = ObjectFlowResult(flow=ObjectFlow(position=torch.from_numpy(position), valid_mask=torch.from_numpy(valid)), initial_pixel_coords=torch.from_numpy(pixels))
    result = planner.plan(flow, initial_joints=initial, initial_pose=initial_pose)
    ee, joints = np.asarray(result.ee_trajectory), np.asarray(result.joint_trajectory)
    if not np.isfinite(ee).all() or not np.isfinite(joints).all():
        raise ValueError('official optimizer returned nonfinite trajectory')
    step = max(1, len(position) // config.max_num_timesteps_for_optimization)
    indices = np.arange(0, len(position), step, dtype=np.int64)
    if len(ee) != len(indices):
        raise ValueError('upstream optimizer subsampling changed')
    # Never bridge invisible native frames merely because upstream subsamples them.
    native_observed = valid.any(axis=1)
    observed = native_observed[indices].copy()
    for i in range(1, len(indices)):
        observed[i] &= bool(native_observed[indices[i-1]:indices[i]+1].all())
    local = Rotation.from_quat(initial_pose[3:]).inv().apply(position[0] - initial_pose[:3])
    fitted = np.stack([Rotation.from_quat(p[3:]).apply(local) + p[:3] for p in ee])
    errors = np.linalg.norm(fitted - position[indices], axis=-1)
    mask = valid[indices]
    residual = errors[mask]
    # Diagnostics only: never use simulator futures, post-fit smoothing or residual-based score tuning.
    limits = {j.get('name'): j.find('limit') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type') != 'fixed'}
    violations = [max(float(np.max(float(limits[n].get('lower')) - joints[:, i])), float(np.max(joints[:, i] - float(limits[n].get('upper')))), 0.) for i, n in enumerate(names)]
    with out.open('xb') as f:
        np.savez_compressed(f, ee_trajectory=ee, joint_trajectory=joints, timestamps=indices / args.fps, source_frame_indices=indices, source_video_fps=args.fps, source_video_frame_count=args.frame_count, observed_frame_mask=observed)
    meta = dict(trajectory_source=TRAJECTORY_SOURCE, evaluation_protocol=PROTOCOL, upstream_commit=UPSTREAM_COMMIT, source_sha256=lock['source_sha256'], coordinate_frame='registered_world', object_name=args.object_name, video_role=args.role, source_video_sha256=sha256(args.video), conditioning_video=str(Path(args.video).resolve()), source_video_fps=args.fps, source_video_frame_count=args.frame_count, flow_path=str(Path(args.flow).resolve()), flow_sha256=sha256(args.flow), registration_path=str(registration_path), registration_sha256=sha256(registration_path), registered_urdf_path=str(urdf), registered_urdf_sha256=sha256(urdf), plan_sha256=sha256(out), action_head_used=False, object_to_robot_ik_used=True, optimizer_config=asdict(config), initial_joint_names=names, initial_joints=initial.tolist(), initial_pose_xyzw=initial_pose.tolist(), initial_pose_source='registered_contact_target' if 'contact_target_pose_xyzw' in registration else 'registered_grasp_target' if 'grasp_target_pose_xyzw' in registration else 'native_measured_position_with_fk_orientation', native_fk_position_error_m=position_error if registration.get('initialization_kind') == 'native_grasped_state_verified' else None, initialization_fk_position_error_m=position_error, initialization_kind=registration.get('initialization_kind','native_frame_zero'), flow_particle_count=position.shape[1], source_frame_indices=indices.tolist(), native_observed_frame_mask=native_observed.tolist(), object_fit_mean_m=float(residual.mean()), object_fit_p95_m=float(np.percentile(residual,95)), max_joint_limit_violation_rad=max(violations), initialization_assumption='official fixed initial particle-to-EE transform; '+registration.get('initialization_kind','native_frame_zero_pregrasp'), jax_enable_x64=os.environ.get('JAX_ENABLE_X64','0'), wrapper_sha256=sha256(__file__), runtime={n: importlib.metadata.version(n) for n in ['jax','jaxlib','jaxls','pyroki','torch','tensordict']})
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'output':str(out),'frames':len(ee),'object_fit_mean_m':meta['object_fit_mean_m'],'native_fk_position_error_m':position_error}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['dream2flow-repo','registration','flow','video','output','object-name']:
        p.add_argument('--'+k,required=True)
    p.add_argument('--role',choices=['reference','imagination'],required=True)
    p.add_argument('--fps',type=float,required=True)
    p.add_argument('--frame-count',type=int,required=True)
    run(p.parse_args())

if __name__ == '__main__': main()
