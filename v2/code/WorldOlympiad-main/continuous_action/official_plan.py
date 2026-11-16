"""Provenance contract for unmodified, pinned Dream2Flow optimizer outputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROTOCOL = 'video_imagination_dream2flow_official_action_trajectory_v2'
TRAJECTORY_SOURCE = 'official_dream2flow_trajectory_optimizer_ee_trajectory'
UPSTREAM_COMMIT = 'f87be21190015f49bacf5c03b81c7192b92cc567'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_plan(path, video_path, role):
    path, video_path = Path(path), Path(video_path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('evaluation_protocol') != PROTOCOL:
        raise ValueError('retired or missing official optimizer source protocol')
    if meta.get('trajectory_source') != TRAJECTORY_SOURCE or meta.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('official pinned Dream2Flow optimizer provenance is required')
    if meta.get('video_role') != role or meta.get('source_video_sha256') != sha256(video_path):
        raise ValueError('official plan/video provenance mismatch')
    if meta.get('plan_sha256') != sha256(path):
        raise ValueError('official plan digest mismatch')
    lock = json.loads((Path(__file__).parents[1] / 'scripts/official_dream2flow_source_lock.json').read_text())
    if meta.get('source_sha256') != lock['source_sha256']:
        raise ValueError('official planner source receipt differs from pinned upstream')
    if not all(meta.get('optimizer_config',{}).get(k) == v for k,v in lock['optimizer_numerical_defaults'].items()):
        raise ValueError('official packaged optimizer settings changed')
    if meta.get('initialization_kind') == 'frame_zero_kinematic_grasp':
        if meta.get('initial_pose_source') != 'registered_grasp_target':
            raise ValueError('official grasp plan must use the independently registered initial pose')
    elif meta.get('initialization_kind') == 'frame_zero_kinematic_contact':
        if meta.get('initial_pose_source') != 'registered_contact_target':
            raise ValueError('contact plan requires an independently registered contact target')
    elif meta.get('initialization_kind') != 'native_grasped_state_verified':
        raise ValueError('rest-pose initialization is not a rigid grasp')
    for key in ('flow', 'registration', 'registered_urdf'):
        source = Path(meta[key + '_path'])
        if sha256(source) != meta.get(key + '_sha256'):
            raise ValueError(f'official plan {key} provenance mismatch')
    if meta.get('object_to_robot_ik_used') is not True or meta.get('action_head_used') is not False:
        raise ValueError('official video-to-plan source flags are invalid')
    if meta.get('coordinate_frame') != 'registered_world':
        raise ValueError('official plans must share the registered world frame')
    return meta

