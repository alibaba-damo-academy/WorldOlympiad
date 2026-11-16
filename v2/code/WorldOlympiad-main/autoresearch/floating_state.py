"""Measured floating mechanism state, independent of Isaac imports.

Inputs are live simulator link-frame poses/velocities (not COM poses). The
object frame is reconstructed from its topological root's declared zero pose;
child frames remain independently measured, never reconstructed from zero joints.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .isaac_articulation import floating_tree_root
from .isaac_compiler import _safe_name


def measured_floating_state(body, *, link_names, positions, quaternions_wxyz,
                            linear_velocities, angular_velocities,
                            joint_names, joint_positions, joint_velocities):
    root = floating_tree_root(body)
    expected_links = {link['prim_name'] for link in body['links']}
    joint_identities = {_safe_name(joint['name']): joint['name'] for joint in body['joints']}
    if len(joint_identities) != len(body['joints']):
        raise ValueError('compiled floating joint names alias after sanitization')
    expected_joints = set(joint_identities)
    if len(set(link_names)) != len(link_names) or set(link_names) != expected_links:
        raise ValueError('live floating link identities differ from compiled mechanism')
    if len(set(joint_names)) != len(joint_names) or set(joint_names) != expected_joints:
        raise ValueError('live floating joint identities differ from compiled mechanism')
    def array(value, shape):
        result = np.asarray(value, dtype=float)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError('missing or nonfinite live floating measurement')
        return result
    n, m = len(link_names), len(joint_names)
    p = array(positions, (n, 3)); q = array(quaternions_wxyz, (n, 4))
    v = array(linear_velocities, (n, 3)); w = array(angular_velocities, (n, 3))
    j = array(joint_positions, (m,)); jd = array(joint_velocities, (m,))
    if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-4):
        raise ValueError('live floating quaternions are not unit rotations')
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    root_link = next(link for link in body['links'] if link['name'] == root)
    index = link_names.index(root_link['prim_name'])
    zero = root_link['pose']
    # Compiler serializes Pose as xyz + quat_xyzw (not URDF Euler angles).
    zero_q = array(zero['quat_xyzw'], (4,))
    if not np.isclose(np.linalg.norm(zero_q), 1., atol=1e-4):
        raise ValueError('compiled floating root quaternion is not a unit rotation')
    object_rotation = rotations[index] * Rotation.from_quat(zero_q).inv()
    offset = object_rotation.apply(zero['xyz'])
    object_position = p[index] - offset
    links = {}
    for link in body['links']:
        i = link_names.index(link['prim_name'])
        links[link['name']] = {'position': p[i].tolist(),
            'orientation_xyzw': rotations[i].as_quat().tolist(),
            'linear_velocity': v[i].tolist(), 'angular_velocity': w[i].tolist()}
    return {'position': object_position.tolist(),
            'orientation_xyzw': object_rotation.as_quat().tolist(),
            'linear_velocity': (v[index] - np.cross(w[index], offset)).tolist(),
            'angular_velocity': w[index].tolist(), 'link_states': links,
            'joint_states': {joint_identities[name]: {'position': float(j[i]), 'velocity': float(jd[i])}
                             for i, name in enumerate(joint_names)},
            'measurement_source': 'native_floating_articulation_link_frames'}
