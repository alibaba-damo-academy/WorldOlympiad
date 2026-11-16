"""Convert model-frame WorldIR joints into USD local X-axis joint frames."""
import numpy as np
from scipy.spatial.transform import Rotation


def joint_local_frames(parent_pose, child_pose, joint_pose, axis):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-9:
        raise ValueError('joint axis must be finite and nonzero')
    axis = axis / np.linalg.norm(axis)
    # Shortest rotation from USD +X to the actual joint-local signed axis.
    cross = np.cross([1., 0., 0.], axis)
    if axis[0] < -1 + 1e-12:
        alignment = Rotation.from_rotvec([0., np.pi, 0.])
    else:
        quat = np.r_[cross, 1 + axis[0]]
        alignment = Rotation.from_quat(quat / np.linalg.norm(quat))

    def pose(value):
        xyz = np.asarray(value.get('xyz', [0., 0., 0.]), dtype=float)
        quat = np.asarray(value.get('quat_xyzw', [0., 0., 0., 1.]), dtype=float)
        if (xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all()
                or not np.isfinite(quat).all() or np.linalg.norm(quat) <= 1e-9):
            raise ValueError('joint frame pose must be finite with nonzero quaternion')
        return xyz, Rotation.from_quat(quat)

    xyz, rotation = pose(joint_pose or {})
    frame_rotation = rotation * alignment
    result = []
    for link in (parent_pose, child_pose):
        link_xyz, link_rotation = pose(link)
        inverse = link_rotation.inv()
        result.append({'xyz': inverse.apply(xyz - link_xyz).tolist(),
                       'quat_xyzw': (inverse * frame_rotation).as_quat().tolist()})
    return result
