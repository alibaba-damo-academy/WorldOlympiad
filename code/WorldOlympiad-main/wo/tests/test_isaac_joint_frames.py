import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autoresearch.isaac_joint_frames import joint_local_frames


def pose(xyz, rpy):
    return {'xyz': xyz, 'quat_xyzw': Rotation.from_euler('xyz', rpy).as_quat().tolist()}


@pytest.mark.parametrize('axis', [[1,0,0], [-1,0,0], [0,-1,0], [1,2,-3], [0,0,1]])
def test_model_frame_reconstruction_with_rotated_parent_child_and_joint(axis):
    parent = pose([1,2,3], [.2,.7,-.3])
    child = pose([-.2,1,4], [-.5,.1,.9])
    joint = pose([.3,2.5,3.1], [.6,-.2,.5])
    frames = joint_local_frames(parent, child, joint, axis)
    expected_axis = Rotation.from_quat(joint['quat_xyzw']).apply(np.array(axis)/np.linalg.norm(axis))
    reconstructed = []
    for link, frame in zip((parent, child), frames):
        rotation = Rotation.from_quat(link['quat_xyzw'])
        np.testing.assert_allclose(rotation.apply(frame['xyz']) + link['xyz'], joint['xyz'], atol=1e-12)
        world_rot = rotation * Rotation.from_quat(frame['quat_xyzw'])
        np.testing.assert_allclose(world_rot.apply([1,0,0]), expected_axis, atol=1e-12)
        reconstructed.append(world_rot.as_matrix())
    np.testing.assert_allclose(*reconstructed, atol=1e-12)


@pytest.mark.parametrize('axis', [[0,0,0], [float('nan'),0,0], [1,2]])
def test_invalid_axis_rejected(axis):
    with pytest.raises(ValueError, match='axis'):
        joint_local_frames({}, {}, {}, axis)


def test_invalid_pose_rejected():
    with pytest.raises(ValueError, match='pose'):
        joint_local_frames({'quat_xyzw':[0,0,0,0]}, {}, {}, [1,0,0])


def test_unrotated_drawer_preserves_signed_displacement():
    frames = joint_local_frames({}, {'xyz':[0,2,0]}, {'xyz':[0,2,0]}, [0,-1,0])
    np.testing.assert_allclose(frames[0]['xyz'], [0,2,0])
    np.testing.assert_allclose(frames[1]['xyz'], [0,0,0])
    np.testing.assert_allclose(Rotation.from_quat(frames[0]['quat_xyzw']).apply([1,0,0]), [0,-1,0], atol=1e-12)
