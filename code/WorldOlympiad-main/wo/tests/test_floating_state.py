import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autoresearch.floating_state import measured_floating_state


def fixture():
    body = {'static': False, 'links': [
        {'name': 'cap', 'prim_name': 'cap', 'pose': {'xyz': [0, 0, 1], 'quat_xyzw': [0, 0, 0, 1]}},
        {'name': 'base', 'prim_name': 'base', 'pose': {'xyz': [1, 0, 0], 'quat_xyzw': Rotation.from_euler('z', np.pi/2).as_quat().tolist()}}],
        'joints': [{'name': 'slide', 'parent': 'base', 'child': 'cap', 'type': 'prismatic'}]}
    # Object frame rotates 90 degrees; root zero frame rotates another90.
    args = dict(link_names=['cap', 'base'], positions=[[4, 5, 8], [4, 6, 6]],
        quaternions_wxyz=[[1, 0, 0, 0], [0, 0, 0, 1]],
        linear_velocities=[[0, 0, 0], [-1, 2, 3]], angular_velocities=[[0, 0, 0], [0, 0, 1]],
        joint_names=['slide'], joint_positions=[.2], joint_velocities=[.03])
    return body, args


def test_measured_child_and_root_offset_not_zero_pose_or_first_link():
    body, args = fixture()
    state = measured_floating_state(body, **args)
    assert state['position'] == pytest.approx([4, 5, 6])
    assert Rotation.from_quat(state['orientation_xyzw']).apply([1, 0, 0]) == pytest.approx([0, 1, 0])
    assert state['linear_velocity'] == pytest.approx([0, 2, 3])
    assert state['link_states']['cap']['position'] == [4, 5, 8]
    assert state['joint_states']['slide'] == {'position': .2, 'velocity': .03}


@pytest.mark.parametrize('field,value', [
    ('link_names', ['base', 'base']), ('link_names', ['cap', 'other']),
    ('joint_names', ['other']), ('positions', [[0, 0, 0]]),
    ('joint_positions', [float('nan')]),
    ('quaternions_wxyz', [[0, 0, 0, 0], [1, 0, 0, 0]])])
def test_missing_invalid_or_unmeasured_state_rejected(field, value):
    body, args = fixture(); args[field] = value
    with pytest.raises(ValueError):
        measured_floating_state(body, **args)


@pytest.mark.parametrize('quaternion', [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, float('nan'), 1]])
def test_compiled_root_rotation_must_be_finite_unit_quaternion(quaternion):
    body, args = fixture()
    body['links'][1]['pose']['quat_xyzw'] = quaternion
    with pytest.raises(ValueError):
        measured_floating_state(body, **args)
