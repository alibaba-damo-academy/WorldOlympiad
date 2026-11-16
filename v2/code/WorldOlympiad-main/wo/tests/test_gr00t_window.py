import numpy as np
import pytest
from wo.models.gr00t_window import eef_9d, build_request, unpack_actions, run_window


def obs():
    return dict(exterior_left=np.zeros((180,320,3), dtype=np.uint8),
                wrist=np.full((360,640,3), 127, dtype=np.uint8),
                joints=np.arange(7), gripper=0.2, ee_position=[1,2,3],
                ee_quat_wxyz=[1,0,0,0], prompt='Place the cube on the tray.')


def test_request_matches_robolab_droid_contract():
    request = build_request(**obs(), version='n1.7')
    n16 = build_request(**obs())
    assert 'state.eef_9d' not in n16
    assert request['video.wrist_image_left'].shape == (1,1,180,320,3)
    assert request['state.joint_position'].shape == (1,1,7)
    assert request['state.gripper_position'].shape == (1,1,1)
    np.testing.assert_array_equal(request['state.eef_9d'][0,0], [1,2,3,0,0,-1,-1,0,0])
    np.testing.assert_array_equal(eef_9d([1,2,3], [2,0,0,0]), request['state.eef_9d'][0,0])


def test_action_shape_horizon_gripper():
    response = ({'action.joint_position': np.zeros((1,16,7)),
                 'action.gripper_position': np.linspace(0,1,16).reshape(1,16,1)}, {})
    a = unpack_actions(response)
    assert a.shape == (8,8)
    assert set(a[:,7]) == {0}
    assert unpack_actions(response, 16)[-1,7] == 1
    with pytest.raises(ValueError, match='insufficient'):
        unpack_actions(response, 32)


def test_invalid_observations_and_actions():
    for key, value in [('joints',[0]*6), ('gripper',2), ('ee_quat_wxyz',[0]*4),
                       ('ee_position',[0,0,float('nan')]), ('prompt','')]:
        data = obs(); data[key] = value
        with pytest.raises(ValueError):
            build_request(**data, version='n1.7')
    with pytest.raises(ValueError, match='nonfinite'):
        unpack_actions({'joint_position': np.full((8,7), np.nan), 'gripper_position': np.ones(8)})


def test_native_window_uses_separate_rgb_and_prepends_measured_state(tmp_path):
    from PIL import Image
    from wo.models.gr00t_window import run_native_window
    for short, value in [('left', 23), ('wrist', 91)]:
        Image.fromarray(np.full((360,640,3), value, dtype=np.uint8)).save(
            tmp_path / f'step_000012_{short}_image.png')
    closed = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs == dict(host='localhost', port=5555)
        def get_action(self, request):
            assert np.all(request['video.exterior_image_1_left'] == 23)
            assert np.all(request['video.wrist_image_left'] == 91)
            return {'joint_position': np.ones((16,7)), 'gripper_position': np.ones((16,1))}
        def close(self):
            closed.append(True)
    a, raw, meta = run_native_window(window_root=tmp_path/'window', observation_root=tmp_path,
        frame_index=12, joints=np.arange(7), gripper=.25, prompt='Move cube',
        host='localhost', port=5555, client_factory=Client)
    assert a.shape == (9,8)
    np.testing.assert_array_equal(a[0], [0,1,2,3,4,5,6,.25])
    np.testing.assert_array_equal(a[1:], np.ones((8,8)))
    assert closed == [True]
    assert meta['model'] == 'gr00t_n1.6_droid'


def test_window_records_policy_not_executed_trajectory(tmp_path):
    class Client:
        def get_action(self, request):
            assert 'state.eef_9d' not in request
            return {'joint_position':np.zeros((16,7)), 'gripper_position':np.ones((16,1))}
    a, meta = run_window(client=Client(), window_root=tmp_path, **obs())
    assert a.shape == (8,8)
    assert (tmp_path/'policy_actions.npy').is_file()
    assert not (tmp_path/'executed_actions.npy').exists()
    assert meta['trajectory_source'] == 'native_simulator'
