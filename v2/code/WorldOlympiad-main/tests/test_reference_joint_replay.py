import copy

import pytest

from autoresearch.articulated_native_task import reference_actions_for_native_replay


def payload():
    names=[f'panda_joint{i}' for i in range(7,0,-1)]
    return {'sample_hz':15,'frames':[
        {'time_s':t/15,'articulated':{'joint_name':'door'},
         'robot':{'joint_names':names,'joint_positions':[i*.1 for i in range(7,0,-1)],'droid_gripper_position':.25+t*.25}}
        for t in range(2)]}


def test_reference_replay_maps_names_and_preserves_gripper():
    result=reference_actions_for_native_replay(payload(),'door')
    assert result[0,:7]==pytest.approx([.1,.2,.3,.4,.5,.6,.7])
    assert result[:,7]==pytest.approx([.25,.5])


def test_reference_replay_rejects_changed_time_and_joint():
    data=payload();data['frames'][1]['time_s']=1.
    with pytest.raises(ValueError,match='timestamps'):reference_actions_for_native_replay(data,'door')
    with pytest.raises(ValueError,match='differs'):reference_actions_for_native_replay(payload(),'drawer')
