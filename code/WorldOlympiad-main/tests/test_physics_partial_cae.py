import numpy as np
import pytest
from continuous_action.physics_partial import score_partial_pair

P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])

def pair(tmp_path,missing):
    info={"roles":{}}
    for role in ["reference","imagination"]:
        root=tmp_path/role;root.mkdir()
        shift=np.arange(11)/10;shift[5:]+=100
        positions=P[None]+shift[:,None,None]*np.array([1.,0,0])
        mask=np.ones((11,5),bool)
        if role=="reference" and missing:mask[3:5]=False
        np.savez(root/"object_flow_result.npz",position=positions,valid_mask=mask)
        info["roles"][role]={"dir":str(root),"fps":10,"frames":11,"video_sha256":"test"}
    return info

def test_partial_identity_scores_one_without_filling_gap(tmp_path):
    result,tracks=score_partial_pair(pair(tmp_path,True))
    assert result["continuous_score"]==pytest.approx(1.)
    assert result["reference_coverage"]==pytest.approx(.7)
    assert result["prediction_coverage"]==pytest.approx(.7)
    assert result["partial_observation"] is True
    for t in tracks.values():
        assert np.linalg.norm(np.diff(t.positions,axis=0),axis=1).sum()==pytest.approx(.7)
        assert t.duration==pytest.approx(.7)

def test_complete_pair_keeps_full_native_score(tmp_path):
    result,_=score_partial_pair(pair(tmp_path,False))
    assert result["continuous_score"]==pytest.approx(1.)
    assert result["reference_coverage"]==1.
    assert result["partial_observation"] is False

def test_partial_half_motion_uses_current_two_component_score(tmp_path):
    info=pair(tmp_path,True)
    file=tmp_path/"imagination"/"object_flow_result.npz"
    with np.load(file) as d:
        position=d["position"].copy()
        mask=d["valid_mask"].copy()
    position=P[None]+(position-P[None])*0.5
    np.savez(file,position=position,valid_mask=mask)
    result,_=score_partial_pair(info)
    assert result["continuous_score"]==pytest.approx(np.sqrt((2/3)*.8))
    assert result["observation_coverage"]=="partial"
    assert result["reference_coverage"]==pytest.approx(.7)

def test_isolated_visible_frames_do_not_create_motion(tmp_path):
    info=pair(tmp_path,True)
    file=tmp_path/"reference"/"object_flow_result.npz"
    with np.load(file) as d:
        position=d["position"].copy()
        mask=d["valid_mask"].copy()
    mask[:]=False
    mask[::2]=True
    np.savez(file,position=position,valid_mask=mask)
    with pytest.raises(ValueError,match="no jointly observed adjacent frames"):
        score_partial_pair(info)
