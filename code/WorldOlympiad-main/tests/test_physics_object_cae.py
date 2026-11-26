import numpy as np
import pytest
from continuous_action.physics_object import object_trajectory, score_object_pair

POINTS=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])

def make(tmp_path,name,positions,role,valid=None,fps=4):
    path=tmp_path/(name+".npz")
    np.savez(path,position=positions,valid_mask=np.ones(positions.shape[:2],bool) if valid is None else valid)
    return object_trajectory(path,fps=fps,expected_frames=len(positions),video_sha256="test",role=role)

def test_rigid_motion_with_changing_visibility(tmp_path):
    poses=[];expected=[];anchor=POINTS.mean(0)
    for a in np.linspace(0,np.pi/2,9):
        rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        translation=np.array([a,0,0])
        poses.append(POINTS@rot.T+translation);expected.append(rot@anchor+translation)
    poses=np.asarray(poses);valid=np.ones((9,5),bool);valid[3:,0]=False
    result=make(tmp_path,"motion",poses,"reference",valid)
    np.testing.assert_allclose(result.positions,expected,atol=1e-12)

def test_same_motion_scores_one_and_static_prediction_zero(tmp_path):
    poses=POINTS[None]+np.linspace(0,1,9)[:,None,None]*np.array([1.,0.,0.])
    ref=make(tmp_path,"ref",poses,"reference")
    pred=make(tmp_path,"pred",poses.copy(),"imagination")
    assert score_object_pair(ref,pred)["continuous_score"]==pytest.approx(1.)
    stopped=make(tmp_path,"stopped",np.repeat(POINTS[None],9,axis=0),"imagination")
    assert score_object_pair(ref,stopped)["continuous_score"]==0.

def test_incomplete_coverage_is_not_model_zero(tmp_path):
    poses=np.repeat(POINTS[None],9,axis=0);valid=np.ones((9,5),bool);valid[5,:4]=False
    with pytest.raises(ValueError,match="coverage"):
        make(tmp_path,"missing",poses,"imagination",valid)

def test_native_timing_is_preserved(tmp_path):
    poses=POINTS[None]+np.linspace(0,1,9)[:,None,None]*np.array([1.,0.,0.])
    result=make(tmp_path,"timing",poses,"reference",fps=16)
    np.testing.assert_array_equal(result.timestamps,np.arange(9)/16)
    assert result.metadata["source_frame_indices"]==list(range(9))


def test_cli_accepts_distinct_reference_and_world_model_timelines(tmp_path, monkeypatch):
    import json
    import sys
    from scripts.score_physics_object_cae import main

    reference = tmp_path / "reference"
    imagination = tmp_path / "imagination"
    reference.mkdir()
    imagination.mkdir()
    for root, frames, fps in ((reference, 9, 15), (imagination, 12, 16)):
        times = np.arange(frames) / fps
        positions = POINTS[None] + times[:, None, None] * np.array([1., 0, 0])
        np.savez(root / "object_flow_result.npz", position=positions,
                 valid_mask=np.ones(positions.shape[:2], bool))
    result = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", ["score_physics_object_cae.py",
        "--reference-dir", str(reference), "--imagination-dir", str(imagination),
        "--reference-fps", "15", "--reference-frames", "9",
        "--imagination-fps", "16", "--imagination-frames", "12",
        "--reference-video-sha256", "a" * 64,
        "--imagination-video-sha256", "b" * 64,
        "--output", str(result)])
    main()
    assert 0 <= json.loads(result.read_text())["continuous_score"] <= 1
