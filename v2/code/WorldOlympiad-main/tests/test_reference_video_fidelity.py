import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from autoresearch.media import write_h264_rgb
from autoresearch.reference_video_fidelity import inspect_source_fidelity
from autoresearch.robotics_quality import certify_reference_views, inspect_video, certify_wam_output


def sequence(tmp_path, name="source", static=False):
    folder=tmp_path/name;folder.mkdir()
    frames=[];paths=[]
    for i in range(30):
        frame=np.full((96,160,3),100,np.uint8)
        # A short slow motion followed by a hold: legitimate low episode mean.
        x=12 if static else 12+min(i,10)
        frame[38:46,x:x+8]=220
        path=folder/f"{i:05d}.png";cv2.imwrite(str(path),frame)
        frames.append(frame);paths.append(path)
    return frames,paths


def encode(path,frames):
    write_h264_rgb(path,frames,15,crf=14)
    return path


def test_slow_motion_is_verified_from_source_even_when_old_average_gate_rejects(tmp_path):
    frames,paths=sequence(tmp_path)
    video=encode(tmp_path/"slow.mp4",frames)
    old=inspect_video(video,min_width=160,min_height=96,min_duration_sec=0)
    assert not old["checks"]["non_static"]
    report=inspect_source_fidelity(video,paths,fps=15)
    assert report["passed"] and report["source_has_local_motion"]


@pytest.mark.parametrize("corruption",["frozen","reversed","missing","extra","wrong_fps"])
def test_corrupt_encoding_is_rejected_against_actual_render_frames(tmp_path,corruption):
    frames,paths=sequence(tmp_path)
    altered={"frozen":[frames[0]]*len(frames),"reversed":frames[::-1],
             "missing":frames[:-1],"extra":frames+[frames[-1]],"wrong_fps":frames}[corruption]
    video=tmp_path/"corrupt.mp4"
    write_h264_rgb(video,altered,10 if corruption=="wrong_fps" else 15,crf=14)
    assert not inspect_source_fidelity(video,paths,fps=15)["passed"]


def test_static_view_can_be_valid_but_all_static_episode_cannot(tmp_path):
    dynamic,dynamic_paths=sequence(tmp_path,"dynamic")
    static,static_paths=sequence(tmp_path,"static",static=True)
    moving=encode(tmp_path/"moving.mp4",dynamic);still=encode(tmp_path/"still.mp4",static)
    names=["top","wrist","exterior_left","exterior_right"]
    videos={n:still for n in names};sources={n:static_paths for n in names}
    sizes={n:(160,96) for n in names}
    path=certify_reference_views(videos,tmp_path/"static_report.json",expected_sizes=sizes,
                                source_frames=sources,expected_fps=15)
    assert not json.loads(path.read_text())["passed"]
    videos["exterior_left"]=moving;sources["exterior_left"]=dynamic_paths
    path=certify_reference_views(videos,tmp_path/"mixed_report.json",expected_sizes=sizes,
                                source_frames=sources,expected_fps=15)
    result=json.loads(path.read_text())
    assert result["passed"] and result["schema_version"]=="robotics_reference_quality.v3"
    # Correctly encoded short reference timelines are not rejected by an
    # unrelated universal eight-second minimum.
    assert result["views"]["top"]["passed"]


def test_returning_to_start_is_not_misclassified_as_static(tmp_path):
    frames,_=sequence(tmp_path)
    cycle=frames[:11]+frames[:10][::-1]
    folder=tmp_path/"cycle";folder.mkdir();paths=[]
    for i,frame in enumerate(cycle):
        p=folder/f"{i:05d}.png";cv2.imwrite(str(p),frame);paths.append(p)
    video=encode(tmp_path/"cycle.mp4",cycle)
    report=inspect_source_fidelity(video,paths,fps=15)
    assert report["passed"] and report["source_has_local_motion"]


def test_lossless_reference_encoding_preserves_high_chroma_edges_exactly(tmp_path):
    frames=[]; paths=[]
    for i in range(4):
        rgb=np.zeros((96,160,3),np.uint8)
        rgb[:,::2]=[255,0,0]
        rgb[20:60,20+i:55+i]=[0,0,255]
        p=tmp_path/f"{i}.png"
        cv2.imwrite(str(p),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        paths.append(p);frames.append(rgb)
    video=tmp_path/"lossless.mp4"
    write_h264_rgb(video,frames,15,lossless=True)
    report=inspect_source_fidelity(video,paths,fps=15)
    assert report["passed"]
    assert report["maximum_frame_mse_8bit"] == 0
    assert report["maximum_local_codec_mae_8bit"] == 0


def test_stationary_model_prediction_is_scored_not_filtered_as_invalid_media(tmp_path):
    frames,_=sequence(tmp_path,static=True)
    video=encode(tmp_path/"prediction.mp4",frames)
    output=tmp_path/"model_quality.json"
    certify_wam_output(video,output,expected_width=160,expected_height=96,
                       expected_frame_count=30,expected_fps=15)
    report=json.loads(output.read_text())
    assert report["passed"] and report["policy"]["task_success_certified"] is False
    assert report["video"]["motion_diagnostics"]["legacy_non_static"] is False
    certify_wam_output(video,output,expected_width=160,expected_height=96,
                       expected_frame_count=31,expected_fps=15)
    assert json.loads(output.read_text())["passed"] is False
