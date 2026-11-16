import json
import pytest
from autoresearch.native_reference_capture import encode_views


def test_real_encoding_preserves_measured_frame_count_and_clock(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("cv2")
    import numpy as np
    from PIL import Image
    folder = tmp_path/"observations"; folder.mkdir()
    trajectory = tmp_path/"trajectory.json"
    trajectory.write_text(json.dumps({"fps":15, "frames":[
        {"frame":i,"time_s":i/15} for i in range(4)]}))
    for i in range(4):
        for view in ("wrist","left","right"):
            Image.fromarray(np.full((24,32,3),i*60,dtype=np.uint8)).save(folder/f"step_{i:06d}_{view}_image.png")
    receipt = encode_views(tmp_path,frame_count=4,trajectory=trajectory)
    assert receipt["frame_count"] == 4 and receipt["physical_duration_s"] == .2
    assert all(row["width"] == 32 and len(row["source_frame_sha256"]) == 4
               for row in receipt["views"].values())
    (folder/"step_000002_right_image.png").unlink()
    with pytest.raises(ValueError,match="lacks a measured frame"):
        encode_views(tmp_path,frame_count=4,trajectory=trajectory)


def test_capture_rejects_retimed_states(tmp_path):
    trajectory = tmp_path/"trajectory.json"
    trajectory.write_text(json.dumps({"fps":15,"frames":[
        {"frame":0,"time_s":0},{"frame":1,"time_s":.2}]}))
    with pytest.raises(ValueError,match="consecutive measured"):
        encode_views(tmp_path,frame_count=2,trajectory=trajectory)
