"""Encode a complete measured replay without missing frames or time rescaling."""
from __future__ import annotations
import json
from pathlib import Path
from .io import sha256_file, write_json
from .media import write_h264_rgb


def encode_views(output: Path, *, frame_count: int, trajectory: Path) -> dict:
    output = Path(output)
    data = json.loads(Path(trajectory).read_text())
    if (isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 2
            or data.get("fps") != 15 or len(data["frames"]) != frame_count
            or any(frame["frame"] != i or abs(frame["time_s"]-i/15)>1e-8
                   for i, frame in enumerate(data["frames"]))):
        raise ValueError("reference video requires complete consecutive measured 15 Hz states")
    paths = {view: [output/"observations"/f"step_{i:06d}_{short}_image.png"
                   for i in range(frame_count)]
             for view, short in (("wrist", "wrist"), ("exterior_left", "left"), ("exterior_right", "right"))}
    for files in paths.values():
        if any(not file.is_file() for file in files):
            raise ValueError("native reference capture lacks a measured frame; no padding is allowed")
    receipt = {"schema_version":"native_reference_rgb_capture.v1", "control_hz":15,
               "frame_count":frame_count, "physical_duration_s":(frame_count-1)/15,
               "trajectory_sha256":sha256_file(trajectory), "time_rescaled":False,
               "interpolated_frames":False, "formal_reference_published":False, "views":{}}
    for view, files in paths.items():
        import cv2
        video = output/f"reference_native_{view}.mp4"
        write_h264_rgb(video, (cv2.cvtColor(cv2.imread(str(file)),cv2.COLOR_BGR2RGB)
                              for file in files), fps=15, crf=18)
        capture = cv2.VideoCapture(str(video))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            width, height = (int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            decoded = 0
            while capture.read()[0]: decoded += 1
        finally:
            capture.release()
        if decoded != frame_count or abs(fps-15)>1e-8:
            raise ValueError("encoded reference changed the measured frame/time window")
        receipt["views"][view] = {"path":video.name, "sha256":sha256_file(video),
            "width":width, "height":height, "decoded_frame_count":decoded, "decoded_fps":fps,
            "source_frame_sha256":[sha256_file(file) for file in files]}
    write_json(output/"reference_capture_receipt.json",receipt)
    return receipt
