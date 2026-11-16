from __future__ import annotations

import cv2
import numpy as np

from autoresearch.media import write_h264_rgb


def test_write_h264_rgb(tmp_path) -> None:
    frames = [
        np.full((64, 96, 3), (index * 30, 80, 180), dtype=np.uint8)
        for index in range(4)
    ]
    output = write_h264_rgb(tmp_path / "video.mp4", frames, 10, crf=16)
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    capture.release()
