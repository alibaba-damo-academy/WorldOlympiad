"""Consistent high-quality video encoding for reference and WAM artifacts."""

from __future__ import annotations

import subprocess
from itertools import chain
from pathlib import Path
from typing import Iterable

import numpy as np


def write_h264_rgb(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float,
    *,
    crf: int = 16,
    lossless: bool = False,
) -> Path:
    """Stream RGB frames into a deterministic high-quality H.264 MP4."""

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is required for benchmark MP4 encoding") from exc
    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator), dtype=np.uint8)
    except StopIteration as exc:
        raise ValueError("cannot encode an empty video") from exc
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape HxWx3, got {first.shape}")
    height, width = first.shape[:2]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264rgb" if lossless else "libx264",
        "-preset",
        "medium",
        "-crf",
        "0" if lossless else str(crf),
        "-pix_fmt",
        "rgb24" if lossless else "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in chain((first,), iterator):
            value = np.asarray(frame, dtype=np.uint8)
            if value.shape != first.shape:
                raise ValueError("video frames must have one fixed RGB shape")
            process.stdin.write(np.ascontiguousarray(value).tobytes())
    except BaseException:
        process.stdin.close()
        process.kill()
        process.wait()
        raise
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"H.264 encoding failed with exit code {return_code}: {stderr[-2000:]}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"H.264 encoder did not produce a video: {destination}")
    return destination
