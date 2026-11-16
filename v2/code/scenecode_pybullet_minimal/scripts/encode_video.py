#!/usr/bin/env python3
"""Encode and verify sequential SceneCode render frames with OpenCV."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2


FRAME_PATTERN = re.compile(r"frame_(\d{5})\.png$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    indexed = []
    for path in args.frames_dir.glob("frame_*.png"):
        match = FRAME_PATTERN.search(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed:
        raise RuntimeError("no sequential render frames found")
    expected_indices = list(range(indexed[-1][0] + 1))
    actual_indices = [index for index, _ in indexed]
    if actual_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(actual_indices))
        raise RuntimeError(f"frame sequence is incomplete; missing={missing[:20]}")

    first = cv2.imread(str(indexed[0][1]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot read {indexed[0][1]}")
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 writer")
    try:
        for _, path in indexed:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"invalid frame geometry: {path}")
            writer.write(image)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(args.output))
    encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    encoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    encoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if (
        encoded_frames != len(indexed)
        or abs(encoded_fps - args.fps) > 0.01
        or encoded_width != width
        or encoded_height != height
    ):
        raise RuntimeError(
            "encoded video verification failed: "
            f"frames={encoded_frames}, fps={encoded_fps}, "
            f"size={encoded_width}x{encoded_height}"
        )

    manifest = {
        "video": str(args.output.resolve()),
        "codec": "mp4v",
        "frame_count": encoded_frames,
        "fps": encoded_fps,
        "duration_s": encoded_frames / encoded_fps,
        "width": encoded_width,
        "height": encoded_height,
        "size_bytes": args.output.stat().st_size,
        "sha256": _sha256(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
