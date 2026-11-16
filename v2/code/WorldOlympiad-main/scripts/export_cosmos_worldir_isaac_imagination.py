#!/usr/bin/env python3
"""Export a task-terminated Cosmos DROID rollout from closed-loop windows.

Each 33-frame model window contains the decoded conditioning frame followed by
32 imagined frames.  The formal rollout keeps 32 decoded frames per window
(including the refreshed conditioning frame). The requested count is the
actual simulator-terminated rollout length, never the reference-video length.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from autoresearch.cosmos3_droid_window import restore_decoded_droid_panels
from autoresearch.io import sha256_file, write_json
from autoresearch.cosmos_window_timeline import publication_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def write_h264(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {path}")


def _owner_relative(path: Path, owner: Path) -> str:
    """Return a relocation-safe path owned by ``owner``.

    The export directory and capture directory are siblings in the formal WAM
    bundle.  ``Path.relative_to`` cannot express that relationship, whereas a
    normalized relative path remains valid when the complete bundle moves.
    """

    value = Path(os.path.relpath(path.resolve(), owner.resolve())).as_posix()
    if value.startswith("/") or "\\" in value:
        raise ValueError(f"could not encode owner-relative path: {path}")
    return value


def main() -> None:
    args = parse_args()
    capture_root = args.capture_root.resolve()
    output_dir = args.output_dir.resolve()
    window_paths = sorted(capture_root.glob("windows/window_*/model/droid_concat_rollout.npy"))
    if not window_paths:
        raise FileNotFoundError(f"No decoded Cosmos windows under {capture_root}")

    lengths = None
    progress_path = capture_root / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        records = progress.get("windows", [])
        if records and all("conditioning_frame" in item and "executed_count" in item for item in records):
            if len(records) != len(window_paths):
                raise ValueError("Raw model windows differ from recorded feedback windows")
            lengths = publication_counts(records, args.frame_count)

    chunks: list[np.ndarray] = []
    stitch_records: list[dict[str, object]] = []
    cursor = 0
    for window_index, path in enumerate(window_paths):
        if cursor >= args.frame_count:
            break
        video = np.load(path, allow_pickle=False)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Unexpected video shape at {path}: {video.shape}")
        if video.dtype != np.uint8:
            video = np.clip(video, 0, 255).astype(np.uint8)
        count = lengths[window_index] if lengths is not None else min(
            32 + int(window_index == len(window_paths) - 1), args.frame_count - cursor, video.shape[0]
        )
        if count > video.shape[0]:
            raise ValueError("Decoded window lacks the recorded terminal frame")
        chunks.append(video[:count])
        stitch_records.append(
            {
                "window_index": window_index,
                "source": _owner_relative(path, output_dir),
                "source_sha256": sha256_file(path),
                "source_shape": list(video.shape),
                "source_frame_range": [0, count - 1],
                "output_frame_range": [cursor, cursor + count - 1],
            }
        )
        cursor += count

    if cursor != args.frame_count:
        raise RuntimeError(f"Stitched {cursor} frames, expected {args.frame_count}")
    concat = np.concatenate(chunks, axis=0)
    height, width = concat.shape[1:3]
    wrist, left_native, right_native, panel_restoration = (
        restore_decoded_droid_panels(concat)
    )
    left_presentation = np.stack(
        [cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) for frame in left_native]
    )
    right_presentation = np.stack(
        [cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) for frame in right_native]
    )
    wrist_presentation = wrist

    outputs = {
        "raw_concat": output_dir / "raw_concat" / "imagination.mp4",
        "wrist": output_dir / "wrist" / "imagination_640x360.mp4",
        "exterior_left_native": output_dir / "exterior_left" / "imagination_native_320x180.mp4",
        "exterior_left_presentation": output_dir / "exterior_left" / "imagination_640x360.mp4",
        "exterior_right_native": output_dir / "exterior_right" / "imagination_native_320x180.mp4",
        "exterior_right_presentation": output_dir / "exterior_right" / "imagination_640x360.mp4",
    }
    write_h264(outputs["raw_concat"], concat, args.fps)
    write_h264(outputs["wrist"], wrist_presentation, args.fps)
    write_h264(outputs["exterior_left_native"], left_native, args.fps)
    write_h264(outputs["exterior_left_presentation"], left_presentation, args.fps)
    write_h264(outputs["exterior_right_native"], right_native, args.fps)
    write_h264(outputs["exterior_right_presentation"], right_presentation, args.fps)

    metadata = {
        "schema_version": "cosmos_worldir_isaac_export.v2",
        "status": "complete",
        "capture_root": _owner_relative(capture_root, output_dir),
        "frame_count": args.frame_count,
        "fps": args.fps,
        "duration_seconds": args.frame_count / args.fps,
        "window_count": len(stitch_records),
        "stitch_policy": "keep raw predicted frames until the next recorded feedback time; retain final terminal frame; no blending",
        "panel_layout": {
            "concat_shape": [height, width],
            "wrist": [0, 360, 0, width],
            "exterior_left_decoded": [360, height, 0, 320],
            "exterior_right_decoded": [360, height, 320, 640],
            "restoration": panel_restoration,
        },
        "resize_policy": (
            "preserve decoded pixels; restore Cosmos bottom-cropped exterior rows "
            "with bottom padding; exact 2x presentation resize"
        ),
        "outputs": {
            key: _owner_relative(value, output_dir) for key, value in outputs.items()
        },
        "output_sha256": {key: sha256_file(value) for key, value in outputs.items()},
        "windows": stitch_records,
    }
    capture_metadata = capture_root / "capture_metadata.json"
    if capture_metadata.is_file():
        metadata["capture_metadata"] = {
            "path": _owner_relative(capture_metadata, output_dir),
            "sha256": sha256_file(capture_metadata),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "export_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
