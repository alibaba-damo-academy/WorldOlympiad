#!/usr/bin/env python3
"""Score a physics reference/imagination object-flow pair on the CAE metric.

Both roles must be Dream2Flow object-flow exports for the same registered
camera. Complete pairs use the full native timelines; partially observed pairs
are scored on the common observed intervals only.
"""
import argparse, json, sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--imagination-dir", required=True)
    parser.add_argument("--fps", type=float, help="Legacy: both roles have the same FPS")
    parser.add_argument("--frames", type=int, help="Legacy: both roles have the same frame count")
    parser.add_argument("--reference-fps", type=float)
    parser.add_argument("--reference-frames", type=int)
    parser.add_argument("--imagination-fps", type=float)
    parser.add_argument("--imagination-frames", type=int)
    parser.add_argument("--reference-video-sha256", required=True)
    parser.add_argument("--imagination-video-sha256", required=True)
    parser.add_argument("--tracked-entity", default="primary task object")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference_fps = args.reference_fps if args.reference_fps is not None else args.fps
    imagination_fps = args.imagination_fps if args.imagination_fps is not None else args.fps
    reference_frames = args.reference_frames if args.reference_frames is not None else args.frames
    imagination_frames = args.imagination_frames if args.imagination_frames is not None else args.frames
    if any(value is None for value in (reference_fps, imagination_fps,
                                       reference_frames, imagination_frames)):
        parser.error("provide native FPS and frame count for each role")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from continuous_action.physics_partial import score_partial_pair
    pair = {"tracked_entity": args.tracked_entity, "roles": {
        "reference": {"dir": str(Path(args.reference_dir).resolve()), "fps": reference_fps,
                      "frames": reference_frames, "video_sha256": args.reference_video_sha256},
        "imagination": {"dir": str(Path(args.imagination_dir).resolve()), "fps": imagination_fps,
                        "frames": imagination_frames, "video_sha256": args.imagination_video_sha256}}}
    result, _ = score_partial_pair(pair)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("continuous_score", "path_score", "velocity_score", "partial_observation",
                       "reference_coverage", "prediction_coverage") if key in result}))


if __name__ == "__main__":
    main()
