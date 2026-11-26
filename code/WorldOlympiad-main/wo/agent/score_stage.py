"""Fresh, retry-safe score stage with full or VRAM-split track execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ..paths import WORKSPACE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_receipt(case: Path) -> bool:
    try:
        receipt = json.loads((case / "three_track_receipt.json").read_text())
        record = json.loads((case / "worldolympiad_case.json").read_text())
        video = case / record["generated_video"]
        reference = case / record["reference_video"]
        chunks = case / (video.stem + "_chunk_timestamps.json")
        inputs = {str(p.relative_to(case)): _sha(p)
                  for p in (video, reference, case / "prompt.json", chunks)}
        return (receipt["inputs"] == inputs
                and receipt["score_sha256"] == _sha(case / "score.json"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def archive_case(case: Path) -> Path | None:
    if not case.exists():
        return None
    history = case.parent / ".score_history"
    history.mkdir(parents=True, exist_ok=True)
    target = history / f"{case.name}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    shutil.move(str(case), str(target))
    return target


def run_stage(ref: Path, episode: Path, out: Path, workspace: Path,
              gpu_id: str = "0", compute_gpu_id: str | None = None,
              score_mode: str = "full") -> dict:
    from ..eval.case import build_case

    if score_mode not in {"full", "split"}:
        raise ValueError("score_mode must be full or split")
    compute_gpu_id = str(gpu_id if compute_gpu_id is None else compute_gpu_id)
    if valid_receipt(out):
        return json.loads((out / "score.json").read_text())
    archive_case(out)
    out = build_case(ref, episode, out)
    if not (out / "worldolympiad_case.json").is_file():
        return json.loads((out / "score.json").read_text())
    command = [sys.executable, "-m", "scripts.score_robotics_cases",
               "--workspace", str(workspace), "--gpu-id", str(gpu_id),
               "--compute-gpu-id", compute_gpu_id,
               "--score-mode", score_mode, "--case-root", str(out)]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    if not valid_receipt(out):
        raise RuntimeError("scoring process exited without a valid three-track receipt")
    return json.loads((out / "score.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--gpu-id", default="0", help="physical GPU index for the VLM service")
    parser.add_argument("--compute-gpu-id", help="physical GPU index for SAM/DA/CLIP work")
    parser.add_argument("--score-mode", choices=("full", "split"), default="full")
    args = parser.parse_args()
    score = run_stage(args.ref, args.episode, args.out, args.workspace,
                      args.gpu_id, args.compute_gpu_id, args.score_mode)
    print(json.dumps({"score": score.get("combined_score"),
                      "worldolympiad2_score": score.get("worldolympiad2_score"),
                      "case": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
