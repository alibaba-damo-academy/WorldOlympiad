"""Build a scoring case from a wo reference + a model episode, then score it.

    python -m wo.eval.case --ref REF_DIR --episode EPISODE_DIR --out CASE_DIR [--score]

The case directory follows the registered scorer contract
(scripts/score_robotics_cases.py): worldolympiad_case.json, reference and generated
videos, prompt.json and chunk timestamps. Native observation views retain their actual
camera IDs in the manifest instead of being relabeled as exterior views.
Track applicability is declared up front from the task, never from the score.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ..scene import WORKSPACE
from ..spec import TaskSpec
from .reference_binding import verify_binding, verify_episode_binding, write_binding

R = WORKSPACE / "code/WorldOlympiad-main"
CHUNK = 32
FPS = 15


def _frames(video: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _generated_video(ep: dict, episode: Path, preferred_view: str) -> tuple[Path, str]:
    """Resolve only the exact view required by the baseline scoring contract."""
    capture = ep.get("capture") or (ep.get("native_result") or {}).get("capture") or {}
    videos = ep.get("videos") or capture.get("videos") or {}
    if preferred_view not in videos:
        raise FileNotFoundError(
            f"episode has no generated video for required matching view {preferred_view!r}")
    path = Path(videos[preferred_view])
    if not path.is_absolute():
        path = episode / path
    return path, preferred_view


def build_case(ref: Path, episode: Path, out: Path, *, view: str = "exterior_left") -> Path:
    ep = json.loads((episode / "episode.json").read_text())
    verify_episode_binding(episode, ref)
    cert = json.loads((ref / "certificate.json").read_text())
    spec = TaskSpec.load(Path(cert["task"]))
    from .report import applicability_result
    applicability = applicability_result(ep)
    if applicability:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'score.json').write_text(json.dumps({
            **applicability,
            'physical': None, 'three_d': None, 'interaction': None}, indent=1))
        return out
    if (out / "reference_binding.json").is_file() or (out / "worldolympiad_case.json").is_file():
        verify_binding(episode, ref, case=out)
    model = ep["model"]
    ref_video = ref / "replay" / f"{view}.mp4"
    gen_video, generated_view = _generated_video(ep, episode, view)
    if not ref_video.is_file() or not gen_video.is_file():
        raise FileNotFoundError(
            f"missing scoring videos: reference {view!r}={ref_video}, "
            f"generated {generated_view!r}={gen_video}")
    out.mkdir(parents=True, exist_ok=True)
    cid = spec.case_id
    refname, genname = f"ref_{cid}.mp4", f"{model}_gen_{cid}.mp4"
    shutil.copy2(ref_video, out / refname)
    shutil.copy2(gen_video, out / genname)
    n = _frames(out / genname)
    chunks, prompts = [], []
    for k, start in enumerate(range(0, n, CHUNK)):
        end = min(start + CHUNK, n)
        interval = f"[00:{start / FPS:07.4f}, 00:{end / FPS:07.4f})"
        chunks.append({"chunk_index": k, "source_interval": interval, "frame_start": start,
                       "frame_end": end, "generated_start_sec": start / FPS,
                       "generated_end_sec": end / FPS})
        prompts.append({"chunk_index": k, "interval": interval, "action": spec.instruction,
                        "caption": spec.instruction})
    (out / "prompt.json").write_text(json.dumps(prompts, indent=1))
    (out / f"{Path(genname).stem}_chunk_timestamps.json").write_text(json.dumps(
        {"version": 1, "video_path": genname, "fps": FPS, "total_frames": n,
         "duration_sec": n / FPS, "chunks": chunks}, indent=1))
    (out / "worldolympiad_case.json").write_text(json.dumps({
        "schema_version": "worldolympiad_robotics_case.v2",
        "case_id": cid, "model_id": model, "task_family": spec.family,
        "event_type": "robot_manipulation", "reference_video": refname, "generated_video": genname,
        "reference_view_id": view, "generated_view_id": generated_view,
        "reference_view_count": 1, "generated_view_count": 1,
        "view_contract": "matched_single_physical_view_v2",
        "evaluator_pipeline_prefix": model.split("_")[0], "interaction_mode": "robot_contact",
        "task_success": ep.get("task_success"),
        "tracks": {"physical": "required", "geometry": "required", "interaction": "required",
                   "continuous_action": "pending_symmetric_dream2flow_manifest"},
        "interaction_provenance": {"reference": str(ref), "episode": str(episode)},
    }, indent=1))
    write_binding(ref, out, view)
    return out


def score(case: Path, gpu: int = 0) -> dict:
    existing = case / 'score.json'
    if existing.is_file():
        payload = json.loads(existing.read_text())
        if payload.get('status') in {'not_applicable', 'pending'} and not (case / 'worldolympiad_case.json').is_file():
            return payload
    subprocess.run([str(WORKSPACE / "envs/autoresearch/bin/python"), "-m", "scripts.score_robotics_cases",
                    "--workspace", str(WORKSPACE), "--case-root", str(case), "--gpu-id", str(gpu)],
                   cwd=R, check=True)
    return json.loads((case / "score.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    case = build_case(a.ref, a.episode, a.out)
    print(case)
    if a.score:
        print(json.dumps(score(case), indent=1)[:2000])


if __name__ == "__main__":
    main()
