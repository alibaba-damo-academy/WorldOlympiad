"""Read exact-attempt scene evidence without treating previews as acceptance."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import time


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run and case must be single path components")
    return value


def _tail(path: Path, limit: int = 6000) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode("utf-8", errors="replace")


def _record(path: Path, kind: str) -> dict:
    stat = path.stat()
    return {"path": str(path), "kind": kind, "bytes": stat.st_size,
            "mtime": stat.st_mtime}


def _log_started(path: Path) -> float | None:
    with path.open("rb") as stream:
        header = stream.read(16384).decode("utf-8", errors="replace")
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", header)
    if match:
        return time.mktime(time.strptime(match.group(), "%Y-%m-%d %H:%M:%S"))
    return None


def collect_scene_evidence(workspace: Path, run: str, case: str) -> dict:
    """Return logs and image paths belonging to the latest scene job for a case.

    Queued retries have no current artifact. Earlier published generations and
    concept images remain explicitly separate; callers must view actual meshes
    and native cameras and validate task binding before accepting a scene.
    """
    run, case = _identifier(run), _identifier(case)
    output = Path(workspace) / "outputs"
    database = output / "wo_agent/jobs.sqlite3"
    job = None
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("select * from jobs where kind='scene' order by id desc"):
            args = json.loads(row["args"])
            if args.get("run") == run and args.get("case") == case:
                job = dict(row)
                break
    result = {"run": run, "case": case, "checked_at": time.time(),
              "latest_scene_job": job, "current_attempts": [],
              "published_generation": None, "acceptance": "not_assessed"}
    task_path = output / "wo_runs" / run / "tasks" / f"{case}.json"
    task = json.loads(task_path.read_text()) if task_path.is_file() else {}
    if task.get("generation"):
        result["published_generation"] = {
            "path": task["generation"],
            "belongs_to_latest_job": False,
            "warning": "Published path alone does not establish current task or job binding.",
        }
    if not job or job.get("started") is None:
        return result
    started, finished = job["started"], job.get("finished")
    logs = []
    if job.get("log") and Path(job["log"]).is_file():
        result["job_log_tail"] = _tail(Path(job["log"]))
    batch = f"wo_{case}"[:120]
    root = output / "robotics_scenecode_generations" / batch
    logs.extend(root.glob("*/scene_run/main.log"))
    for folder in ["robotics_scenecode_recoveries", "robotics_scenecode_partial_recoveries"]:
        logs.append(output / folder / batch / case / "scene_run/main.log")
    logs = [p for p in logs if p.is_file() and p.stat().st_mtime >= started - 1
            and (finished is None or p.stat().st_mtime <= finished + 1)]
    # A previous timed-out child can keep appending during a newer retry. Its
    # recent mtime must not bind the old log to the new job.
    logs = [p for p in logs if (origin := _log_started(p)) is not None
            and origin >= started - 2 and (finished is None or origin <= finished + 1)]
    for log in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        scene = log.parent / "scene_000"
        attempt = {"log": _record(log, "scene_log"), "tail": _tail(log),
                   "geometry_renders": [], "concept_images": [], "scene_exports": []}
        # Most recent files from this attempt only. A furniture render still
        # cannot prove that task objects have been constructed or placed.
        physical = sorted(scene.rglob("full_object.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in physical[:8]:
            kind = "manipuland_geometry" if "manipuland" in path.parts else "furniture_geometry"
            attempt["geometry_renders"].append(_record(path, kind))
        concepts = list(scene.rglob("context_edited.png")) + list(scene.rglob("image.png"))
        for path in sorted(concepts, key=lambda p: p.stat().st_mtime, reverse=True)[:4]:
            attempt["concept_images"].append(_record(path, "concept_only"))
        for path in scene.rglob("scene.sdf"):
            attempt["scene_exports"].append(_record(path, "scene_sdf_requires_validation"))
        result["current_attempts"].append(attempt)
    return result
