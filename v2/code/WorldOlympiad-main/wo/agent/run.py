"""Autonomous benchmark agent: rounds of (observe -> decide -> act), jobs run in between.

    envs/scenecode/bin/python -m wo.agent.run --run RUN --goal GOAL.md [--rounds 200]

Every round is a fresh agent session whose input is: the goal, the knowledge
files, the persistent notes and the current status/queue.  No transcript grows
across rounds.  Between rounds the supervisor waits while jobs are running.
"""

from __future__ import annotations

import argparse
import hashlib
import asyncio
import json
import os
from itertools import count
import re
import sys
import time
from pathlib import Path

from ..scene import WORKSPACE
from . import jobs as J
from . import tools as T

KNOW = Path(__file__).parent / "knowledge"
LOG = T.AGENT_DIR / "rounds"
MODEL = os.environ.get("WO_AGENT_MODEL", "gpt-5.6-sol")
EVAL_ONLY = False

INSTRUCTIONS = """You are the WorldOlympiad auto-research benchmark agent. You build and run a robot
manipulation benchmark end to end: author tasks, generate SceneCode scenes, produce expert references,
run native model episodes, score them, research failures, and improve the shared pipeline code.

How you work:
- Pipeline work only through `submit` with registered stages (scene, reference, expert_check, scene_check, policy_check, episode, score, report, showcase).
  Stages are long jobs; queue many at once (use `after` for dependencies), then end the round. You will
  be called again with fresh status once jobs progress. Never wait or poll inside a round.
- Failures: read the job error and log, find the root cause, fix the GENERAL capability (never a per-case
  hack or tuned constant for one sample). Code changes: edit_file in the agent worktree, add/extend a test
  under wo/tests, run_tests, then promote. Then resubmit the failed stage.
- Use scene_check for shared RoboTwin scene import/framing diagnostics without policy inference;
  it takes run/case and preserves previous diagnostics. A pass verifies initialization, not policy success.
- Engineering for new models (adapters, envs, downloads): use `shell` in the worktree; long installs or
  downloads go to the background with nohup and a log under outputs/wo_agent/logs; check later.
  New adapters live in wo/models/<name>.py with the same CLI as wo/models/cosmos.py and must be
  registered in wo/agent/stages.py MODELS (promote). Run a positive control first.
- Start scene diagnosis with `inspect_scene(run, case)` to bind current logs and renders to the latest job; queued retries have no current artifact. View returned task-object geometry, distinguish furniture and concept images, and reuse unchanged observations instead of repeatedly viewing the same partial.
- Inspect actual asset/native-camera images with `view_image`; for videos use `shell`/ffmpeg to extract timestamped frames. Record visible defects and physical evidence; a pretty concept image is not an executed task.
- Record every finding, decision and open problem with `note` (they persist; this round's memory is lost).
- Be honest: a model failing a task is a valid result to score; an infrastructure bug is not.
- Keep the GPU busy but respect the job queue; do not start GPU work outside jobs except short smoke tests.
- Distinguish a candidate patch, passing unit tests, and a repair verified by its real registered stage. Do not say a task is repaired while its fresh validation is queued or failed.
- End the round with a short summary: what you queued, what you learned, what is next.
- For the standard benchmark goal, end only with GOAL COMPLETE after its report and coverage audit. If the supplied goal defines a different completion marker and a narrower endpoint, verify that endpoint and finish with that exact marker on the final line.

REFERENCE VALIDATION SCOPE: keep repairs small. For an expert/scene fix, run the failed task and at most one closely related previously successful control, plus focused tests. Choose a control that executes the changed path: scene_check for native loader/layout/camera changes, expert_check for expert/planning changes. Preserve its reference. Reuse live jobs and preserve predicates, collision checks, declared initial relations and historical evidence. All references declared in the current run must be ready before its model episodes; bench_v1 requires its 24 references, while an isolated pilot follows its own declared scope. Do not rerun every reference or the full test suite for an unrelated change.

TASK AUTHORING: follow knowledge/benchmark.md for household assets, graspable parts, direction clearance and native-camera visibility. Put authoring facts in existing descriptions/geometry_constraints; keep model instructions free of privileged grasp coordinates. Calibrate basic-task difficulty before evaluation, never filter a frozen evaluation set by an evaluated model's success. Preserve working execution paths and use separate runs for single-factor variants.
"""

EVAL_INSTRUCTIONS = """You run a user-requested WorldOlympiad evaluation. Follow the supplied goal and use only
the configured tested-model adapter. Use registered pipeline stages through submit, inspect their results,
and report model failures separately from infrastructure failures. Do not edit source code, change benchmark
prompts or scoring policy, tune thresholds, install packages, download weights, or inspect secret values.
Preserve every task, episode, score, and partial artifact. Use a fresh run id for a new evaluation.
Submit independent ready stages with their declared dependencies and end each round while jobs run.
Do not start stages outside the supplied goal's scope. Verify the requested endpoint before using its
completion marker; a reference-only goal does not authorize episode or scoring jobs."""


def _read(p: Path, limit: int = 20000) -> str:
    if not p.is_file():
        return ""
    t = p.read_text(errors="replace")
    return t if len(t) <= limit else "...(older notes cut)...\n" + t[-limit:]


def round_input(run: str, goal: Path) -> str:
    know = "\n\n".join(f"### {p.name}\n{_read(p)}" for p in sorted(KNOW.glob("*.md")))
    return (f"# GOAL\n{goal.read_text()}\n\n# KNOWLEDGE\n{know}\n\n# YOUR NOTES (persistent)\n"
            f"{_read(T.NOTES) or '(none yet)'}\n\n# STATUS\n{T.status(run)}\n\n# RECENT FAILED JOBS\n"
            f"{T.jobs('failed', 8)}\n\n# RUNNING/QUEUED\n{T.jobs('running', 10)}\n{T.jobs('queued', 15)}\n\n"
            f"Run name for this goal: {run}. Time now: {time.strftime('%Y-%m-%d %H:%M')}.")



def _trace_text(value, limit=4000):
    text = str(value)
    text = re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[image omitted]", text)
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._-]+", "Bearer [redacted]", text)
    text = re.sub(r"\b(?:sk-|ghp_)[A-Za-z0-9_-]{16,}", "[redacted]", text)
    if len(text) > limit:
        return text[:limit // 2] + "\n...[trace shortened]...\n" + text[-limit // 2:]
    return text


def round_actions(items):
    """Retain actual mutation/test/job returns, not only the model's summary.

    Record bounded metadata; image pixels and edit/task bodies remain in their
    original artifacts rather than being duplicated in the round log.
    """
    selected = {"edit_file", "author_task", "run_tests", "promote", "submit", "cancel", "view_image"}
    fields = {"run", "case", "model", "seed", "kind", "after", "priority", "path", "paths",
              "test_paths", "message", "job_id"}
    def get(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    actions, outputs = [], {}
    for item in items:
        raw = get(item, "raw_item", {})
        if get(item, "type") == "tool_call_output_item":
            outputs[get(raw, "call_id")] = get(raw, "output", get(item, "output"))
        if get(item, "type") != "tool_call_item" or get(raw, "name") not in selected:
            continue
        name = get(raw, "name")
        try:
            args = json.loads(get(raw, "arguments", "{}"))
            if not isinstance(args, dict):
                args = {}
        except (TypeError, ValueError):
            args = {}
        meta = {k: _trace_text(v, 1000) if isinstance(v, str) else v
                for k, v in args.items() if k in fields and isinstance(v, (str, int, float, bool, type(None)))}
        if name == "submit":
            try:
                scope = json.loads(args.get("args_json", "{}"))
                meta["job_args"] = {k: _trace_text(v, 1000) if isinstance(v, str) else v
                                    for k, v in scope.items() if k in {"run", "case", "model", "seed", "layout"}
                                    and isinstance(v, (str, int, float, bool, type(None)))}
            except (TypeError, ValueError, AttributeError):
                meta["job_args_unavailable"] = True
        for key in ("old", "new", "task_spec_json"):
            if key in args:
                data = str(args[key]).encode()
                meta[key + "_sha256"] = hashlib.sha256(data).hexdigest()
                meta[key + "_bytes"] = len(data)
        actions.append({"tool": name, "call_id": get(raw, "call_id"), "arguments": meta})
    for action in actions:
        if action["call_id"] not in outputs:
            continue
        output = outputs[action["call_id"]]
        if action["tool"] == "view_image" and not isinstance(output, str):
            action["output"] = {"image_payload_omitted": True}
        else:
            action["output"] = _trace_text(output)
    return actions


def refresh_tools(schema_module=None, tool_module=None):
    """Rebuild tool functions from promoted code at each round boundary.

    Function-tool wrappers capture Python function objects. A fresh Agent alone
    cannot see edited tool implementations or types imported by the old module.
    """
    import importlib

    if schema_module is None:
        from .. import spec as schema_module
    tool_module = T if tool_module is None else tool_module
    importlib.invalidate_caches()
    importlib.reload(schema_module)
    return importlib.reload(tool_module)


async def one_round(run: str, goal: Path, n: int, *, eval_only: bool = False) -> str:
    sys.path.insert(0, os.environ.get("SCENECODE_RUNTIME_ROOT",
                                     str(WORKSPACE / "runtime/scenecode")))
    from agents import Agent, Runner, function_tool
    from scenecode.utils.openai import create_run_config

    current_tools = refresh_tools()
    engineering = {"shell", "edit_file", "run_tests", "promote"}
    functions = [f for f in current_tools.ALL if not eval_only or f.__name__ not in engineering]
    agent = Agent(name="wo-benchmark-agent",
                  instructions=EVAL_INSTRUCTIONS if eval_only else INSTRUCTIONS, model=MODEL,
                  tools=[function_tool(f) for f in functions])
    cfg = create_run_config(api_base=os.environ.get("WO_AGENT_API_BASE"))
    result = await Runner.run(agent, round_input(run, goal), max_turns=60, run_config=cfg)
    out = str(result.final_output)
    LOG.mkdir(parents=True, exist_ok=True)
    calls = [getattr(i.raw_item, "name", "") for i in result.new_items
             if getattr(i, "type", "") == "tool_call_item"]
    (LOG / f"{n:04d}.json").write_text(json.dumps({
        "schema_version": "wo_agent_round.v2", "round": n, "run": run,
        "goal_path": str(goal), "time": time.time(), "calls": calls,
        "actions": round_actions(result.new_items), "final": out}, indent=1))
    return out


def busy() -> bool:
    con = J.connect()
    return bool(con.execute("select count(*) from jobs where status in ('queued','running')").fetchone()[0])


def completion_verified(summary: str, run: str) -> bool:
    """An explicit final marker plus the existing completed report audit ends the loop."""
    lines = summary.strip().splitlines()
    if not lines or not re.fullmatch(r"(?:\*\*)?GOAL COMPLETE(?:\*\*)?[.!]?", lines[-1].strip()):
        return False
    try:
        latest = next((status for args, status in J.connect().execute(
            "select args,status from jobs where kind='report' order by id desc")
            if json.loads(args).get("run") == run), None)
        if latest != "done":
            return False
        root = T.run_dir(run)
        document = json.loads((root / "results.json").read_text())
        if document.get("coverage", {}).get("matrix_complete") is not True:
            return False
        # Reuse the same current-manifest/reference audit as the report. This
        # also rejects an old table when required rows or references changed.
        from ..eval.report import coverage
        return coverage(root, document["episodes"]).get("matrix_complete") is True
    except Exception as exc:
        print(f"Completion audit unavailable ({type(exc).__name__}); continuing.", flush=True)
        return False


def round_numbers(start: int, limit: int):
    """Zero runs until verified completion; a positive limit is an explicit cap."""
    if limit < 0:
        raise ValueError("round limit must be nonnegative")
    return count(start) if limit == 0 else range(start, start + limit)


def load_agent_environment(path=None):
    """Load private legacy values and map the two-interface user profile."""
    from .config import apply_environment, load_private_environment

    path = load_private_environment(os.environ, path)
    apply_environment(os.environ, phase=os.environ.get("WO_RESOURCE_PHASE", "evaluation"))
    return path


def main() -> None:
    global MODEL, EVAL_ONLY
    load_agent_environment()
    MODEL = os.environ.get("WO_AGENT_MODEL", "gpt-5.6-sol")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--goal", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=0, help="0: continue until verified completion; positive: explicit round limit")
    ap.add_argument("--min-gap-s", type=int, default=600, help="max wait between rounds while jobs run")
    ap.add_argument("--completion-marker", help="stop after the goal's final output ends with this marker")
    ap.add_argument("--no-engineering", action="store_true",
                    help="evaluation mode: remove shell, code-edit, test, and promotion tools")
    a = ap.parse_args()
    EVAL_ONLY = a.no_engineering
    # resolve before any worktree/cwd change: a relative goal path otherwise
    # breaks every later round (the file is re-read each round)
    a.goal = a.goal.resolve()
    if not EVAL_ONLY:
        T.ensure_worktree()
    start = len(list(LOG.glob("*.json"))) if LOG.is_dir() else 0
    marker_reached = False
    for n in round_numbers(start, a.rounds):
        try:
            out = asyncio.run(one_round(a.run, a.goal, n, eval_only=EVAL_ONLY))
        except Exception as exc:  # a failed round is logged, never fatal
            out = f"ROUND ERROR: {type(exc).__name__}: {exc}"
            LOG.mkdir(parents=True, exist_ok=True)
            (LOG / f"{n:04d}.json").write_text(json.dumps({"round": n, "error": out}))
        print(f"[{time.strftime('%H:%M')}] round {n}: {out[:500]}", flush=True)
        if a.completion_marker and out.strip().splitlines()[-1:] == [a.completion_marker]:
            marker_reached = True
            break
        if completion_verified(out, a.run):
            break
        t0 = time.time()  # let jobs progress before the next look
        while busy() and time.time() - t0 < a.min_gap_s:
            time.sleep(30)
        if not busy():
            time.sleep(20)
    if a.completion_marker and not marker_reached:
        raise SystemExit(f"goal did not produce completion marker {a.completion_marker!r}")


if __name__ == "__main__":
    main()
