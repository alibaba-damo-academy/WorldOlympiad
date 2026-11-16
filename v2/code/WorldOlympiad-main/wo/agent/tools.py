"""The agent's tools: few, typed, and safe to call repeatedly.

status / jobs / read / list_dir / view_image   observe
author_task / submit / cancel     act on the pipeline (registered stages only)
shell                             research and engineering in the agent worktree
edit_file / run_tests / promote   change shared code on a branch; promote only when tests pass
note / declare_done               persistent research notes; finish
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from ..scene import WORKSPACE
from ..spec import TaskSpec
from . import jobs as J
from .config import load_profile
from .stages import RUNS, STAGES, bind_scene_source_profile, run_dir

MAIN = WORKSPACE / "code/WorldOlympiad-main"
WORKTREE = WORKSPACE / "code/wo-agent-worktree"
BRANCH = "agent/dev"
AGENT_DIR = WORKSPACE / "outputs/wo_agent"
NOTES = AGENT_DIR / "notes.md"
DENY = re.compile(r"(\brm\s+-[a-z]*r[a-z]*\s+(/|~|\$W|/root)(\s|$))|git\s+push|pkill|killall|"
                  r"\bkill\s|shutdown|reboot|mkfs|>\s*/dev/sd|outputs/wo_runs/[^ ]*\s*$.*rm|"
                  r"\.secrets|scenecode\.env")


def _clip(text: str, n: int = 6000) -> str:
    return text if len(text) <= n else text[: n // 2] + f"\n...[{len(text) - n} chars cut]...\n" + text[-n // 2:]


def _inside(path: str) -> Path:
    p = Path(path)
    p = (p if p.is_absolute() else WORKSPACE / p).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError("path must be inside the workspace")
    if ".secrets" in p.parts:
        raise ValueError("secrets are not readable")
    return p



def registered_models() -> list[str]:
    """Model names the episode stage accepts *now* - re-read so a long-running
    agent process sees adapters promoted after it started."""
    import importlib

    from . import stages as stages_mod

    return sorted(importlib.reload(stages_mod).MODELS)


# ------------------------------------------------------------------ observe
def status(run: str) -> str:
    """Per-case pipeline status of a run plus queue summary (compact text table)."""
    root = run_dir(run)
    lines = [f"run {run}  ({root})"]
    con = J.connect()
    by_case: dict[str, list[dict]] = {}
    for row in con.execute("select id,kind,args,status,error from jobs order by id"):
        a = json.loads(row["args"])
        if a.get("run") == run:
            by_case.setdefault(a.get("case", "-"), []).append(dict(row) | {"a": a})
    cases = sorted({p.stem for p in (root / "tasks").glob("*.json")} | set(by_case) - {"-"})
    for c in cases:
        latest: dict[str, dict] = {}
        for j in by_case.get(c, []):
            key = j["kind"] + (f":{j['a'].get('model')}_s{j['a'].get('seed', 0)}" if j["kind"] in ("episode", "score") else "")
            latest[key] = j
        cells = []
        for key, j in sorted(latest.items()):
            mark = {"done": "ok", "failed": "FAIL", "running": "run", "queued": "q", "cancelled": "x"}[j["status"]]
            cells.append(f"{key}={mark}#{j['id']}")
        lines.append(f"  {c}: " + (" ".join(cells) or "no jobs"))
    counts = {s: n for s, n in con.execute("select status,count(*) from jobs group by status")}
    lines.append(f"queue: {counts}")
    res = root / "results.md"
    if res.is_file():
        lines.append("results.md exists (read it for the table)")
    lines.append("episode-stage model adapters registered right now: " + ", ".join(registered_models()))
    return "\n".join(lines)


def jobs(status_filter: str = "", limit: int = 20) -> str:
    """Recent jobs (optionally only 'failed'/'running'/...), with error tails for failures."""
    out = []
    for j in J.listing(status_filter or None, limit):
        e = (j.get("error") or "").splitlines()
        out.append(f"#{j['id']} {j['kind']} {j['status']} args={j['args']}"
                   + (f"\n   error: {' | '.join(e[:1] + e[-6:])}" if e else "")
                   + (f"\n   log: {j['log']}" if j.get("log") else ""))
    return _clip("\n".join(out) or "no jobs")


def read(path: str, start: int = 1, lines: int = 200) -> str:
    """Read a text file inside the workspace (line-numbered window)."""
    p = _inside(path)
    text = p.read_text(errors="replace").splitlines()
    sel = text[max(0, start - 1): start - 1 + lines]
    return _clip("\n".join(f"{start + i}: {l}" for i, l in enumerate(sel)) + f"\n[{len(text)} lines total]", 12000)



def view_image(path: str) -> list[dict]:
    """See a local PNG/JPEG/WebP image, including generated assets and video frames.

    For video inspection, extract a few timestamped frames with ffmpeg via shell,
    then view the frames. Visual inspection does not replace native success checks.
    """
    import base64
    from PIL import Image

    p = _inside(path)
    if p.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("image exceeds 8 MiB; save a smaller inspection frame")
    with Image.open(p) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("view_image supports PNG, JPEG and WebP")
        mime = Image.MIME[image.format]
        image.verify()
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return [{"type": "text", "text": f"Image: {p}"},
            {"type": "image", "image_url": f"data:{mime};base64,{data}", "detail": "high"}]



def inspect_scene(run: str, case: str) -> str:
    """Inspect the latest scene job's exact-attempt logs and labeled image paths.

    View returned geometry_renders with view_image. Furniture and concept images
    do not establish task-object/native-camera quality or reference acceptance.
    Queued retries deliberately return no current artifacts; old published paths
    require independent task binding before reuse.
    """
    from .scene_evidence import collect_scene_evidence

    return json.dumps(collect_scene_evidence(WORKSPACE, run, case), ensure_ascii=False)


def list_dir(path: str) -> str:
    p = _inside(path)
    items = sorted(p.iterdir())[:300]
    return "\n".join(f"{'d' if i.is_dir() else 'f'} {i.name}" for i in items) or "(empty)"


# ------------------------------------------------------------------ act
def _initialize_authored_scope(manifest: Path, tasks_dir: Path) -> None:
    """Complete an unscoped run manifest when its first task is authored.

    Explicit case scopes are frozen. A manifest without cases cannot support a
    coverage audit, so initialize it from the active task directory. When the
    launcher supplied its all-adapters default, narrow that unscoped default to
    the configured tested model for the single-interface evaluation run.
    """
    declared = (json.loads(manifest.read_text()) if manifest.exists() else
                {"schema_version": "wo_run_manifest.v1", "models": registered_models(), "seeds": [0]})
    if declared.get("cases"):
        return
    cases = sorted(path.stem for path in tasks_dir.glob("*.json") if path.is_file())
    if not cases:
        return
    declared["cases"] = cases
    profile = load_profile()
    if profile:
        adapter = profile["tested_model"]["adapter"]
        if not declared.get("models") or declared.get("models") == registered_models():
            declared["models"] = [adapter]
    declared.setdefault("models", registered_models())
    declared.setdefault("seeds", [0])
    declared.setdefault("schema_version", "wo_run_manifest.v1")
    manifest.write_text(json.dumps(declared, indent=1))


def author_task(run: str, task_spec_json: str) -> str:
    """Validate and save a wo_task_spec.v1 into RUN/tasks/<case_id>.json."""
    # The tool server is long-lived: imported schema classes can predate a
    # promotion. Validate in a fresh interpreter against promoted main, rather
    # than mixing cached TaskSpec/TaskObject classes with current source.
    import sys

    payload = json.loads(task_spec_json)
    validation = subprocess.run(
        [sys.executable, "-c",
         "import json,sys; from wo.spec import TaskSpec; "
         "p=json.load(sys.stdin); p.setdefault('schema_version', TaskSpec.schema_version); "
         "s=TaskSpec.from_dict(p); "
         "print(json.dumps({'payload':p,'case_id':s.case_id,'family':s.family,'steps':len(s.steps)}))"],
        cwd=MAIN, input=json.dumps(payload), capture_output=True, text=True, timeout=30,
    )
    if validation.returncode:
        raise ValueError(f"task validation against promoted schema failed: {validation.stderr.strip()}")
    validated = json.loads(validation.stdout)
    payload = validated["payload"]
    path = run_dir(run) / "tasks" / f"{validated['case_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_text = path.read_text() if path.exists() else None
    if previous_text is not None:
        previous = json.loads(previous_text)
        if previous.get("generation"):
            payload.setdefault("generation", previous["generation"])
        if previous != payload:
            history = path.parent.parent / "task_history" / validated["case_id"]
            history.mkdir(parents=True, exist_ok=True)
            (history / f"{time.time_ns()}.json").write_text(previous_text)
    path.write_text(json.dumps(payload, indent=1))
    manifest = path.parent.parent / "manifest.json"
    _initialize_authored_scope(manifest, path.parent)
    return f"saved {path} (family={validated['family']}, steps={validated['steps']})"


def submit(kind: str, args_json: str, after: str = "", priority: int = 0) -> str:
    """Queue a registered stage. kind in scene|reference|expert_check|scene_check|policy_check|episode|score|report|showcase.
    after: comma-separated job ids that must finish first.
    priority: admitted before lower-priority queued work; use it for a positive
    control or a diagnostic that must not wait behind a long chain of bulk runs,
    and leave it 0 for ordinary matrix work."""
    if os.environ.get("WO_RESOURCE_PHASE") == "preparation" and kind not in {
        "scene", "reference", "expert_check", "scene_check"
    }:
        return f"not queued ({kind}): preparation phase is restricted to scene and reference stages"
    args = json.loads(args_json)
    if kind == "scene":
        try:
            args = bind_scene_source_profile(args)
        except (KeyError, ValueError) as exc:
            return f"not queued (scene): {exc}"
    deps = [int(x) for x in after.split(",") if x.strip()]
    if kind == "reference":
        STAGES[kind].validate(args)
        root = run_dir(args["run"])
        ref = root / "references" / args["case"]
        cert = ref / "certificate.json"
        required = ["trajectory.json", "scene_inputs.json", "replay_plan.json", "replay/exterior_left.mp4"]
        if cert.is_file() and json.loads(cert.read_text()).get("success") and all(
            (ref / name).is_file() and (ref / name).stat().st_size for name in required
        ):
            from ..eval.reference_binding import verify_task, capture
            verify_task(ref, root / "tasks" / f"{args['case']}.json")
            capture(ref)
            return f"reused complete reference {args['case']}; use expert_check for a control or a new case version"
    for row in J.connect().execute(
        "select id,status,args from jobs where kind=? and status in ('queued','running') order by id",
        (kind,),
    ):
        existing = json.loads(row["args"])
        same_reference_output = kind == "reference" and all(
            existing.get(key) == args.get(key) for key in ("run", "case"))
        if existing == args or same_reference_output:
            return f"reused {row['status']} job #{row['id']} ({kind})"
    if kind == "scene_check":
        for row in J.connect().execute(
            "select id,args from jobs where kind='reference' and status in ('queued','running')",
        ):
            parent = json.loads(row["args"])
            if all(parent.get(key) == args.get(key) for key in ("run", "case")):
                deps.append(row["id"])
        deps = sorted(set(deps))
    if kind == "episode":
        # The tool server is long-lived and may have imported episode_stage
        # before a promoted case-scope or admission fix. Refresh this gate just
        # as jobs.submit refreshes stage registrations, so admission uses the
        # current shared pipeline implementation.
        import importlib

        from . import episode_stage

        episode_stage = importlib.reload(episode_stage)
        STAGES[kind].validate(args)
        try:
            episode_stage.require_submission_references(run_dir(args["run"]), J.connect(), deps)
        except ValueError as exc:
            return f"not queued (episode): {exc}"
    jid = J.submit(kind, args, after=deps, submitted_by="agent", priority=priority)
    return f"queued job #{jid} ({kind})" + (f" priority {priority}" if priority else "")


def cancel(job_id: int) -> str:
    J.cancel(job_id)
    return f"cancelled #{job_id}"


# ------------------------------------------------------------------ engineer
def _git(*args: str, cwd: Path = WORKTREE) -> str:
    r = subprocess.run(["git", "-c", "user.name=wo-agent", "-c", "user.email=agent@localhost", *args],
                       cwd=cwd, capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip()


def ensure_worktree() -> None:
    if not WORKTREE.is_dir():
        _git("worktree", "add", "-B", BRANCH, str(WORKTREE), "HEAD", cwd=MAIN)
    for name in ("weights", "Depth-Anything-3"):
        source = MAIN / name
        if not (source.exists() or source.is_symlink()):
            continue
        target = source.resolve(strict=True)
        destination = WORKTREE / name
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve(strict=True) != target:
                raise RuntimeError(f"RSI runtime path is occupied; left untouched: {destination}")
        else:
            destination.symlink_to(target, target_is_directory=True)
    shared = WORKSPACE / "outputs"
    shared.mkdir(parents=True, exist_ok=True)
    local = WORKTREE / "outputs"
    if local.exists() and local.samefile(shared):
        return
    if local.exists() or local.is_symlink():
        history = shared / "wo_agent/worktree_data_history"
        history.mkdir(parents=True, exist_ok=True)
        local.rename(history / f"{WORKTREE.name}_{time.time_ns()}")
    local.symlink_to(shared, target_is_directory=True)


def shell(command: str, timeout_s: int = 600) -> str:
    """Run a shell command (cwd = agent worktree) for research, downloads, env setup.
    Destructive commands, process killing and secrets are refused."""
    if DENY.search(command):
        return "refused: command matches the safety denylist"
    ensure_worktree()
    try:
        tmp = WORKSPACE / "tmp/agent"
        tmp.mkdir(parents=True, exist_ok=True)
        child_env = {key: value for key, value in os.environ.items()
                     if key not in {"WO_AGENT_API_KEY", "OPENAI_API_KEY"}}
        r = subprocess.run(["bash", "-lc", f"export TMPDIR={tmp}; {command}"], cwd=WORKTREE,
                           capture_output=True, text=True, timeout=min(int(timeout_s), 3600), env=child_env)
        return _clip(f"cwd={WORKTREE}\nshared_outputs={WORKSPACE / 'outputs'}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s}s (use a 'scene'/'episode' job or nohup for long work)"


def edit_file(path: str, old: str, new: str) -> str:
    """Exact string replacement in a file of the agent worktree (relative path). old='' creates a file."""
    ensure_worktree()
    p = (WORKTREE / path).resolve()
    if WORKTREE not in p.parents:
        raise ValueError("edit_file paths are relative to the agent worktree")
    if old == "":
        if p.exists():
            raise ValueError("file exists; give old text to replace")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new)
        return f"created {path}"
    text = p.read_text()
    if text.count(old) != 1:
        return f"old text found {text.count(old)} times; must be exactly once"
    p.write_text(text.replace(old, new))
    return f"edited {path}"


def run_tests(paths: str = "wo/tests") -> str:
    """Run pytest in the agent worktree (e.g. 'wo/tests' or 'tests/test_x.py')."""
    ensure_worktree()
    r = subprocess.run([str(WORKSPACE / "envs/autoresearch/bin/python"), "-m", "pytest", "-q", "-x",
                        "-p", "no:cacheprovider", *paths.split()], cwd=WORKTREE, capture_output=True,
                       text=True, timeout=1800, env={"PYTHONPATH": str(WORKTREE), "PATH": "/usr/bin:/bin"})
    return _clip(f"rc={r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}")


def promote(message: str, test_paths: str = "") -> str:
    """Commit changes after focused tests pass, then merge into the main code.
    test_paths: relevant pytest files for this patch; empty runs a small sanity set.
    (which pipeline jobs run).  Running jobs keep the code they started with."""
    ensure_worktree()
    # User requested lean checks: callers choose relevant tests; fallback is a
    # small pipeline sanity check, not the full benchmark/test suite.
    tests = run_tests(test_paths or "wo/tests/test_model_registry.py wo/tests/test_not_applicable.py")
    if not tests.startswith("rc=0"):
        return "not promoted: tests failed\n" + tests
    _git("add", "-A")
    if _git("status", "--porcelain"):
        _git("commit", "-m", message)
    out = _git("merge", "--no-edit", BRANCH, cwd=MAIN)
    return f"promoted: {message}\n{out[-1500:]}"


# ------------------------------------------------------------------ memory
def note(text: str) -> str:
    """Append a finding to the persistent research notes (read at the start of every round)."""
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    with NOTES.open("a") as fh:
        fh.write(f"\n- [{time.strftime('%m-%d %H:%M')}] {text.strip()}\n")
    return "noted"


ALL = [status, jobs, inspect_scene, read, list_dir, view_image, author_task, submit, cancel, shell, edit_file, run_tests,
       promote, note]
