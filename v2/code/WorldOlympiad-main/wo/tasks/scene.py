"""Generate the SceneCode scene for a wo TaskSpec.

    python -m wo.tasks.scene --task RUN/tasks/CASE.json

TaskSpec -> legacy robotics_task_instruction.v3 + batch manifest -> registered
scene-prompt compiler -> run_scenecode_case.sh.  On success the TaskSpec file
gets its "generation" field.  Object sizes from TaskObject descriptions are kept
verbatim, so authors should state explicit dimensions.
"""

from __future__ import annotations

import argparse
import codecs
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import tempfile
from pathlib import Path

from ..scene import WORKSPACE
from ..spec import TaskSpec

OUT = WORKSPACE / "outputs"
R = WORKSPACE / "code/WorldOlympiad-main"

_KIND = {"inside": "inside", "on": "on", "on_region": "on_region", "stacked": "stacked",
         "inserted": "inserted", "hung": "hung", "covering": "covering", "contact": "contact",
         "pressed": "pressed", "joint_closed": "closed", "upright": "on_region",
         "inverted": "inverted", "pose_delta": "pose_delta"}
_PLACE_OP = {"inside": "place", "on": "place", "on_region": "place", "stacked": "stack",
             "inserted": "insert", "hung": "hang", "covering": "cover"}






def _scene_category_symbol(category: str | None) -> str:
    """Map an authored semantic category to the identifier form required by SceneCode."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", (category or "object").strip()).strip("_.-")
    if not value:
        value = "object"
    if not value[0].isalpha():
        value = f"object_{value}"
    return value[:64]


def legacy_task(spec: TaskSpec) -> dict:
    from autoresearch.scenecode_object_requirements import check_text_limit, DESCRIPTION_MAX_CHARS, SCENE_MAX_CHARS
    initial_state = spec.initial_state or "All task objects rest stably on the tabletop; the task is not solved."
    check_text_limit(initial_state, field="scene_brief", max_chars=SCENE_MAX_CHARS)
    for obj in spec.objects:
        check_text_limit(obj.description, field="visual_description", max_chars=DESCRIPTION_MAX_CHARS)
    program, preds = [], []
    for st in spec.steps:
        g = st.goal
        target = g.target or st.target
        if st.primitive == "transport":
            program += [{"operation": "pick", "subjects": [st.subject], "constraints": {}},
                        {"operation": _PLACE_OP[g.kind], "subjects": [st.subject], "target": target,
                         "constraints": {k: v for k, v in st.constraints.items() if k != "keep_upright"}
                         | ({"upright": True} if st.constraints.get("keep_upright") else {})}]
        elif st.primitive == "reorient":
            program += [{"operation": "pick", "subjects": [st.subject], "constraints": {}},
                        {"operation": "place", "subjects": [st.subject], "target": target or spec.support.object_id,
                         "constraints": {"upright": g.kind == "upright"}}]
        elif st.primitive == "push":
            program.append({"operation": "push", "subjects": [st.subject], "target": target, "constraints": {}})
        elif st.primitive == "articulate":
            op = {"joint_open": "pull", "joint_closed": "close", "pressed": "press"}[g.kind]
            host = next(a.host_object_id for a in spec.articulations if a.component_id == st.subject)
            program.append({"operation": op, "subjects": [st.subject], "target": host, "constraints": {}})
        elif st.primitive == "tool_contact":
            program += [{"operation": "pick", "subjects": [st.subject], "constraints": {}},
                        {"operation": "strike", "subjects": [st.subject], "target": target, "constraints": {}}]
        if g.kind == "joint_open":
            kind = "drawer_open" if any(
                a.joint_type == "prismatic" and a.component_id == st.subject
                for a in spec.articulations) else "door_open"
        else:
            kind = _KIND[g.kind]
        obj = target
        if st.primitive == "articulate":
            obj = next(a.host_object_id for a in spec.articulations if a.component_id == st.subject)
        if g.kind == "upright":
            obj = target or spec.support.object_id
        predicate = {"kind": kind, "subject": g.subject, "object": obj}
        if g.params:
            predicate["params"] = dict(g.params)
        preds.append(predicate)
    # Abstract regions have no mesh for the inventory binder. Realize them as
    # visible thin mats while retaining their identity and on_region semantics.
    objects = [{"object_id": o.object_id,
                "category": "mat" if o.category == "region" else _scene_category_symbol(o.category),
                "role": {"articulated_host": "destination", "distractor": "destination"}.get(o.role, o.role),
                "geometry_constraints": o.geometry_constraints,
                "visual_description": (o.description + "; physical flat marker mat, 1 mm thick, resting on the support"
                                       if o.category == "region" else o.description)} for o in spec.objects]
    return {
        "schema_version": "robotics_task_instruction.v3", "case_id": spec.case_id,
        # Generated tasks require both deterministic binding fields.
        "case_seed": spec.seed, "case_parameters": {"layout_seed": spec.seed},
        "instruction": spec.instruction,
        "scene_brief": initial_state,
        "action_program": program, "robot": "franka_robotiq_2f85", "simulator": "pybullet",
        "objects": objects,
        "articulation_components": [{"component_id": a.component_id, "host_object_id": a.host_object_id,
                                     "semantic_kind": "drawer" if a.joint_type == "prismatic" else "door",
                                     "joint_type": a.joint_type, "visual_description": a.description}
                                    for a in spec.articulations],
        "initial_state": initial_state,
        "success_predicates": preds, "duration_sec": 12.0, "fps": 15.0,
        "requested_wams": ["cosmos3_nano_droid"],
    }


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _scenecode_logs_mention(markers, since: float, roots) -> bool:
    """SceneCode logs upstream errors to its own experiment/main logs, not stdout."""
    for root in roots:
        for log in Path(root).glob("**/scene_run/*.log"):
            try:
                if log.stat().st_mtime < since:
                    continue
                text = log.read_text(errors="replace")
            except OSError:
                continue
            if any(m in text for m in markers):
                return True
    return False


def _cleanup_scene_log_directory(path: Path, attempts: int = 5) -> None:
    """Retry transient ENOTEMPTY results from the shared data filesystem."""
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            if not path.exists():
                return
            if attempt + 1 == attempts:
                raise
        except OSError as exc:
            if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST} or attempt + 1 == attempts:
                raise
        time.sleep(0.05 * (2 ** attempt))


def run_with_service_retries(command, *, cwd, env, attempts=3, scene_log_roots=()):
    """Retry upstream availability failures, not invalid scenes/code.

    Spool output to disk (scene logs are large) and relay it live into the job log.
    Backoff runs only within the registered scene job.
    """
    transient = ("did not become ready",  # local SceneCode server startup under load
                 "Service temporarily unavailable", "HTTP 503", "Error code: 503",
                 "HTTP 502", "Error code: 502", "HTTP 429", "Error code: 429",
                 "Server returned nothing (no headers, no data)")
    for attempt in range(attempts):
        started = time.time()
        retryable = False
        window = ""
        # Separate file handles keep reader seeks away from the child's write
        # offset. Poll the parent, so background Blender stdout cannot hold EOF.
        # SceneCode may leave short-lived descendants holding stdout open after
        # its parent exits. On the shared NFS job scratch, unlinking that spool
        # then fails with EBUSY (.nfs*); keep this disposable, already-relayed
        # spool on node-local storage so cleanup does not invalidate a scene.
        log_dir = Path(tempfile.mkdtemp(prefix="wo-scene-log-", dir="/tmp"))
        try:
            path = log_dir / "output.log"
            with path.open("wb") as output, path.open("rb") as reader:
                with subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                      stderr=subprocess.STDOUT) as process:
                    decoder = codecs.getincrementaldecoder("utf-8")("replace")
                    while True:
                        chunk = reader.read(65536)
                        if chunk:
                            text = decoder.decode(chunk)
                            print(text, end="", flush=True)
                            window += text
                            retryable |= any(marker in window for marker in transient)
                            window = window[-max(map(len, transient)):]
                        elif process.poll() is not None:
                            print(decoder.decode(b"", final=True), end="", flush=True)
                            break
                        else:
                            time.sleep(0.2)
                    returncode = process.wait()
        finally:
            _cleanup_scene_log_directory(log_dir)
        if returncode == 0:
            return
        retryable = retryable or _scenecode_logs_mention(transient, started, scene_log_roots)
        if not retryable or attempt + 1 == attempts:
            raise subprocess.CalledProcessError(returncode, command)
        print(f"Upstream availability failure; retry {attempt + 2}/{attempts}", flush=True)
        time.sleep(30 * (2 ** attempt))


def archive_stale_task_recoveries(root: Path, batch: str, case: str, task_hash: str) -> None:
    """Keep prior checkpoints as evidence, but never resume another task revision."""
    for folder, manifest in (("robotics_scenecode_recoveries", "recovery_manifest.json"),
                             ("robotics_scenecode_partial_recoveries", "partial_recovery_manifest.json")):
        recovery = root / folder / batch / case
        metadata = recovery / manifest
        if not metadata.is_file():
            continue
        previous = json.loads(metadata.read_text()).get('task_sha256')
        if previous and previous != task_hash:
            archive = recovery.with_name(f"{case}.stale_{time.time_ns()}")
            recovery.rename(archive)
            print(f"Archived incompatible task checkpoint: {archive}", flush=True)


def archive_transient_incomplete_recoveries(root: Path, batch: str, case: str) -> None:
    """Do not resume a completed scene that swallowed asset-conversion outages.

    Require both missing live inventory and an explicit failed conversion caused
    by a transient service response. Ordinary geometry/name failures are retained.
    Archives preserve all diagnostics and assets for subsequent investigation.
    """
    failure = root / "robotics_scenecode_failures" / batch / f"{case}.json"
    if not failure.is_file():
        return
    data = json.loads(failure.read_text())
    if data.get("failed_stage") != "task_asset_inventory" or not any(
            m.get("status") == "missing" for m in data.get("matches", [])):
        return
    folders = ("robotics_scenecode_recoveries", "robotics_scenecode_partial_recoveries")
    recoveries = [root / folder / batch / case for folder in folders]
    markers = tuple(f"VLM analysis failed: Error code: {code}" for code in (429, 502, 503, 504))
    conversion_outage = any(
        any(marker in log.read_text(errors="replace") for marker in markers)
        for recovery in recoveries for log in recovery.glob("**/room.log"))
    if not conversion_outage:
        return
    for recovery in recoveries:
        if recovery.exists():
            archive = recovery.with_name(f"{case}.service_incomplete_{time.time_ns()}")
            recovery.rename(archive)
            print(f"Archived incomplete service-failed checkpoint: {archive}", flush=True)


def _pinned_generation(payload: dict, spec: TaskSpec) -> Path | None:
    """Validate and reuse an explicitly pinned SceneCode generation.

    TaskSpec deliberately contains only authored fields; ``generation`` is a
    runtime binding kept in the raw task payload. Material-only derivatives
    must be routed from the payload before any compiler invocation.
    """
    raw = payload.get("generation")
    if not raw:
        return None
    generation = Path(raw).expanduser().resolve()
    certificate = generation / "task_inventory_certificate.json"
    if not certificate.is_file():
        raise FileNotFoundError(f"pinned SceneCode generation is incomplete: {generation}")
    from .visual_buffer_recovery import recover_overwritten_visual_buffers
    recover_overwritten_visual_buffers(generation)
    from ..scene import resolve_scenecode
    from .geometry_contract import validate_size_contracts
    resolved = resolve_scenecode(spec, generation)
    if any(o.geometry_constraints.get("overall_dimensions_m") is not None for o in spec.objects):
        validation = validate_size_contracts(spec, resolved)
        if not validation["success"]:
            raise RuntimeError(f"pinned generation violates declared physical sizes: {validation}")
    return generation



def _reference_sidecar(task_path: Path, directory: str, schema: str) -> tuple[Path, dict] | None:
    # Keep the caller's run even when its TaskSpec is a symlink into an older run.
    task_path = Path(os.path.abspath(task_path))
    manifest = task_path.parent.parent / directory / f"{task_path.stem}.json"
    if not manifest.is_file():
        return None
    if manifest.resolve().parent != manifest.parent.resolve():
        raise ValueError("reference manifest must stay inside the current run")
    payload = json.loads(manifest.read_text())
    if payload.get("schema_version") != schema or payload.get("case_id") != task_path.stem:
        raise ValueError("reference manifest does not bind the current case")
    return manifest, payload


def context_reference_env(task_path: Path) -> dict[str, str]:
    bound = _reference_sidecar(task_path, "context_references", "scenecode_context_reference.v1")
    if bound is None:
        return {}
    manifest, _ = bound
    return {"SCENECODE_CONTEXT_REFERENCE_MANIFEST": str(manifest.resolve()),
            "SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256": _sha(manifest)}


def object_reference_env(task_path: Path) -> dict[str, str]:
    bound = _reference_sidecar(task_path, "object_references", "cosmos_seven_object_reference_images.v1")
    if bound is None:
        return {}
    manifest, payload = bound
    if payload.get("task_sha256") != _sha(Path(task_path)):
        raise ValueError("provided object references do not bind the current TaskSpec")
    return {"SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST": str(manifest.resolve()),
            "SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256": _sha(manifest)}



def source_floor_env(task_path: Path) -> dict[str, str]:
    bound = _reference_sidecar(task_path, "source_floors", "scenecode_source_floor.v1")
    if bound is None:
        return {}
    manifest, payload = bound
    if payload.get("task_sha256") != _sha(Path(task_path)):
        raise ValueError("source floor does not bind the current TaskSpec")
    return {"SCENECODE_SOURCE_FLOOR_MANIFEST": str(manifest.resolve()),
            "SCENECODE_SOURCE_FLOOR_MANIFEST_SHA256": _sha(manifest)}


def source_config_env(task_path: Path, config_root: Path | None) -> dict[str, str]:
    """Select existing native CodeObject profiles for this run without editing defaults."""
    if config_root is None:
        return {}
    import yaml
    run = Path(os.path.abspath(task_path)).parent.parent.resolve()
    root = Path(config_root).resolve()
    root.relative_to(run)
    names = ("config.yaml", "config_WallArt.yaml", "config_simplemanip.yaml", "config_structmanip.yaml")
    models = set()
    for name in names:
        payload = yaml.safe_load((root / name).read_text())
        model = payload.get("model", {})
        for role in ("llm", "planner_llm", "constructor_llm", "vision"):
            settings = model.get(role, model.get("llm", {}))
            if settings.get("provider") != "openai" or not settings.get("model_name"):
                raise ValueError("source profile must explicitly select an OpenAI author model")
            models.add(settings["model_name"])
    if len(models) != 1:
        raise ValueError("source profiles must agree on the declared author model")
    return {"SCENECODE_OPENAI_MODEL": models.pop(),
            "SCENECODE_CODE_OBJECT_CONFIG_ROOT": str(root),
            "SCENECODE_CODE_OBJECT_API": "responses",
            "SCENECODE_CODE_OBJECT_STREAM": "1",
            "SCENECODE_CODE_OBJECT_RESPONSES_STREAM": "0",
            "SCENECODE_CODE_OBJECT_READ_TIMEOUT": "none"}


def generate(task_path: Path, *, source_config_root: Path | None = None) -> Path:
    payload = json.loads(task_path.read_text())
    spec = TaskSpec.from_dict(payload)
    pinned = _pinned_generation(payload, spec)
    if pinned is not None:
        return pinned
    batch = f"wo_{spec.case_id}"[:120]
    case = spec.case_id
    bdir = OUT / "robotics_task_batches" / batch
    (bdir / "tasks").mkdir(parents=True, exist_ok=True)
    tpath = bdir / "tasks" / f"{case}.json"
    tpath.write_text(json.dumps(legacy_task(spec), indent=1))
    manifest = bdir / "batch_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "robotics_task_batch.v2", "batch_id": batch,
        "generation_mode": "model_authored_declarative", "generator_id": "wo_task_spec_v1",
        "identity_seed": spec.seed, "requested_wam": "cosmos3_nano_droid",
        "proposal_sha256": _sha(task_path),
        "cases": [{"case_id": case, "case_seed": spec.seed, "compiler_id": "agentic_worldir_compiler_v1",
                   "task_path": str(tpath.relative_to(OUT)), "task_sha256": _sha(tpath),
                   "semantic_signature": _sha(tpath), "reference_run_id": f"ref_{case}",
                   "wam_run_id": f"wam_{case}"}]}, indent=1))
    env = dict(os.environ, PYTHONPATH=str(R))
    source_env = source_config_env(task_path, source_config_root)
    env.update(source_env)
    if source_env:
        env.pop("SCENECODE_CODE_OBJECT_REASONING_EFFORT", None)
        (bdir / "source_author_input.json").write_text(json.dumps({
            "schema_version": "scenecode_source_author_input.v1", "case_id": case,
            "author_model": source_env["SCENECODE_OPENAI_MODEL"],
            "profiles": {path.name: _sha(path) for path in Path(source_env["SCENECODE_CODE_OBJECT_CONFIG_ROOT"]).glob("*.yaml")},
            "profile_root": source_env["SCENECODE_CODE_OBJECT_CONFIG_ROOT"],
            "api": "responses", "stream": False, "read_timeout": None,
        }, indent=2))
    env.update(source_floor_env(task_path))
    context_env = context_reference_env(task_path)
    env.update(context_env)
    object_env = object_reference_env(task_path)
    env.update(object_env)
    if object_env:
        (bdir / "object_reference_input.json").write_text(json.dumps({
            "schema_version": "scenecode_object_reference_input.v1",
            "case_id": case, "task_sha256": _sha(task_path),
            "manifest": object_env["SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST"],
            "manifest_sha256": object_env["SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256"],
            "scope": "external product design references; no native-server generation or physical success claim",
        }, indent=2))
    if context_env:
        (bdir / "context_reference_input.json").write_text(json.dumps({
            "schema_version": "scenecode_context_reference_input.v1",
            "case_id": case, "task_sha256": _sha(task_path),
            "manifest": context_env["SCENECODE_CONTEXT_REFERENCE_MANIFEST"],
            "manifest_sha256": context_env["SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256"],
            "scope": "semantic generation guidance; no physical success claim",
        }, indent=2))
    py = str(WORKSPACE / "envs/autoresearch/bin/python")
    subprocess.run([py, "-m", "autoresearch.robotics_scene_prompt_compiler", "--batch-manifest",
                    str(manifest), "--output-root", str(OUT), "--supersede-existing"],
                   cwd=R, env=env, check=True)
    archive_stale_task_recoveries(OUT, batch, case, _sha(tpath))
    archive_transient_incomplete_recoveries(OUT, batch, case)
    env.update(WORLDOLYMPIAD_SCENECODE_CONCURRENT="1", WORLDOLYMPIAD_SCENECODE_CODE_RETRIES="5",
               WORLDOLYMPIAD_SCENECODE_SUPERSEDE="1",
               SCENECODE_AUTHORED_TASK_REQUIREMENTS=tpath.read_text())
    run_with_service_retries(
        ["bash", str(R / "deployment/ziplab5090/run_scenecode_case.sh"), batch, case],
        cwd=R, env=env)
    gen = OUT / "robotics_scenecode_generations" / batch / case
    if not (gen / "task_inventory_certificate.json").is_file():
        raise RuntimeError(f"SceneCode produced no task inventory for {case}")
    from .visual_buffer_recovery import recover_overwritten_visual_buffers
    recover_overwritten_visual_buffers(gen)
    if any(o.geometry_constraints.get("overall_dimensions_m") is not None for o in spec.objects):
        from ..scene import resolve_scenecode
        from .asset_size_normalization import normalize_asset_sizes
        from .collision_envelope import conform_collision_envelopes
        from .geometry_contract import validate_size_contracts
        resolved = resolve_scenecode(spec, gen)
        normalize_asset_sizes(spec, resolved, gen)
        resolved = resolve_scenecode(spec, gen)
        conform_collision_envelopes(spec, resolved, gen)
        validation = validate_size_contracts(spec, resolve_scenecode(spec, gen))
        (gen / "physical_size_validation.json").write_text(json.dumps(validation, indent=1))
        if not validation["success"]:
            raise RuntimeError(f"exported assets violate declared physical sizes: {validation}")
    payload["generation"] = str(gen)
    task_path.write_text(json.dumps(payload, indent=1))
    return gen


def validate_source(task_path: Path) -> Path | None:
    """Run the authoritative physical admission gate without compiling a scene.

    Downstream registered stages must not trust a copied validation JSON: the
    source, task and current resolver are re-bound and measured here.
    """
    payload = json.loads(task_path.read_text())
    spec = TaskSpec.from_dict(payload)
    return _pinned_generation(payload, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--source-config-root", type=Path)
    ap.add_argument("--validate", action="store_true",
                    help="validate a pinned source and physical contract without compiling")
    a = ap.parse_args()
    if a.validate:
        generation = validate_source(a.task)
        if generation is not None:
            print(generation)
        return
    print(generate(a.task, source_config_root=a.source_config_root))


if __name__ == "__main__":
    main()
