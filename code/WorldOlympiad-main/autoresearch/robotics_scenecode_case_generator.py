"""Generate and inventory-certify one selected declarative SceneCode case."""

from __future__ import annotations

from contextlib import nullcontext

import argparse
from datetime import datetime, timezone
import errno
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

from wo.scene_objects import physical_objects

from .generate_scene import _hydra_string_override, _run
from .robotics_contracts import RoboticsTaskBatchManifest, TaskInstructionSpec
from .robotics_resource_lock import exclusive_blender_job
from .robotics_views import sha256_file


IGNORED_TOKENS = {
    "a", "an", "and", "the", "with", "matte", "solid", "flat", "small",
    "large", "low", "raised", "ceramic", "plastic", "wood", "wooden",
}

TOKEN_CANONICAL_FORMS = {
    # SceneCode may name the same manipulation support by its common robotics
    # synonym even when the selected task uses ``worktable``.
    "workbench": "worktable",
    "table": "worktable",
    # Natural-language generators freely alternate these equivalent quantity
    # words ("one vertical hole" vs "a single vertical hole").
    "single": "one",
    # Generators describe the same rigid primitive with different shape nouns
    # ("blue cube" requested, "blue cuboid task block" generated).  Colour and
    # size words still rank competing candidates through identity overlap.
    "cuboid": "block",
    "cube": "block",
    "cubes": "block",
    "blocks": "block",
    "brick": "block",
    "bricks": "block",
    # common plural / near-synonym container nouns
    "bowls": "bowl",
    "bins": "bin",
    "boxes": "box",
    "mugs": "mug",
    "cups": "cup",
    "mat": "pad",
    "mats": "pad",
    "pads": "pad",
}

PHYSICAL_CATEGORY_ALIASES = {"cabinet": {"cupboard"}, "cupboard": {"cabinet"}}


def _physical_category_tokens(item) -> set[str]:
    """Keep category gates physical when legacy specs use semantic role labels.

    Appearance words never become category evidence. A concrete category is
    unchanged; generic role labels use only recognized physical nouns from the
    description/id, and container requests only use receptacle nouns.
    """
    category = _tokens(item.category)
    generic = {"distractor", "contents", "container", "manipuland", "object"}
    if not category or not category <= generic:
        return category
    receptacles = {"cup", "mug", "bowl", "bin", "box", "tray", "basket", "bucket", "bottle"}
    shapes = receptacles | {"block", "bead", "beads", "ball", "sphere", "puck", "disk", "peg", "pad", "lid", "book", "plate", "tool"}
    nouns = _tokens(f"{item.object_id} {item.visual_description}") & (receptacles if "container" in category else shapes)
    return nouns or category


PUBLISHED_TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".sdf",
    ".txt",
    ".urdf",
    ".xml",
    ".yaml",
    ".yml",
}

GENERATED_CANDIDATE_CONTAINERS = {"code_object", "geometry", "sdf"}
# Robotics references keep SceneCode's architectural and background diversity,
# but optional context must not become a blocking dependency of a declared task.
# The generic indoor-population profile asks another VLM to select nearby props
# before manipuland placement, so disable that extra population pass.  Wall and
# ceiling assets are omitted by the launcher unless explicitly enabled.
# Validation remains mandatory; task assets allow one candidate repair.
# A generated task asset is authored as code and validated against its declared
# PartPlan.  The failure is usually mechanical (a custom mesh is not centered,
# so its bounds do not match the plan), and SceneCode's repair loop fixes it by
# re-prompting with the error.  Two repair passes were not enough for an
# articulated host: every attempt reached the last part and then discarded the
# whole asset, which costs tens of minutes per restart.  The budget is the
# number of *cheap* in-place repairs, so it is raised and made tunable rather
# than paying for a fresh scene each time.
CODE_GENERATED_RETRIES = max(1, int(os.environ.get("WORLDOLYMPIAD_SCENECODE_CODE_RETRIES", "5")))

def _code_object_profile_overrides(profile_root: str | None) -> list[str]:
    profile_overrides = []
    if profile_root:
        profiles = {"config_path": "config.yaml", "wall_art_config_path": "config_WallArt.yaml",
                    "simple_object_config_path": "config_simplemanip.yaml",
                    "manipuland_config_path": "config_structmanip.yaml"}
        for agent in ("furniture_agent", "manipuland_agent", "wall_agent", "ceiling_agent"):
            for key, filename in profiles.items():
                path = Path(profile_root).resolve() / filename
                if not path.is_file():
                    raise ValueError(f"source author profile is missing: {filename}")
                profile_overrides.append(_hydra_string_override(
                    f"++{agent}.asset_manager.code_object.{key}", str(path)))
    return profile_overrides


ROBOTICS_SCENECODE_OVERRIDES = (
    # Benchmark generation must not depend on a separately provisioned image
    # account just to texture the room shell.  SceneCode's checked-in default
    # wall/floor materials are deterministic and sufficient for a robotics
    # reference; task assets still keep their authored matte materials.
    "floor_plan_agent.materials.generator.enabled=false",
    "manipuland_agent.context_furniture.enabled=false",
    # The two-interface profile provides text generation and a local tested model,
    # not a separate image-generation endpoint. SceneCode can author these assets
    # from text, so context-image generation stays optional.
    "furniture_agent.context_image_generation.enabled=false",
    "manipuland_agent.context_image_generation.enabled=false",
    f"furniture_agent.asset_manager.router.strategies.code_generated.max_retries={CODE_GENERATED_RETRIES}",
    f"manipuland_agent.asset_manager.router.strategies.code_generated.max_retries={CODE_GENERATED_RETRIES}",
    # SceneCode's furniture critic sees the whole prompt but can only call
    # furniture tools. In task scenes it misreads later-stage manipulands as
    # missing furniture and may remove a valid support while trying to repair
    # the impossible request. Candidate assets still undergo mesh/VLM/physics
    # validation, and the completed scene remains subject to harness inventory,
    # physics, identity, and visibility gates.
    "furniture_agent.max_critique_rounds=0",
    "wall_agent.max_critique_rounds=0",
    "ceiling_agent.max_critique_rounds=0",
    # A completed-state critic can reinterpret the typed goal relation as an
    # instruction to pre-solve the static reference. The initial designer still
    # places all requested objects and checks collisions; the harness certifies
    # the unsolved relation after SceneCode returns.
    "manipuland_agent.max_critique_rounds=0",
    "wall_agent.asset_manager.router.strategies.code_generated.max_retries=1",
    "wall_agent.asset_manager.code_object.max_attempts=1",
    "ceiling_agent.asset_manager.router.strategies.code_generated.max_retries=1",
    "ceiling_agent.asset_manager.code_object.max_attempts=1",
)
DEPENDENCY_SUFFIXES = {
    ".bin",
    ".dae",
    ".glb",
    ".gltf",
    ".jpeg",
    ".jpg",
    ".mtl",
    ".obj",
    ".png",
    ".sdf",
    ".stl",
    ".urdf",
    ".webp",
}


def classify_generation_exception(
    *,
    stage: str,
    exception: BaseException,
    log_tails: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return an evidence-based failure class and owning stage."""

    if stage != "scenecode_generation":
        return "asset_geometry", "scenecode_candidate"
    if isinstance(exception, subprocess.CalledProcessError) and exception.returncode in {
        -signal.SIGINT,
        -signal.SIGTERM,
        -signal.SIGKILL,
        130,
        120,
    }:
        return "runtime_interrupted", "scenecode_runtime"
    diagnostics = "\n".join((log_tails or {}).values()).lower()
    if any(
        marker in diagnostics
        for marker in (
            "service temporarily unavailable",
            "502 bad gateway",
            "503 service unavailable",
            "internalservererror: error code: 502",
            "internalservererror: error code: 503",
        )
    ):
        return "provider_unavailable", "model_provider"
    return "prompt_semantics", "scenecode_candidate"


def collect_failure_log_tails(
    *, staging: Path, max_lines_per_file: int = 40, max_total_chars: int = 16_000
) -> dict[str, str]:
    """Preserve bounded text diagnostics before failed staging is discarded."""

    tails: dict[str, str] = {}
    remaining = max_total_chars
    for path in sorted(staging.rglob("*.log")):
        if remaining <= 0 or not path.is_file():
            break
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        value = "\n".join(lines[-max_lines_per_file:]).replace(
            str(staging), "<discarded-staging>"
        )
        if not value:
            continue
        value = value[-remaining:]
        tails[path.relative_to(staging).as_posix()] = value
        remaining -= len(value)
    return tails


def completed_despite_broken_stdout(*, scene_run: Path, exception: BaseException) -> bool:
    """Recognize only a fully materialized SceneCode run lost at stdout flush.

    CPython may exit 120 when its final stdout flush sees a closed supervising
    pipe.  That transport failure happens after application work.  We accept it
    only when SceneCode's own terminal marker and final combined state both
    exist; all normal inventory and semantic gates still run afterwards.
    """

    if not (
        isinstance(exception, subprocess.CalledProcessError)
        and exception.returncode == 120
    ):
        return False
    house_state = scene_run / "scene_000" / "combined_house" / "house_state.json"
    if not house_state.is_file():
        return False
    for path in (scene_run / "main.log", scene_run / "experiment.log"):
        if path.is_file() and "ALL SCENES COMPLETED!" in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return True
    return False


def _tokens(value: str) -> set[str]:
    return {
        TOKEN_CANONICAL_FORMS.get(token, token)
        for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
        if len(token) >= 3 and token not in IGNORED_TOKENS
    }


def generation_recovery_path(*, root: Path, batch_id: str, case_id: str) -> Path:
    """Return the deterministic completed-SceneCode recovery location."""

    return root / "robotics_scenecode_recoveries" / batch_id / case_id


def partial_generation_recovery_path(
    *, root: Path, batch_id: str, case_id: str
) -> Path:
    """Return the deterministic pre-manipuland checkpoint recovery location."""

    return root / "robotics_scenecode_partial_recoveries" / batch_id / case_id


def _manipuland_resume_checkpoints(scene_run: Path) -> list[Path]:
    return sorted(
        scene_run.glob(
            "scene_*/room_*/scene_states/scene_after_ceiling_objects/scene_state.json"
        )
    )


PARTIAL_RESUME_CHECKPOINTS = (
    ("manipuland", "scene_after_ceiling_objects"),
    ("ceiling_mounted", "scene_after_wall_objects"),
    ("wall_mounted", "scene_after_furniture"),
)



def _prepare_source_floor(root: Path, staging: Path, prompt: str,
                          prompt_sha256: str) -> Path | None:
    """Bind an explicit floor input, separately from same-task recovery."""
    configured = os.environ.get("SCENECODE_SOURCE_FLOOR_MANIFEST")
    if not configured:
        return None
    manifest = Path(configured).resolve()
    if sha256_file(manifest) != os.environ.get("SCENECODE_SOURCE_FLOOR_MANIFEST_SHA256"):
        raise ValueError("source floor manifest changed")
    record = json.loads(manifest.read_text())
    if (record.get("schema_version") != "scenecode_source_floor.v1"
            or record.get("scene_prompt_sha256") != prompt_sha256
            or not isinstance(record.get("source_job_id"), int)):
        raise ValueError("source floor input lacks current prompt/source-job binding")
    source = Path(record["source_scene_dir"]).resolve()
    source.relative_to(root.resolve())
    layout_path = source / "house_layout.json"
    layout = json.loads(layout_path.read_text())
    rooms = layout.get("rooms", [])
    geometry = layout.get("room_geometries", {})
    if (not layout.get("placement_valid") or not layout.get("connectivity_valid")
            or len(rooms) != 1 or set(geometry) != {rooms[0].get("id")}):
        raise ValueError("source floor must have a completed single-room layout")
    size = record.get("room_size_m", [])
    actual = geometry[rooms[0]["id"]]
    if (len(size) != 2 or sorted(map(float, size)) !=
            sorted((float(actual["width"]), float(actual["length"])))):
        raise ValueError("source floor dimensions differ from the declared input")
    if (any(p.is_file() for p in source.glob("room_*/generated_assets/**/*"))
            or any(p.is_file() for p in source.glob("room_*/scene_states/**/*"))):
        raise ValueError("source floor must not include task/furniture assets or checkpoints")
    paths = [layout_path]
    for directory in ("room_geometry", "floor_plans", "materials"):
        paths += [p for p in (source / directory).rglob("*") if p.is_file()]
    actual_files = {str(p.relative_to(source)): sha256_file(p) for p in sorted(paths)}
    if (not any(k.startswith("room_geometry/") for k in actual_files)
            or not any(k.startswith("floor_plans/") for k in actual_files)
            or actual_files != record.get("source_files")):
        raise ValueError("source floor file inventory/hash changed")
    target = staging / "source_floor" / "scene_000"
    target.mkdir(parents=True)
    for relative, digest in actual_files.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        if sha256_file(destination) != digest:
            raise ValueError("source floor copy changed resource bytes")
    floor_size = None
    if actual.get("sdf_path"):
        floor_size = ET.parse(source / actual["sdf_path"]).findtext(
            ".//collision[@name='floor_collision']/geometry/box/size")
    floor_xy = list(map(float, floor_size.split()))[:2] if floor_size else None
    rewritten = json.loads(json.dumps(layout))
    rewritten["house_prompt"] = prompt
    rewritten["rooms"][0]["prompt"] = prompt
    (target / "house_layout.json").write_text(json.dumps(rewritten, indent=2) + "\n")
    (staging / "source_floor_input.json").write_text(json.dumps({
        "schema_version": "scenecode_source_floor_input.v1", "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest), "source_job_id": record["source_job_id"],
        "source_scene_dir": str(source), "source_files": actual_files,
        "current_prompt_sha256": prompt_sha256,
        "source_floor_sdf_xy_m": floor_xy,
        "source_room_geometry_width_length_m": [actual["width"], actual["length"]],
        "current_scene_prompt_size_m": size,
        "current_house_layout_sha256": sha256_file(target / "house_layout.json"),
        "changed_fields": ["house_prompt", "rooms[0].prompt"],
        "scope": "explicit reused floor source; not a same-task recovery or furniture success",
    }, indent=2) + "\n")
    return target.parent


def _partial_resume_stage(scene_run: Path) -> str | None:
    """Return the latest unique SceneCode stage that can resume cleanly."""

    for stage, checkpoint in PARTIAL_RESUME_CHECKPOINTS:
        matches = sorted(
            scene_run.glob(
                f"scene_*/room_*/scene_states/{checkpoint}/scene_state.json"
            )
        )
        if len(matches) == 1:
            return stage
        if len(matches) > 1:
            raise ValueError(f"partial SceneCode recovery has ambiguous {checkpoint}")
    layouts=sorted(scene_run.glob('scene_*/house_layout.json'))
    if len(layouts)>1:
        raise ValueError('partial SceneCode recovery has ambiguous floor plans')
    if len(layouts)==1:
        scene=layouts[0].parent
        layout=json.loads(layouts[0].read_text())
        if (isinstance(layout,dict) and isinstance(layout.get('rooms'),list) and layout['rooms']
                and all(isinstance(room,dict) and room.get('id') for room in layout['rooms'])
                and any(p.is_file() for p in (scene/'room_geometry').rglob('*'))
                and any(p.is_file() for p in (scene/'floor_plans').rglob('*'))):
            # The released SceneCode resume interface accepts furniture directly
            # from house_layout and the two floor-plan geometry directories.
            return 'furniture'
    return None


def _validate_partial_recovery_candidate(
    *, recovery: Path, task_sha256: str, prompt_sha256: str
) -> str:
    manifest = json.loads(
        (recovery / "partial_recovery_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != "robotics_scenecode_partial_recovery.v2":
        raise ValueError("unsupported partial SceneCode recovery manifest")
    if manifest.get("task_sha256") != task_sha256:
        raise ValueError("partial SceneCode recovery belongs to another selected task")
    if manifest.get("scene_prompt_sha256") != prompt_sha256:
        raise ValueError("partial SceneCode recovery belongs to another scene prompt")
    resume_stage = _partial_resume_stage(recovery / "scene_run")
    if resume_stage is None or manifest.get("resume_stage") != resume_stage:
        raise ValueError("partial SceneCode recovery lacks its declared checkpoint")
    return resume_stage


def _preserve_partial_generation(
    *,
    staging: Path,
    recovery: Path,
    task_sha256: str,
    prompt_sha256: str,
) -> Path:
    """Move a resumable pre-manipuland SceneCode checkpoint into recovery."""

    resume_stage = _partial_resume_stage(staging / "scene_run")
    if resume_stage is None:
        raise ValueError("SceneCode staging has no unique resumable checkpoint")
    recovery.parent.mkdir(parents=True, exist_ok=True)
    if recovery.exists():
        raise FileExistsError(
            f"refusing to overwrite partial SceneCode recovery: {recovery}"
        )
    rewrite_published_references(
        root=staging, staging=staging, destination=recovery
    )
    (staging / "partial_recovery_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "robotics_scenecode_partial_recovery.v2",
                "task_sha256": task_sha256,
                "scene_prompt_sha256": prompt_sha256,
                "resume_stage": resume_stage,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(staging, recovery)
    return recovery


def _validate_recovery_candidate(
    *, recovery: Path, task_sha256: str, prompt_sha256: str
) -> None:
    manifest_path = recovery / "recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "robotics_scenecode_recovery.v1":
        raise ValueError("unsupported SceneCode recovery manifest")
    if manifest.get("task_sha256") != task_sha256:
        raise ValueError("SceneCode recovery belongs to a different selected task")
    if manifest.get("scene_prompt_sha256") != prompt_sha256:
        raise ValueError("SceneCode recovery belongs to a different scene prompt")
    house_state = recovery / "scene_run" / "scene_000" / "combined_house" / "house_state.json"
    if not house_state.is_file():
        raise ValueError("SceneCode recovery lacks completed combined house state")


def _preserve_completed_generation(
    *,
    staging: Path,
    recovery: Path,
    task_sha256: str,
    prompt_sha256: str,
    failed_stage: str,
) -> Path:
    """Move a completed SceneCode candidate to reusable recovery storage."""

    recovery.parent.mkdir(parents=True, exist_ok=True)
    if staging != recovery:
        if recovery.exists():
            raise FileExistsError(
                f"refusing to overwrite SceneCode recovery candidate: {recovery}"
            )
        rewrite_published_references(
            root=staging, staging=staging, destination=recovery
        )
    (staging / "recovery_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "robotics_scenecode_recovery.v1",
                "task_sha256": task_sha256,
                "scene_prompt_sha256": prompt_sha256,
                "failed_stage": failed_stage,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if staging != recovery:
        os.replace(staging, recovery)
    return recovery


def runtime_executable(value: str | Path) -> Path:
    """Return an absolute executable path without dereferencing a venv symlink."""

    return Path(os.path.abspath(os.fspath(value)))


def rewrite_published_references(
    *, root: Path, staging: Path, destination: Path
) -> int:
    """Replace dead atomic-staging prefixes in runtime-readable text assets."""

    replacements = (
        (str(staging), str(destination)),
        (staging.name, destination.name),
    )
    rewritten = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PUBLISHED_TEXT_SUFFIXES:
            continue
        try:
            value = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = value
        for source, target in replacements:
            updated = updated.replace(source, target)
        if updated == value:
            continue
        path.write_text(updated, encoding="utf-8")
        rewritten += 1
    return rewritten


def remove_staging_latest_run_link(*, staging: Path, destination: Path) -> bool:
    """Remove SceneCode's batch-level symlink when it still targets staging."""

    latest = destination.parent / "latest-run"
    if not latest.is_symlink():
        return False
    raw_target = Path(os.readlink(latest))
    target = raw_target if raw_target.is_absolute() else latest.parent / raw_target
    target = Path(os.path.abspath(target))
    staging_absolute = Path(os.path.abspath(staging))
    if target != staging_absolute and staging_absolute not in target.parents:
        return False
    latest.unlink()
    return True


def remove_self_referential_batch_link(*, destination: Path) -> bool:
    """Remove SceneCode's batch link when it points back at its own directory.

    SceneCode publishes a batch-level link inside the batch directory.  When it
    resolves to that same directory, every recursive scan of the published
    output tree descends it without bound, so the self-loop is removed before
    the batch is published.
    """

    link = destination / destination.name
    if not link.is_symlink():
        return False
    raw_target = Path(os.readlink(link))
    target = raw_target if raw_target.is_absolute() else link.parent / raw_target
    if Path(os.path.abspath(target)) != Path(os.path.abspath(destination)):
        return False
    link.unlink()
    return True


def find_unvalidated_task_assets(
    *, task: TaskInstructionSpec, scene_run: Path
) -> list[dict[str, str]]:
    """Find task assets that SceneCode explicitly accepted without validation."""

    pattern = re.compile(
        r"final attempt accepted without validation for ['\"]([^'\"]+)['\"]",
        flags=re.IGNORECASE,
    )
    descriptions: set[str] = set()
    for log_path in scene_run.rglob("*.log"):
        try:
            value = log_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        descriptions.update(pattern.findall(value))

    findings: list[dict[str, str]] = []
    for description in sorted(descriptions):
        description_tokens = _tokens(description)
        candidates: list[tuple[tuple[int, int], object]] = []
        for item in task.objects:
            category_tokens = _tokens(f"{item.object_id} {item.category}")
            category_overlap = len(category_tokens & description_tokens)
            if not category_overlap:
                continue
            identity_tokens = _tokens(
                f"{item.object_id} {item.category} {item.visual_description}"
            )
            candidates.append(
                ((category_overlap, len(identity_tokens & description_tokens)), item)
            )
        if not candidates:
            continue
        _, item = max(candidates, key=lambda value: value[0])
        findings.append(
            {
                "task_object_id": item.object_id,
                "category": item.category,
                "visual_description": item.visual_description,
                "unvalidated_asset_description": description,
            }
        )
    return findings


def _nested_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _nested_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _nested_strings(child)]
    return []


def _resolve_existing_reference(
    value: str, *, room_root: Path, scene_root: Path, relative_to: Path | None = None
) -> Path | None:
    raw = value.removeprefix("file://")
    path = Path(raw)
    # House-state dictionaries also contain natural-language descriptions.
    # Never submit an overlong prose component to stat(2): Linux reports
    # ENAMETOOLONG before we can classify it as a non-path, which used to
    # discard an otherwise successful SceneCode generation during cleanup.
    if any(len(os.fsencode(part)) > 255 for part in path.parts):
        return None
    candidates = [path] if path.is_absolute() else []
    if relative_to is not None and not path.is_absolute():
        candidates.append(relative_to / path)
    if not path.is_absolute():
        candidates.extend((room_root / path, scene_root / path))
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate))
        try:
            if absolute.exists():
                return absolute
        except OSError as exc:
            # Malformed registry prose is not a runtime dependency.  Actual
            # declared asset paths still fail later inventory/certificate gates.
            if exc.errno == errno.ENAMETOOLONG:
                continue
            raise
    return None


def _dependency_references(path: Path) -> list[str]:
    if not path.is_file() or path.suffix.lower() not in PUBLISHED_TEXT_SUFFIXES | {".gltf"}:
        return []
    try:
        value = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    # Covers XML URIs and JSON path values without mistaking ordinary prompt
    # prose for dependencies: the token must end in a known asset suffix.
    suffixes = "|".join(re.escape(item.lstrip(".")) for item in DEPENDENCY_SUFFIXES)
    return re.findall(
        rf"(?:file://)?(?:/[^<>'\"\s]+|[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)\.(?:{suffixes})",
        value,
        flags=re.IGNORECASE,
    )


def prune_unreferenced_generated_assets(*, house_state: Path) -> dict[str, Any]:
    """Delete rejected SceneCode candidates outside the final scene dependency graph.

    Only immediate children of SceneCode's candidate containers are eligible.
    The final house registry is authoritative, and referenced text assets are
    followed transitively so SDF/URDF/GLTF dependencies remain intact.
    """

    payload = json.loads(house_state.read_text(encoding="utf-8"))
    scene_root = house_state.parent.parent
    removed: list[str] = []
    kept: list[str] = []
    registry_entries_removed = 0

    for room_id, room in payload.get("rooms", {}).items():
        room_root = scene_root / f"room_{room_id}"
        generated_root = room_root / "generated_assets"
        if not generated_root.is_dir():
            continue

        references: set[Path] = set()
        queue: list[Path] = []
        for value in _nested_strings(room.get("objects", {})):
            resolved = _resolve_existing_reference(
                value, room_root=room_root, scene_root=scene_root
            )
            if resolved is not None:
                queue.append(resolved)
        while queue:
            path = queue.pop()
            if path in references:
                continue
            references.add(path)
            for value in _dependency_references(path):
                resolved = _resolve_existing_reference(
                    value,
                    room_root=room_root,
                    scene_root=scene_root,
                    relative_to=path.parent,
                )
                if resolved is not None and resolved not in references:
                    queue.append(resolved)

        candidate_children: list[Path] = []
        for asset_kind in generated_root.iterdir():
            if not asset_kind.is_dir():
                continue
            for container_name in GENERATED_CANDIDATE_CONTAINERS:
                container = asset_kind / container_name
                if container.is_dir():
                    candidate_children.extend(container.iterdir())
        kept_paths: set[Path] = set()
        for candidate in sorted(set(candidate_children)):
            candidate_abs = Path(os.path.abspath(candidate))
            live = any(
                reference == candidate_abs or candidate_abs in reference.parents
                for reference in references
            )
            if live:
                kept_paths.add(candidate_abs)
                kept.append(str(candidate))
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
            elif candidate.exists():
                candidate.unlink()
            removed.append(str(candidate))

        # Registries are audit indexes, not runtime authorities. Remove entries
        # whose every recorded artifact disappeared with a rejected candidate.
        for registry_path in generated_root.rglob("asset_registry.json"):
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            if not isinstance(registry, dict):
                raise ValueError(f"unsupported SceneCode asset registry: {registry_path}")
            filtered: dict[str, Any] = {}
            for key, value in registry.items():
                entry_live = False
                for raw in _nested_strings(value):
                    resolved = _resolve_existing_reference(
                        raw, room_root=room_root, scene_root=scene_root
                    )
                    if resolved is not None and any(
                        root == resolved or root in resolved.parents
                        for root in kept_paths
                    ):
                        entry_live = True
                        break
                if entry_live:
                    filtered[key] = value
                else:
                    registry_entries_removed += 1
            if filtered != registry:
                registry_path.write_text(
                    json.dumps(filtered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    return {
        "schema_version": "scenecode_generated_asset_cleanup.v1",
        "house_state": str(house_state),
        "kept_candidate_roots": sorted(kept),
        "removed_candidate_roots": sorted(removed),
        "registry_entries_removed": registry_entries_removed,
    }


def _scene_inventory(house_state: Path) -> list[dict[str, Any]]:
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    scene_root = house_state.parent.parent
    inventory: list[dict[str, Any]] = []
    for room_id, room in payload.get("rooms", {}).items():
        room_root = scene_root / f"room_{room_id}"
        objects = physical_objects(room.get("objects") or {})
        live_object_ids = {
            str(item.get('metadata', {}).get('source_asset_id', object_id))
            for object_id, item in objects.items()
        }
        registry_entries: dict[str, list[dict[str, Any]]] = {}
        generated_root = room_root / "generated_assets"
        if generated_root.is_dir():
            for registry_path in generated_root.rglob("asset_registry.json"):
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                if not isinstance(registry, dict):
                    raise ValueError(
                        f"unsupported SceneCode asset registry: {registry_path}"
                    )
                for key, candidate in registry.items():
                    if not isinstance(candidate, dict):
                        continue
                    candidate_id = str(candidate.get("object_id", key))
                    # A registry is never allowed to resurrect rejected assets.
                    # It can only fill paths for an object that remains present
                    # in the final authoritative house_state.
                    if candidate_id in live_object_ids:
                        registry_entries.setdefault(candidate_id, []).append(candidate)
        for object_id, item in objects.items():
            source_id = str(item.get('metadata', {}).get('source_asset_id', object_id))
            live_registry = registry_entries.get(source_id, [])
            sdf_paths = [
                value
                for value in [
                    item.get("sdf_path"),
                    *(candidate.get("sdf_path") for candidate in live_registry),
                ]
                if value
            ]
            geometry_paths = [
                value
                for value in [
                    item.get("geometry_path"),
                    *(candidate.get("geometry_path") for candidate in live_registry),
                ]
                if value
            ]
            # Metadata paths contain task names, not physical object identity.
            text = " ".join(
                (
                    str(object_id),
                    str(item.get("object_type", "")),
                    str(item.get("name", "")),
                    str(item.get("description", "")),
                )
            )
            inventory.append(
                {
                    "room_id": str(room_id),
                    "object_id": str(object_id),
                    "name": str(item.get("name", "")),
                    "description": str(item.get("description", "")),
                    "object_type": str(item.get("object_type", "")),
                    "has_support_surfaces": bool(item.get("support_surfaces")),
                    "tokens": sorted(_tokens(text)),
                    "sdf_path_candidates": list(dict.fromkeys(sdf_paths)),
                    "geometry_path_candidates": list(dict.fromkeys(geometry_paths)),
                    "metadata": item.get("metadata", {}),
                }
            )
    return inventory


def _resolve_scene_asset_path(
    *, house_state: Path, room_id: str, value: str | None
) -> Path | None:
    """Resolve an asset exactly as the canonical SceneCode importer does."""

    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve() if path.is_file() else None
    scene_root = house_state.parent.parent
    for candidate in (scene_root / f"room_{room_id}" / path, scene_root / path):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _sdf_articulation_joint_types(asset_path: Path) -> set[str]:
    """Return the moving joint types declared by one SDF or URDF asset.

    A task that declares an articulation component requires the *host asset* to
    actually contain a matching moving joint. SceneCode asset names and category
    tokens cannot establish that: a cabinet-shaped static box and a real hinged
    cupboard can share an appearance while only one exposes the joint the
    reference executor must drive. This reads the asset that the canonical
    importer would consume and reports the declared joint semantics, so host
    selection can be qualified by joint type instead of appearance alone.
    """

    try:
        root = ET.parse(asset_path).getroot()
    except (ET.ParseError, OSError):
        return set()
    declared: set[str] = set()
    for joint in root.iter("joint"):
        joint_type = str(joint.get("type", "")).strip().lower()
        if joint_type in {"revolute", "continuous", "prismatic"}:
            # ``continuous`` is a revolute joint without a limit.
            declared.add("revolute" if joint_type == "continuous" else joint_type)
    return declared


SUPPORT_REGION_TOKENS = {"area", "region", "surface", "tabletop"}


def _bind_declared_support_regions(
    *, matches: list[dict[str, Any]], inventory: list[dict[str, Any]], task: TaskInstructionSpec
) -> None:
    """Bind a declared support *region* destination to its already matched support.

    A task may declare the support object and, separately, the region of that
    support where the object must end up (for example a ``table_region``
    destination on the same tabletop). SceneCode authors one physical support,
    so requiring a second distinct scene asset would reject a correct scene.
    The destination is bound to the single already matched support object that
    exposes the load-bearing surface, and the binding is recorded explicitly so
    the shared dependency is auditable instead of silently dropped.
    """

    task_by_id = {item.object_id: item for item in task.objects}
    scene_by_id = {item["object_id"]: item for item in inventory}
    for match in matches:
        if match.get("status") != "missing":
            continue
        item = task_by_id.get(str(match.get("task_object_id")))
        if item is None or item.role != "destination":
            continue
        destination_tokens = _tokens(
            f"{item.object_id} {item.category} {item.visual_description}"
        )
        # A container description may mention an open top or a table surface.
        # Only the declared category can establish that this object is a region.
        if not _tokens(item.category) & SUPPORT_REGION_TOKENS:
            continue
        candidates: list[tuple[dict[str, Any], set[str]]] = []
        for other in matches:
            if other is match or other.get("status") != "matched":
                continue
            support_item = task_by_id.get(str(other.get("task_object_id")))
            if support_item is None or support_item.role != "support":
                continue
            scene_object = scene_by_id.get(str(other.get("scene_object_id")))
            if scene_object is None or not scene_object["has_support_surfaces"]:
                continue
            support_tokens = _tokens(
                f"{support_item.object_id} {support_item.category} "
                f"{support_item.visual_description}"
            )
            if not destination_tokens & _tokens(support_item.category):
                continue
            candidates.append((other, support_tokens))
        if len(candidates) != 1:
            continue
        support, support_tokens = candidates[0]
        match.update(
            {
                "status": "matched",
                "scene_object_id": support.get("scene_object_id"),
                "scene_name": support.get("scene_name"),
                "scene_object_type": support.get("scene_object_type"),
                "category_token_overlap": sorted(
                    set(match.get("required_category_tokens", []))
                    & _tokens(item.category)
                ),
                "identity_token_overlap": sorted(destination_tokens & support_tokens),
                "structural_support_fallback": True,
                "asset_resolution_source": "declared_support_region",
                "role": item.role,
                "region_of_scene_object_id": support.get("scene_object_id"),
                "sdf_path": support.get("sdf_path"),
                "sdf_sha256": support.get("sdf_sha256"),
            }
        )


def certify_task_inventory(
    *, task: TaskInstructionSpec, house_state: Path
) -> dict[str, Any]:
    inventory = _scene_inventory(house_state)
    matches: list[dict[str, Any]] = []
    requested: list[dict[str, Any]] = []
    # A declared articulation component is a hard constraint on its host asset:
    # the reference executor must be able to drive the declared moving joint.
    # Selecting a jointless object with matching tokens makes the whole case
    # uncertifiable, so joint qualification is resolved from the asset the
    # canonical importer would consume and cached per inventory entry.
    articulation_hosts = {
        component.host_object_id: component for component in task.articulation_components
    }
    joint_type_cache: dict[int, set[str]] = {}

    def _scene_joint_types(index: int) -> set[str]:
        cached = joint_type_cache.get(index)
        if cached is None:
            scene_object = inventory[index]
            cached = set()
            for sdf_value in scene_object["sdf_path_candidates"]:
                resolved = _resolve_scene_asset_path(
                    house_state=house_state,
                    room_id=scene_object["room_id"],
                    value=str(sdf_value),
                )
                if resolved is not None:
                    cached = _sdf_articulation_joint_types(resolved)
                    break
            joint_type_cache[index] = cached
        return cached

    for item in task.objects:
        identity = _tokens(
            f"{item.object_id} {item.category} {item.visual_description}"
        )
        # Object ids often contain appearance words (for example blue_bin).
        # They are identity evidence, never category evidence: otherwise a
        # blue cube can incorrectly satisfy a blue sorting-bin request.
        category_tokens = _physical_category_tokens(item)
        category_match_tokens = category_tokens | {
            alias for token in category_tokens
            for alias in PHYSICAL_CATEGORY_ALIASES.get(token, ())
        }
        articulation_component = articulation_hosts.get(item.object_id)
        candidate_scores: list[tuple[int, int]] = []
        for index, scene_object in enumerate(inventory):
            tokens = set(inventory[index]["tokens"])
            category_overlap = len(tokens & category_match_tokens)
            total_overlap = len(tokens & identity)
            joint_qualified = bool(
                articulation_component is not None
                and articulation_component.joint_type in _scene_joint_types(index)
            )
            # Joint type ranks compatible objects; it cannot establish identity.
            if category_overlap == 0:
                if not (
                    item.role == "support"
                    and scene_object["object_type"] == "furniture"
                    and scene_object["has_support_surfaces"]
                ):
                    continue
                # SceneCode may omit the requested noun from a generated
                # primary support's final name. A furniture object with an
                # explicit load-bearing support surface remains structurally
                # eligible; identity overlap still ranks competing supports.
                category_overlap = 1
            expected_type = (
                "furniture" if item.role == "support" else "manipuland"
            )
            role_type_match = int(scene_object["object_type"] == expected_type)
            # Matched cardinality is optimized separately below. Within one
            # complete assignment, category coverage dominates appearance,
            # while role/type consistency breaks otherwise ambiguous ties.
            score = (
                10_000 * category_overlap
                + 100 * total_overlap
                + 10 * role_type_match
                + 50_000 * int(joint_qualified)
            )
            candidate_scores.append((index, score))
        requested.append(
            {
                "item": item,
                "identity": identity,
                "category_tokens": category_tokens,
                "category_match_tokens": category_match_tokens,
                "candidates": tuple(sorted(candidate_scores)),
            }
        )

    # Solve the one-to-one assignment globally. A greedy pass can consume the
    # only compatible bin for the wrong colored request and then misclassify a
    # cube as the remaining bin. The task object count is deliberately small;
    # memoized bitmask search is exact and deterministic for this inventory.
    order = tuple(
        sorted(
            range(len(requested)),
            key=lambda index: (
                len(requested[index]["candidates"]),
                -len(requested[index]["identity"]),
                requested[index]["item"].object_id,
            ),
        )
    )

    @lru_cache(maxsize=None)
    def assign(position: int, used_mask: int) -> tuple[int, int, tuple[int, ...]]:
        if position == len(order):
            return 0, 0, ()
        request = requested[order[position]]
        # Missing is a real assignment option. Maximizing matched cardinality
        # first prevents a high-overlap item from stealing another item's only
        # compatible asset.
        missing_matched, missing_score, missing_tail = assign(
            position + 1, used_mask
        )
        best = (missing_matched, missing_score, (-1, *missing_tail))
        for candidate_index, candidate_score in request["candidates"]:
            bit = 1 << candidate_index
            if used_mask & bit:
                continue
            child_matched, child_score, child_tail = assign(
                position + 1, used_mask | bit
            )
            option = (
                child_matched + 1,
                child_score + candidate_score,
                (candidate_index, *child_tail),
            )
            if option[:2] > best[:2] or (
                option[:2] == best[:2] and option[2] < best[2]
            ):
                best = option
        return best

    _, _, ordered_assignment = assign(0, 0)
    assignment = {
        request_index: candidate_index
        for request_index, candidate_index in zip(order, ordered_assignment, strict=True)
    }

    for request_index, request in enumerate(requested):
        item = request["item"]
        identity = request["identity"]
        category_tokens = request["category_tokens"]
        index = assignment[request_index]
        if index < 0:
            matches.append(
                {
                    "task_object_id": item.object_id,
                    "category": item.category,
                    "status": "missing",
                    "required_tokens": sorted(identity),
                    "required_category_tokens": sorted(category_tokens),
                }
            )
            continue
        scene_object = inventory[index]
        scene_tokens = set(scene_object["tokens"])
        category_overlap = sorted(scene_tokens & category_tokens)
        identity_overlap = sorted(scene_tokens & identity)
        sdf_path = None
        resolution_source = None
        for path_index, sdf_value in enumerate(scene_object["sdf_path_candidates"]):
            sdf_path = _resolve_scene_asset_path(
                house_state=house_state,
                room_id=scene_object["room_id"],
                value=str(sdf_value),
            )
            if sdf_path is not None:
                resolution_source = (
                    "house_state" if path_index == 0 else "live_object_registry"
                )
                break
        sdf_ok = sdf_path is not None
        articulation_component = articulation_hosts.get(item.object_id)
        host_joint_types = sorted(_scene_joint_types(index)) if sdf_ok else []
        match: dict[str, Any] = {
            "task_object_id": item.object_id,
            "category": item.category,
            "role": item.role,
            "status": "matched" if sdf_ok else "matched_without_sdf",
            "scene_object_id": scene_object["object_id"],
            "scene_name": scene_object["name"],
            "scene_object_type": scene_object["object_type"],
            "category_token_overlap": category_overlap,
            "identity_token_overlap": identity_overlap,
            "structural_support_fallback": bool(
                item.role == "support"
                and not category_overlap
                and scene_object["object_type"] == "furniture"
                and scene_object["has_support_surfaces"]
            ),
            "asset_resolution_source": resolution_source,
            "sdf_path": str(sdf_path) if sdf_path is not None else None,
            "sdf_sha256": sha256_file(sdf_path) if sdf_ok else None,
        }
        alias_overlap = scene_tokens & (request["category_match_tokens"] - category_tokens)
        if alias_overlap:
            match["category_alias_token_overlap"] = sorted(alias_overlap)
        if articulation_component is not None:
            match["articulation_component_id"] = articulation_component.component_id
            match["articulation_required_joint_type"] = articulation_component.joint_type
            match["articulation_host_joint_types"] = host_joint_types
            match["articulation_joint_qualified"] = (
                articulation_component.joint_type in host_joint_types
            )
        matches.append(match)
    _bind_declared_support_regions(matches=matches, inventory=inventory, task=task)
    passed = all(item["status"] == "matched" for item in matches)
    return {
        "schema_version": "robotics_scenecode_inventory.v1",
        "case_id": task.case_id,
        "house_state": str(house_state),
        "house_state_sha256": sha256_file(house_state),
        "passed": passed,
        "task_object_count": len(task.objects),
        "matched_object_count": sum(item["status"] == "matched" for item in matches),
        "matches": sorted(matches, key=lambda item: item["task_object_id"]),
        "scene_inventory_count": len(inventory),
    }


def restore_validated_fallen_task_objects(
    *,
    task: TaskInstructionSpec,
    house_state: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Restore a required asset removed only by SceneCode's final drop test.

    SceneCode can successfully generate and validate an asset, place it in the
    authored scene, and then remove it when the final feasibility simulation
    drops it through a support surface. The reference executor owns the later
    physical placement, so we may recover that already-validated visual asset
    without regenerating the room. Recovery is deliberately strict: the object
    must appear in a prior live render snapshot, have a resolvable SDF, have an
    exact validation/conversion record for the selected generated asset, and
    have an explicit ``Removing fallen manipuland`` record. Rejected candidates
    and registry-only objects remain ineligible.
    """

    missing_ids = {
        match["task_object_id"]
        for match in inventory.get("matches", [])
        if match.get("status") == "missing"
    }
    report: dict[str, Any] = {
        "schema_version": "robotics_scenecode_fallen_task_recovery.v1",
        "case_id": task.case_id,
        "restored": [],
        "skipped": [],
        "changed": False,
    }
    if not missing_ids:
        return report

    scene_root = house_state.parent.parent
    # SceneCode currently records mesh validation/conversion in the room log,
    # while launchers may also emit scene.log or main.log. Recovery evidence
    # must cover the log that actually owns the asset; an absent scene.log must
    # not make an otherwise validated fallen task asset unrecoverable.
    scene_log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (scene_root / "scene.log", scene_root / "main.log")
        if path.is_file()
    )
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    rooms = payload.get("rooms", {})
    task_by_id = {item.object_id: item for item in task.objects}

    for task_object_id in sorted(missing_ids):
        item = task_by_id[task_object_id]
        category_tokens = _tokens(item.category)
        object_id_tokens = _tokens(item.object_id)
        identity_tokens = _tokens(
            f"{item.object_id} {item.category} {item.visual_description}"
        )
        candidates: dict[str, tuple[int, str, dict[str, Any], Path]] = {}
        for room_id in sorted(rooms):
            room_root = scene_root / f"room_{room_id}"
            room_log = room_root / "room.log"
            room_log_text = (
                room_log.read_text(encoding="utf-8", errors="replace")
                if room_log.is_file()
                else ""
            )
            current_objects = rooms[room_id].get("objects", {})
            for snapshot in sorted(room_root.rglob("scene_state.json")):
                snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
                for object_id, record in snapshot_payload.get("objects", {}).items():
                    if object_id in current_objects or not isinstance(record, dict):
                        continue
                    if f"Removing fallen manipuland {object_id}" not in room_log_text:
                        continue
                    text = " ".join(
                        (
                            str(object_id),
                            str(record.get("name", "")),
                            str(record.get("description", "")),
                        )
                    )
                    tokens = _tokens(text)
                    category_overlap = len(tokens & category_tokens)
                    # The selected task description also contains state and
                    # relation language (e.g. "contains the apple after
                    # release"); those words are not expected in a physical
                    # asset description. Require the physical category and
                    # task-object identity, then leave detailed appearance and
                    # geometry qualification to the downstream task checks.
                    object_identity = object_id_tokens - category_tokens
                    if (
                        category_overlap == 0
                        or (object_identity and not (tokens & object_identity))
                    ):
                        continue
                    candidate_description = str(record.get("description", "")).strip()
                    if not candidate_description:
                        continue
                    validation_log_text = "\n".join(
                        value
                        for value in (scene_log_text, room_log_text)
                        if value
                    )
                    if (
                        f"validation passed for '{candidate_description}'"
                        not in validation_log_text
                        or f"Successfully converted asset: '{candidate_description}'"
                        not in validation_log_text
                    ):
                        continue
                    sdf_path = _resolve_scene_asset_path(
                        house_state=house_state,
                        room_id=str(room_id),
                        value=record.get("sdf_path"),
                    )
                    if sdf_path is None:
                        continue
                    score = 10_000 * category_overlap + 100 * len(
                        tokens & identity_tokens
                    )
                    previous = candidates.get(str(object_id))
                    if previous is None or score > previous[0] or (
                        score == previous[0] and str(snapshot) > str(previous[3])
                    ):
                        candidates[str(object_id)] = (
                            score,
                            str(room_id),
                            record,
                            snapshot,
                        )
        if not candidates:
            report["skipped"].append(
                {
                    "task_object_id": task_object_id,
                    "reason": "no previously-live validated fallen asset is recoverable",
                }
            )
            continue

        ranked = sorted(
            candidates.items(),
            key=lambda value: (-value[1][0], value[0]),
        )
        object_id, (score, room_id, record, snapshot) = ranked[0]
        room = rooms[room_id]
        support_records = [
            value
            for value in room.get("objects", {}).values()
            if isinstance(value, dict) and value.get("support_surfaces")
        ]
        if len(support_records) != 1:
            report["skipped"].append(
                {
                    "task_object_id": task_object_id,
                    "reason": "recovery requires exactly one declared support object",
                }
            )
            continue
        surface = support_records[0]["support_surfaces"][0]
        support_z = float(surface["transform"]["translation"][2])
        bbox_min = record.get("bbox_min")
        transform = record.get("transform")
        if (
            not isinstance(bbox_min, list)
            or len(bbox_min) != 3
            or not isinstance(transform, dict)
            or not isinstance(transform.get("translation"), list)
            or len(transform["translation"]) != 3
        ):
            report["skipped"].append(
                {
                    "task_object_id": task_object_id,
                    "reason": "fallen asset lacks placement bounds",
                }
            )
            continue

        restored = json.loads(json.dumps(record))
        restored["transform"]["translation"][2] = (
            support_z - float(bbox_min[2]) + 0.002
        )
        metadata = dict(restored.get("metadata") or {})
        metadata["worldolympiad_recovery"] = {
            "policy": "validated_fallen_task_object_support_reprojection",
            "task_object_id": task_object_id,
            "source_snapshot": str(snapshot.relative_to(scene_root)),
            "support_surface_id": surface.get("surface_id"),
        }
        restored["metadata"] = metadata
        room.setdefault("objects", {})[object_id] = restored
        report["restored"].append(
            {
                "task_object_id": task_object_id,
                "scene_object_id": object_id,
                "room_id": room_id,
                "source_snapshot": str(snapshot.relative_to(scene_root)),
                "semantic_score": score,
                "support_surface_id": surface.get("surface_id"),
                "compiled_translation": restored["transform"]["translation"],
                "sdf_path": restored.get("sdf_path"),
            }
        )

    if report["restored"]:
        temporary = house_state.with_suffix(f"{house_state.suffix}.repair.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, house_state)
        report["changed"] = True
    return report


PREFILLED_COMPOSITE_TYPE = "filled_container"


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        return None


def _quaternion4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        return None


def _quaternion_conjugate(quaternion: list[float]) -> list[float]:
    w, x, y, z = quaternion
    return [w, -x, -y, -z]


def _quaternion_rotate(quaternion: list[float], vector: list[float]) -> list[float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _load_scene_asset_registry(
    *, house_state: Path, room_id: str
) -> dict[str, dict[str, Any]]:
    """Load every generated-asset registry record for one room, keyed by asset id."""

    room_root = house_state.parent.parent / f"room_{room_id}"
    registry: dict[str, dict[str, Any]] = {}
    generated_root = room_root / "generated_assets"
    if not generated_root.is_dir():
        return registry
    for path in sorted(generated_root.rglob("asset_registry.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if isinstance(value, dict):
                registry[str(value.get("object_id", key))] = value
    return registry


def _match_composite_asset_to_task_object(
    asset: dict[str, Any], candidates: list[Any]
) -> str | None:
    """Match one recorded composite member to exactly one declared task object."""

    text = " ".join(
        str(asset.get(field, "")) for field in ("asset_id", "name", "description")
    )
    tokens = _tokens(text) | {
        token[:-1] for token in _tokens(text) if len(token) > 3 and token.endswith("s")
    }
    if not tokens:
        return None
    scored: list[tuple[int, int, str]] = []
    for item in candidates:
        # Composite members must use the same physical category gate as live
        # inventory objects. Legacy role labels (container/contents) are not
        # physical nouns and cannot identify a cup or bead asset on their own.
        category_tokens = _physical_category_tokens(item)
        category_tokens |= {
            token[:-1]
            for token in category_tokens
            if len(token) > 3 and token.endswith("s")
        }
        category_overlap = len(tokens & category_tokens)
        if category_overlap == 0:
            continue
        identity_tokens = _tokens(
            f"{item.object_id} {item.category} {item.visual_description}"
        )
        identity_tokens |= {
            token[:-1]
            for token in identity_tokens
            if len(token) > 3 and token.endswith("s")
        }
        identity_overlap = len(tokens & identity_tokens)
        scored.append((category_overlap, identity_overlap, item.object_id))
    if not scored:
        return None
    scored.sort(key=lambda value: (-value[0], -value[1], value[2]))
    if len(scored) > 1 and scored[1][:2] == scored[0][:2]:
        return None
    return scored[0][2]


def _validated_registered_asset(
    *,
    asset_id: str,
    registry_entry: dict[str, Any] | None,
    declared_path: str | None,
    scene_log_text: str,
) -> str | None:
    """Accept only a registry asset with a real SDF and explicit conversion evidence.

    SceneCode merges the declared container and its declared manipuland into one
    geometry-less ``filled_container`` composite when the manipuland agent calls
    its ``fill_container`` tool. That pre-solves the task. The composite record
    still names the real per-asset SDFs it was built from, so the unsolved state
    can be restored from those exact validated assets instead of from the merged
    composite or a regenerated room.
    """

    if not isinstance(registry_entry, dict) or registry_entry.get("object_id") != asset_id:
        return None
    sdf_value = registry_entry.get("sdf_path")
    if not sdf_value or not Path(str(sdf_value)).is_file():
        return None
    if declared_path and str(declared_path) != str(sdf_value):
        return None
    description = str(registry_entry.get("description", "")).strip()
    if not description:
        return None
    if f"Successfully converted asset: '{description}'" not in scene_log_text:
        return None
    return description


def _declared_contents_outside(initial_state: str) -> bool:
    """Return true when the typed initial state keeps contents out of the container.

    ``transfer`` declares the cubes fully inside the small container, so the
    composite arrangement is the correct initial state and must be expanded in
    place. ``pick_place`` declares the bottle is not inside the basket, so the
    members must be separated onto the support instead.

    The decision is clause-local. An explicit, non-negated "inside" declaration
    for the contents always wins over an unrelated "beside"/"separate"/"apart"
    clause that describes where the *container* sits relative to some other
    object; otherwise a room statement such as "the container sits beside the
    bin, and the cubes are fully inside it" would wrongly spill the contents.
    """

    text = str(initial_state or "").strip().lower()
    if not text:
        return False
    for clause in re.split(r"[.;,]", text):
        if "inside" not in clause and " within " not in clause:
            continue
        if any(
            negation in clause
            for negation in ("not inside", "n't inside", "outside", "not in")
        ):
            continue
        return False
    if _tokens(text) & {"outside", "beside", "separate", "separated", "apart"}:
        return True
    return "not inside" in text


def _separated_manipuland_record(
    *,
    asset_id: str,
    registry_entry: dict[str, Any],
    transform: dict[str, Any],
    placement_info: dict[str, Any] | None,
    recovery_note: dict[str, Any],
    internal_model_pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in (registry_entry.get("metadata") or {}).items()
        if key != "worldolympiad_recovery"
    }
    metadata["worldolympiad_recovery"] = recovery_note
    return {
        "object_id": asset_id,
        "object_type": "manipuland",
        "name": str(registry_entry.get("name") or asset_id),
        "description": str(registry_entry.get("description") or ""),
        "transform": transform,
        "internal_model_pose": internal_model_pose
        or {
            "translation": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "geometry_path": registry_entry.get("geometry_path"),
        "sdf_path": registry_entry.get("sdf_path"),
        "image_path": None,
        "support_surfaces": [],
        "placement_info": placement_info,
        "metadata": metadata,
        "bbox_min": [0.0, 0.0, 0.0],
        "bbox_max": [0.0, 0.0, 0.0],
        "immutable": False,
        "scale_factor": float(registry_entry.get("scale_factor") or 1.0),
    }


def _composite_support_surface(
    *, objects: dict[str, Any], container_placement: Any
) -> dict[str, Any] | None:
    supports = [
        value
        for value in objects.values()
        if isinstance(value, dict) and value.get("support_surfaces")
    ]
    if len(supports) != 1:
        return None
    surfaces = supports[0]["support_surfaces"]
    surface_id = (
        container_placement.get("parent_surface_id")
        if isinstance(container_placement, dict)
        else None
    )
    if surface_id is not None:
        surfaces = [
            surface for surface in surfaces if surface.get("surface_id") == surface_id
        ] or surfaces
    if len(surfaces) != 1:
        return None
    return surfaces[0]


def separate_prefilled_container_task_objects(
    *,
    task: TaskInstructionSpec,
    house_state: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """Restore the declared unsolved initial state from a pre-filled composite.

    SceneCode's manipuland designer may call ``fill_container`` and place the
    declared manipuland into the declared destination, producing a single
    ``filled_container`` composite whose own ``sdf_path`` is empty. That
    pre-solves the reference task, so the composite can never be certified or
    imported. This recovery re-separates the composite's own recorded
    constituent assets: the container keeps its authored placement, and each
    declared manipuland is restored upright beside it on the same support
    surface. It never relabels the merged composite as two task assets; both
    records must resolve to distinct, already-validated SDFs named by the
    composite metadata and by the live generation registry.
    """

    report: dict[str, Any] = {
        "schema_version": "robotics_scenecode_prefilled_separation.v1",
        "case_id": task.case_id,
        "separated": [],
        "skipped": [],
        "changed": False,
    }
    manipulands = [item for item in task.objects if item.role == "manipuland"]
    destinations = [item for item in task.objects if item.role == "destination"]
    if not manipulands or not destinations:
        return report
    manipuland_ids = {item.object_id for item in manipulands}
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    scene_root = house_state.parent.parent
    scene_log = scene_root / "scene.log"
    scene_log_text = (
        scene_log.read_text(encoding="utf-8", errors="replace")
        if scene_log.is_file()
        else ""
    )
    for room_id in sorted(payload.get("rooms", {})):
        room = payload["rooms"][room_id]
        objects = room.get("objects")
        if not isinstance(objects, dict):
            continue
        registry = _load_scene_asset_registry(house_state=house_state, room_id=room_id)
        for object_id in sorted(objects):
            record = objects[object_id]
            metadata = record.get("metadata") if isinstance(record, dict) else None
            if (
                not isinstance(metadata, dict)
                or metadata.get("composite_type") != PREFILLED_COMPOSITE_TYPE
            ):
                continue
            container_asset = metadata.get("container_asset")
            fill_assets = metadata.get("fill_assets")
            if (
                not isinstance(container_asset, dict)
                or not isinstance(fill_assets, list)
                or not fill_assets
            ):
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "composite lacks complete constituent asset records",
                    }
                )
                continue
            # The container may itself be a declared manipuland (a pouring
            # container that the robot grasps) rather than a destination bin, so
            # resolve it against every declared object. Repeated instances of one
            # member asset (three cubes sharing a single asset id) all map to the
            # same declared object, so only the container must stay distinct from
            # every fill member.
            container_task_id = _match_composite_asset_to_task_object(
                container_asset, list(task.objects)
            )
            fill_task_ids = [
                _match_composite_asset_to_task_object(asset, manipulands)
                for asset in fill_assets
            ]
            mapped_ids = [container_task_id, *fill_task_ids]
            if any(value is None for value in mapped_ids) or (
                container_task_id in fill_task_ids
            ):
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "composite members do not map to declared task objects",
                    }
                )
                continue
            if manipuland_ids - set(mapped_ids):
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "composite does not cover every declared manipuland",
                    }
                )
                continue
            container_id = str(container_asset.get("asset_id") or "")
            container_entry = registry.get(container_id)
            container_description = _validated_registered_asset(
                asset_id=container_id,
                registry_entry=container_entry,
                declared_path=container_asset.get("sdf_path"),
                scene_log_text=scene_log_text,
            )
            fill_entries: list[tuple[str, dict[str, Any], str]] = []
            evidence_ok = container_description is not None
            for asset in fill_assets:
                fill_id = str(asset.get("asset_id") or "")
                entry = registry.get(fill_id)
                description = _validated_registered_asset(
                    asset_id=fill_id,
                    registry_entry=entry,
                    declared_path=asset.get("sdf_path"),
                    scene_log_text=scene_log_text,
                )
                if description is None or not isinstance(entry, dict):
                    evidence_ok = False
                    break
                fill_entries.append((fill_id, entry, description))
            if not evidence_ok or container_entry is None:
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "constituent assets lack distinct validated SDF evidence",
                    }
                )
                continue
            container_sdf = Path(str(container_entry["sdf_path"]))
            if any(container_sdf.samefile(str(entry["sdf_path"])) for _, entry, _ in fill_entries):
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "container and manipuland refer to the same SDF asset",
                    }
                )
                continue
            # A distinct member asset must resolve to a distinct declared
            # object, while repeated instances of one member asset (three cubes
            # sharing an asset id) are legal as long as they are genuinely
            # distinct placements; an exact duplicate entry is corrupt.
            declared_by_member: dict[str, str] = {}
            seen_instances: dict[str, set[str]] = {}
            member_mapping_ambiguous = False
            for fill_id, asset, task_object_id in zip(
                (value[0] for value in fill_entries), fill_assets, fill_task_ids, strict=True
            ):
                if (
                    declared_by_member.setdefault(fill_id, task_object_id)
                    != task_object_id
                ):
                    member_mapping_ambiguous = True
                placement = json.dumps(asset.get("transform"), sort_keys=True)
                if placement in seen_instances.setdefault(fill_id, set()):
                    member_mapping_ambiguous = True
                seen_instances[fill_id].add(placement)
            if len(set(declared_by_member.values())) != len(declared_by_member):
                member_mapping_ambiguous = True
            if member_mapping_ambiguous:
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "composite members do not map to declared task objects",
                    }
                )
                continue
            # SceneCode reuses one asset id for repeated instances (three cubes
            # share blue_wooden_cube_0), so expanded members need distinct scene
            # ids; the first instance keeps its authored id.
            member_ids: dict[int, str] = {-1: container_id}
            used_ids = {container_id}
            for member_index, (fill_id, _entry, _description) in enumerate(fill_entries):
                candidate = fill_id
                suffix = 1
                while candidate in used_ids:
                    candidate = f"{fill_id}_instance_{suffix}"
                    suffix += 1
                used_ids.add(candidate)
                member_ids[member_index] = candidate
            new_ids = list(member_ids.values())
            if any(value in objects for value in new_ids):
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "separated asset ids are not distinct scene objects",
                    }
                )
                continue
            if not _declared_contents_outside(task.initial_state):
                # The typed initial state keeps the contents inside the
                # container. Expand the geometry-less composite in place into
                # its own recorded members, preserving each member's authored
                # transform and internal pose; nothing is re-solved or moved.
                container_record = _separated_manipuland_record(
                    asset_id=member_ids[-1],
                    registry_entry=container_entry,
                    transform={
                        "translation": list(container_translation)
                        if (container_translation := _vector3(
                            (container_asset.get("transform") or {}).get("translation")
                        ))
                        else [0.0, 0.0, 0.0],
                        "rotation_wxyz": list(container_rotation)
                        if (container_rotation := _quaternion4(
                            (container_asset.get("transform") or {}).get("rotation_wxyz")
                        ))
                        else [1.0, 0.0, 0.0, 0.0],
                    },
                    internal_model_pose=container_asset.get("internal_model_pose"),
                    placement_info=record.get("placement_info"),
                    recovery_note={
                        "policy": "filled_container_member_expansion",
                        "source_composite": object_id,
                        "role": "container",
                        "task_object_id": container_task_id,
                        "asset_description": container_description,
                    },
                )
                member_records = [container_record]
                for fill_index, (
                    asset,
                    (fill_id, entry, description),
                    task_object_id,
                ) in enumerate(zip(fill_assets, fill_entries, fill_task_ids, strict=True)):
                    member_records.append(
                        _separated_manipuland_record(
                            asset_id=member_ids[fill_index],
                            registry_entry=entry,
                            transform=asset.get("transform")
                            or {
                                "translation": [0.0, 0.0, 0.0],
                                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                            },
                            internal_model_pose=asset.get("internal_model_pose"),
                            placement_info={
                                "parent_surface_id": (
                                    (record.get("placement_info") or {}).get(
                                        "parent_surface_id"
                                    )
                                ),
                                "placement_method": "composite_member_expansion",
                            },
                            recovery_note={
                                "policy": "filled_container_member_expansion",
                                "source_composite": object_id,
                                "role": "contents",
                                "task_object_id": task_object_id,
                                "asset_description": description,
                            },
                        )
                    )
                del objects[object_id]
                for member in member_records:
                    objects[member["object_id"]] = member
                report["separated"].append(
                    {
                        "room_id": room_id,
                        "source_composite": object_id,
                        "mode": "member_expansion",
                        "assets": [member["object_id"] for member in member_records],
                    }
                )
                continue
            surface = _composite_support_surface(
                objects=objects, container_placement=record.get("placement_info")
            )
            surface_transform = (surface or {}).get("transform") or {}
            surface_translation = _vector3(surface_transform.get("translation"))
            surface_rotation = _quaternion4(surface_transform.get("rotation_wxyz"))
            surface_min = _vector3((surface or {}).get("bounding_box_min"))
            surface_max = _vector3((surface or {}).get("bounding_box_max"))
            container_transform = container_asset.get("transform") or {}
            container_translation = _vector3(container_transform.get("translation"))
            container_rotation = _quaternion4(container_transform.get("rotation_wxyz"))
            if None in (
                surface_translation,
                surface_rotation,
                surface_min,
                surface_max,
                container_translation,
                container_rotation,
            ):
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "composite placement lacks a measurable support frame",
                    }
                )
                continue
            bbox_min = _vector3(record.get("bbox_min")) or [0.0, 0.0, 0.0]
            bbox_max = _vector3(record.get("bbox_max")) or [0.0, 0.0, 0.0]
            container_radius = max(
                abs(bbox_min[0]),
                abs(bbox_max[0]),
                abs(bbox_min[1]),
                abs(bbox_max[1]),
            )
            spacing = container_radius + 0.18
            local = _quaternion_rotate(
                _quaternion_conjugate(surface_rotation),
                [
                    container_translation[index] - surface_translation[index]
                    for index in range(3)
                ],
            )
            margin = 0.05
            assigned: list[tuple[float, float]] = []
            separated: list[dict[str, Any]] = []
            fill_task_by_asset = {
                fill_id: task_id
                for fill_id, task_id in zip(
                    (value[0] for value in fill_entries), fill_task_ids, strict=True
                )
            }
            for fill_index, (fill_id, entry, description) in enumerate(fill_entries):
                member_id = member_ids[fill_index]
                distance = spacing * (fill_index + 1)
                chosen = None
                for candidate in (
                    (local[0] + distance, local[1]),
                    (local[0] - distance, local[1]),
                    (local[0], local[1] + distance),
                    (local[0], local[1] - distance),
                ):
                    if not (
                        surface_min[0] + margin <= candidate[0] <= surface_max[0] - margin
                        and surface_min[1] + margin
                        <= candidate[1]
                        <= surface_max[1] - margin
                    ):
                        continue
                    if any(
                        (
                            (candidate[0] - other[0]) ** 2
                            + (candidate[1] - other[1]) ** 2
                        )
                        < spacing**2
                        for other in assigned
                    ):
                        continue
                    chosen = candidate
                    break
                if chosen is None:
                    separated = []
                    break
                assigned.append(chosen)
                world = _quaternion_rotate(
                    surface_rotation, [chosen[0], chosen[1], local[2]]
                )
                world_translation = [
                    surface_translation[index] + world[index] for index in range(3)
                ]
                task_object_id = fill_task_by_asset[fill_id]
                separated.append(
                    _separated_manipuland_record(
                        asset_id=member_id,
                        registry_entry=entry,
                        transform={
                            "translation": world_translation,
                            "rotation_wxyz": list(container_rotation),
                        },
                        placement_info={
                            "parent_surface_id": (surface or {}).get("surface_id"),
                            "position_2d": [chosen[0], chosen[1]],
                            "rotation_2d": 0.0,
                            "placement_method": "reference_unsolved_separation",
                        },
                        recovery_note={
                            "policy": "prefilled_container_unsolved_separation",
                            "source_composite": object_id,
                            "role": "manipuland",
                            "task_object_id": task_object_id,
                            "asset_description": description,
                            "support_plane_z": container_translation[2],
                        },
                    )
                )
            if not separated:
                report["skipped"].append(
                    {
                        "scene_object_id": object_id,
                        "reason": "no reachable separated placement exists on the support",
                    }
                )
                continue
            container_record = _separated_manipuland_record(
                asset_id=member_ids[-1],
                registry_entry=container_entry,
                transform={
                    "translation": list(container_translation),
                    "rotation_wxyz": list(container_rotation),
                },
                placement_info=record.get("placement_info"),
                recovery_note={
                    "policy": "prefilled_container_unsolved_separation",
                    "source_composite": object_id,
                    "role": "destination",
                    "task_object_id": container_task_id,
                    "asset_description": container_description,
                },
            )
            del objects[object_id]
            objects[container_id] = container_record
            for item in separated:
                objects[item["object_id"]] = item
            report["separated"].append(
                {
                    "room_id": room_id,
                    "source_composite": object_id,
                    "container": {
                        "asset_id": container_id,
                        "task_object_id": container_task_id,
                        "transform": container_record["transform"],
                    },
                    "manipulands": [
                        {
                            "asset_id": item["object_id"],
                            "transform": item["transform"],
                        }
                        for item in separated
                    ],
                    "support_surface_id": (surface or {}).get("surface_id"),
                    "spacing_m": spacing,
                }
            )
    if report["separated"]:
        temporary = house_state.with_suffix(f"{house_state.suffix}.repair.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, house_state)
        report["changed"] = True
    return report


def _write_failure_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    from .research_failures import capture_failure
    for parent in path.parents:
        if parent.name == "robotics_scenecode_failures":
            capture_failure(parent.parent, path)
            break


def can_retry_inventory_at_manipuland_stage(
    *, task: TaskInstructionSpec, inventory: dict[str, Any]
) -> bool:
    """Return true only when every failed task asset belongs to the late stage."""

    roles = {item.object_id: item.role for item in task.objects}
    categories = {item.object_id: _tokens(item.category) for item in task.objects}
    structural_support_categories = {
        "table", "workbench", "worktable", "floor", "counter", "shelf", "cabinet",
    }
    failed_ids = {
        str(match.get("task_object_id"))
        for match in inventory.get("matches", [])
        if match.get("status") != "matched"
        or not (match.get("sdf_path") or match.get("sdf_present"))
    }
    def authored_in_late_stage(object_id: str) -> bool:
        role = roles.get(object_id)
        if role in {"manipuland", "destination"}:
            return True
        # A source container supports its contents semantically, but SceneCode
        # authors it with the other interactive task objects, not as room
        # furniture.  Keep true architectural supports in the earlier stage.
        return (
            role == "support"
            and not (categories.get(object_id, set()) & structural_support_categories)
        )

    return bool(failed_ids) and all(authored_in_late_stage(x) for x in failed_ids)


def _arbitrate_inventory_recovery(
    *, task: TaskInstructionSpec, failure_path: Path,
    recovery: Path, partial_recovery: Path,
) -> bool:
    if not failure_path.is_file():
        return False
    try:
        prior = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if prior.get("failed_stage") != "task_asset_inventory":
        return False
    candidate_state = recovery / "scene_run/scene_000/combined_house/house_state.json"
    routing_inventory = (
        certify_task_inventory(task=task, house_state=candidate_state)
        if candidate_state.is_file() else prior
    )
    if routing_inventory.get("passed"):
        return False
    if routing_inventory.get("matches") and not can_retry_inventory_at_manipuland_stage(
        task=task, inventory=routing_inventory,
    ):
        # A late-stage checkpoint cannot regenerate missing room furniture.
        for candidate in (recovery, partial_recovery):
            if candidate.exists():
                archived = candidate.with_name(f"{candidate.name}.superseded_{time.time_ns()}")
                candidate.rename(archived)
                print(f"Archived inventory-invalid early-stage checkpoint: {archived}", flush=True)
        return False
    return (
        recovery.exists() and partial_recovery.exists()
        and Path(prior.get("recoverable_candidate", "")).resolve() == partial_recovery.resolve()
    )


def generate_case_with_bounded_inventory_resume(**kwargs: Any) -> Path:
    """Retry once from a newly preserved pre-manipuland checkpoint."""
    try:
        return generate_case(**kwargs)
    except ValueError:
        root = Path(kwargs["output_root"]).resolve()
        batch_id, case_id = kwargs["batch_id"], kwargs["case_id"]
        failure = root / "robotics_scenecode_failures" / batch_id / f"{case_id}.json"
        partial = partial_generation_recovery_path(
            root=root, batch_id=batch_id, case_id=case_id,
        )
        if not failure.is_file() or not partial.is_dir():
            raise
        report = json.loads(failure.read_text(encoding="utf-8"))
        if (
            report.get("classification") != "asset_geometry"
            or report.get("failed_stage") != "task_asset_inventory"
            or Path(report.get("recoverable_candidate", "")).resolve() != partial
        ):
            raise
        return generate_case(**kwargs)


def write_final_inventory_binding(
    *, inventory: dict[str, Any], inventory_path: Path, house_state: Path
) -> None:
    """Bind the inventory certificate to the final rewritten house state.

    Publication rewrites staging paths inside ``house_state.json``.  The
    certificate must therefore be serialized after that rewrite; otherwise its
    embedded digest describes a pre-publication file that no longer exists.
    """

    inventory["house_state_sha256"] = sha256_file(house_state)
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_case(
    *, output_root: Path, batch_id: str, case_id: str
) -> tuple[TaskInstructionSpec, Path, Path]:
    batch_path = output_root / "robotics_task_batches" / batch_id / "batch_manifest.json"
    batch = RoboticsTaskBatchManifest.model_validate_json(
        batch_path.read_text(encoding="utf-8")
    )
    rows = [item for item in batch.cases if item.case_id == case_id]
    if len(rows) != 1:
        raise ValueError(f"case is not present exactly once in batch {batch_id}: {case_id}")
    row = rows[0]
    task_path = (output_root / row.task_path).resolve()
    if sha256_file(task_path) != row.task_sha256:
        raise ValueError("selected task digest changed")
    task = TaskInstructionSpec.model_validate_json(task_path.read_text(encoding="utf-8"))
    prompt_root = output_root / "robotics_scene_prompts" / batch_id / case_id
    prompt_bundle = prompt_root / "scene_prompt_bundle.json"
    prompt_manifest_path = (
        output_root / "robotics_scene_prompts" / batch_id / "scene_prompt_manifest.json"
    )
    prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
    prompt_rows = [item for item in prompt_manifest.get("cases", []) if item.get("case_id") == case_id]
    if len(prompt_rows) != 1 or sha256_file(prompt_bundle) != prompt_rows[0]["bundle_sha256"]:
        raise ValueError("SceneCode prompt bundle is absent or changed")
    return task, task_path, prompt_root


def articulated_host_view_envelope(task: TaskInstructionSpec) -> dict[str, Any] | None:
    """Return the registered native-view size bound for an articulated host.

    ``None`` for every task without articulation components: a tabletop
    manipuland is framed by the same cameras through its own fit, while a
    self-standing host is generated once and cannot be resized afterwards.
    """

    if not task.articulation_components:
        return None
    from .robotics_reference_materializer import native_destination_view_requirements
    from .robotics_native_cameras import native_view_box_envelope

    required_views = native_destination_view_requirements(task)
    envelope = native_view_box_envelope(
        center_xy_m=(0.0, 0.55), required_views=required_views
    )
    limit = round(max(float(value) for value in envelope), 3)
    views = ", ".join(sorted(required_views))
    return {
        "maximum_extent_m": limit,
        "workcell_center_xy_m": [0.0, 0.55],
        "required_views": sorted(required_views),
        "prompt_requirement": (
            "Registered render constraint: this scene is rendered by the fixed "
            f"cameras {views} of the registered robot, which frame only a bounded "
            "volume of the workcell. The articulated host must therefore be a "
            f"task-scale object that fits inside a {limit:g} m cube at the "
            "workcell centre (0.0, 0.55) m - no dimension larger than "
            f"{limit:g} m, and in particular no taller than {limit:g} m - with its "
            "declared handle at a standard graspable size rather than scaled down "
            "with the body. A host larger than that cannot be framed by those "
            "cameras at any later stage, so it would be rejected after the whole "
            "scene was generated."
        ),
    }


def supersede_case_generation(
    *, root: Path, batch_id: str, case_id: str, destination: Path,
    require_explicit: bool = True,
) -> Path:
    """Archive the artifacts a SceneCode regeneration invalidates.

    A case whose generation already exists cannot be regenerated in place: the
    published assets, the identity verdicts derived from them and the
    articulation bindings that reference their digests all belong to that
    revision.  Regeneration is therefore an explicit, auditable revision move -
    never a silent overwrite - and it refuses to touch a case a reference bundle
    already consumed, so published cells keep the assets they were scored with.
    """

    if not (destination / "generation_manifest.json").is_file():
        if require_explicit:
            raise ValueError(
                "refusing to archive a SceneCode generation without a published "
                f"manifest; inspect it first: {destination}"
            )
        # An interrupted earlier attempt leaves an unpublished directory behind.
        # It carries no certificate, so archiving it cannot lose published work.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = (root / "robotics_scenecode_generation_archive" / batch_id / case_id
                   / f"{stamp}_unpublished")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(archive))
        return archive
    batch = RoboticsTaskBatchManifest.model_validate_json(
        (root / "robotics_task_batches" / batch_id / "batch_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    records = [item for item in batch.cases if item.case_id == case_id]
    if len(records) != 1:
        raise ValueError("selected batch must contain the requested case exactly once")
    record = records[0]
    bundle = root / "robotics_reference_bundles" / record.reference_run_id
    if bundle.exists():
        raise ValueError(
            "cannot regenerate a case consumed by a reference bundle; the "
            f"published reference keeps its assets: {bundle}"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "robotics_scenecode_generation_archive" / batch_id / case_id / stamp
    archive.mkdir(parents=True)
    shutil.move(str(destination), str(archive / "generation"))
    bindings = root / "robotics_articulation_bindings" / batch_id / case_id
    if bindings.exists():
        shutil.move(str(bindings), str(archive / "articulation_bindings"))
    failure = root / "robotics_reference_failures" / f"{record.reference_run_id}.json"
    if failure.is_file():
        shutil.move(str(failure), str(archive / "reference_failure.json"))
    return archive


def generate_case(
    *,
    output_root: str | Path,
    batch_id: str,
    case_id: str,
    scenecode_root: str | Path,
    scenecode_python: str | Path,
    supersede_existing: bool = False,
) -> Path:
    root = Path(output_root).resolve()
    task, task_path, prompt_root = _load_case(
        output_root=root, batch_id=batch_id, case_id=case_id
    )
    # Missing native operations are evidence for the shared RSI loop. They do
    # not prevent the task from receiving an environment and an execution trace.
    prompt_path = prompt_root / "scene_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    view_envelope = articulated_host_view_envelope(task)
    if view_envelope is not None:
        # The registered cameras bound how large a self-standing host may be.
        # Publishing that bound with the generation request lets the generator
        # satisfy it while it authors the asset, instead of every sample being
        # discovered as oversized after a full scene generation.
        prompt = f"{prompt}\n\n{view_envelope['prompt_requirement']}"
    destination = root / "robotics_scenecode_generations" / batch_id / case_id
    if destination.exists():
        # Regeneration is a revision move, not an error: the previous revision
        # keeps its assets, identity verdicts and articulation bindings under a
        # timestamped archive, and the new revision is written into the freed
        # path.  A case whose reference bundle is already published is the one
        # case that still refuses, because that would silently change a scored
        # cell.
        supersede_case_generation(
            root=root, batch_id=batch_id, case_id=case_id, destination=destination,
            require_explicit=not supersede_existing,
        )
    failure_path = (
        root / "robotics_scenecode_failures" / batch_id / f"{case_id}.json"
    )
    task_sha256 = sha256_file(task_path)
    prompt_sha256 = sha256_file(prompt_path)
    recovery = generation_recovery_path(
        root=root, batch_id=batch_id, case_id=case_id
    )
    partial_recovery = partial_generation_recovery_path(
        root=root, batch_id=batch_id, case_id=case_id
    )
    # A retry owns this exact case identity.  If the inventory report names a
    # valid partial checkpoint, do not let an older completed-but-invalid
    # recovery win merely because it has the preferred directory name.  This
    # arbitration is evidence based (the failed inventory explicitly points at
    # the resumable candidate), and preserves the old candidate for audit.
    prefer_partial = _arbitrate_inventory_recovery(
        task=task, failure_path=failure_path,
        recovery=recovery, partial_recovery=partial_recovery,
    )
    if prefer_partial:
        archived = recovery.with_name(f"{case_id}.superseded_{time.time_ns()}")
        recovery.rename(archived)
        print(f"Selected evidence-backed partial recovery; archived {archived}", flush=True)
    failure_path.unlink(missing_ok=True)
    reused_recovery = recovery.exists()
    resumed_partial = False
    partial_resume_stage: str | None = None
    if reused_recovery:
        _validate_recovery_candidate(
            recovery=recovery,
            task_sha256=task_sha256,
            prompt_sha256=prompt_sha256,
        )
        staging = recovery
        stage = "task_asset_inventory"
    else:
        if partial_recovery.exists():
            partial_resume_stage = _validate_partial_recovery_candidate(
                recovery=partial_recovery,
                task_sha256=task_sha256,
                prompt_sha256=prompt_sha256,
            )
            resumed_partial = True
        staging = destination.with_name(f".{case_id}.partial.{os.getpid()}")
        if staging.exists():
            raise FileExistsError(
                f"stale SceneCode staging requires inspection: {staging}"
            )
        staging.mkdir(parents=True)
        stage = "scenecode_generation"
    failure_written = False
    retry_completed_as_partial = False
    generation_completed = reused_recovery
    try:
        scene_run = staging / "scene_run"
        scene_root = Path(scenecode_root).resolve()
        # Resolving a virtual-environment executable follows its symlink to the
        # base interpreter and loses the venv's site-packages (for example
        # Hydra). Keep the venv entrypoint identity intact.
        python = runtime_executable(scenecode_python)
        main = scene_root / "main.py"
        if not main.is_file() or not python.is_file():
            raise FileNotFoundError("SceneCode runtime or Python is missing")
        model = os.environ.get("SCENECODE_OPENAI_MODEL", "gpt-5.6-sol")
        profile_overrides = _code_object_profile_overrides(
            os.environ.get("SCENECODE_CODE_OBJECT_CONFIG_ROOT"))
        if profile_overrides:
            (staging / "source_author_observed.json").write_text(json.dumps({
                "scope": "observed original generator environment after launcher API-env loading",
                "scene_author_model": model,
                "code_object_profile_root": os.environ["SCENECODE_CODE_OBJECT_CONFIG_ROOT"],
                "code_object_api": os.environ.get("SCENECODE_CODE_OBJECT_API"),
                "code_object_collector": os.environ.get("SCENECODE_CODE_OBJECT_STREAM"),
                "responses_http_stream": os.environ.get("SCENECODE_CODE_OBJECT_RESPONSES_STREAM"),
                "read_timeout": os.environ.get("SCENECODE_CODE_OBJECT_READ_TIMEOUT"),
                "reasoning_effort_override": os.environ.get("SCENECODE_CODE_OBJECT_REASONING_EFFORT"),
            }, indent=2) + "\n")
        command = [
            str(python),
            str(main),
            f"+name=worldolympiad_{hashlib.sha256(case_id.encode()).hexdigest()[:12]}",
            "experiment.single_scene_id=0",
            _hydra_string_override("experiment.single_prompt", prompt),
            "experiment.num_workers=1",
            f"floor_plan_agent.openai.model={model}",
            f"floor_plan_agent.session_memory.summarization_model={model}",
            f"furniture_agent.openai.model={model}",
            f"furniture_agent.session_memory.summarization_model={model}",
            f"wall_agent.openai.model={model}",
            f"wall_agent.session_memory.summarization_model={model}",
            f"ceiling_agent.openai.model={model}",
            f"ceiling_agent.session_memory.summarization_model={model}",
            f"manipuland_agent.openai.model={model}",
            f"manipuland_agent.session_memory.summarization_model={model}",
            *ROBOTICS_SCENECODE_OVERRIDES,
            *profile_overrides,
            f"hydra.run.dir={scene_run}",
        ]
        if not reused_recovery:
            source_floor = (None if resumed_partial else
                            _prepare_source_floor(root, staging, prompt, prompt_sha256))
            if source_floor is not None:
                command.extend([
                    _hydra_string_override("experiment.pipeline.resume_from_path", str(source_floor)),
                    "experiment.pipeline.start_stage=furniture",
                    "experiment.pipeline.stop_stage=manipuland",
                ])
            if resumed_partial:
                command.extend(
                    [
                        "experiment.pipeline.resume_from_path="
                        f"{partial_recovery / 'scene_run'}",
                        "experiment.pipeline.start_stage="
                        f"{partial_resume_stage}",
                        "experiment.pipeline.stop_stage=manipuland",
                    ]
                )
            # SceneCode allocates its Blender, MCP and convex-decomposition
            # servers from port ranges per worker, so scene authoring is
            # designed to coexist.  The host-level job lock exists to be safe
            # about server startup/shutdown and can be released explicitly when
            # a deployment wants several generations in flight; everything else
            # (renders that must own the machine) stays serialized.
            concurrent = os.environ.get("WORLDOLYMPIAD_SCENECODE_CONCURRENT") == "1"
            job_lock = nullcontext() if concurrent else exclusive_blender_job(root)
            with job_lock:
                try:
                    _run(
                        command,
                        scene_root,
                        env_file=root.parent / ".secrets" / "scenecode.env",
                    )
                except subprocess.CalledProcessError as exc:
                    if not completed_despite_broken_stdout(
                        scene_run=scene_run, exception=exc
                    ):
                        raise
            generation_completed = True
        stage = "task_asset_inventory"
        house_state = scene_run / "scene_000" / "combined_house" / "house_state.json"
        if not house_state.is_file():
            raise RuntimeError("SceneCode completed without combined house_state.json")
        inventory = certify_task_inventory(task=task, house_state=house_state)
        recovery_report_path: Path | None = None
        recovery_reports: list[dict[str, Any]] = []
        if not inventory["passed"]:
            recovery_report = restore_validated_fallen_task_objects(
                task=task,
                house_state=house_state,
                inventory=inventory,
            )
            # A declined recovery is evidence too: preserve its skip reasons
            # without treating it as a mutation or recertifying unchanged assets.
            recovery_reports.append(recovery_report)
            if recovery_report["changed"]:
                inventory = certify_task_inventory(task=task, house_state=house_state)
        if not inventory["passed"]:
            # SceneCode's manipuland designer can pre-solve a container task by
            # merging the declared manipuland into the declared destination.
            # Restore the typed unsolved relation from the composite's own
            # validated constituent assets before declaring the case failed.
            separation_report = separate_prefilled_container_task_objects(
                task=task,
                house_state=house_state,
                inventory=inventory,
            )
            recovery_reports.append(separation_report)
            if separation_report["changed"]:
                inventory = certify_task_inventory(task=task, house_state=house_state)
        if recovery_reports:
            recovery_report_path = staging / "task_object_recovery.json"
            recovery_payload: dict[str, Any]
            if len(recovery_reports) == 1:
                recovery_payload = recovery_reports[0]
            else:
                recovery_payload = {
                    "schema_version": "robotics_scenecode_task_object_recovery.v1",
                    "case_id": case_id,
                    "reports": recovery_reports,
                }
            recovery_report_path.write_text(
                json.dumps(
                    recovery_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        # The staging directory is atomically renamed below. Publish only final
        # paths so a successful artifact never contains a dead `.partial` path.
        inventory["house_state"] = str(
            destination / house_state.relative_to(staging)
        )
        for match in inventory["matches"]:
            sdf_path = match.get("sdf_path")
            if sdf_path:
                path = Path(sdf_path)
                if path.is_relative_to(staging):
                    match["sdf_path"] = str(destination / path.relative_to(staging))
        inventory_path = staging / "task_inventory_certificate.json"
        inventory_path.write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not inventory["passed"]:
            retry_completed_as_partial = can_retry_inventory_at_manipuland_stage(
                task=task, inventory=inventory
            )
            diagnostic_matches = []
            for match in inventory["matches"]:
                diagnostic = {
                    key: value
                    for key, value in match.items()
                    if key not in {"sdf_path", "sdf_sha256"}
                }
                diagnostic["sdf_present"] = bool(match.get("sdf_path"))
                diagnostic_matches.append(diagnostic)
            _write_failure_report(
                failure_path,
                {
                    "schema_version": "robotics_scenecode_failure.v1",
                    "batch_id": batch_id,
                    "case_id": case_id,
                    "classification": "asset_geometry",
                    "owner_stage": "scenecode_candidate",
                    "failed_stage": stage,
                    "reason": "final live task object lacks a resolvable SDF asset",
                    "task_sha256": task_sha256,
                    "scene_inventory_count": inventory["scene_inventory_count"],
                    "matches": diagnostic_matches,
                    "heavy_staging_discarded": False,
                    "recoverable_candidate": str(
                        partial_recovery
                        if retry_completed_as_partial
                        else recovery
                    ),
                },
            )
            failure_written = True
            raise ValueError(
                "SceneCode task-asset inventory failed; "
                f"structured_report={failure_path}"
            )
        stage = "final_visual_identity"
        unvalidated = find_unvalidated_task_assets(task=task, scene_run=scene_run)
        if unvalidated:
            _write_failure_report(
                failure_path,
                {
                    "schema_version": "robotics_scenecode_failure.v1",
                    "batch_id": batch_id,
                    "case_id": case_id,
                    "classification": "asset_geometry",
                    "owner_stage": "scenecode_candidate",
                    "failed_stage": stage,
                    "reason": (
                        "a declared task asset was accepted by SceneCode without "
                        "final validation and therefore has no visual-identity certificate"
                    ),
                    "task_sha256": task_sha256,
                    "unvalidated_task_assets": unvalidated,
                    "heavy_staging_discarded": False,
                    "recoverable_candidate": str(recovery),
                },
            )
            failure_written = True
            raise ValueError(
                "SceneCode final task asset is explicitly unvalidated; "
                f"structured_report={failure_path}"
            )
        stage = "publication_cleanup"
        (staging / "recovery_manifest.json").unlink(missing_ok=True)
        cleanup = prune_unreferenced_generated_assets(house_state=house_state)
        cleanup_path = staging / "generated_asset_cleanup.json"
        cleanup_path.write_text(
            json.dumps(cleanup, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rewritten_reference_files = rewrite_published_references(
            root=staging, staging=staging, destination=destination
        )
        write_final_inventory_binding(
            inventory=inventory,
            inventory_path=inventory_path,
            house_state=house_state,
        )
        stage = "publication"
        generation_manifest = staging / "generation_manifest.json"
        generation_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "robotics_scenecode_generation.v1",
                    "batch_id": batch_id,
                    "case_id": case_id,
                    "task": str(task_path),
                    "task_sha256": task_sha256,
                    "scene_prompt_bundle": str(prompt_root / "scene_prompt_bundle.json"),
                    "scene_prompt_bundle_sha256": sha256_file(prompt_root / "scene_prompt_bundle.json"),
                    "house_state": str(destination / house_state.relative_to(staging)),
                    "house_state_sha256": sha256_file(house_state),
                    "task_inventory_certificate": str(destination / inventory_path.name),
                    "task_inventory_certificate_sha256": sha256_file(inventory_path),
                    "generated_asset_cleanup": str(destination / cleanup_path.name),
                    "generated_asset_cleanup_sha256": sha256_file(cleanup_path),
                    "task_object_recovery": (
                        str(destination / recovery_report_path.name)
                        if recovery_report_path is not None
                        else None
                    ),
                    "task_object_recovery_sha256": (
                        sha256_file(recovery_report_path)
                        if recovery_report_path is not None
                        else None
                    ),
                    "published_reference_rewrites": rewritten_reference_files,
                    "reused_completed_generation": reused_recovery,
                    "passed": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        remove_staging_latest_run_link(staging=staging, destination=destination)
        remove_self_referential_batch_link(destination=destination)
        os.replace(staging, destination)
        shutil.rmtree(partial_recovery, ignore_errors=True)
        failure_path.unlink(missing_ok=True)
        return destination / generation_manifest.name
    except BaseException as exc:
        if not failure_written:
            try:
                log_tails = collect_failure_log_tails(staging=staging)
                partial_checkpoint_available = (
                    _partial_resume_stage(staging / "scene_run") is not None
                    or partial_recovery.exists()
                )
                classification, owner_stage = classify_generation_exception(
                    stage=stage, exception=exc, log_tails=log_tails
                )
                _write_failure_report(
                    failure_path,
                    {
                        "schema_version": "robotics_scenecode_failure.v1",
                        "batch_id": batch_id,
                        "case_id": case_id,
                        "classification": classification,
                        "owner_stage": owner_stage,
                        "failed_stage": stage,
                        "exception_type": type(exc).__name__,
                        "reason": str(exc).replace(str(staging), "<discarded-staging>"),
                        "task_sha256": task_sha256,
                        "log_tails": log_tails,
                        "heavy_staging_discarded": not (
                            generation_completed or partial_checkpoint_available
                        ),
                        **(
                            {"recoverable_candidate": str(recovery)}
                            if generation_completed
                            else (
                                {"recoverable_candidate": str(partial_recovery)}
                                if partial_checkpoint_available
                                else {}
                            )
                        ),
                    },
                )
            except BaseException:
                pass
        remove_staging_latest_run_link(staging=staging, destination=destination)
        remove_self_referential_batch_link(destination=destination)
        if generation_completed and retry_completed_as_partial:
            try:
                _preserve_partial_generation(
                    staging=staging,
                    recovery=partial_recovery,
                    task_sha256=task_sha256,
                    prompt_sha256=prompt_sha256,
                )
            except BaseException:
                # Keep the complete candidate recoverable if narrowing it to the
                # failed late stage cannot be committed atomically.
                _preserve_completed_generation(
                    staging=staging,
                    recovery=recovery,
                    task_sha256=task_sha256,
                    prompt_sha256=prompt_sha256,
                    failed_stage=stage,
                )
        elif generation_completed:
            try:
                _preserve_completed_generation(
                    staging=staging,
                    recovery=recovery,
                    task_sha256=task_sha256,
                    prompt_sha256=prompt_sha256,
                    failed_stage=stage,
                )
            except BaseException:
                # Never delete a completed SceneCode candidate merely because
                # moving it into deterministic recovery storage also failed.
                pass
        elif _partial_resume_stage(staging / "scene_run") is not None and not partial_recovery.exists():
            try:
                _preserve_partial_generation(
                    staging=staging,
                    recovery=partial_recovery,
                    task_sha256=task_sha256,
                    prompt_sha256=prompt_sha256,
                )
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--scenecode-root", required=True)
    parser.add_argument("--scenecode-python", required=True)
    parser.add_argument(
        "--supersede-existing",
        action="store_true",
        help=(
            "archive an existing generation revision (assets, identity verdicts "
            "and articulation bindings) and regenerate this case"
        ),
    )
    args = parser.parse_args()
    print(generate_case_with_bounded_inventory_resume(**vars(args)))


if __name__ == "__main__":
    main()
