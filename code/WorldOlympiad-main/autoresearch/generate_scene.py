"""Run the fixed SceneCode pilot prompt with bounded manipuland-only retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from .scene_prompting import (
    DEFAULT_PROFILES_PATH,
    PILOT_PROFILE_ID,
    PILOT_PROMPT,
    assess_scene_prompt,
    compile_scene_prompt,
    design_scene_prompt_with_api,
    load_profile,
)


REQUIRED_ACTORS = {
    "sorting_block": ("sorting block", "sort block", "sorting piece"),
    "cup": ("cup", "mug"),
    "slider": (
        "slider",
        "sliding block",
        "push block",
        "natural wood block",
        "wooden block",
        "wood block",
        "wooden prism",
    ),
    "stack": ("stack", "stacked", "building block", "cube block"),
}
REQUIRED_COUNTS = {"sorting_block": 1, "cup": 1, "slider": 1, "stack": 3}


def _hydra_string_override(name: str, value: str) -> str:
    """Encode arbitrary prompt text as one OmegaConf string override argv."""

    return f"{name}={json.dumps(value, ensure_ascii=False)}"


def _inventory(house_state: Path) -> dict[str, list[str]]:
    raw = json.loads(house_state.read_text(encoding="utf-8"))
    inventory: dict[str, list[str]] = {}
    for room in raw.get("rooms", {}).values():
        for object_id, item in room.get("objects", {}).items():
            metadata = item.get("metadata", {})
            composite_members = (
                metadata.get("member_assets", [])
                if metadata.get("composite_type") == "stack"
                else []
            )
            if composite_members:
                inventory.setdefault("stack", []).extend(
                    f"{object_id}::member_{index}:{member.get('asset_id', 'unknown')}"
                    for index, member in enumerate(composite_members)
                )
            text = " ".join(
                [str(object_id), str(item.get("name", "")), str(item.get("description", "")), json.dumps(metadata, sort_keys=True)]
            ).lower()
            for actor, aliases in REQUIRED_ACTORS.items():
                if actor == "stack" and composite_members:
                    continue
                if any(alias in text for alias in aliases):
                    inventory.setdefault(actor, []).append(str(object_id))
    return inventory


def _missing_actors(inventory: dict[str, list[str]]) -> list[str]:
    return sorted(
        actor
        for actor, count in REQUIRED_COUNTS.items()
        if len(inventory.get(actor, [])) < count
    )


def _environment_with_optional_secret(env_file: Path | None = None) -> dict[str, str]:
    """Build a child environment without printing or persisting secret values."""

    env = os.environ.copy()
    if env_file is not None and env_file.is_file():
        if env_file.stat().st_mode & 0o077:
            raise PermissionError(f"secret environment file must be mode 0600: {env_file}")
        for line_number, raw in enumerate(
            env_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise ValueError(f"invalid secret environment line {line_number}")
            key, encoded = line.split("=", 1)
            key = key.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid secret environment key on line {line_number}")
            values = shlex.split(encoded, comments=False, posix=True)
            if len(values) != 1:
                raise ValueError(f"invalid secret environment value on line {line_number}")
            env.setdefault(key, values[0])
        # Robotics scene generation uses SceneCode's vendored text-conditioned
        # Code_Object route.  Reference-image generation is an optional style
        # aid and must not make dataset generation depend on another API.
        env.setdefault("SCENECODE_CODE_OBJECT_TEXT_ONLY", "1")
        workspace = env_file.parent.parent
        blender_candidates = sorted(
            path
            for path in (workspace / "runtime").glob("blender-*/blender")
            if path.is_file() and os.access(path, os.X_OK)
        )
        if len(blender_candidates) == 1:
            blender = blender_candidates[0]
            env["PATH"] = str(blender.parent) + os.pathsep + env.get("PATH", "")
            env.setdefault("BLENDER_PATH", str(blender))
        elif len(blender_candidates) > 1 and shutil.which("blender", path=env.get("PATH")) is None:
            raise RuntimeError("multiple workspace Blender runtimes require disambiguation")
    # Third-party OpenAI tracing is not part of the benchmark artifact and some
    # compatible gateways do not expose its ingestion endpoint. Disable it so
    # each SceneCode agent turn does not wait through three non-fatal timeouts.
    env.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    # The lab gateway exposes its OpenAI-compatible API below /v1 while its root
    # path serves the web UI. Responses happened to be proxied at both paths,
    # but Chat Completions returned HTML at the root and broke Code_Object's
    # planner at response.choices. Normalize once for every SceneCode child.
    api_base = env.get("OPENAI_API_BASE") or env.get("OPENAI_BASE_URL")
    if api_base:
        normalized = api_base.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        env["OPENAI_API_BASE"] = normalized
        env["OPENAI_BASE_URL"] = normalized
    return env


def _run(command: list[str], cwd: Path, *, env_file: Path | None = None) -> None:
    env = _environment_with_optional_secret(env_file)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def _can_resume_manipuland(run_dir: Path) -> bool:
    return any(
        run_dir.glob(
            "scene_*/room_*/scene_states/scene_after_ceiling_objects/scene_state.json"
        )
    )


def generate(
    scenecode_root: Path,
    output_root: Path,
    python: Path,
    *,
    resume_from: Path | None = None,
    start_stage: str = "floor_plan",
    prompt: str = PILOT_PROMPT,
) -> Path:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be supplied through the process environment")
    if not shutil.which("blender"):
        raise RuntimeError("SceneCode requires Blender 4.4 on PATH")
    main = scenecode_root / "main.py"
    if not main.exists():
        raise FileNotFoundError(main)
    previous_run = resume_from
    attempts: list[dict[str, object]] = []
    model = os.environ.get("SCENECODE_OPENAI_MODEL", "gpt-5.6-sol")
    if previous_run is not None:
        existing_house_state = (
            previous_run / "scene_000" / "combined_house" / "house_state.json"
        )
        if existing_house_state.exists():
            inventory = _inventory(existing_house_state)
            missing = _missing_actors(inventory)
            attempts.append(
                {
                    "attempt": "existing_resume_audit",
                    "run_directory": str(previous_run),
                    "start_stage": "existing_house_state",
                    "inventory": inventory,
                    "missing": missing,
                }
            )
            (previous_run / "pilot_actor_inventory.json").write_text(
                json.dumps(
                    {"inventory": inventory, "missing": missing},
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if not missing:
                _write_generation_summary(
                    output_root=output_root,
                    house_state=existing_house_state,
                    prompt=prompt,
                    model=model,
                    attempts=attempts,
                )
                return existing_house_state
    for attempt in range(3):
        if attempt == 0:
            run_name = "initial" if previous_run is None else f"resume_{start_stage}"
        else:
            run_name = f"manipuland_retry_{attempt}"
        run_dir = output_root / run_name
        command = [
            str(python),
            str(main),
            "+name=scenecode_pybullet_pilot",
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
            f"hydra.run.dir={run_dir}",
        ]
        if previous_run is not None:
            resumed_stage = start_stage if attempt == 0 else "manipuland"
            command.extend(
                [
                    f"experiment.pipeline.resume_from_path={previous_run}",
                    f"experiment.pipeline.start_stage={resumed_stage}",
                    "experiment.pipeline.stop_stage=manipuland",
                ]
            )
        try:
            _run(command, scenecode_root)
        except subprocess.CalledProcessError as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "run_directory": str(run_dir),
                    "start_stage": (
                        start_stage if attempt == 0 and resume_from is not None else
                        ("floor_plan" if attempt == 0 else "manipuland")
                    ),
                    "process_exit_code": exc.returncode,
                    "missing": sorted(REQUIRED_ACTORS),
                }
            )
            resume_candidate = run_dir if _can_resume_manipuland(run_dir) else previous_run
            if attempt == 2 or resume_candidate is None:
                raise RuntimeError(
                    f"SceneCode process failed with exit code {exc.returncode}; "
                    "no further safe manipuland retry is available"
                ) from exc
            previous_run = resume_candidate
            continue
        house_state = run_dir / "scene_000" / "combined_house" / "house_state.json"
        if not house_state.exists():
            attempts.append(
                {
                    "attempt": attempt,
                    "run_directory": str(run_dir),
                    "start_stage": (
                        start_stage if attempt == 0 and resume_from is not None else
                        ("floor_plan" if attempt == 0 else "manipuland")
                    ),
                    "error": "completed_without_house_state",
                    "missing": sorted(REQUIRED_ACTORS),
                }
            )
            resume_candidate = run_dir if _can_resume_manipuland(run_dir) else previous_run
            if attempt == 2 or resume_candidate is None:
                raise RuntimeError(f"SceneCode completed without house_state.json: {run_dir}")
            previous_run = resume_candidate
            continue
        inventory = _inventory(house_state)
        missing = _missing_actors(inventory)
        attempts.append(
            {
                "attempt": attempt,
                "run_directory": str(run_dir),
                "start_stage": (
                    start_stage if attempt == 0 and resume_from is not None else
                    ("floor_plan" if attempt == 0 else "manipuland")
                ),
                "inventory": inventory,
                "missing": missing,
            }
        )
        (run_dir / "pilot_actor_inventory.json").write_text(
            json.dumps({"inventory": inventory, "missing": missing}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if not missing:
            _write_generation_summary(
                output_root=output_root,
                house_state=house_state,
                prompt=prompt,
                model=model,
                attempts=attempts,
            )
            return house_state
        previous_run = run_dir
    raise RuntimeError(f"SceneCode pilot is missing required actors after two manipuland retries: {missing}")


def _write_generation_summary(
    *,
    output_root: Path,
    house_state: Path,
    prompt: str,
    model: str,
    attempts: list[dict[str, object]],
) -> None:
    raw_base = os.environ.get("OPENAI_API_BASE") or os.environ.get(
        "OPENAI_BASE_URL", ""
    )
    parsed_base = urlsplit(raw_base)
    gateway = (
        f"{parsed_base.scheme}://{parsed_base.hostname}"
        if parsed_base.scheme and parsed_base.hostname
        else "configured_openai_compatible_gateway"
    )
    summary = {
        "schema_version": "scenecode_generation_summary.v1",
        "prompt": prompt,
        "model": model,
        "api_gateway": gateway,
        "house_state": str(house_state),
        "house_state_sha256": hashlib.sha256(house_state.read_bytes()).hexdigest(),
        "code_object_conditioning": (
            "text_only"
            if os.environ.get("SCENECODE_CODE_OBJECT_TEXT_ONLY") == "1"
            else "image_with_text_fallback"
        ),
        "wall_policy": (
            "bare"
            if os.environ.get("SCENECODE_BARE_WALLS") == "1"
            else "scenecode_agent"
        ),
        "ceiling_policy": (
            "bare"
            if os.environ.get("SCENECODE_BARE_CEILING") == "1"
            else "scenecode_agent"
        ),
        "attempts": attempts,
    }
    (output_root / "scene_generation_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenecode-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenecode-python", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default=os.environ.get("AUTORESEARCH_SCENE_PROMPT"),
        help="Explicit SceneCode prompt; bypasses the Prompt Architect but retains actor checks",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 SceneCode prompt file; retains actor checks without shell quoting",
    )
    parser.add_argument(
        "--prompt-profile",
        default=os.environ.get("AUTORESEARCH_SCENE_PROFILE", PILOT_PROFILE_ID),
        help="Validated profile id from scene_prompt_profiles_v1.json",
    )
    parser.add_argument(
        "--prompt-brief",
        help="Ask the configured API Prompt Architect to design a typed scene from this brief",
    )
    parser.add_argument(
        "--prompt-profiles",
        type=Path,
        default=DEFAULT_PROFILES_PATH,
        help="Prompt profile catalog used for deterministic profiles and API reference patterns",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Previous SceneCode Hydra run directory containing scene_000",
    )
    parser.add_argument(
        "--start-stage",
        choices=("floor_plan", "furniture", "wall_mounted", "ceiling_mounted", "manipuland"),
        default="floor_plan",
        help="Stage to resume; requires --resume-from unless floor_plan",
    )
    args = parser.parse_args()
    if args.start_stage != "floor_plan" and args.resume_from is None:
        parser.error("--start-stage other than floor_plan requires --resume-from")
    selected_sources = sum(
        value is not None for value in (args.prompt, args.prompt_file, args.prompt_brief)
    )
    if selected_sources > 1:
        parser.error("--prompt, --prompt-file and --prompt-brief are mutually exclusive")
    if args.prompt_brief:
        spec, selected_prompt, quality = design_scene_prompt_with_api(
            args.prompt_brief,
            args.prompt_profiles,
        )
        prompt_source = "api_prompt_architect"
    elif args.prompt_file:
        selected_prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        if not selected_prompt:
            parser.error("--prompt-file must contain a non-empty prompt")
        spec = None
        quality = None
        prompt_source = f"explicit_prompt_file:{args.prompt_file.resolve()}"
    elif args.prompt:
        spec = None
        selected_prompt = args.prompt
        quality = None
        prompt_source = "explicit_prompt"
    else:
        spec = load_profile(args.prompt_profiles, args.prompt_profile)
        selected_prompt = compile_scene_prompt(spec)
        quality = assess_scene_prompt(spec, selected_prompt)
        if not quality.passed:
            raise ValueError(f"prompt profile failed quality gate: {quality.problems}")
        prompt_source = f"validated_profile:{args.prompt_profile}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    prompt_record = {
        "schema_version": "scene_prompt_record.v1",
        "source": prompt_source,
        "brief": args.prompt_brief,
        "prompt": selected_prompt,
        "spec": spec.model_dump(mode="json") if spec else None,
        "quality": quality.model_dump(mode="json") if quality else None,
    }
    (args.output_root / "scene_prompt_record.json").write_text(
        json.dumps(prompt_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    # Do not resolve the venv launcher symlink into its base CPython binary:
    # executing that resolved target would bypass the virtual environment and
    # make all installed SceneCode dependencies disappear from sys.path.
    scenecode_python = args.scenecode_python.expanduser().absolute()
    print(
        generate(
            args.scenecode_root.resolve(),
            args.output_root.resolve(),
            scenecode_python,
            resume_from=(args.resume_from.resolve() if args.resume_from else None),
            start_stage=args.start_stage,
            prompt=selected_prompt,
        )
    )


if __name__ == "__main__":
    main()
