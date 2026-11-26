"""Auditable SceneCode prompt architecture for world and robotics scenes.

SceneCode remains the scene generator.  This module turns a typed scene brief
into a detailed natural-language prompt, and rejects the sparse, contradictory
or already-solved task setups that make poor benchmark references.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Literal
from urllib import request

from pydantic import Field, field_validator, model_validator

from .models import Contract


SceneDomain = Literal["world_physics", "robotics_manipulation"]
SCENECODE_ROBOTICS_PROMPT_STYLE_ID = "scenecode_robotics_prompt_authoring"
SCENECODE_ROBOTICS_PROMPT_STYLE_VERSION = "2.0.0"

# Clauses that describe an affordance the static scene has to provide -- reach
# or clearance -- rather than the task outcome the model has to produce. Fit,
# insertion, covering, stacking, and placement outcomes stay in the typed task
# contract, so the scene is never pre-solved by prompt prose.
SCENE_AFFORDANCE_RELATION_MARKERS = (
    "prominent handle",
    "protruding handle",
    "open space around the handle",
    "reachable support surface",
)


def scene_affordance_relation(relation: str) -> bool:
    """Return whether a task relation is an affordance of the static scene."""

    return any(marker in relation for marker in SCENE_AFFORDANCE_RELATION_MARKERS)


class SceneObjectRequest(Contract):
    name: str
    category: str
    count: int = 1
    support: str | None = None
    placement: str
    attributes: list[str] = Field(default_factory=list)
    task_role: Literal[
        "environment", "support", "support_target", "target", "distractor", "robot"
    ] = "environment"

    @field_validator("count")
    @classmethod
    def positive_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("scene object count must be positive")
        return value


class ScenePromptSpec(Contract):
    schema_version: Literal["scene_prompt_spec.v1"] = "scene_prompt_spec.v1"
    scene_id: str
    domain: SceneDomain
    room_type: str
    room_dimensions_m: tuple[float, float]
    style: str
    activity_zones: list[str]
    scene_brief: str | None = None
    objects: list[SceneObjectRequest]
    task_geometry_relations: list[str] = Field(default_factory=list)
    wall_and_ceiling: list[str] = Field(default_factory=list)
    lighting_and_materials: list[str]
    task_setup: str
    camera_readability: list[str]
    negative_constraints: list[str] = Field(default_factory=list)
    required_semantics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def benchmark_ready(self) -> "ScenePromptSpec":
        if min(self.room_dimensions_m) < 2.5:
            raise ValueError("benchmark rooms must be at least 2.5 m on each side")
        if not 1 <= len(self.activity_zones) <= 3:
            raise ValueError("use one to three coherent activity zones")
        if not self.objects:
            raise ValueError("a benchmark scene needs declared task objects")
        if not any(
            item.task_role in {"support", "support_target"}
            for item in self.objects
        ):
            raise ValueError("scene prompt needs an explicit support surface")
        if not any(
            item.task_role in {"target", "support_target"}
            for item in self.objects
        ):
            raise ValueError("scene prompt needs at least one task target")
        if self.domain == "robotics_manipulation":
            lowered = self.task_setup.lower()
            if not any(
                word in lowered
                for word in ("not complete", "not solved", "task-ready", "unsolved")
            ):
                raise ValueError("robotics initial state must explicitly remain unsolved")
        return self


class PromptQualityReport(Contract):
    schema_version: Literal["scene_prompt_quality.v1"] = "scene_prompt_quality.v1"
    scene_id: str
    passed: bool
    score: float
    checks: dict[str, bool]
    problems: list[str]


def _object_phrase(item: SceneObjectRequest) -> str:
    number_words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    amount = f"{number_words.get(item.count, str(item.count))} " if item.count != 1 else ""
    object_name = item.name.strip()
    if item.count != 1:
        for prefix in (number_words.get(item.count, str(item.count)), str(item.count)):
            marker = prefix + " "
            if object_name.lower().startswith(marker):
                object_name = object_name[len(marker) :].lstrip()
                break
    phrase = f"{amount}{object_name}"
    if item.support:
        phrase += f" on or relative to {item.support}"
    phrase += f", {item.placement.strip().rstrip('.')}"
    return phrase


def _named_object_phrase(item: SceneObjectRequest) -> str:
    number_words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    name = item.name.strip().rstrip(".")
    has_article = name.lower().startswith(("a ", "an ", "the "))
    if has_article and item.count != 1:
        name = name.split(" ", 1)[1]
    if item.count != 1:
        phrase = f"{number_words.get(item.count, str(item.count))} {name}"
    elif has_article:
        phrase = name
    else:
        article = "an" if name[:1].lower() in "aeiou" else "a"
        phrase = f"{article} {name}"
    if item.attributes:
        phrase += ", " + _natural_list(item.attributes)
    return phrase


def _natural_list(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _prose_words(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def _compile_scenecode_native_robotics_prompt(spec: ScenePromptSpec) -> str:
    """Compile 1--5 natural sentences in SceneCode's official sample idiom."""

    width, depth = spec.room_dimensions_m
    supports = [
        item for item in spec.objects
        if item.task_role in {"support", "support_target"}
    ]
    task_objects = [
        item for item in spec.objects
        if item.task_role not in {"support", "support_target"}
    ]
    primary_support = supports[0]
    initial_state = spec.task_setup.strip().rstrip(".")
    initial_state = initial_state[:1].lower() + initial_state[1:]
    sentences = [
        f"A {spec.style} {spec.room_type}, {width:g} m by {depth:g} m, has "
        f"{_named_object_phrase(primary_support)} {primary_support.placement.strip().rstrip('.')}.",
    ]
    if task_objects:
        sentences.append(
            "On or beside the main support are "
            + _natural_list([_named_object_phrase(item) for item in task_objects])
            + "."
        )
    # Affordance clauses belong in the SceneCode scene prompt; outcome clauses
    # stay in the typed task contract.  Dropping every relation except the
    # articulated handle used to make a placement task unrepairable from the
    # scene side: nothing in the prompt could change, so the SceneCode
    # manipuland repair could only ever answer "would make no change".
    prompt_geometry_relations = [
        relation
        for relation in spec.task_geometry_relations
        if scene_affordance_relation(relation)
    ]
    if prompt_geometry_relations:
        sentences.append(
            "The task geometry shows "
            + _natural_list(prompt_geometry_relations)
            + "."
        )
    sentences.append("Initially, " + initial_state + ".")
    # The authored brief can carry functional collision-geometry requirements
    # beyond the inventory and initial state. Do not silently discard them in
    # the native robotics path. Avoid repeating a brief already in the setup.
    if spec.scene_brief:
        brief = spec.scene_brief.strip().rstrip(".")
        if _prose_words(brief) not in _prose_words(" ".join(sentences)):
            sentences.append(brief + ".")
    closing = _natural_list(spec.lighting_and_materials)
    closing = closing[:1].upper() + closing[1:]
    closing += " keep " + _natural_list(spec.camera_readability)
    if spec.negative_constraints:
        closing += "; keep the scene free of " + _natural_list(
            [value.strip().rstrip(".") for value in spec.negative_constraints]
        )
    sentences.append(
        closing.rstrip(".") + "."
    )
    return " ".join(sentences)


def compile_scene_prompt(spec: ScenePromptSpec) -> str:
    """Compile one validated spec into SceneCode's natural-language interface."""

    if spec.domain == "robotics_manipulation":
        return _compile_scenecode_native_robotics_prompt(spec)

    width, depth = spec.room_dimensions_m
    sentences = [
        f"Create a {width:g} m by {depth:g} m {spec.room_type} in a {spec.style} style.",
        "Organize it into " + "; ".join(spec.activity_zones) + ".",
    ]
    if spec.scene_brief:
        sentences.append(
            "Treat this case-authored scene brief as authoritative: "
            + spec.scene_brief.strip().rstrip(".")
            + "."
        )
    number_words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    for item in spec.objects:
        amount = f"{number_words.get(item.count, str(item.count))} " if item.count != 1 else ""
        object_name = item.name.strip()
        if item.count != 1:
            # Compatible APIs sometimes redundantly encode the quantity in both
            # the structured count and the noun phrase ("three three cubes").
            # The compiled SceneCode prompt is canonical and never repeats it.
            for prefix in (number_words.get(item.count, str(item.count)), str(item.count)):
                marker = prefix + " "
                if object_name.lower().startswith(marker):
                    object_name = object_name[len(marker) :].lstrip()
                    break
        description = f"Place {amount}{object_name}"
        if item.attributes:
            description += f" with {', '.join(item.attributes)}"
        if item.support:
            description += f" on or relative to {item.support}"
        description += f", {item.placement}."
        sentences.append(description)
    if spec.wall_and_ceiling:
        sentences.append("Complete the room with " + "; ".join(spec.wall_and_ceiling) + ".")
    sentences.append(
        "Use the following lighting and material treatment: "
        + "; ".join(spec.lighting_and_materials)
        + "."
    )
    sentences.append(spec.task_setup.rstrip(".") + ".")
    sentences.append("For the main camera, " + "; ".join(spec.camera_readability) + ".")
    if spec.negative_constraints:
        normalized = []
        for constraint in spec.negative_constraints:
            value = constraint.strip().rstrip(".")
            for prefix in ("do not include ", "avoid ", "no "):
                if value.lower().startswith(prefix):
                    value = value[len(prefix) :]
                    break
            normalized.append(value)
        sentences.append("Do not include " + "; ".join(normalized) + ".")
    return " ".join(sentences)


def assess_scene_prompt(spec: ScenePromptSpec, prompt: str | None = None) -> PromptQualityReport:
    text = (prompt or compile_scene_prompt(spec)).lower()
    checks = {
        "room_function": spec.room_type.lower() in text,
        "metric_scale": all(f"{value:g} m" in text for value in spec.room_dimensions_m),
        "spatial_anchors": sum(
            token in text
            for token in ("center", "wall", "window", "beside", "across", "corner", "aisle")
        ) >= (1 if spec.domain == "robotics_manipulation" else 2),
        "support_surface": any(
            item.task_role in {"support", "support_target"}
            for item in spec.objects
        ),
        "task_targets": any(
            item.task_role in {"target", "support_target"}
            for item in spec.objects
        ),
        "declared_object_coverage": all(
            item.name.strip().lower() in text for item in spec.objects
        ),
        "task_geometry_relations": spec.domain == "robotics_manipulation" or all(
            relation.strip().lower() in text
            for relation in spec.task_geometry_relations
        ),
        "scene_brief_preserved": not spec.scene_brief
        or _prose_words(spec.scene_brief) in _prose_words(text),
        "materials_and_lighting": len(spec.lighting_and_materials) >= 2,
        "camera_readability": len(spec.camera_readability) >= 2,
        "task_unsolved": spec.domain != "robotics_manipulation"
        or any(
            token in text
            for token in ("not complete", "not solved", "task-ready", "unsolved")
        ),
        "no_toy_showcase_language": not any(
            token in text for token in ("toy room", "physics showcase", "keep every wall bare")
        ),
    }
    problems = [name for name, passed in checks.items() if not passed]
    score = sum(checks.values()) / len(checks)
    return PromptQualityReport(
        scene_id=spec.scene_id,
        passed=not problems,
        score=score,
        checks=checks,
        problems=problems,
    )


PROMPT_ARCHITECT_SYSTEM = """You are the SceneCode Prompt Architect.
Return only JSON matching scene_prompt_spec.v1. Design a realistic executable
indoor scene, not a mood-board and not a toy showcase. Follow these rules:
1. State room function, metric size, style, and one to three activity zones.
2. Name exact furniture and manipulands, their support surfaces, and useful
   relative placements such as center, beside a window, across from, or along a wall.
   Keep each object name singular or quantity-free; encode quantity only in count.
3. Add only a few contextual objects with a clear home; do not use random clutter.
4. Describe materials and practical lighting. Keep target objects visually distinct.
5. For robotics, reserve robot reach and camera sight lines, and make the initial
   state task-ready but explicitly not complete. The simulator adds the certified
   robot and motion, so never invent an already-moving robot in the static scene.
6. Avoid glass, mirrors, reflective clutter, text decals, severe occlusion, and
   physically impossible placements unless the brief explicitly requires them.
The output is validated before SceneCode receives a deterministically compiled prompt."""


def architect_request(brief: str, profiles_path: str | Path) -> list[dict[str, str]]:
    profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    query = set(re.findall(r"[a-z0-9_]+", brief.lower()))
    ranked = sorted(
        profiles["profiles"],
        key=lambda profile: len(
            query
            & set(
                re.findall(
                    r"[a-z0-9_]+",
                    (profile["id"] + " " + profile["brief"]).lower(),
                )
            )
        ),
        reverse=True,
    )
    examples = [
        {"brief": profile["brief"], "spec": profile["spec"]}
        for profile in ranked[:2]
    ]
    return [
        {"role": "system", "content": PROMPT_ARCHITECT_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"brief": brief, "reference_patterns": examples},
                ensure_ascii=False,
            ),
        },
    ]


def design_scene_prompt_with_api(
    brief: str,
    profiles_path: str | Path,
    *,
    model: str | None = None,
) -> tuple[ScenePromptSpec, str, PromptQualityReport]:
    """Ask the configured API for a typed brief, then compile and verify it."""

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be configured")
    endpoint = api_base.rstrip("/")
    payload = {
        "model": model or os.environ.get("SCENECODE_OPENAI_MODEL", "gpt-5.6-sol"),
        "input": architect_request(brief, profiles_path),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "scene_prompt_spec",
                "strict": True,
                "schema": ScenePromptSpec.model_json_schema(),
            }
        },
    }
    req = request.Request(
        endpoint + "/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python WorldOlympiad2-PromptArchitect/1.0",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    raw = body.get("output_text")
    if not raw:
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    raw = content["text"]
                    break
            if raw:
                break
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Responses API returned no structured scene prompt")
    spec = ScenePromptSpec.model_validate_json(raw)
    prompt = compile_scene_prompt(spec)
    report = assess_scene_prompt(spec, prompt)
    if not report.passed:
        raise ValueError(f"API scene prompt failed quality gate: {report.problems}")
    return spec, prompt, report


def load_profile(profiles_path: str | Path, profile_id: str) -> ScenePromptSpec:
    payload = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    for item in payload["profiles"]:
        if item["id"] == profile_id:
            return ScenePromptSpec.model_validate(item["spec"])
    raise KeyError(f"unknown scene prompt profile: {profile_id}")


def write_prompt_bundle(
    output_dir: str | Path,
    spec: ScenePromptSpec,
    prompt: str,
    report: PromptQualityReport,
    *,
    source: str,
    brief: str | None = None,
    authoring_contract: dict[str, str] | None = None,
) -> Path:
    if spec.domain == "robotics_manipulation":
        from .scenecode_object_requirements import check_text_limit, SCENE_MAX_CHARS
        check_text_limit(prompt, field="compiled scene prompt", max_chars=SCENE_MAX_CHARS)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scene prompt bundle: {output}")
    output.mkdir(parents=True)
    spec.write_json(output / "scene_prompt_spec.json")
    report.write_json(output / "scene_prompt_quality.json")
    (output / "scene_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    record = {
        "schema_version": "scene_prompt_bundle.v1",
        "scene_id": spec.scene_id,
        "source": source,
        "brief": brief,
        "authoring_contract": authoring_contract,
        "prompt_spec": "scene_prompt_spec.json",
        "prompt_quality": "scene_prompt_quality.json",
        "prompt_text": "scene_prompt.txt",
        "passed": report.passed,
    }
    path = output / "scene_prompt_bundle.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


DEFAULT_PROFILES_PATH = Path(__file__).parent / "configs" / "scene_prompt_profiles_v1.json"
PILOT_PROFILE_ID = "robotics_tabletop_physics_v1"
PILOT_PROMPT = compile_scene_prompt(load_profile(DEFAULT_PROFILES_PATH, PILOT_PROFILE_ID))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile-profile")
    compile_parser.add_argument("--profile", required=True)
    compile_parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    compile_parser.add_argument("--output-dir", required=True)
    design_parser = subparsers.add_parser("design")
    design_parser.add_argument("--brief", required=True)
    design_parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    design_parser.add_argument("--model")
    design_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "compile-profile":
        spec = load_profile(args.profiles, args.profile)
        prompt = compile_scene_prompt(spec)
        report = assess_scene_prompt(spec, prompt)
        source = f"validated_profile:{args.profile}"
        brief = None
    else:
        spec, prompt, report = design_scene_prompt_with_api(
            args.brief, args.profiles, model=args.model
        )
        source = "api_prompt_architect"
        brief = args.brief
    print(
        write_prompt_bundle(
            args.output_dir,
            spec,
            prompt,
            report,
            source=source,
            brief=brief,
        )
    )


if __name__ == "__main__":
    main()
