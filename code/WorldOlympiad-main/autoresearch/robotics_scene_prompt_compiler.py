"""Compile declarative robotics cases into deterministic SceneCode prompts.

This is a generic data transform, not a task catalog.  It preserves every
model-authored object and task fact, adds only reusable room/workcell context,
and validates the result with the common SceneCode prompt quality gate.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import re
import shutil

from .robotics_contracts import RoboticsTaskBatchManifest, TaskInstructionSpec
from .robotics_views import sha256_file
from .scene_prompting import (
    SCENECODE_ROBOTICS_PROMPT_STYLE_ID,
    SCENECODE_ROBOTICS_PROMPT_STYLE_VERSION,
    SceneObjectRequest,
    ScenePromptSpec,
    assess_scene_prompt,
    compile_scene_prompt,
    write_prompt_bundle,
)


PROMPT_AUTHORING_SKILL = (
    Path(__file__).parents[1]
    / "skills"
    / "scenecode-robotics-prompt-authoring"
    / "SKILL.md"
)


PRIMARY_SUPPORT_TERMS = {
    "table",
    "worktable",
    "workbench",
    "bench",
    "counter",
    "countertop",
    "desk",
    "platform",
}


_QUANTITY_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def _canonical_object_name(value: str) -> tuple[str, int]:
    """Move a leading quantity into the structured object count.

    Model-authored descriptions often say ``three blue cubes`` even though
    the typed object leaves ``count`` at its default. Keeping the quantity in
    both fields produces malformed prose such as ``a three blue cubes`` and
    makes downstream asset inventory ambiguous.
    """

    name = value.strip().rstrip(".")
    match = re.match(r"^(one|two|three|four|five|six|[0-9]+)\s+(.+)$", name, re.I)
    if not match:
        return name, 1
    raw_count, remainder = match.groups()
    count = _QUANTITY_WORDS.get(raw_count.lower())
    if count is None:
        count = int(raw_count)
    return remainder.strip().rstrip("."), count


def _primary_support_id(task: TaskInstructionSpec) -> str:
    supports = [item for item in task.objects if item.role == "support"]
    if not supports:
        raise ValueError(f"task has no support object: {task.case_id}")

    def score(item: object) -> tuple[int, int]:
        text = (
            f"{item.object_id} {item.category} {item.visual_description}"
            .lower()
            .replace("_", " ")
        )
        words = set(text.split())
        return (len(words & PRIMARY_SUPPORT_TERMS), -supports.index(item))

    return max(supports, key=score).object_id


def _component_support_id(
    task: TaskInstructionSpec, *, object_id: str, primary_support: str
) -> str:
    # Only an operation that presupposes an existing mechanical attachment may
    # define the initial host. A place/insert/stack target describes the final
    # state and must never move the initial object onto or into that target.
    for action in task.action_program:
        if (
            action.operation == "close"
            and object_id in action.subjects
            and action.target
        ):
            return action.target

    # For installed components such as a button on a panel or a drawer in a
    # cabinet, the sole secondary support is their host. This is a reusable
    # object-graph rule, not a task-specific recipe.
    secondary = [
        item.object_id
        for item in task.objects
        if item.role == "support" and item.object_id != primary_support
    ]
    return secondary[0] if len(secondary) == 1 else primary_support


def _compact_clauses(value: str) -> str:
    clauses = [item.strip() for item in re.split(r"\.\s+", value.strip().rstrip("."))]
    return "; ".join(
        item if index == 0 or not item else item[0].lower() + item[1:]
        for index, item in enumerate(clauses)
        if item
    )


def _structural_identity_attributes(visual_description: str) -> list[str]:
    """Translate identity-bearing words into concise visible geometry clauses.

    SceneCode's asset planner can repeat words such as ``open-top`` in its plan
    while still emitting a capped primitive.  These clauses preserve the
    selected object meaning without adding task goals or construction recipes.
    They are derived from the description itself and therefore scale across
    cases rather than selecting a task-specific template.
    """

    lowered = visual_description.lower()
    attributes: list[str] = []
    if any(token in lowered for token in ("open-top", "open top", "hollow")):
        attributes.append(
            "visibly showing a bottom, four enclosing side walls, an unobstructed "
            "empty interior, and no top panel"
        )
    if any(token in lowered for token in ("through-hole", "through hole")):
        attributes.append("with the through-hole visibly open from end to end")
    return attributes


def _geometry_constraint_attributes(constraints: dict[str, object]) -> list[str]:
    """Turn task-authored geometry facts into concise, model-visible clauses."""
    clauses: list[str] = []
    labels = {
        "flush_with_host": "flush with the host front plane",
        "recessed_from_host_m": "recessed from the host front plane",
        "handle_orientation": "handle orientation",
        "handle_clearance_m": "clearance between handle bar and panel",
        "handle_mount_count": "handle mounting post count",
        "handle_shape": "handle shape",
    }
    for key, label in labels.items():
        if key not in constraints:
            continue
        value = constraints[key]
        if key == "flush_with_host" and value is True:
            clauses.append(label)
        elif key == "recessed_from_host_m":
            clauses.append(f"{label} of {float(value):.3g} m")
        else:
            clauses.append(f"{label}: {value}")
    dimensions = constraints.get("overall_dimensions_m")
    if dimensions is not None:
        if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3
                or any(not isinstance(value, (int, float)) or float(value) <= 0
                       for value in dimensions)):
            raise ValueError("overall_dimensions_m must be three positive width/depth/height values")
        clauses.append(
            "physical width × depth × height "
            + " × ".join(f"{float(value):.3g} m" for value in dimensions)
        )
    feature_dimensions = constraints.get("feature_dimensions_m")
    if feature_dimensions is not None:
        if not isinstance(feature_dimensions, dict) or not feature_dimensions:
            raise ValueError("feature_dimensions_m must be a nonempty feature-to-size map")
        for feature, values in sorted(feature_dimensions.items()):
            dimensions_m = values if isinstance(values, (list, tuple)) else [values]
            if (not feature or not dimensions_m
                    or any(not isinstance(value, (int, float)) or float(value) <= 0
                           for value in dimensions_m)):
                raise ValueError(f"feature size for {feature!r} must contain positive metre values")
            feature_label = str(feature).removesuffix("_m").replace("_", " ")
            clauses.append(
                f"{feature_label}: "
                + " × ".join(f"{float(value):.3g} m" for value in dimensions_m)
            )
    return clauses


def _task_geometry_relations(task: TaskInstructionSpec) -> list[str]:
    objects = {item.object_id: item for item in task.objects}
    components = {
        item.component_id: item for item in task.articulation_components
    }
    relations: list[str] = []
    for action in task.action_program:
        if len(action.subjects) != 1:
            continue
        subject = objects.get(action.subjects[0])
        component = components.get(action.subjects[0])
        if subject is None and component is None:
            continue
        target = objects.get(action.target) if action.target else None
        subject_name = (
            subject.object_id if subject is not None else component.component_id
        ).replace("_", " ")
        target_name = target.object_id.replace("_", " ") if target is not None else ""
        if action.operation == "insert" and target is not None:
            relations.append(
                f"the {target_name} opening admits the {subject_name}, "
                f"which remains easy to grasp"
            )
        elif action.operation == "cover" and target is not None:
            relations.append(
                f"the {subject_name} matches and fully covers the {target_name} opening"
            )
        elif action.operation == "stack" and target is not None:
            relations.append(
                f"the {target_name} top fits the {subject_name} footprint"
            )
        elif action.operation == "close" and target is not None:
            relations.append(
                f"the {subject_name} remains installed on and closes the {target_name} opening"
            )
        elif action.operation == "pull" and component is not None:
            host_name = component.host_object_id.replace("_", " ")
            relations.append(
                f"the {subject_name} is a closed movable component of the {host_name}, "
                "with a prominent handle projecting clear of its front face and open "
                "space around the handle for a two-sided grasp"
            )
        elif (
            action.operation == "place"
            and target is not None
            and any(
                predicate.kind == "on_region"
                and predicate.subject == action.subjects[0]
                and predicate.object == action.target
                for predicate in task.success_predicates
            )
        ):
            # An upright/supported placement is a paired-geometry relation even
            # though the destination is a flat region rather than a container:
            # the subject has to admit its own standing footprint on that
            # support, and the support has to stay reachable and clear around
            # it.  Without this the SceneCode manipuland repair has no reusable
            # relation to regenerate the task object from.
            relations.append(
                f"the {target_name} has a reachable support surface and clear "
                f"space around the {subject_name}"
            )
    return relations


def compile_task_prompt(task: TaskInstructionSpec) -> ScenePromptSpec:
    from .scenecode_object_requirements import check_text_limit, DESCRIPTION_MAX_CHARS, SCENE_MAX_CHARS
    check_text_limit(task.scene_brief or "", field="scene_brief", max_chars=SCENE_MAX_CHARS)
    for item in [*task.objects, *task.articulation_components]:
        check_text_limit(item.visual_description, field="visual_description", max_chars=DESCRIPTION_MAX_CHARS)
    primary_support = _primary_support_id(task)
    objects: list[SceneObjectRequest] = []
    for item in task.objects:
        if item.object_id == primary_support:
            placement = "near the center with open approach space on the near side"
            support = None
            prompt_role = (
                "support_target"
                if any(
                    component.host_object_id == item.object_id
                    for component in task.articulation_components
                )
                else "support"
            )
            name = item.visual_description
            attributes = _structural_identity_attributes(item.visual_description) + _geometry_constraint_attributes(item.geometry_constraints)
        elif item.role == "support":
            placement = "in the installed position described by the initial state"
            support = primary_support
            prompt_role = "target"
            name = item.visual_description
            attributes = _structural_identity_attributes(item.visual_description) + _geometry_constraint_attributes(item.geometry_constraints)
        elif item.role == "manipuland":
            placement = "in the reachable non-goal position described below"
            support = _component_support_id(
                task, object_id=item.object_id, primary_support=primary_support
            )
            prompt_role = "target"
            name = item.visual_description
            attributes = _structural_identity_attributes(item.visual_description) + _geometry_constraint_attributes(item.geometry_constraints)
        elif item.role == "destination":
            placement = "reachable and visibly separated from the moved object"
            support = primary_support
            # A freestanding articulated host (for example a cabinet whose door
            # is the articulated task target) is a manipulation target even
            # though its declared role is "destination"; otherwise the prompt
            # would declare no task target at all.
            hosts_component = any(
                component.host_object_id == item.object_id
                for component in task.articulation_components
            )
            prompt_role = "target" if hosts_component else "environment"
            name = item.visual_description
            attributes = _structural_identity_attributes(item.visual_description) + _geometry_constraint_attributes(item.geometry_constraints)
        else:
            placement = (
                "near the far side of the task area without blocking robot reach or the "
                "main camera"
            )
            support = primary_support
            prompt_role = "environment"
            name = item.visual_description
            attributes = _structural_identity_attributes(item.visual_description) + _geometry_constraint_attributes(item.geometry_constraints)
        name, count = _canonical_object_name(name)
        for component in task.articulation_components:
            if component.host_object_id == item.object_id:
                component_attributes = _geometry_constraint_attributes(component.geometry_constraints)
                component_description = component.visual_description.strip()
                if component_attributes:
                    component_description += ", " + ", ".join(component_attributes)
                attributes.append(component_description)
        objects.append(
            SceneObjectRequest(
                name=name,
                category=item.category,
                count=count,
                support=support,
                placement=placement,
                attributes=attributes,
                task_role=prompt_role,
            )
        )

    initial_state = _compact_clauses(task.initial_state)
    lowered_initial_state = initial_state.lower()
    if not any(
        marker in lowered_initial_state
        for marker in ("unsolved", "not solved", "not complete", "task-ready")
    ):
        initial_state += "; the task is not solved"
    return ScenePromptSpec(
        scene_id=task.case_id,
        domain="robotics_manipulation",
        room_type="task-focused robotics manipulation workspace",
        room_dimensions_m=(4.8, 4.0),
        style="realistic uncluttered",
        activity_zones=[
            "the case-authored manipulation area with a clear robot approach aisle",
        ],
        scene_brief=task.scene_brief,
        objects=objects,
        task_geometry_relations=_task_geometry_relations(task),
        wall_and_ceiling=[],
        lighting_and_materials=[
            "neutral diffuse lighting",
            "matte task surfaces with clear contrast",
            *(
                ["give every moving articulated panel and its handle a visible "
                 "material or edge contrast against the static host and room "
                 "while preserving the task's declared host material"]
                if task.articulation_components else []
            ),
        ],
        task_setup=initial_state,
        camera_readability=[
            "all task objects plainly visible",
            "their silhouettes separated",
            *(
                ["the moving component, its handle, and the static host remain "
                 "separately identifiable at native robot-camera scale"]
                if task.articulation_components else []
            ),
        ],
        negative_constraints=[
            "unrelated clutter, robots, and floating, intersecting, or pre-solved geometry",
        ],
        required_semantics=[item.category for item in task.objects],
    )


def archive_superseded_prompts(*, output: Path, root: Path) -> Path:
    """Move a published prompt batch aside before an explicit re-authoring.

    Recompiling a prompt is a revision move, never an overwrite: the superseded
    revision stays auditable under ``robotics_scene_prompt_archive``.
    """

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "robotics_scene_prompt_archive" / output.name / stamp
    if archive.exists():
        raise FileExistsError(f"superseded prompt archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(output, archive)
    return archive


def compile_batch_prompts(
    *,
    batch_manifest: str | Path,
    output_root: str | Path,
    supersede_existing: bool = False,
) -> Path:
    manifest_path = Path(batch_manifest).resolve()
    manifest = RoboticsTaskBatchManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    root = Path(output_root).resolve()
    output = root / "robotics_scene_prompts" / manifest.batch_id
    if not PROMPT_AUTHORING_SKILL.is_file():
        raise FileNotFoundError(PROMPT_AUTHORING_SKILL)
    authoring_contract = {
        "skill_id": SCENECODE_ROBOTICS_PROMPT_STYLE_ID,
        "skill_version": SCENECODE_ROBOTICS_PROMPT_STYLE_VERSION,
        "skill_path": PROMPT_AUTHORING_SKILL.relative_to(
            Path(__file__).parents[1]
        ).as_posix(),
        "skill_sha256": sha256_file(PROMPT_AUTHORING_SKILL),
    }
    superseded: Path | None = None
    if output.exists():
        # The scheduler contract reuses the published prompts whenever every
        # prompt it would re-emit is byte-identical to the published one, and
        # fails closed the moment recompiling would actually change a prompt.
        # The authoring contract is recorded rather than compared, so a style
        # skill that only restates the same rules no longer blocks the batch.
        # A caller that has decided the change is intended asks for the
        # revision explicitly instead of editing the published prompt by hand.
        if not supersede_existing:
            return _reuse_prompt_batch(
                output=output,
                root=root,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        superseded = archive_superseded_prompts(output=output, root=root)
    staging = output.with_name(f".{manifest.batch_id}.partial.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale SceneCode prompt staging requires inspection: {staging}")
    staging.mkdir(parents=True)
    try:
        records: list[dict[str, object]] = []
        for case in manifest.cases:
            task_path = (root / case.task_path).resolve()
            if sha256_file(task_path) != case.task_sha256:
                raise ValueError(f"batch task digest changed: {case.case_id}")
            task = TaskInstructionSpec.model_validate_json(
                task_path.read_text(encoding="utf-8")
            )
            spec = compile_task_prompt(task)
            prompt = compile_scene_prompt(spec)
            from .scenecode_object_requirements import check_text_limit, SCENE_MAX_CHARS
            check_text_limit(prompt, field="compiled scene prompt", max_chars=SCENE_MAX_CHARS)
            report = assess_scene_prompt(spec, prompt)
            if not report.passed:
                raise ValueError(
                    f"compiled SceneCode prompt failed for {case.case_id}: {report.problems}"
                )
            case_root = staging / case.case_id
            bundle = write_prompt_bundle(
                case_root,
                spec,
                prompt,
                report,
                source="deterministic_robotics_task_compiler",
                brief=task.scene_brief,
                authoring_contract=authoring_contract,
            )
            final_case_root = output / case.case_id
            records.append(
                {
                    "case_id": case.case_id,
                    "task_sha256": case.task_sha256,
                    "bundle": (
                        final_case_root / "scene_prompt_bundle.json"
                    ).relative_to(root).as_posix(),
                    "bundle_sha256": sha256_file(bundle),
                    "prompt_sha256": sha256_file(case_root / "scene_prompt.txt"),
                    "quality_score": report.score,
                    "required_semantics": spec.required_semantics,
                    "authoring_contract": authoring_contract,
                }
            )
        manifest_name = "scene_prompt_manifest.json"
        (staging / manifest_name).write_text(
            json.dumps(
                {
                    "schema_version": "robotics_scene_prompt_batch.v2",
                    "batch_id": manifest.batch_id,
                    "source_batch": str(manifest_path),
                    "source_batch_sha256": sha256_file(manifest_path),
                    "case_count": len(records),
                    "cases": records,
                    "passed": True,
                    "superseded_archive": (
                        superseded.relative_to(root).as_posix()
                        if superseded is not None
                        else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return output / manifest_name
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _reuse_prompt_batch(
    *,
    output: Path,
    root: Path,
    manifest: RoboticsTaskBatchManifest,
    manifest_path: Path,
) -> Path:
    """Return the published prompt batch unless recompiling would change it."""

    published_path = output / "scene_prompt_manifest.json"
    unpublished = FileExistsError(
        f"refusing to overwrite SceneCode prompts: {output}"
    )
    if not published_path.is_file():
        raise unpublished
    published = json.loads(published_path.read_text(encoding="utf-8"))
    if published.get("passed") is not True:
        raise ValueError(f"published SceneCode prompts did not pass: {output}")
    if published.get("source_batch_sha256") != sha256_file(manifest_path):
        raise FileExistsError(
            "refusing to overwrite SceneCode prompts compiled from another batch: "
            f"{output}"
        )
    recorded = {
        item.get("case_id"): item for item in published.get("cases", [])
    }
    if set(recorded) != {case.case_id for case in manifest.cases}:
        raise FileExistsError(
            "refusing to overwrite SceneCode prompts for a different case set: "
            f"{output}"
        )
    changed: list[str] = []
    for case in manifest.cases:
        record = recorded[case.case_id]
        if record.get("task_sha256") != case.task_sha256:
            raise FileExistsError(
                f"published SceneCode prompt has another task digest: {case.case_id}"
            )
        task_path = (root / case.task_path).resolve()
        if sha256_file(task_path) != case.task_sha256:
            raise ValueError(f"batch task digest changed: {case.case_id}")
        task = TaskInstructionSpec.model_validate_json(
            task_path.read_text(encoding="utf-8")
        )
        prompt = compile_scene_prompt(compile_task_prompt(task))
        case_prompt = output / case.case_id / "scene_prompt.txt"
        if (
            not case_prompt.is_file()
            or sha256_file(case_prompt) != record.get("prompt_sha256")
            or case_prompt.read_text(encoding="utf-8") != prompt + "\n"
        ):
            changed.append(case.case_id)
    if changed:
        raise FileExistsError(
            "refusing to overwrite SceneCode prompts whose recompiled content "
            f"changed: {sorted(changed)}"
        )
    return published_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--supersede-existing",
        action="store_true",
        help=(
            "Archive the published prompt batch under robotics_scene_prompt_archive "
            "and re-author it.  Required when recompiling legitimately changes a "
            "prompt; without it the compiler keeps failing closed."
        ),
    )
    args = parser.parse_args()
    print(
        compile_batch_prompts(
            batch_manifest=args.batch_manifest,
            output_root=args.output_root,
            supersede_existing=args.supersede_existing,
        )
    )


if __name__ == "__main__":
    main()
