"""Typed, model-specific prompts derived from robotics task semantics.

Prompt text is deliberately independent of reference and WAM rollout timing.
Only reviewed semantic relations can select a template; unsupported or
ambiguous relations fail before model inference.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import field_validator, model_validator

from .models import Contract
from .io import sha256_file
from .robotics_contracts import (
    RoboticsReferenceManifest,
    TaskInstructionSpec,
    TaskObject,
)


PromptWAM = Literal["cosmos3_nano_droid", "lingbot_va_base"]
PromptRelation = Literal["inside"]

_TIMELINE_LANGUAGE = re.compile(
    r"\b(?:duration|fps|frames?|seconds?|secs?|minutes?|hz|rollout|(?:time)?steps?)\b",
    flags=re.IGNORECASE,
)
_SAFE_OBJECT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_timing_free(value: str, *, field: str) -> None:
    if _TIMELINE_LANGUAGE.search(value):
        raise ValueError(
            f"{field} may not encode reference or WAM rollout timing"
        )


class RoboticsPromptTemplate(Contract):
    """One reviewed prompt syntax for a released WAM and task relation."""

    template_id: str
    template_version: str
    wam: PromptWAM
    relation: PromptRelation
    format_string: str
    max_utf8_bytes: int

    @model_validator(mode="after")
    def reviewed_template(self) -> "RoboticsPromptTemplate":
        if not self.template_id or not self.template_version:
            raise ValueError("prompt templates need an id and version")
        if set(re.findall(r"{([^{}]+)}", self.format_string)) != {
            "manipuland",
            "destination",
        }:
            raise ValueError(
                "robotics prompt templates must bind manipuland and destination"
            )
        _assert_timing_free(self.format_string, field="prompt template")
        if self.max_utf8_bytes <= 0:
            raise ValueError("prompt byte budget must be positive")
        return self


class RoboticsPromptObjectBinding(Contract):
    object_id: str
    category: str
    role: str
    prompt_label: str

    @field_validator("object_id", "category", "prompt_label")
    @classmethod
    def nonempty_single_line(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned or "\n" in value or "\r" in value:
            raise ValueError("prompt object bindings must be non-empty single lines")
        return cleaned

    @field_validator("prompt_label")
    @classmethod
    def timing_free_label(cls, value: str) -> str:
        _assert_timing_free(value, field="prompt object label")
        return value


class RoboticsPromptActionBinding(Contract):
    """One selected task-program step carried into a model prompt record."""

    step_index: int
    operation: str
    subjects: tuple[str, ...]
    target: str | None = None

    @model_validator(mode="after")
    def valid_program_step(self) -> "RoboticsPromptActionBinding":
        if self.step_index < 0 or not self.operation or not self.subjects:
            raise ValueError("prompt action steps need an index, operation, and subjects")
        symbols = [*self.subjects, *([self.target] if self.target else [])]
        if any(not _SAFE_OBJECT_ID.fullmatch(value) for value in symbols):
            raise ValueError("prompt action steps reference unsafe object symbols")
        return self


class RoboticsPromptSemanticBindings(Contract):
    relation: PromptRelation
    predicate_index: int
    manipuland: RoboticsPromptObjectBinding
    destination: RoboticsPromptObjectBinding
    supports: tuple[RoboticsPromptObjectBinding, ...]

    @model_validator(mode="after")
    def role_consistency(self) -> "RoboticsPromptSemanticBindings":
        if self.predicate_index < 0:
            raise ValueError("prompt predicate index must be non-negative")
        if self.manipuland.role != "manipuland":
            raise ValueError("prompt subject must bind the manipuland role")
        if self.destination.role != "destination":
            raise ValueError("prompt object must bind the destination role")
        if not self.supports or any(item.role != "support" for item in self.supports):
            raise ValueError("prompt semantics need explicit support-role bindings")
        ids = [
            self.manipuland.object_id,
            self.destination.object_id,
            *(item.object_id for item in self.supports),
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("prompt semantic roles must bind distinct task objects")
        return self


class RoboticsPromptManifest(Contract):
    """Content-addressed record of a rendered model prompt."""

    schema_version: Literal[
        "robotics_prompt_manifest.v1", "robotics_prompt_manifest.v2"
    ] = "robotics_prompt_manifest.v2"
    case_id: str
    task_contract_sha256: str
    wam: PromptWAM
    template_id: str
    template_version: str
    rendered_prompt: str
    semantic_bindings: RoboticsPromptSemanticBindings | None = None
    source_instruction: str | None = None
    source_instruction_sha256: str | None = None
    object_bindings: tuple[RoboticsPromptObjectBinding, ...] = ()
    object_bindings_sha256: str | None = None
    action_program: tuple[RoboticsPromptActionBinding, ...] = ()
    action_program_sha256: str | None = None
    prompt_sha256: str
    semantic_bindings_sha256: str | None = None

    @model_validator(mode="after")
    def prompt_identity(self) -> "RoboticsPromptManifest":
        if not self.case_id:
            raise ValueError("robotics prompt manifest must bind one case")
        if not re.fullmatch(r"[0-9a-f]{64}", self.task_contract_sha256):
            raise ValueError(
                "robotics prompt task contract needs an explicit SHA-256 digest"
            )
        prompt = " ".join(self.rendered_prompt.split())
        if prompt != self.rendered_prompt or not prompt:
            raise ValueError("rendered robotics prompt must be one normalized line")
        _assert_timing_free(prompt, field="rendered robotics prompt")
        if self.prompt_sha256 != _sha256_text(prompt):
            raise ValueError("rendered robotics prompt SHA-256 changed")
        if self.schema_version == "robotics_prompt_manifest.v1":
            if self.semantic_bindings is None:
                raise ValueError("v1 prompt manifest lacks semantic bindings")
            key = (self.wam, self.semantic_bindings.relation)
            template = ROBOTICS_PROMPT_REGISTRY.get(key)
            if template is None:
                raise ValueError(f"no reviewed robotics prompt template for {key!r}")
            if (self.template_id, self.template_version) != (
                template.template_id,
                template.template_version,
            ):
                raise ValueError("robotics prompt manifest changes its reviewed template")
            if len(prompt.encode("utf-8")) > template.max_utf8_bytes:
                raise ValueError(
                    f"rendered prompt exceeds the {template.max_utf8_bytes}-byte "
                    f"budget for {self.wam}"
                )
            expected_prompt = template.format_string.format(
                manipuland=self.semantic_bindings.manipuland.prompt_label,
                destination=self.semantic_bindings.destination.prompt_label,
            )
            if prompt != expected_prompt:
                raise ValueError(
                    "rendered robotics prompt differs from its semantic template"
                )
            if self.semantic_bindings_sha256 != self.semantic_bindings.content_hash():
                raise ValueError("robotics prompt semantic bindings SHA-256 changed")
            return self

        style = MODEL_PROMPT_STYLES.get(self.wam)
        if style is None or (self.template_id, self.template_version) != style[:2]:
            raise ValueError("v2 prompt manifest changes its model prompt style")
        if len(prompt.encode("utf-8")) > style[2]:
            raise ValueError(
                f"rendered prompt exceeds the {style[2]}-byte budget for {self.wam}"
            )
        if self.semantic_bindings is not None or self.semantic_bindings_sha256 is not None:
            raise ValueError("v2 prompt manifest must not collapse a task to one relation")
        if not self.source_instruction or prompt != " ".join(self.source_instruction.split()):
            raise ValueError("v2 prompt must preserve the selected short task instruction")
        if self.source_instruction_sha256 != _sha256_text(self.source_instruction):
            raise ValueError("v2 source instruction SHA-256 changed")
        if not self.object_bindings or not self.action_program:
            raise ValueError("v2 prompt manifest needs the complete object/program record")
        if self.object_bindings_sha256 != _sha256_text(
            "|".join(item.content_hash() for item in self.object_bindings)
        ):
            raise ValueError("v2 prompt object record changed")
        if self.action_program_sha256 != _sha256_text(
            "|".join(item.content_hash() for item in self.action_program)
        ):
            raise ValueError("v2 prompt action program changed")
        return self


_PROMPT_TEMPLATES = (
    RoboticsPromptTemplate(
        template_id="cosmos3_droid.put_inside.short_command",
        template_version="1.0.0",
        wam="cosmos3_nano_droid",
        relation="inside",
        # This short DROID command syntax is the form validated by the native
        # cup-to-tray closed-loop run. Object labels remain task-derived.
        format_string="Put the {manipuland} in the {destination}.",
        max_utf8_bytes=160,
    ),
    RoboticsPromptTemplate(
        template_id="lingbot_va_base.pick_place_inside.task_only",
        template_version="1.0.0",
        wam="lingbot_va_base",
        relation="inside",
        format_string=(
            "Pick up the {manipuland} and place it inside the {destination}."
        ),
        max_utf8_bytes=220,
    ),
)

ROBOTICS_PROMPT_REGISTRY: Mapping[
    tuple[PromptWAM, PromptRelation], RoboticsPromptTemplate
] = MappingProxyType({(item.wam, item.relation): item for item in _PROMPT_TEMPLATES})

MODEL_PROMPT_STYLES: Mapping[PromptWAM, tuple[str, str, int]] = MappingProxyType(
    {
        "cosmos3_nano_droid": (
            "cosmos3_droid.short_task_instruction",
            "2.0.0",
            180,
        ),
        "lingbot_va_base": (
            "lingbot_va_base.short_task_instruction",
            "2.0.0",
            240,
        ),
    }
)


def _prompt_label(item: TaskObject) -> str:
    if not _SAFE_OBJECT_ID.fullmatch(item.object_id):
        raise ValueError(
            f"task object id {item.object_id!r} is unsafe for a model prompt"
        )
    object_label = re.sub(r"[_-]+", " ", item.object_id).strip().lower()
    category_label = re.sub(r"[_-]+", " ", item.category).strip().lower()
    if not category_label:
        raise ValueError(f"task object {item.object_id!r} has no semantic category")
    # Reviewed task ids such as red_cup and blue_tray preserve useful visual
    # disambiguation. Opaque ids fall back to their declared category.
    category_head = category_label.split()[-1]
    label = (
        object_label
        if category_head in object_label.split()
        else category_label
    )
    _assert_timing_free(label, field="task-derived prompt label")
    return label


def _binding(item: TaskObject) -> RoboticsPromptObjectBinding:
    if not _SAFE_OBJECT_ID.fullmatch(item.role):
        raise ValueError(
            f"task object {item.object_id!r} has unsafe prompt role {item.role!r}"
        )
    return RoboticsPromptObjectBinding(
        object_id=item.object_id,
        category=item.category,
        role=item.role,
        prompt_label=_prompt_label(item),
    )


def _resolve_semantics(task: TaskInstructionSpec) -> RoboticsPromptSemanticBindings:
    objects: dict[str, TaskObject] = {}
    for item in task.objects:
        if item.object_id in objects:
            raise ValueError(f"duplicate task object id: {item.object_id!r}")
        objects[item.object_id] = item

    terminal = [
        (index, predicate)
        for index, predicate in enumerate(task.success_predicates)
        if predicate.kind in {"inside", "on"}
    ]
    if len(terminal) != 1:
        relations = [predicate.kind for _index, predicate in terminal]
        raise ValueError(
            "robotics prompt requires exactly one reviewed terminal relation; "
            f"found {relations!r}"
        )
    predicate_index, predicate = terminal[0]
    if predicate.kind != "inside":
        raise ValueError(
            f"unsupported robotics prompt relation: {predicate.kind!r}"
        )
    if predicate.object is None:
        raise ValueError("inside prompt relation needs a destination object")
    try:
        manipuland = objects[predicate.subject]
        destination = objects[predicate.object]
    except KeyError as exc:
        raise ValueError(
            f"prompt relation references unknown task object: {exc.args[0]!r}"
        ) from exc
    if manipuland.role != "manipuland" or destination.role != "destination":
        raise ValueError(
            "inside prompt relation must bind manipuland to destination roles"
        )

    # Auxiliary success predicates may certify motion, contact, and release,
    # but they may not silently introduce a second subject or destination.
    for auxiliary in task.success_predicates:
        if auxiliary is predicate:
            continue
        if auxiliary.kind not in {"pose_delta", "contact", "released"}:
            raise ValueError(
                f"unsupported robotics prompt relation: {auxiliary.kind!r}"
            )
        if auxiliary.subject != manipuland.object_id:
            raise ValueError("prompt success predicates disagree on the manipuland")
        if auxiliary.kind == "contact" and auxiliary.object != destination.object_id:
            raise ValueError("prompt contact predicate disagrees on the destination")
        if (
            auxiliary.kind in {"pose_delta", "released"}
            and auxiliary.object is not None
        ):
            raise ValueError(
                f"{auxiliary.kind} prompt predicate may not bind another object"
            )

    supports = tuple(
        _binding(item)
        for item in sorted(task.objects, key=lambda candidate: candidate.object_id)
        if item.role == "support"
    )
    return RoboticsPromptSemanticBindings(
        relation="inside",
        predicate_index=predicate_index,
        manipuland=_binding(manipuland),
        destination=_binding(destination),
        supports=supports,
    )


def render_robotics_prompt(
    task: TaskInstructionSpec, wam: PromptWAM | str
) -> RoboticsPromptManifest:
    """Render one short model prompt without collapsing the task program.

    Current adapters consume the selected declarative instruction exactly as a
    normalized one-line command. The complete object table and action program
    remain in the manifest for audit; case names and legacy relation templates
    do not select prompt text.
    """

    style = MODEL_PROMPT_STYLES.get(wam)  # type: ignore[arg-type]
    if style is None:
        raise ValueError(f"no registered robotics prompt style for WAM {wam!r}")
    prompt = " ".join(task.instruction.split())
    if not prompt:
        raise ValueError("task instruction is empty")
    _assert_timing_free(prompt, field="task-derived model prompt")
    bindings = tuple(
        _binding(item)
        for item in sorted(task.objects, key=lambda candidate: candidate.object_id)
    )
    known = {item.object_id for item in task.objects}
    program = tuple(
        RoboticsPromptActionBinding(
            step_index=index,
            operation=step.operation,
            subjects=tuple(step.subjects),
            target=step.target,
        )
        for index, step in enumerate(task.action_program)
    )
    if not program:
        # Older typed tasks may predate action_program. Preserve compatibility
        # only when one unambiguous terminal predicate supplies the step.
        terminal = [
            predicate
            for predicate in task.success_predicates
            if predicate.object is not None
            and predicate.kind
            in {
                "inside",
                "in_container",
                "on",
                "on_region",
                "stacked",
                "inserted",
                "covering",
            }
        ]
        if len(terminal) != 1:
            raise ValueError("task prompt needs a declarative action program")
        predicate = terminal[0]
        if predicate.subject not in known or predicate.object not in known:
            raise ValueError("task prompt predicate references an unknown object")
        program = (
            RoboticsPromptActionBinding(
                step_index=0,
                operation=predicate.kind,
                subjects=(predicate.subject,),
                target=predicate.object,
            ),
        )
    return RoboticsPromptManifest(
        schema_version="robotics_prompt_manifest.v2",
        case_id=task.case_id,
        task_contract_sha256=task.content_hash(),
        wam=wam,  # type: ignore[arg-type]
        template_id=style[0],
        template_version=style[1],
        rendered_prompt=prompt,
        source_instruction=task.instruction,
        source_instruction_sha256=_sha256_text(task.instruction),
        object_bindings=bindings,
        object_bindings_sha256=_sha256_text(
            "|".join(item.content_hash() for item in bindings)
        ),
        action_program=program,
        action_program_sha256=_sha256_text(
            "|".join(item.content_hash() for item in program)
        ),
        prompt_sha256=_sha256_text(prompt),
    )


def load_reference_prompt_task(
    reference_manifest: str | Path,
    *,
    expected_case_id: str | None = None,
    expected_task_instruction: str | None = None,
) -> tuple[Path, RoboticsReferenceManifest, Path, TaskInstructionSpec]:
    """Load the typed task colocated with a certified reference manifest."""

    reference_path = Path(reference_manifest).resolve()
    reference = RoboticsReferenceManifest.model_validate_json(
        reference_path.read_text(encoding="utf-8")
    )
    task_path = reference_path.parent / "task_instruction.json"
    if not task_path.is_file():
        raise FileNotFoundError(
            f"reference bundle is missing typed task_instruction.json: {task_path}"
        )
    task = TaskInstructionSpec.model_validate_json(
        task_path.read_text(encoding="utf-8")
    )
    if task.case_id != reference.case_id:
        raise ValueError("reference manifest and typed task identify different cases")
    if task.instruction != reference.task_instruction:
        raise ValueError(
            "reference manifest instruction differs from its typed task contract"
        )
    if expected_case_id is not None and reference.case_id != expected_case_id:
        raise ValueError("WAM input and reference prompt identify different cases")
    if (
        expected_task_instruction is not None
        and reference.task_instruction != expected_task_instruction
    ):
        raise ValueError(
            "WAM input task instruction differs from its reference prompt contract"
        )
    return reference_path, reference, task_path, task


def render_reference_robotics_prompt(
    reference_manifest: str | Path,
    wam: PromptWAM | str,
) -> RoboticsPromptManifest:
    """Render a model prompt from the reference bundle's typed task contract."""

    _reference_path, _reference, _task_path, task = load_reference_prompt_task(
        reference_manifest
    )
    return render_robotics_prompt(task, wam)


def copy_reference_prompt_snapshot(
    reference_manifest: str | Path, output_root: str | Path
) -> tuple[str, str, str]:
    """Copy the exact prompt authority needed to validate a relocated WAM bundle."""

    reference_path, _reference, task_path, _task = load_reference_prompt_task(
        reference_manifest
    )
    root = Path(output_root).resolve()
    snapshot_root = root / "reference_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    reference_copy = snapshot_root / "reference_manifest.json"
    task_copy = snapshot_root / "task_instruction.json"
    reference_copy.write_bytes(reference_path.read_bytes())
    task_copy.write_bytes(task_path.read_bytes())
    return (
        reference_copy.relative_to(root).as_posix(),
        task_copy.relative_to(root).as_posix(),
        sha256_file(task_copy),
    )


def load_bound_robotics_prompt(
    input_manifest: str | Path,
    *,
    schema_version: str,
    reference_manifest: str,
    reference_manifest_sha256: str | None,
    reference_snapshot: str | None,
    reference_task_snapshot: str | None,
    reference_task_sha256: str | None,
    case_id: str,
    task_instruction: str,
    wam: str,
    prompt_manifest: str | None,
    prompt_manifest_sha256: str | None,
    rendered_prompt: str | None,
) -> tuple[Path, RoboticsPromptManifest]:
    """Reload and verify the prompt contract bound by a formal WAM input."""

    if schema_version != "wam_input_bundle.v5":
        raise ValueError(
            "formal WAM inference requires typed prompt input manifest v5"
        )
    if prompt_manifest is None or prompt_manifest_sha256 is None:
        raise ValueError("WAM input v5 is missing its prompt manifest binding")
    input_path = Path(input_manifest).resolve()
    raw_prompt_path = Path(prompt_manifest)
    if raw_prompt_path.is_absolute():
        raise ValueError("WAM input prompt manifest path must be relative")
    prompt_path = (input_path.parent / raw_prompt_path).resolve()
    try:
        prompt_path.relative_to(input_path.parent)
    except ValueError as exc:
        raise ValueError("WAM input prompt manifest escapes its bundle") from exc
    if not prompt_path.is_file():
        raise FileNotFoundError(f"WAM prompt manifest is missing: {prompt_path}")
    if sha256_file(prompt_path) != prompt_manifest_sha256:
        raise ValueError("WAM prompt manifest digest changed")

    if not all(
        isinstance(value, str) and value
        for value in (
            reference_manifest_sha256,
            reference_snapshot,
            reference_task_snapshot,
            reference_task_sha256,
        )
    ):
        raise ValueError("WAM input v5 is missing its reference prompt snapshot")

    def snapshot_path(value: str, label: str) -> Path:
        raw = Path(value)
        if raw.is_absolute() or ".." in raw.parts or value != raw.as_posix():
            raise ValueError(f"WAM {label} must be bundle-relative")
        resolved = (input_path.parent / raw).resolve()
        try:
            resolved.relative_to(input_path.parent)
        except ValueError as exc:
            raise ValueError(f"WAM {label} escapes its bundle") from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(resolved)
        return resolved

    assert isinstance(reference_snapshot, str)
    assert isinstance(reference_task_snapshot, str)
    assert isinstance(reference_manifest_sha256, str)
    assert isinstance(reference_task_sha256, str)
    reference_path = snapshot_path(reference_snapshot, "reference snapshot")
    task_path = snapshot_path(reference_task_snapshot, "reference task snapshot")
    if sha256_file(reference_path) != reference_manifest_sha256:
        raise ValueError("WAM reference snapshot digest changed")
    if sha256_file(task_path) != reference_task_sha256:
        raise ValueError("WAM reference task snapshot digest changed")
    reference = RoboticsReferenceManifest.model_validate_json(
        reference_path.read_text(encoding="utf-8")
    )
    task = TaskInstructionSpec.model_validate_json(
        task_path.read_text(encoding="utf-8")
    )
    if reference.case_id != case_id or task.case_id != case_id:
        raise ValueError("WAM input and reference prompt identify different cases")
    if (
        reference.task_instruction != task_instruction
        or task.instruction != task_instruction
    ):
        raise ValueError("WAM input task differs from its reference prompt snapshot")
    expected = render_robotics_prompt(task, wam)
    actual = RoboticsPromptManifest.model_validate_json(
        prompt_path.read_text(encoding="utf-8")
    )
    if actual.case_id != case_id:
        raise ValueError("prompt manifest and WAM input identify different cases")
    if actual.task_contract_sha256 != task.content_hash():
        raise ValueError("prompt manifest task contract digest changed")
    if actual.canonical_json() != expected.canonical_json():
        raise ValueError(
            "prompt manifest differs from the reference task's reviewed rendering"
        )
    if rendered_prompt != actual.rendered_prompt:
        raise ValueError("WAM input rendered prompt differs from its prompt manifest")
    return prompt_path, actual


# ``build`` is a convenient contract-oriented spelling for callers that write
# the returned manifest as an artifact.
build_robotics_prompt_manifest = render_robotics_prompt
