"""Author typed robotics eval cases with OpenAI, then freeze them deterministically.

The configured model chooses task content from a high-level request. The
coordinator never authors case payloads; it only launches this bounded
author/validate operation
and orchestrates the downstream harness stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from pydantic import Field, model_validator

from .io import write_json
from .models import Contract
from .robotics_case_generator import generate_batch
from .robotics_contracts import (
    ActionStep,
    ArticulationComponentSpec,
    SuccessPredicate,
    TaskObject,
    WAMName,
)
from .robotics_reference_materializer import compiled_execution_steps
from .robotics_task_compiler import ACTION_OPERATORS, compile_task
from .robotics_task_intent import review_task_payloads, require_review_passed
from .wam_adapter_contracts import reference_requirements_for_wam


AUTHORING_SKILL = (
    Path(__file__).parents[1] / "skills" / "robotics-eval-task-authoring" / "SKILL.md"
)
DEFAULT_REQUEST = (
    "Generate a diverse suite of high-quality, single-manipulant robotics "
    "evaluation tasks suitable for action-trajectory comparison."
)
# The configured Responses proxy is fronted by a WAF that rejects urllib's
# default signature with Cloudflare error 1010. Every OpenAI-backed stage sends
# this explicit User-Agent so a credential-valid request is not blocked.
OPENAI_USER_AGENT = "WorldOlympiad2-RoboticsOpenAI/1.0"

# Exactly the reusable operators that the shared compiler and reference
# materializer already realize.  The author is never offered an operator or a
# program shape that has no downstream implementation; adding a name here is not
# a capability claim by itself.
SHARED_ACTION_OPERATORS: tuple[str, ...] = (
    "pick",
    "place",
    "stack",
    "push",
    "insert",
    "cover",
    "press",
    "pull",
    "close",
    "flip",
    "strike",
    "pour",
)
SHARED_PROGRAM_SHAPES: tuple[tuple[str, ...], ...] = (
    ("push",),
    ("press",),
    ("pull",),
    ("close",),
    ("pick", "place"),
    ("pick", "stack"),
    ("pick", "insert"),
    ("pick", "cover"),
    ("pick", "flip"),
    ("pick", "strike"),
    ("pick", "pour"),
)


class AuthoredCase(Contract):
    instruction: str
    scene_brief: str
    initial_state: str
    objects: list[TaskObject]
    action_program: list[ActionStep]
    success_predicates: list[SuccessPredicate]
    articulation_components: list[ArticulationComponentSpec] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_family_label(cls, value: Any) -> Any:
        if isinstance(value, dict) and "task_family" in value:
            value = dict(value)
            value.pop("task_family", None)
        return value

    @model_validator(mode="after")
    def single_visible_manipuland(self) -> "AuthoredCase":
        manipulands = [item for item in self.objects if item.role == "manipuland"]
        if len(manipulands) > 1:
            raise ValueError("authored cases allow at most one primary manipuland")
        if len(manipulands) != 1 and not self.articulation_components:
            raise ValueError(
                "authored cases need exactly one primary manipuland or a declared "
                "articulation component"
            )
        if len(self.success_predicates) != 1:
            raise ValueError("authored cases need exactly one terminal predicate")
        return self


class AuthoredSuite(Contract):
    schema_version: str = "robotics_task_suite_proposal.v2"
    cases: list[AuthoredCase] = Field(min_length=1)

    @model_validator(mode="after")
    def supported_version(self) -> "AuthoredSuite":
        if self.schema_version != "robotics_task_suite_proposal.v2":
            raise ValueError("unsupported authored suite schema")
        return self


def _strict_output_schema(*, case_count: int, predicates: list[str]) -> dict[str, Any]:
    symbol = {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_.-]{0,63}$"}
    nullable_symbol = {"anyOf": [symbol, {"type": "null"}]}
    nullable_number = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    object_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "object_id": symbol,
            "category": symbol,
            "role": {"type": "string", "enum": ["manipuland", "destination", "support"]},
            "visual_description": {"type": "string"},
        },
        "required": ["object_id", "category", "role", "visual_description"],
    }
    action_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(SHARED_ACTION_OPERATORS),
            },
            "subjects": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": symbol,
            },
            "target": nullable_symbol,
            "constraints": {"type": "string", "description": "A JSON object containing reusable action constraints. Use {} for no constraints."},
        },
        "required": ["operation", "subjects", "target", "constraints"],
    }
    articulation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_id": symbol,
            "host_object_id": symbol,
            "semantic_kind": symbol,
            "joint_type": {"type": "string", "enum": ["prismatic", "revolute"]},
            "visual_description": {"type": "string"},
        },
        "required": [
            "component_id",
            "host_object_id",
            "semantic_kind",
            "joint_type",
            "visual_description",
        ],
    }
    predicate_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": predicates},
            "subject": symbol,
            "object": nullable_symbol,
            "threshold": nullable_number,
            "params": {"type": "object"},
        },
        "required": ["kind", "subject", "object", "threshold"],
    }
    case_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "instruction": {"type": "string"},
            "scene_brief": {"type": "string"},
            "initial_state": {"type": "string"},
            "objects": {"type": "array", "minItems": 2, "items": object_schema},
            "articulation_components": {
                "type": "array",
                "maxItems": 2,
                "items": articulation_schema,
            },
            "action_program": {"type": "array", "minItems": 1, "items": action_schema},
            "success_predicates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": predicate_schema,
            },
        },
        "required": [
            "instruction",
            "scene_brief",
            "initial_state",
            "objects",
            "articulation_components",
            "action_program",
            "success_predicates",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {
                "type": "string",
                "const": "robotics_task_suite_proposal.v2",
            },
            "cases": {
                "type": "array",
                "minItems": case_count,
                "maxItems": case_count,
                "items": case_schema,
            },
        },
        "required": ["schema_version", "cases"],
    }


def _response_text(body: dict[str, Any]) -> str:
    raw = body.get("output_text")
    if isinstance(raw, str) and raw.strip():
        return raw
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if (
                isinstance(content, dict)
                and content.get("type") == "output_text"
                and isinstance(content.get("text"), str)
                and content["text"].strip()
            ):
                return content["text"]
    raise RuntimeError("Responses API returned no structured robotics task suite")


def _api_configuration() -> tuple[str, str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be configured")
    api_base = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    model = os.environ.get(
        "ROBOTICS_TASK_AUTHOR_MODEL",
        os.environ.get("SCENECODE_OPENAI_MODEL", "gpt-5.6-sol"),
    )
    return api_key, api_base, model


def _decode_authored_suite(text: str) -> AuthoredSuite:
    payload = json.loads(text)
    for case in payload.get("cases", []):
        for step in case.get("action_program", []):
            if isinstance(step.get("constraints"), str):
                value = json.loads(step["constraints"])
                if not isinstance(value, dict):
                    raise ValueError("action constraints must decode to a JSON object")
                step["constraints"] = value
    return AuthoredSuite.model_validate(payload)


def _author_once(
    *,
    case_count: int,
    requested_wam: WAMName,
    evaluation_request: str,
    feedback: str | None,
) -> tuple[AuthoredSuite, dict[str, Any]]:
    if not AUTHORING_SKILL.is_file():
        raise FileNotFoundError(AUTHORING_SKILL)
    api_key, api_base, model = _api_configuration()
    requirements = reference_requirements_for_wam(requested_wam)
    predicates = list(requirements.supported_terminal_predicates)
    author_input = {
        "evaluation_request": evaluation_request,
        "case_count": case_count,
        "requested_wam": requested_wam,
        "runtime_capabilities": {
            "program_shapes": [list(shape) for shape in SHARED_PROGRAM_SHAPES],
            "terminal_predicates": predicates,
            "maximum_manipulands": requirements.maximum_manipulands,
            "maximum_terminal_predicates": requirements.maximum_terminal_predicates,
            "articulation_contract": {
                "use_when": (
                    "the task moves a jointed component such as a door, drawer, "
                    "handle or a movable control surface"
                ),
                "declaration": (
                    "declare one articulation_components entry with component_id, "
                    "host_object_id (must be a declared object), semantic_kind, "
                    "joint_type and visual_description"
                ),
                "addressing": (
                    "the operator subject and the success predicate subject are the "
                    "component_id; the success predicate object is the "
                    "host_object_id"
                ),
                "shapes": ["pull", "press", "close"],
            },
            "native_cameras": [
                {"view": item.source_view, "size_wh": list(item.size_wh)}
                for item in requirements.cameras
            ],
        },
        "previous_validation_feedback": feedback,
    }
    payload = {
        "model": model,
        "instructions": AUTHORING_SKILL.read_text(encoding="utf-8"),
        "input": json.dumps(author_input, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "robotics_task_suite_proposal",
                "strict": True,
                "schema": _strict_output_schema(
                    case_count=case_count, predicates=predicates
                ),
            }
        },
    }
    req = request.Request(
        api_base + "/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": OPENAI_USER_AGENT,
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    suite = _decode_authored_suite(_response_text(body))
    if len(suite.cases) != case_count:
        raise ValueError(
            f"case_count={case_count} but OpenAI authored {len(suite.cases)} cases"
        )
    receipt = {
        "response_id": body.get("id"),
        "model": body.get("model") or model,
        "request_status": body.get("status"),
    }
    return suite, receipt


def _validate_compilable_suite(
    suite: AuthoredSuite,
    *,
    batch_id: str,
    requested_wam: WAMName,
) -> list[dict[str, Any]]:
    # Reuse the frozen task generator's complete contract, duplicate check, and
    # deterministic ids without publishing anything during author retries.
    from .robotics_case_generator import validate_proposals

    proposals = [item.model_dump(mode="json", exclude_none=False) for item in suite.cases]
    tasks = validate_proposals(
        batch_id=batch_id,
        case_count=len(proposals),
        requested_wam=requested_wam,
        proposals=proposals,
    )
    for task in tasks:
        compiled_execution_steps(compile_task(task))
    return proposals


def author_and_generate_batch(
    *,
    output_root: str | Path,
    batch_id: str,
    case_count: int,
    requested_wam: WAMName,
    evaluation_request: str = DEFAULT_REQUEST,
) -> Path:
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    evaluation_request = " ".join(evaluation_request.split())
    if not evaluation_request:
        evaluation_request = DEFAULT_REQUEST
    root = Path(output_root).resolve()
    proposal_root = root / "robotics_suite_proposals"
    proposal_path = proposal_root / f"{batch_id}.json"
    receipt_path = proposal_root / f"{batch_id}.authoring.json"
    manifest_path = root / "robotics_task_batches" / batch_id / "batch_manifest.json"
    proposal_exists = proposal_path.exists()
    receipt_exists = receipt_path.exists()
    manifest_exists = manifest_path.exists()
    if proposal_exists or receipt_exists or manifest_exists:
        if not (proposal_exists and receipt_exists):
            raise RuntimeError(
                "incomplete authored batch requires inspection before retry: "
                f"{batch_id}"
            )
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            prior.get("requested_wam") != requested_wam
            or prior.get("case_count") != case_count
            or prior.get("evaluation_request") != evaluation_request
        ):
            raise ValueError("batch id is already frozen to a different authoring request")
        if prior.get("proposal_sha256") != hashlib.sha256(proposal_path.read_bytes()).hexdigest():
            raise ValueError("cached authored proposal digest changed")
        if "task_intent_review" not in prior:
            prior["task_intent_review"] = review_task_payloads(
                json.loads(proposal_path.read_text())["cases"], evaluation_request=evaluation_request
            )
            write_json(receipt_path, prior)
        require_review_passed(prior["task_intent_review"])
        return generate_batch(
            output_root=root,
            batch_id=batch_id,
            case_count=case_count,
            requested_wam=requested_wam,
            proposals=proposal_path,
        )
    maximum_attempts = max(1, int(os.environ.get("OPENAI_TASK_AUTHOR_MAX_ATTEMPTS", "3")))
    feedback: str | None = None
    suite: AuthoredSuite | None = None
    receipt: dict[str, Any] = {}
    proposals: list[dict[str, Any]] | None = None
    last_error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            suite, receipt = _author_once(
                case_count=case_count,
                requested_wam=requested_wam,
                evaluation_request=evaluation_request,
                feedback=feedback,
            )
            proposals = _validate_compilable_suite(
                suite, batch_id=batch_id, requested_wam=requested_wam
            )
            intent_review = review_task_payloads(proposals, evaluation_request=evaluation_request)
            require_review_passed(intent_review)
            receipt["task_intent_review"] = intent_review
            receipt["attempt_count"] = attempt
            break
        except (ValueError, RuntimeError, error.URLError) as exc:
            proposals = None
            last_error = exc
            feedback = f"The previous proposal was rejected: {exc}"
    if suite is None or proposals is None:
        raise RuntimeError(
            f"OpenAI task author failed after {maximum_attempts} attempts: {last_error}"
        ) from last_error

    proposal_root.mkdir(parents=True, exist_ok=True)
    proposal_payload = suite.model_dump(mode="json", exclude_none=False)
    proposal_text = json.dumps(
        proposal_payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if proposal_path.exists():
        if proposal_path.read_text(encoding="utf-8") != proposal_text:
            raise ValueError("batch id is already bound to a different authored proposal")
    else:
        proposal_path.write_text(proposal_text, encoding="utf-8")

    skill_sha256 = hashlib.sha256(AUTHORING_SKILL.read_bytes()).hexdigest()
    write_json(
        receipt_path,
        {
            "schema_version": "robotics_task_authoring_receipt.v1",
            "batch_id": batch_id,
            "requested_wam": requested_wam,
            "case_count": case_count,
            "evaluation_request": evaluation_request,
            "authoring_skill": AUTHORING_SKILL.relative_to(
                Path(__file__).parents[1]
            ).as_posix(),
            "authoring_skill_sha256": skill_sha256,
            "proposal": proposal_path.relative_to(root).as_posix(),
            "proposal_sha256": hashlib.sha256(proposal_text.encode("utf-8")).hexdigest(),
            "batch_manifest": manifest_path.relative_to(root).as_posix(),
            **receipt,
        },
    )
    manifest = generate_batch(
        output_root=root,
        batch_id=batch_id,
        case_count=case_count,
        requested_wam=requested_wam,
        proposals=proposal_path,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--case-count", type=int, required=True)
    parser.add_argument("--evaluation-request", default=DEFAULT_REQUEST)
    parser.add_argument(
        "--requested-wam",
        choices=["cosmos3_nano_droid", "lingbot_va_base", "lingbot_va_robotwin"],
        required=True,
    )
    args = parser.parse_args()
    print(
        author_and_generate_batch(
            output_root=args.output_root,
            batch_id=args.batch_id,
            case_count=args.case_count,
            requested_wam=args.requested_wam,
            evaluation_request=args.evaluation_request,
        )
    )


if __name__ == "__main__":
    main()
