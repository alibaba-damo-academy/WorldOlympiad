"""Task and stage contracts shared by Physics→WM and Robotics→WAM."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from .models import Contract
from .robotics_contracts import (ActionStep, ArticulationComponentSpec,
                                 SuccessPredicate, TaskObject)


STAGES = ("scene", "physical_reference", "reference_video", "model", "evaluation")
MODEL_PROFILES = {
    "lingbot_world2_14b_causal_fast": {
        "kind": "wm", "protocol": "open_loop_first_frame_text_camera_video",
    },
    "cosmos3_nano_droid": {"kind": "wam", "protocol": "simulator_closed_loop"},
    "lingbot_va_robotwin": {"kind": "wam", "protocol": "simulator_closed_loop"},
    "lingbot_va_base": {
        "kind": "wam", "protocol": "open_loop_autoregressive_video_action",
    },
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


class AgentTask(Contract):
    """One task request; the program and embodiment select capabilities, not a family."""

    schema_version: Literal["worldolympiad_task.v1"] = "worldolympiad_task.v1"
    task_id: str
    instruction: str
    scene_brief: str
    initial_state: str
    objects: list[TaskObject]
    articulation_components: list[ArticulationComponentSpec] = Field(default_factory=list)
    operation_program: list[ActionStep]
    success_predicates: list[SuccessPredicate]
    requested_models: list[str]
    model_prompts: dict[str, str] = Field(default_factory=dict)
    embodiment: str | None = None
    seed: int | None = None
    reference_duration_s: float = 4.0
    reference_fps: float = 15.0
    source: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_task(self) -> "AgentTask":
        if not _IDENTIFIER.fullmatch(self.task_id):
            raise ValueError("unsafe task_id")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if (not math.isfinite(self.reference_duration_s)
                or self.reference_duration_s <= 0
                or not math.isfinite(self.reference_fps)
                or self.reference_fps < 1):
            raise ValueError("invalid reference timing")
        if not self.objects or not self.operation_program or not self.success_predicates:
            raise ValueError("task needs objects, operations and success predicates")
        ids = [item.object_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task object id")
        component_ids = [item.component_id for item in self.articulation_components]
        if len(component_ids) != len(set(component_ids)) or set(component_ids) & set(ids):
            raise ValueError("duplicate or overlapping articulation component id")
        if any(item.host_object_id not in ids for item in self.articulation_components):
            raise ValueError("articulation host is not a task object")
        known = set(ids) | set(component_ids)
        referenced = {name for step in self.operation_program for name in
                      [*step.subjects, *([step.target] if step.target else [])]}
        referenced.update(predicate.subject for predicate in self.success_predicates)
        referenced.update(predicate.object for predicate in self.success_predicates
                          if predicate.object)
        if referenced - known:
            raise ValueError(f"task references unknown objects: {sorted(referenced - known)}")
        if not self.requested_models or len(self.requested_models) != len(set(self.requested_models)):
            raise ValueError("requested_models must be nonempty and unique")
        unknown = set(self.requested_models) - MODEL_PROFILES.keys()
        if unknown:
            raise ValueError(f"unknown model profile: {sorted(unknown)}")
        if set(self.model_prompts) - set(self.requested_models):
            raise ValueError("model prompt names an unrequested profile")
        if any(not value.strip() for value in self.model_prompts.values()):
            raise ValueError("model prompt must be nonempty")
        if any(MODEL_PROFILES[name]["kind"] == "wam" for name in self.requested_models):
            if not self.embodiment:
                raise ValueError("WAM task needs a robot embodiment")
        return self

    def content_hash(self) -> str:
        return digest_json(self.model_dump(mode="json"))


def from_robotics_task(value: Any) -> AgentTask:
    """Read a frozen robotics v2/v3 task without rewriting the source file."""
    from .robotics_contracts import TaskInstructionSpec

    old = value if isinstance(value, TaskInstructionSpec) else TaskInstructionSpec.model_validate(value)
    return AgentTask(
        task_id=old.case_id,
        instruction=old.instruction,
        scene_brief=old.scene_brief or old.initial_state,
        initial_state=old.initial_state,
        objects=old.objects,
        articulation_components=old.articulation_components,
        operation_program=old.action_program,
        success_predicates=old.success_predicates,
        requested_models=list(old.requested_wams),
        embodiment=old.robot,
        seed=old.case_seed,
        reference_duration_s=old.duration_sec,
        reference_fps=old.fps,
        source={"schema_version": old.schema_version, "content_hash": old.content_hash()},
    )


class StageReceipt(Contract):
    schema_version: Literal["worldolympiad_stage_receipt.v1"] = "worldolympiad_stage_receipt.v1"
    task_id: str
    run_id: str
    stage: Literal["scene", "physical_reference", "reference_video", "model", "evaluation"]
    model_profile: str | None = None
    status: Literal["complete", "failed", "blocked"]
    task_sha256: str
    input_sha256: str
    code_sha256: str
    skill_versions: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)
    failure_owner: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def check_receipt(self) -> "StageReceipt":
        if not _IDENTIFIER.fullmatch(self.task_id) or not _IDENTIFIER.fullmatch(self.run_id):
            raise ValueError("unsafe receipt identity")
        if self.stage in {"reference_video", "model", "evaluation"}:
            if self.model_profile not in MODEL_PROFILES:
                raise ValueError("video/model/evaluation stage needs a registered profile")
        elif self.model_profile is not None:
            raise ValueError("pre-model stage must not name a model profile")
        for value in (self.task_sha256, self.input_sha256, self.code_sha256,
                      *self.skill_versions.values(), *self.artifacts.values(),
                      *self.evidence.values()):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("receipt hashes must be SHA-256 digests")
        if self.status == "complete" and not self.artifacts:
            raise ValueError("complete stage needs artifact digests")
        if self.status != "complete" and not self.failure_owner:
            raise ValueError("failed stage needs an owner")
        return self
