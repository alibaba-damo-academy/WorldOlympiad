"""Case-routed, evidence-first evaluation planning for the WorldOlympiad harness.

The design follows HarnessEval-W's useful separation of skill registry,
case-only routing, immutable plans, evidence collection, and final audit.  It
does not add an outer research loop or let model identity influence routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .models import Contract


SkillKind = Literal["observation", "core", "diagnostic"]
SkillStatus = Literal["pass", "fail", "not_applicable", "error"]


class EvaluationSkill(Contract):
    skill_id: str
    kind: SkillKind
    question: str
    required_inputs: list[str]
    supported_events: list[str] = Field(default_factory=lambda: ["*"])


class SkillSelection(Contract):
    skill_id: str
    role: SkillKind
    reason: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class SkippedSkill(Contract):
    skill_id: str
    reason: str


class EvaluationPlan(Contract):
    schema_version: Literal["worldolympiad_evaluation_plan.v1"] = (
        "worldolympiad_evaluation_plan.v1"
    )
    case_id: str
    event_type: str
    view_contract: Literal[
        "single_main_view_v1", "matched_single_physical_view_v2"
    ] = "single_main_view_v1"
    routing_inputs: dict[str, Any]
    selected_skills: list[SkillSelection]
    skipped_skills: list[SkippedSkill]
    aggregation_policy: Literal["required_skills_fail_closed"] = (
        "required_skills_fail_closed"
    )

    @model_validator(mode="after")
    def valid_plan(self) -> "EvaluationPlan":
        if "model_id" in self.routing_inputs:
            raise ValueError("evaluation routing must not depend on model identity")
        selected = [item.skill_id for item in self.selected_skills]
        skipped = [item.skill_id for item in self.skipped_skills]
        if len(selected) != len(set(selected)) or set(selected) & set(skipped):
            raise ValueError("skills must be uniquely selected or skipped")
        required = {"render_quality", "motion_quality", "physical", "geometry", "interaction"}
        if (
            self.routing_inputs.get("continuous_action_applicable", True)
            and self.routing_inputs.get("continuous_action_available", True)
        ):
            required.add("continuous_action")
        missing = required - set(selected)
        if missing:
            raise ValueError(f"evaluation plan omits required skills: {sorted(missing)}")
        return self


class SkillEvidence(Contract):
    skill_id: str
    status: SkillStatus
    artifact_paths: list[str] = Field(default_factory=list)
    measurements: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class CaseEvidenceBundle(Contract):
    schema_version: Literal["worldolympiad_case_evidence.v1"] = (
        "worldolympiad_case_evidence.v1"
    )
    case_id: str
    plan_path: str
    evidence: list[SkillEvidence]
    passed: bool


SKILLS: tuple[EvaluationSkill, ...] = (
    EvaluationSkill(skill_id="render_quality", kind="observation", question="Is the single main-view video sharp, stable, decodable, and temporally complete?", required_inputs=["reference_video", "generated_video"]),
    EvaluationSkill(skill_id="motion_quality", kind="observation", question="Is motion temporally coherent without freezing, cuts, or camera drift?", required_inputs=["reference_video", "generated_video"]),
    EvaluationSkill(skill_id="target_visibility", kind="observation", question="Can the target be tracked from the first frame through the event?", required_inputs=["reference_video", "generated_video", "object_name"]),
    EvaluationSkill(skill_id="physical", kind="core", question="Does the imagined event obey gravity, rigidity, stability, and plausible dynamics?", required_inputs=["reference_video", "generated_video", "prompt"]),
    EvaluationSkill(skill_id="geometry", kind="core", question="Are scene geometry, depth, scale, and object identity preserved?", required_inputs=["reference_video", "generated_video"]),
    EvaluationSkill(skill_id="interaction", kind="core", question="Does the commanded interaction occur with the correct participants and outcome?", required_inputs=["reference_video", "generated_video", "prompt"]),
    EvaluationSkill(skill_id="continuous_action", kind="core", question="Do independently extracted Dream2Flow action trajectories match in path, velocity, and contact evolution?", required_inputs=["reference_flow", "generated_flow"]),
    EvaluationSkill(skill_id="target_zone_completion", kind="diagnostic", question="Does the manipulated object finish inside the instructed tabletop target zone?", required_inputs=["reference_video", "generated_video", "prompt"], supported_events=["block_sort_tray", "block_push_target"]),
    EvaluationSkill(skill_id="orientation_transition", kind="diagnostic", question="Does target orientation change in the correct direction and magnitude?", required_inputs=["reference_video", "generated_video"], supported_events=["cup_tip", "stack_collapse"]),
    EvaluationSkill(skill_id="contact_topology", kind="diagnostic", question="Do multi-object contacts form and break in the correct order?", required_inputs=["reference_video", "generated_video"], supported_events=["stack_collapse"]),
    EvaluationSkill(skill_id="robot_task_completion", kind="diagnostic", question="Does the robot cause the instructed state transition without an unrelated action?", required_inputs=["reference_video", "generated_video", "prompt"], supported_events=["robot_pick_place", "robot_push", "robot_articulation", "robot_stack"]),
)


def plan_case_evaluation(case: dict[str, Any]) -> EvaluationPlan:
    """Route skills from case semantics only; model_id is provenance, never input."""

    case_id = str(case["case_id"])
    event_type = str(case.get("event_type") or case_id)
    continuous_action_status = case.get("tracks", {}).get(
        "continuous_action", "required"
    )
    routing_inputs = {
        "event_type": event_type,
        "interaction_mode": case.get("interaction_mode", "external_intervention"),
        "benchmark_status": case.get("benchmark_status"),
        "evaluation_scope": case.get("evaluation_scope"),
        "reference_view_count": int(case.get("reference_view_count", 1)),
        "generated_view_count": int(case.get("generated_view_count", 1)),
        "reference_view_id": str(case.get("reference_view_id", "main")),
        "generated_view_id": str(case.get("generated_view_id", "main")),
        "continuous_action_applicable": (
            continuous_action_status
            != "not_applicable_to_official_rigid_grasp_optimizer"
        ),
        "continuous_action_available": continuous_action_status == "required",
        "continuous_action_status": continuous_action_status,
    }
    if routing_inputs["reference_view_count"] != 1 or routing_inputs["generated_view_count"] != 1:
        raise ValueError("formal WorldOlympiad CAE requires exactly one main view on each side")
    if routing_inputs["reference_view_id"] != routing_inputs["generated_view_id"]:
        raise ValueError("formal WorldOlympiad CAE requires one matched physical view")
    selected: list[SkillSelection] = []
    skipped: list[SkippedSkill] = []
    for skill in SKILLS:
        applies = "*" in skill.supported_events or event_type in skill.supported_events
        if skill.skill_id == "robot_task_completion":
            applies = routing_inputs["interaction_mode"] == "robot_contact"
        if skill.skill_id == "continuous_action":
            applies = (
                applies
                and routing_inputs["continuous_action_applicable"]
                and routing_inputs["continuous_action_available"]
            )
        if applies:
            reason = (
                "mandatory observation gate"
                if skill.kind == "observation"
                else "mandatory WorldOlympiad track"
                if skill.kind == "core"
                else f"diagnostic matches event_type={event_type}"
            )
            parameters = {
                "view_id": routing_inputs["reference_view_id"],
                "fail_closed": True,
            }
            if skill.skill_id == "continuous_action":
                parameters["frontend"] = "symmetric_dream2flow"
            selected.append(
                SkillSelection(
                    skill_id=skill.skill_id,
                    role=skill.kind,
                    reason=reason,
                    parameters=parameters,
                )
            )
        else:
            skipped.append(
                SkippedSkill(
                    skill_id=skill.skill_id,
                    reason=(
                        "not applicable to the official rigid-grasp optimizer"
                        if skill.skill_id == "continuous_action"
                        and not routing_inputs["continuous_action_applicable"]
                        else (
                            "applicable but unavailable: "
                            f"{routing_inputs['continuous_action_status']}"
                            if skill.skill_id == "continuous_action"
                            and routing_inputs["continuous_action_applicable"]
                            and not routing_inputs["continuous_action_available"]
                            else f"not applicable to event_type={event_type}"
                        )
                    ),
                )
            )
    return EvaluationPlan(
        case_id=case_id,
        event_type=event_type,
        view_contract=case.get("view_contract", "single_main_view_v1"),
        routing_inputs=routing_inputs,
        selected_skills=selected,
        skipped_skills=skipped,
    )


def audit_evidence(
    plan: EvaluationPlan,
    evidence: list[SkillEvidence],
    *,
    plan_path: str,
) -> CaseEvidenceBundle:
    expected = {item.skill_id for item in plan.selected_skills}
    actual = {item.skill_id for item in evidence}
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"evidence does not match frozen plan: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    by_id = {item.skill_id: item for item in evidence}
    roles = {item.skill_id: item.role for item in plan.selected_skills}
    invalid_na = [
        skill_id
        for skill_id in expected
        if by_id[skill_id].status == "not_applicable"
        and (roles[skill_id] != "diagnostic" or not by_id[skill_id].reason.strip())
    ]
    if invalid_na:
        raise ValueError(
            "not_applicable is allowed only for diagnostic evidence with an exact reason: "
            + ", ".join(sorted(invalid_na))
        )
    passed = all(
        by_id[skill_id].status == "pass"
        or (
            roles[skill_id] == "diagnostic"
            and by_id[skill_id].status == "not_applicable"
        )
        for skill_id in expected
    )
    return CaseEvidenceBundle(
        case_id=plan.case_id,
        plan_path=plan_path,
        evidence=evidence,
        passed=passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("case")
    plan_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        case = json.loads(Path(args.case).read_text(encoding="utf-8"))
        output = plan_case_evaluation(case).write_json(args.output)
        print(output)


if __name__ == "__main__":
    main()
