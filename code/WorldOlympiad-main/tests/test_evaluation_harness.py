from __future__ import annotations

import pytest

from autoresearch.evaluation_harness import (
    SkillEvidence,
    audit_evidence,
    plan_case_evaluation,
)


def _case(event: str = "block_sort_tray") -> dict:
    return {
        "case_id": event,
        "event_type": event,
        "model_id": "must_not_affect_routing",
        "benchmark_status": "pipeline_validation",
        "evaluation_scope": "synthetic_cross_domain_real_wam",
        "reference_view_count": 1,
        "generated_view_count": 1,
    }


def test_routing_is_model_independent_and_auditable() -> None:
    left = plan_case_evaluation(_case())
    right_case = _case()
    right_case["model_id"] = "different_model"
    right = plan_case_evaluation(right_case)
    assert left.canonical_json() == right.canonical_json()
    assert "model_id" not in left.routing_inputs
    assert any(item.skill_id == "target_zone_completion" for item in left.selected_skills)
    assert all(item.reason for item in left.skipped_skills)


def test_formal_plan_rejects_multiple_views() -> None:
    case = _case()
    case["generated_view_count"] = 2
    with pytest.raises(ValueError, match="exactly one main view"):
        plan_case_evaluation(case)


def test_matched_physical_view_is_frozen_into_every_skill() -> None:
    case = _case("robot_pick_place")
    case.update(
        {
            "view_contract": "matched_single_physical_view_v2",
            "reference_view_id": "wrist",
            "generated_view_id": "wrist",
        }
    )
    plan = plan_case_evaluation(case)
    assert plan.view_contract == "matched_single_physical_view_v2"
    assert all(item.parameters["view_id"] == "wrist" for item in plan.selected_skills)
    case["generated_view_id"] = "top"
    with pytest.raises(ValueError, match="matched physical view"):
        plan_case_evaluation(case)


def test_robot_contact_case_routes_robot_completion_skill() -> None:
    case = _case()
    case["interaction_mode"] = "robot_contact"
    plan = plan_case_evaluation(case)
    assert any(
        item.skill_id == "robot_task_completion" for item in plan.selected_skills
    )


def test_non_rigid_case_skips_continuous_action_before_plan_is_frozen() -> None:
    case = _case("robot_pick_place")
    case["tracks"] = {
        "physical": "required",
        "geometry": "required",
        "interaction": "required",
        "continuous_action": (
            "not_applicable_to_official_rigid_grasp_optimizer"
        ),
    }
    plan = plan_case_evaluation(case)
    assert plan.routing_inputs["continuous_action_applicable"] is False
    assert not any(
        item.skill_id == "continuous_action" for item in plan.selected_skills
    )
    skipped = {item.skill_id: item.reason for item in plan.skipped_skills}
    assert skipped["continuous_action"] == (
        "not applicable to the official rigid-grasp optimizer"
    )


def test_applicable_but_unavailable_continuous_action_is_explicitly_skipped() -> None:
    case = _case("robot_pick_place")
    case["tracks"] = {
        "continuous_action": "unavailable_missing_formal_registration"
    }
    plan = plan_case_evaluation(case)
    assert plan.routing_inputs["continuous_action_applicable"] is True
    assert plan.routing_inputs["continuous_action_available"] is False
    assert not any(
        item.skill_id == "continuous_action" for item in plan.selected_skills
    )
    skipped = next(
        item for item in plan.skipped_skills if item.skill_id == "continuous_action"
    )
    assert "unavailable_missing_formal_registration" in skipped.reason


def test_evidence_bundle_fails_closed_on_missing_skill() -> None:
    plan = plan_case_evaluation(_case())
    with pytest.raises(ValueError, match="missing"):
        audit_evidence(plan, [], plan_path="evaluation_plan.json")
    evidence = [
        SkillEvidence(skill_id=item.skill_id, status="pass", artifact_paths=["result.json"])
        for item in plan.selected_skills
    ]
    bundle = audit_evidence(plan, evidence, plan_path="evaluation_plan.json")
    assert bundle.passed


def test_not_applicable_cannot_hide_missing_core_measurement() -> None:
    plan = plan_case_evaluation(_case())
    evidence = [
        SkillEvidence(
            skill_id=item.skill_id,
            status="not_applicable" if item.role == "core" else "pass",
            reason="backend could not measure it" if item.role == "core" else "",
        )
        for item in plan.selected_skills
    ]
    with pytest.raises(ValueError, match="only for diagnostic evidence"):
        audit_evidence(plan, evidence, plan_path="evaluation_plan.json")


def test_reasoned_diagnostic_not_applicable_does_not_become_zero() -> None:
    plan = plan_case_evaluation(_case())
    evidence = [
        SkillEvidence(
            skill_id=item.skill_id,
            status="not_applicable" if item.role == "diagnostic" else "pass",
            reason=(
                "case has no applicable orientation or contact-topology measurement"
                if item.role == "diagnostic"
                else ""
            ),
        )
        for item in plan.selected_skills
    ]
    bundle = audit_evidence(plan, evidence, plan_path="evaluation_plan.json")
    assert bundle.passed is True
