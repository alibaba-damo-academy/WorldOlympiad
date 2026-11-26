from __future__ import annotations

import json

from autoresearch.full_track_report import build_report, build_report_v2, build_report_v3


def test_native_single_case_score_entrypoint_joins_official_cae(tmp_path):
    _case(tmp_path)
    root = tmp_path / "case"
    (root / "model_judge_case.json").rename(root / "score.json")
    plan = {"schema_version": "worldolympiad_track_applicability.v1", "cases": [{
        "case_id": "case", "task_sha256": "frozen", "tracks": {name: True for name in
            ("physical", "geometry", "interaction", "continuous_action")}}]}
    action = {"cases": [{"case_id": "case", "continuous_action": 0.2,
                         "path_score": 0.4, "velocity_score": 0.1}]}
    report = build_report_v3([root], applicability_plan=plan, action_report=action)
    assert report["complete_count"] == 1
    assert abs(report["mean_score"] - (0.8 + 0.6 + 0.3) / 3) < 1e-12
    assert report["cases"][0]["track_results"]["interaction"]["submetrics"]["continuous_action"]["components"]["speed_curve"] == 0.1


def _case(root, *, cae_score=None):
    case = root / "case"
    case.mkdir(parents=True)
    (case / "worldolympiad_case.json").write_text(
        json.dumps(
            {
                "case_id": "case",
                "tracks": {
                    "continuous_action": (
                        "required"
                        if cae_score is not None
                        else "not_applicable_to_official_rigid_grasp_optimizer"
                    )
                },
            }
        )
    )
    (case / "model_judge_case.json").write_text(
        json.dumps(
            {
                "physical": {"score": 0.8},
                "three_d": {"final_score_normalized": 0.6},
                "interaction": {"score": 0.4},
                "continuous_action": (
                    {"status": "applicable", "score": cae_score}
                    if cae_score is not None
                    else {"status": "not_applicable"}
                ),
                "worldolympiad2_score": 0.6,
            }
        )
    )


def test_missing_cae_gets_explicit_pipeline_coverage_zero(tmp_path) -> None:
    _case(tmp_path)
    record = build_report([tmp_path])["cases"][0]
    assert record["track_scores"]["continuous_action"] == 0.0
    assert record["evaluator_coverage_failure"] is True
    assert record["coverage_enforced_four_track_score"] == 0.45


def test_real_cae_is_preserved(tmp_path) -> None:
    _case(tmp_path, cae_score=0.5)
    action = {"cases": [{"case_id": "case", "continuous_action": 0.5,
              "status": "scored", "score_semantics": "official_dream2flow_optimizer_action_trajectory",
              "evaluator_coverage_failure": False}]}
    record = build_report([tmp_path], action_report=action)["cases"][0]
    assert record["track_scores"]["continuous_action"] == 0.5
    assert record["evaluator_coverage_failure"] is False


def test_judge_only_cae_cannot_replace_official_evidence(tmp_path):
    _case(tmp_path, cae_score=0.9)
    row=build_report([tmp_path])["cases"][0]
    assert row["track_scores"]["continuous_action"]==0.0
    assert row["evaluator_coverage_failure"] is True


def test_coverage_gap_cannot_leak_a_stale_positive_score(tmp_path):
    _case(tmp_path)
    actions={"cases":[{"case_id":"case","continuous_action":0.9,"evaluator_coverage_failure":True}]}
    assert build_report([tmp_path],action_report=actions)["cases"][0]["track_scores"]["continuous_action"]==0.0


def test_duplicate_case_ids_are_not_silently_merged(tmp_path):
    import pytest
    _case(tmp_path/'first');_case(tmp_path/'second')
    with pytest.raises(ValueError,match='duplicate case id'):
        build_report([tmp_path/'first',tmp_path/'second'])
    with pytest.raises(ValueError,match='duplicate action case id'):
        build_report([tmp_path/'first'],action_report={"cases":[{"case_id":"case"},{"case_id":"case"}]})


def test_v2_semantic_na_uses_only_applicable_tracks(tmp_path):
    _case(tmp_path)
    plan = {"schema_version": "worldolympiad_track_applicability.v1", "cases": [{
        "case_id": "case", "task_sha256": "a" * 64,
        "tracks": {"physical": True, "geometry": True,
                   "interaction": {"applicable": False, "reason": "no interaction event"},
                   "continuous_action": True},
    }]}
    actions = {"cases": [{"case_id": "case", "continuous_score": .7,
                           "source": "object-motion-cae"}]}
    report = build_report_v2([tmp_path], applicability_plan=plan,
                             action_report=actions)
    case = report["cases"][0]
    assert report["schema_version"] == "worldolympiad_full_track_report.v2"
    assert case["track_results"]["interaction"]["status"] == "not_applicable"
    assert case["worldolympiad2_score"] == 0.7
    assert build_report([tmp_path])["cases"][0]["track_scores"]["continuous_action"] == 0.


def test_v2_missing_required_evidence_stays_pending(tmp_path):
    _case(tmp_path)
    plan = {"schema_version": "worldolympiad_track_applicability.v1", "cases": [{
        "case_id": "case", "task_sha256": "a" * 64,
        "tracks": {"physical": True, "geometry": True,
                   "interaction": {"applicable": False, "reason": "no interaction event"},
                   "continuous_action": True},
    }]}
    report = build_report_v2([tmp_path], applicability_plan=plan)
    assert report["mean_score"] is None
    assert report["cases"][0]["status"] == "pending"
    assert report["cases"][0]["track_results"]["continuous_action"]["status"] == "pending"


def test_current_report_places_cae_under_interaction_and_retains_raw_measurements(tmp_path):
    import pytest
    _case(tmp_path, cae_score=.5)
    plan = {"schema_version": "worldolympiad_track_applicability.v1", "cases": [{
        "case_id": "case", "task_sha256": "a" * 64,
        "tracks": {"physical": True, "geometry": True,
                   "interaction": True, "continuous_action": True}}]}
    action = {"cases": [{"case_id": "case", "continuous_action": .5,
              "path_score": .8, "velocity_score": .3125,
              "evaluator_coverage_failure": False}]}
    report = build_report_v3([tmp_path], applicability_plan=plan, action_report=action)
    row = report["cases"][0]
    assert set(row["track_results"]) == {"physical", "geometry", "interaction"}
    interaction = row["track_results"]["interaction"]
    assert interaction["score"] == pytest.approx(.45)
    assert interaction["submetrics"]["continuous_action"]["components"]["trajectory_shape"] == .8
    assert report["mean_score"] == pytest.approx((.8+.6+.45)/3)
    assert build_report_v2([tmp_path], applicability_plan=plan, action_report=action)["mean_score"] == .575
