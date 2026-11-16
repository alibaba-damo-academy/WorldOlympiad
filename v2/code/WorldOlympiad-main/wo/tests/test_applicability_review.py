import json
from wo.eval import report


def row(tmp_path, reason, **extra):
    ep = tmp_path / "episode"
    ep.mkdir()
    (ep / "episode.json").write_text(json.dumps({
        "case_id": "case", "model": "model", "status": "not_applicable",
        "task_success": None, "reason": reason, **extra}))
    return report.episode_result(ep)


def test_missing_measurement_does_not_finish_matrix_entry(tmp_path):
    result = row(tmp_path, "Native runner has NO MEASUREMENT for goal 'inserted'")
    assert result["status"] == "pending"
    assert result["applicability_kind"] == "implementation_gap"
    assert result["pending_tracks"] == ["native_task_measurement"]
    assert result["worldolympiad2_score"] is None


def test_structured_adapter_gap_is_pending(tmp_path):
    result = row(tmp_path, "adapter needs a goal implementation", applicability={"kind": "missing_adapter"})
    assert result["status"] == "pending"
    assert result["task_success"] is None


def test_intrinsic_not_applicable_remains_explicit(tmp_path):
    result = row(tmp_path, "native simulator cannot represent deformable bodies", applicability={"kind": "simulation_representation"})
    assert result["status"] == "not_applicable"
    assert result["applicability_kind"] == "simulation_representation"
    assert result["pending_tracks"] == []


def test_rigid_importer_joint_gap_cannot_be_declared_na(tmp_path):
    result = row(tmp_path, "kettle: expert uses a multi-link URDF; Cosmos WorldIR rigid-body importer cannot preserve its joints", applicability={"kind": "simulation_representation"})
    assert result["status"] == "pending"
    assert result["applicability_kind"] == "implementation_gap"
    assert result["task_success"] is None


def test_untyped_or_unknown_exclusion_requires_review(tmp_path):
    result = row(tmp_path, "adapter could not start this task")
    assert result["status"] == "pending"
    assert result["applicability_kind"] == "applicability_review"
    assert result["pending_tracks"] == ["applicability_evidence"]
    assert report.applicability_result({"status":"not_applicable","reason":"unknown restriction",
        "applicability":{"kind":"declared_not_applicable"}})["status"] == "pending"


def test_scene_initialization_failure_is_not_policy_inapplicability(tmp_path):
    result = row(tmp_path, "reference scene is not at rest in SAPIEN: Objects is unstable in seed(0), unstable objects: red_cube",
                 applicability={"kind":"simulation_representation"})
    assert result["status"] == "pending"
    assert result["applicability_kind"] == "implementation_gap"
    assert result["task_success"] is None
