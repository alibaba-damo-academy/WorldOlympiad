import json
import pytest
from wo.eval import report

REASON = "articulated support host: the native RoboTwin table replaces the support"


@pytest.mark.parametrize("model", ["fastwam", "lingbot_va"])
def test_support_substitution_is_pending_and_cannot_finish_a_matrix_row(tmp_path, model):
    epdir = tmp_path / "cabinet" / f"{model}_s0"
    epdir.mkdir(parents=True)
    path = epdir / "episode.json"
    path.write_text(json.dumps({"case_id":"cabinet","model":model,"seed":0,
        "task_success":None,"status":"not_applicable","reason":REASON,
        "termination_reason":"not_applicable: "+REASON,"videos":{}}))
    original = path.read_bytes()
    row = report.episode_result(epdir)
    assert row["status"] == "pending"
    assert row["applicability_kind"] == "implementation_gap"
    assert row["task_success"] is None and row["worldolympiad2_score"] is None
    assert path.read_bytes() == original
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/cabinet.json").write_text(json.dumps({"case_id":"cabinet","family":"articulation"}))
    audit = report.coverage(tmp_path,[row],models=[model],seeds=[0])
    assert len(audit["pending_matrix_entries"]) == 1
    assert audit["pending_matrix_entries"][0]["reason"] == "scoring incomplete"


def test_legacy_reason_suffix_is_read_without_needing_video():
    row = report.applicability_result({"termination_reason":"not_applicable: "+REASON})
    assert row["status"] == "pending"
    assert row["reason"] == REASON


def test_explicit_protocol_exclusion_and_other_terminations_are_preserved():
    row = report.applicability_result({"termination_reason":"not_applicable: incompatible action protocol",
                                      "applicability":{"kind":"action_protocol"}})
    assert row["status"] == "not_applicable"
    assert row["applicability_kind"] == "action_protocol"
    assert report.applicability_result({"termination_reason":"evaluation_window_complete"}) == {}
