import json
import pytest
from wo.eval import report


def make_episode(tmp_path, monkeypatch, generic, probe):
    ep = tmp_path / "sample/model_s0"
    (ep / "case").mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"case_id":"sample", "model":"model", "seed":0,
                                                "task_family":"transfer", "task_success":False}))
    (ep / "case/score.json").write_text(json.dumps({"physical":{"score":generic},
        "three_d":{"final_score_normalized":.6}, "interaction":{"score":.4}}))
    monkeypatch.setattr(report,"episode_probes",lambda _: probe)
    monkeypatch.setattr(report,"episode_cae",lambda _: {"status":"scored","score":.8,"source":"simulator_execution"})
    return ep


@pytest.mark.parametrize("generic,probe", [
    (None, {"status":"scored","score":.8}),
    (.8, {"status":"pending","score":None,"reason":"trajectory missing"}),
    (.8, {"status":"unmeasurable","score":None,"reason":"no recorded state"}),
    (None, {"status":"not_applicable","score":None,"reason":"declared exclusion"}),
])
def test_missing_required_physical_half_cannot_produce_complete_total(tmp_path, monkeypatch, generic, probe):
    ep = make_episode(tmp_path,monkeypatch,generic,probe)
    row = report.episode_result(ep)
    assert row["status"] == "pending"
    assert row["physical"] is None
    assert "physical" in row["pending_tracks"]
    assert row["worldolympiad2_score"] is None
    assert row["task_success"] is False
    assert row["physical_generic"] == generic
    assert row["physical_task"] == probe["score"]


def test_zero_is_a_measured_failure_not_a_missing_score(tmp_path, monkeypatch):
    ep = make_episode(tmp_path,monkeypatch,0.,{"status":"scored","score":0.,"source":"simulator_execution"})
    row = report.episode_result(ep)
    assert row["status"] == "complete"
    assert row["physical"] == 0.
    assert row["task_success"] is False


def test_explicit_physical_na_can_exclude_a_half(tmp_path, monkeypatch):
    ep = make_episode(tmp_path,monkeypatch,.8,{"status":"not_applicable","score":None,"reason":"declared exclusion"})
    row = report.episode_result(ep)
    assert row["status"] == "complete"
    assert row["physical"] == .8


def test_probe_error_preserves_missing_evidence_instead_of_reweighting(tmp_path, monkeypatch):
    ep = make_episode(tmp_path,monkeypatch,.8,{})
    def broken(_):raise ValueError("wrong trajectory binding")
    monkeypatch.setattr(report,"episode_probes",broken)
    row = report.episode_result(ep)
    assert row["worldolympiad2_score"] is None
    assert row["physical_probe_details"]["reason"] == "wrong trajectory binding"


def test_physical_submetric_status_and_effective_weights_are_reported():
    full = report.physical_result({"score":.8}, {"status":"scored","score":.4,"source":"simulator_execution"})
    assert full["score"] == pytest.approx(.6)
    assert full["weights"] == {"generic_video":.5,"task_probes":.5}
    missing = report.physical_result({"score":.8}, {"status":"pending","score":None,"reason":"not recorded"})
    assert missing["submetrics"]["task_probes"]["reason"] == "not recorded"
    excluded = report.physical_result({"score":.8}, {"status":"not_applicable","reason":"declared exclusion"})
    assert excluded["weights"] == {"generic_video":1.,"task_probes":0.}
    assert report.physical_result({"status":"not_applicable","reason":"excluded"},
                                  {"status":"not_applicable","reason":"excluded"})["status"] == "not_applicable"


def test_confirmed_invalid_execution_preserves_history_and_cannot_enter_aggregates(tmp_path, monkeypatch):
    ep=make_episode(tmp_path,monkeypatch,.8,{"status":"scored","score":.8,"source":"simulator_execution"})
    original=(ep/"episode.json").read_bytes();score=(ep/"case/score.json").read_bytes()
    assert report.episode_result(ep)["status"]=="complete"
    (ep/"evaluation_invalidated.json").write_text(json.dumps({"reason":"native environment omitted reference collider"}))
    row=report.episode_result(ep)
    assert row["status"]=="pending" and row["task_success"] is None
    assert row["historical_task_success"] is False and row["worldolympiad2_score"] is None
    aggregate=report.aggregates([row])["by_model"]["model"]
    assert aggregate["success_measured"]==0 and aggregate["physical"] is None
    assert (ep/"episode.json").read_bytes()==original and (ep/"case/score.json").read_bytes()==score
    (ep/"evaluation_invalidated.json").unlink()
    assert report.episode_result(ep)["status"]=="complete"
