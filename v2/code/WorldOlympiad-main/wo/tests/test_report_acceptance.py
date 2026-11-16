import json

import pytest

from wo.agent import stages


def _fixture(tmp_path, monkeypatch, *, status="complete", pending=None, cae_status="scored"):
    case_id, model, seed = "case_a", "lingbot_va", 0
    root = tmp_path / "run"
    ep = root / case_id / f"{model}_s{seed}"
    (ep / "native_run/simulation").mkdir(parents=True)
    (ep / "native_run/simulation/trajectory.json").write_text("{}")
    (ep / "case").mkdir()
    ref = root / "references" / case_id
    ref.mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"reference": str(ref)}))
    v2 = {"case_id": case_id, "model": model, "seed": seed,
          "status": "complete", "pending_tracks": [], "worldolympiad2_score": .41,
          "tracks": {name: {"status": "scored", "score": .4} for name in
                     ("physical", "geometry", "interaction")},
          "cae_details": {"status": cae_status, "score": .6, "source": "simulator_execution"}}
    v2["tracks"]["interaction"]["submetrics"] = {
        "continuous_action": {"status": cae_status, "score": .6}}
    v3_tracks = {name: {"status": "scored", "score": .4} for name in
                 ("physical", "geometry", "interaction")}
    v3_tracks["geometry"]["submetrics"] = {"action_trajectory": {"status": "scored", "score": .5}}
    v3_tracks["interaction"]["submetrics"] = {"action_speed": {"status": "scored", "score": .5}}
    v3 = {"case_id": case_id, "model": model, "seed": seed,
          "status": status, "pending_tracks": pending or [],
          "worldolympiad2_score": .5, "tracks": v3_tracks}
    (root / "manifest.json").write_text(json.dumps({"cases": [case_id], "models": [model], "seeds": [seed]}))
    (root / "results.json").write_text(json.dumps({"episodes": [v2], "coverage": {
        "matrix_complete": True, "missing_references": [], "pending_matrix_entries": []}}))
    (root / "results_v3.json").write_text(json.dumps({"episodes": [v3], "coverage": {
        "matrix_complete": True, "missing_references": [], "pending_matrix_entries": []}}))
    case = ep / "case"
    (case / "worldolympiad_case.json").write_text(json.dumps({
        "view_contract": "matched_single_physical_view_v2",
        "reference_view_id": "exterior_left", "generated_view_id": "exterior_left"}))
    monkeypatch.setattr(stages, "run_dir", lambda _: root)
    monkeypatch.setattr("wo.agent.score_stage.valid_receipt", lambda _: True)
    monkeypatch.setattr("wo.eval.reference_binding.verify_episode_binding", lambda *_: {"status": "verified"})
    monkeypatch.setattr("wo.eval.reference_binding.verify_binding", lambda *_a, **_k: {"status": "verified"})
    monkeypatch.setattr("wo.agent.episode_stage.verify_evaluation_view", lambda *_: {
        "view": "exterior_left", "video_sha256": "video"})
    return root


def test_report_gate_requires_v2_and_v3_complete_rows(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    result = stages._check_report({"run": "test"})
    assert result == {"episodes": 1, "results_v3": str(root / "results_v3.json"), "complete": True}


@pytest.mark.parametrize("status,pending,cae", [
    ("pending", ["geometry"], "scored"),
    ("complete", [], "pending"),
])
def test_report_gate_rejects_pending_report_or_cae(tmp_path, monkeypatch, status, pending, cae):
    _fixture(tmp_path, monkeypatch, status=status, pending=pending, cae_status=cae)
    with pytest.raises(ValueError):
        stages._check_report({"run": "test"})
