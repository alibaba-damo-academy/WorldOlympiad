from wo.eval.metric_audit import metric_audit


def test_unrelated_questions_are_na_missing_questions_pending_and_failures_zero():
    rows = [dict(case_id="a", model="m", seed=0, physical_details={"results": [
        {"question_id": "gravity", "related": True, "compliant": True, "confidence": .8},
        {"question_id": "buoyancy", "related": False, "confidence": 0.},
        {"question_id": "impact", "related": True, "compliant": False},
        {"question_id": "compression", "related": True, "compliant": None}],
        "dimension_scores": {"mechanics": .4}})]
    audit = metric_audit(rows)
    q = lambda key: audit["physical_details.questions." + key]
    assert q("gravity")["mean"] == .8
    assert q("impact")["N"] == 1 and q("impact")["mean"] == 0
    assert q("buoyancy")["NA"] == 1 and q("buoyancy")["N"] == 0
    assert q("compression")["pending"] == 1
    assert q("melting")["pending"] == 1 and q("melting")["NA"] == 0
    assert audit["physical_details.dimension_scores.mechanics"]["pending"] == 1


def test_evidence_is_scorer_specific_and_no_bookkeeping_metrics():
    row = dict(case_id="a", model="m", seed=0, episode="/run/a/m_s0",
               physical_task_source="simulator_execution",
               physical_probe_details={"status": "scored", "score": 0., "source": "simulator_execution"},
               geometry_details={"gs_score": .4, "scorer_type": "qwenvl_server", "reconstruction_model_name": "da3", "num_workers": 4},
               interaction_details={"score": .2, "backend": "qwenvl_server"})
    audit = metric_audit([row])
    assert audit["physical_probe_details.score"]["N"] == 1
    assert audit["physical_probe_details.score"]["mean"] == 0
    assert audit["geometry_details.gs_score"]["source_evidence"] == ["video_reconstruction:da3;judge:qwenvl_server"]
    assert "simulator_execution" not in audit["interaction_details.score"]["source_evidence"]
    assert "geometry_details.num_workers" not in audit
    assert audit["geometry_details.gs_score"]["subset"] == [{"case_id": "a", "model": "m", "seed": 0}]
    assert audit["geometry_details.gs_score"]["observations"][0]["evidence"].endswith("score.json#/three_d/gs_score")


def test_chunk_aggregation_is_per_episode_and_missing_component_is_pending():
    row = lambda case, chunks: dict(case_id=case, model="m", seed=0,
        interaction_details={"chunks": chunks, "transitions": []})
    audit = metric_audit([row("a", [{"visual_quality": 5}]), row("b", [{"visual_quality": 1}] * 4)])
    assert audit["interaction_details.chunks.visual_quality"]["N"] == 2
    assert audit["interaction_details.chunks.visual_quality"]["mean"] == .6
    assert audit["interaction_details.chunks.text_alignment"]["pending"] == 2
    assert audit["interaction_details.transitions.overall"]["NA"] == 1
    assert audit["interaction_details.transitions.overall"]["pending"] == 1


def test_probe_na_needs_complete_declared_measurements_and_track_na_needs_reason():
    base = dict(probe_count=1, measured_probe_count=1, status="scored", source="simulator_execution")
    rows = [dict(case_id="a", model="m", physical_probe_details={**base, "probes": [{"probe": "lift", "score": .9}], "per_probe": {"lift": .9}}),
            dict(case_id="b", model="m", physical_probe_details={**base, "probes": [{"probe": "joint", "score": 0}], "per_probe": {"joint": 0}}),
            dict(case_id="c", model="m", physical_probe_details={"status": "failed"})]
    audit = metric_audit(rows)["physical_probe_details.per_probe.lift"]
    assert (audit["N"], audit["NA"], audit["pending"]) == (1, 1, 1)
    assert metric_audit([dict(model="m", cae_details={"status": "not_applicable", "reason": "declared protocol limitation"})])["cae_details.score"]["NA"] == 1
    assert metric_audit([dict(model="m", cae_details={"status": "not_applicable"})])["cae_details.score"]["pending"] == 1


def test_measured_absence_of_event_is_explicit_na_but_missing_track_is_pending():
    import numpy as np
    from wo.eval.physical_probes import carry_attachment, no_spontaneous_motion, aggregate
    t = np.arange(10) / 15
    stationary = np.zeros((10, 3))
    held = carry_attachment(stationary, stationary, t)
    assert held["status"] == "not_applicable" and held["measured"]["held_frames"] == 0
    missing = no_spontaneous_motion({"object": stationary}, t)
    assert missing["status"] == "pending" and "track" in missing["reason"]
    assert aggregate([held])["subscore_applicability"]["plausibility"]["status"] == "not_applicable"
    assert aggregate([missing])["status"] == "pending"
    measured = {"probe": "lift_rise", "score": 0., "measured": {"max_rise_m": 0}, "status": "scored"}
    result = aggregate([held, measured])
    assert result["score"] == 0 and result["status"] == "scored"


def test_paper_diagnostic_exports_clip_and_does_not_rewrite_recorded_score():
    row = dict(model="m", interaction_details={"score": .25, "summary": {
        "chunk_mean": 5., "transition_mean": 3., "global_score": 4.}},
        clip_details={"clip_model": "ViT-B/32", "summary": {"semantic_adherence": .3}})
    audit = metric_audit([row])
    assert audit["interaction_details.score"]["mean"] == .25
    assert abs(audit["clip_details.aux"]["mean"] - .5) < 1e-12
    assert abs(audit["paper_formula.interaction"]["mean"] - .77) < 1e-12
    assert metric_audit([dict(model="m")])["clip_details.raw"]["pending"] == 1
