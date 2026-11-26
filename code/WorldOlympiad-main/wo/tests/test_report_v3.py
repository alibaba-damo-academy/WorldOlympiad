from copy import deepcopy
from math import isclose
from wo.eval.report_v3 import project, build, markdown


def row():
    return {"case_id": "a", "model": "m", "seed": 0, "episode": "/run/a/m_s0", "status": "complete",
            "physical": .6, "geometry": .5, "interaction": .2, "worldolympiad2_score": .3,
            "geometry_details": {"gs_score": .4, "meta_score": .6, "camera_motion_score": .8},
            "interaction_details": {"summary": {"chunk_mean": 5., "transition_mean": 3., "global_score": 4.}},
            "clip_details": {"summary": {"semantic_adherence": .3}},
            "cae_details": {"status": "scored", "entities": {
                "ee": {"kind": "end_effector", "track": .2, "speed": .6},
                "object": {"kind": "object", "track": .8, "speed": .4}}}}


def test_shape_and_speed_move_to_separate_tracks_without_changing_source():
    original = row(); saved = deepcopy(original); result = project(original)
    assert original == saved
    assert isclose(result["action_trajectory"], .5)
    assert isclose(result["action_speed"], .5)
    assert isclose(result["geometry"], (.4+.6+.8+.5)/4)
    assert isclose(result["semantic_interaction"], .77)
    assert isclose(result["interaction"], (1.+.6+.8+.5+.5)/5)
    assert set(result["tracks"]["interaction"]["submetrics"]) == {"chunk", "transition", "global", "clip_aux", "action_speed"}
    assert all(isclose(w, .2) for w in result["tracks"]["interaction"]["weights"].values())
    assert result["historical_v2_scores"]["geometry"] == .5
    changed = deepcopy(original); changed["cae_details"]["entities"]["object"]["speed"] = .9
    assert project(changed)["geometry"] == result["geometry"]
    assert project(changed)["interaction"] != result["interaction"]
    changed = deepcopy(original); changed["cae_details"]["entities"]["object"]["track"] = 0
    assert project(changed)["interaction"] == result["interaction"]
    assert project(changed)["geometry"] != result["geometry"]


def test_target_count_does_not_overweight_objects_against_end_effector():
    original = row()
    original["cae_details"]["entities"]["object2"] = {"kind": "object", "track": .8, "speed": .4}
    assert isclose(project(original)["action_trajectory"], .5)


def test_missing_action_measurement_or_clip_is_pending_but_zero_is_scored():
    original = row(); original["cae_details"]["entities"]["object"]["track"] = None
    result = project(original)
    assert result["geometry"] is None and result["status"] == "pending"
    assert result["worldolympiad2_score"] is None
    original = row(); original["clip_details"] = {}
    assert project(original)["interaction"] is None
    original = row()
    for e in original["cae_details"]["entities"].values():e.update(track=0., speed=0.)
    result = project(original)
    assert result["status"] == "complete"
    assert result["action_trajectory"] == result["action_speed"] == 0


def test_rank_uses_unrounded_scores_and_compact_tables_have_no_case_rows():
    a = row(); b = row(); b["model"] = "other"; b["physical"] += .00001
    result = build({"episodes": [a,b]})
    assert result["models"]["other"]["rank"] == 1
    assert result["models"]["m"]["rank"] == 2
    text = markdown(result)
    assert "S_traj (camera)" in text and "S_action" in text and "Action Speed" in text
    assert "/run/a/" not in text and "case_id" not in text


def test_new_model_and_non_24_case_counts_are_not_mislabeled():
    example = row()
    example["model"] = "longwam"
    result = build({"episodes": [example]})
    assert result["models"]["longwam"]["cases"] == 1
    text = markdown(result)
    assert "Long-WAM RoboTwin2.0-IDM" in text
    assert "N=24" not in text
