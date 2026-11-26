from copy import deepcopy
import pytest
from autoresearch.score_hierarchy import compose_three_tracks, from_judge_summaries


def measurements():
    return {name: {"status": "scored", "score": value} for name, value in
            (("physical", .9), ("geometry", .6), ("interaction", .8), ("continuous_action", .2))}


def test_cae_is_counted_once_with_three_equal_macro_tracks():
    inputs = measurements()
    frozen = deepcopy(inputs)
    report = compose_three_tracks(inputs)
    assert set(report["track_results"]) == {"physical", "geometry", "interaction"}
    assert report["track_results"]["interaction"]["score"] == .5
    assert report["worldolympiad2_score"] == pytest.approx(2/3)
    assert inputs == frozen
    assert report["track_results"]["interaction"]["submetrics"]["continuous_action"]["score"] == .2


def test_required_unmeasured_motion_keeps_interaction_and_total_pending():
    inputs = measurements()
    inputs["continuous_action"] = {"status": "pending", "score": None}
    report = compose_three_tracks(inputs)
    assert report["track_results"]["interaction"]["score"] is None
    assert report["worldolympiad2_score"] is None


def test_semantic_na_renormalizes_submetrics_without_adding_a_fourth_track():
    inputs = measurements()
    inputs["interaction"] = {"status": "not_applicable", "score": None,
                             "reason": "no contact or chunked interaction"}
    report = compose_three_tracks(inputs)
    assert report["track_results"]["interaction"]["score"] == .2
    assert report["worldolympiad2_score"] == pytest.approx((.9+.6+.2)/3)


def test_zero_motion_score_is_a_real_measurement_and_missing_manifest_is_not_na():
    report = from_judge_summaries({"score": .9}, {"score": .8},
        {"final_score_normalized": .6}, {"status": "applicable", "score": 0.})
    assert report["track_results"]["interaction"]["score"] == .4
    pending = from_judge_summaries({"score": .9}, {"score": .8},
        {"final_score_normalized": .6}, {"status": "not_applicable", "reason": "no_cae_manifest"})
    assert pending["status"] == "pending"


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -.1, 1.1])
def test_invalid_component_cannot_enter_a_macro_score(bad):
    inputs = measurements()
    inputs["continuous_action"]["score"] = bad
    with pytest.raises(ValueError):
        compose_three_tracks(inputs)
