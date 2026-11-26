import numpy as np
import pytest
from continuous_action.schema import ActionTrajectory
from continuous_action.metrics import score_action_trajectories as score
from continuous_action.current_metric import CURRENT_METRIC_VERSION


def trajectory(points, times=None, observed=None):
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([p, np.zeros((len(p), 2))])
    if times is None:
        times = np.linspace(0, 1, len(p))
    meta = {} if observed is None else {"observed_frame_mask": observed}
    return ActionTrajectory(p, np.asarray(times), coordinate_frame="registered_world", metadata=meta)


def test_identical_and_start_translation():
    r = trajectory([[0,0,0],[1,0,0],[1,1,0]])
    p = trajectory(r.positions+[2,3,4])
    s = score(r,p)
    assert s.continuous_score == pytest.approx(1)
    assert s.to_dict()["scored_components"] == ["trajectory_shape", "speed_curve"]


def test_all_direction_and_extent_are_preserved():
    r=trajectory([0,1])
    assert score(r,trajectory([0,-1])).path_score == pytest.approx(0)
    s=score(r,trajectory([0,.5]))
    assert s.path_score == pytest.approx(2/3)
    assert s.velocity_score == pytest.approx(.8)
    assert s.continuous_score == pytest.approx(np.sqrt((2/3)*.8))


def test_different_duration_preserves_absolute_speed():
    r=trajectory([0,1], [0,1]); p=trajectory([0,1], [0,2])
    s=score(r,p)
    assert s.path_score == pytest.approx(1)
    assert s.velocity_score == pytest.approx(.8)
    assert s.continuous_score == pytest.approx(np.sqrt(.8))


def test_equal_speed_cannot_hide_shorter_spatial_path():
    s=score(trajectory([0,1],[0,1]),trajectory([0,.5],[0,.5]))
    assert s.velocity_score == pytest.approx(1)
    assert s.path_score == pytest.approx(2/3)


@pytest.mark.parametrize('p',[[0,0,1],[0,1,1]])
def test_leading_and_trailing_holds_are_scored(p):
    r=trajectory([0,.5,1]); s=score(r,trajectory(p))
    assert s.path_score == pytest.approx(1)
    assert s.velocity_score == pytest.approx(2/3)
    assert s.to_dict()["predicted_source_node_range"] == [0,2]


def test_internal_pause_location_changes_speed_curve():
    r=trajectory([0,.5,.5,1]);p=trajectory([0,0,.5,1])
    assert score(r,p).path_score == pytest.approx(1)
    assert score(r,p).velocity_score == pytest.approx(.5)


def test_complete_path_does_not_select_matching_prefix():
    r=trajectory([[0,0,0],[1,0,0],[1,1,0]])
    p=trajectory([[0,0,0],[1,0,0],[1,-1,0]])
    s=score(r,p)
    assert s.path_score < 1
    assert s.to_dict()["predicted_samples"] == 3


def test_source_sampling_density_does_not_change_polyline_integrals():
    r=trajectory([[0,0,0],[1,0,0],[1,1,0]], [0,.4,1])
    p=trajectory([[0,0,0],[.5,0,0],[1,0,0],[1,.5,0],[1,1,0]], [0,.2,.4,.7,1])
    assert score(r,p).continuous_score == pytest.approx(1)


def test_partial_visibility_does_not_change_plan_score():
    r=trajectory([0,.2,.5,1],observed=[1,1,1,1])
    p=trajectory([0,.1,.3,.5],observed=[1,1,0,0])
    complete=trajectory(p.positions,observed=[1,1,1,1])
    partial=score(r,p)
    assert partial.continuous_score == score(r,complete).continuous_score
    assert partial.to_dict()["observation_coverage"] == "partial"
    assert partial.to_dict()["prediction_observation_coverage"] == pytest.approx(1/3)
    assert "coverage_factor" not in partial.to_dict()
    assert partial.to_dict()["visibility_affects_score"] is False


def test_stationary_motion():
    assert score(trajectory([0,1]),trajectory([0,0])).continuous_score == 0
    assert score(trajectory([0,0]),trajectory([0,0])).continuous_score == 1


def test_no_observed_support_is_unmeasured_and_invalid_masks_rejected():
    for mask in ([1,0,1], [1,2,1], [True]):
        with pytest.raises(ValueError):
            score(trajectory([0,.5,1]),trajectory([0,.5,1],observed=mask))


def test_only_current_version_is_callable():
    r=trajectory([0,1])
    assert score(r,r,metric_version=CURRENT_METRIC_VERSION).continuous_score == 1
    with pytest.raises(ValueError, match='retired'):
        score(r,r,metric_version='independent_metric_shape_extent_geomean_partial_v18')
