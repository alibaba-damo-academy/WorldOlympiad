"""Behavioural tests for CAE v2: noise robustness and sensitivity to real differences."""

import numpy as np

from wo.eval.cae_v2 import Track, score, score_entity

T = np.arange(0, 10, 1 / 15)


def reach_and_carry():
    # move out, lift, carry sideways, lower: a typical pick-place EE path
    x = np.interp(T, [0, 3, 6, 9, 10], [0, 0.3, 0.3, 0.6, 0.6])
    z = np.interp(T, [0, 3, 4, 7, 8, 10], [0, -0.1, 0.1, 0.1, 0.0, 0.0])
    return np.column_stack([x, 0.2 * x, z])


REF = Track(reach_and_carry(), T, "end_effector")
rng = np.random.default_rng(0)


def s(pos, t=T):
    return score_entity(REF, Track(pos, t, "end_effector"))["score"]


def test_identical_is_one():
    assert s(REF.positions) > 0.999


def test_reconstruction_noise_is_tolerated():
    assert s(REF.positions + rng.normal(0, 0.01, REF.positions.shape)) > 0.95
    assert s(REF.positions + rng.normal(0, 0.02, REF.positions.shape)) > 0.9


def test_not_moving_scores_near_zero_even_with_jitter():
    still = np.repeat(REF.positions[:1], len(T), 0) + rng.normal(0, 0.005, REF.positions.shape)
    assert s(still) < 0.15


def test_stopping_halfway_is_penalised():
    half = REF.positions.copy()
    half[len(T) // 2:] = half[len(T) // 2]
    assert 0.4 < s(half) < 0.9


def test_waiting_then_same_motion_is_not_penalised():
    # model hesitates 10 s, then does the reference motion (window 20 s)
    t = np.arange(0, 20, 1 / 15)
    pos = np.vstack([np.repeat(REF.positions[:1], 150, 0), REF.positions])[: len(t)]
    assert s(pos, t) > 0.9


def test_slower_but_complete_is_not_penalised():
    t = np.arange(0, 25, 1 / 15)
    slow = np.column_stack([np.interp(t * 0.4, T, REF.positions[:, i]) for i in range(3)])
    assert s(slow, t) > 0.9


def test_much_slower_and_unfinished_is_penalised():
    slow = np.column_stack([np.interp(T * 0.5, T, REF.positions[:, i]) for i in range(3)])
    assert s(slow) < 0.8


def test_wrong_direction_scores_low():
    assert s(REF.positions * np.array([-1, -1, 1])) < 0.5


def test_aggregate_weights_and_missing_entity():
    obj = Track(REF.positions * 0.5, T, "object")
    full = score({"end_effector": REF, "cup": obj}, {"end_effector": REF, "cup": obj})
    assert full["status"] == "scored" and full["score"] > 0.999
    assert score({"end_effector": REF, "cup": obj}, {"end_effector": REF})["status"] == "pending"
    only_obj = score({"cup": obj}, {"cup": obj})
    assert only_obj["end_effector_score"] is None and only_obj["score"] > 0.999
