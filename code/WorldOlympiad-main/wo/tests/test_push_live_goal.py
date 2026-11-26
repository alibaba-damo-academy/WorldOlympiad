import numpy as np


def test_segment_evidence_distinguishes_tracking_from_contact_stall():
    from wo.expert.push import push_segment_evidence
    from wo.expert.kin import tcp_to_ee
    start = Pose(np.array([0., 0., 0.8]), np.eye(3))
    end = start.moved([0.03, 0., 0.])
    before = np.array([0.07, 0.])
    after = before.copy()
    evidence = push_segment_evidence(before, after, [0.08, 0.], start, end,
                                     0.05, tcp_to_ee(end))
    assert evidence['displacement_m'] == 0.
    assert evidence['endpoint_tracking_error_m'] == 0.
    assert np.isclose(evidence['remaining_distance_m'], 0.01)
    assert evidence['footprint_radius_m'] == 0.05
    assert evidence['end_tcp'] == end.p.tolist()
    np.testing.assert_array_equal(before, after)

from wo.expert.kin import Pose
from wo.expert.push import live_goal_position


def test_goal_follows_target_translation():
    goal = Pose(np.array([0.1, 0.2, 0.04]), np.eye(3))
    target = Pose(np.array([0., 0., 0.]), np.eye(3))
    live = target.moved([0.07, -0.03, 0.])
    np.testing.assert_allclose(live_goal_position(goal, target, live), [0.17, 0.17, 0.04])


def test_goal_follows_target_rotation_about_nonzero_origin():
    target = Pose(np.array([1., 2., 0.]), np.eye(3))
    goal = target @ Pose(np.array([0.1, 0., 0.03]), np.eye(3))
    live = Pose(np.array([2., 3., 0.]), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    np.testing.assert_allclose(live_goal_position(goal, target, live), [2., 3.1, 0.03])


def test_static_target_leaves_certified_goal_unchanged():
    target = Pose.from_pq([0.3, -0.1, 0.6], [0., 0., 0.6, 0.8])
    goal = Pose(np.array([0.4, 0.02, 0.64]), np.eye(3))
    np.testing.assert_allclose(live_goal_position(goal, target, target), goal.p)
