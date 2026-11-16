import numpy as np
from wo.expert.push import push_route, _visible_segment, _push_obstacles, select_push_route


def test_goal_search_keeps_first_routable_rank():
    index, route = select_push_route([0., 0.], [[1., 0.], [2., 0.]], [])
    assert index == 0
    np.testing.assert_allclose(route[-1], [1., 0.])


def test_goal_search_rejects_blocked_placement_without_removing_obstacle():
    boxes = [(np.array([.3, -.1]), np.array([.6, .1]))]
    index, route = select_push_route([0., 0.], [[.4, 0.], [1., 0.]], boxes)
    assert index == 1
    assert all(_visible_segment(a, b, boxes) for a, b in zip(route, route[1:]))


def test_goal_search_reports_exhausted_certified_goals():
    boxes = [(np.array([.3, -.1]), np.array([.6, .1]))]
    assert select_push_route([0., 0.], [[.4, 0.], [.5, 0.]], boxes) == (None, None)



def test_unobstructed_route_unchanged():
    route = push_route([0., 0.], [1., 0.], [])
    np.testing.assert_allclose(route, [[0., 0.], [1., 0.]])


def test_measured_obstacle_detour_all_segments_clear():
    boxes = [(np.array([.3, -.1]), np.array([.6, .1]))]
    route = push_route([0., 0.], [1., 0.], boxes)
    assert len(route) == 4
    assert not _visible_segment(np.array([0., 0.]), np.array([1., 0.]), boxes)
    assert all(_visible_segment(a, b, boxes) for a, b in zip(route, route[1:]))


def test_detour_corners_have_contact_clearance_for_lateral_drift():
    from wo.expert.push import CONTACT_GAP_M
    lo, hi = np.array([.3, -.1]), np.array([.6, .1])
    boxes = [(lo, hi)]
    route = push_route([0., 0.], [1., 0.], boxes)
    for corner in route[1:-1]:
        distance = np.linalg.norm(np.maximum(np.maximum(lo - corner, corner - hi), 0.))
        assert distance >= CONTACT_GAP_M
    # Small physical drift toward the obstacle at a corner remains routable;
    # it must not require removing the forbidden envelope.
    corner = route[1]
    shifted = corner + .004 * np.sign((lo + hi) / 2 - corner)
    recovered = push_route(shifted, [1., 0.], boxes)
    assert recovered is not None
    assert all(_visible_segment(a, b, boxes)
               for a, b in zip(recovered, recovered[1:]))


def test_blocked_start_does_not_bypass_obstacle():
    boxes = [(np.array([-.1, -.1]), np.array([.1, .1]))]
    assert push_route([0., 0.], [1., 0.], boxes) is None


def test_obstacles_preserve_target_and_support_exclusions():
    class World:
        bodies = ['subject', 'target', 'table', 'distractor']
        def aabb(self, oid):
            return tuple(np.array(x) for x in {
                'subject': ([0., 0., .7], [.04, .04, .76]),
                'target': ([.4, 0., .7], [.6, .2, .8]),
                'table': ([-1., -1., .6], [1., 1., .7]),
                'distractor': ([.2, 0., .7], [.24, .04, .8])}[oid])
    boxes = _push_obstacles(World(), 'subject', 'target', .74)
    assert len(boxes) == 1
    assert boxes[0][0][0] < .2 and boxes[0][1][0] > .24
