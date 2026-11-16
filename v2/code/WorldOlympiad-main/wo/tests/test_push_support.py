import numpy as np
from wo.expert.push import push_route, select_push_route, _push_support_bounds, CONTACT_GAP_M


def test_detour_uses_supported_side_only():
    boxes = [(np.array([.3, -.1]), np.array([.6, .1]))]
    bounds = (np.array([-.2, -.3]), np.array([1.2, .11]))
    route = push_route([0., 0.], [1., 0.], boxes, bounds)
    assert route is not None
    assert all(np.all(p >= bounds[0]) and np.all(p <= bounds[1]) for p in route)
    assert route[1][1] < -.1


def test_no_supported_detour_is_failure_not_off_table_route():
    boxes = [(np.array([.3, -.1]), np.array([.6, .1]))]
    bounds = (np.array([-.2, -.11]), np.array([1.2, .11]))
    assert push_route([0., 0.], [1., 0.], boxes, bounds) is None


def test_goal_search_rejects_unsupported_goal():
    bounds = (np.array([-1., -1.]), np.array([1., 1.]))
    index, route = select_push_route([0., 0.], [[2., 0.], [.8, 0.]], [], bounds)
    assert index == 1
    np.testing.assert_allclose(route[-1], [.8, 0.])


def test_measured_support_shrinks_by_full_subject_and_contact_reserve():
    class World:
        support_id = 'table'
        def aabb(self, oid):
            return tuple(np.array(v) for v in (
                ([-1., -1., .6], [1., 1., .7]) if oid == 'table'
                else ([-.03, -.04, .7], [.03, .04, .72])))
    lo, hi = _push_support_bounds(World(), 'puck')
    reserve = .05 + CONTACT_GAP_M
    np.testing.assert_allclose(lo, [-1 + reserve] * 2)
    np.testing.assert_allclose(hi, [1 - reserve] * 2)
