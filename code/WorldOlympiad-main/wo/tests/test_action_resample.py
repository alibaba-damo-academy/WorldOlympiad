"""Motion interpolation preserves empty/no-op paths and endpoint semantics."""
import numpy as np
import pytest

from wo.expert.rollout import _resample, fine_actions, free_actions


@pytest.mark.parametrize('resample', [lambda qs: _resample(qs, .3),
                                     lambda qs: fine_actions(qs, 0.),
                                     free_actions])
def test_empty_motion_is_noop(resample):
    assert resample([]) == []


def test_single_waypoint_preserved():
    q = np.arange(7, dtype=float)
    out = _resample([q], .3)
    assert len(out) == 1
    np.testing.assert_array_equal(out[0], q)


def test_motion_endpoint_and_finite_interpolation():
    start, end = np.zeros(7), np.ones(7)
    out = _resample([start, end], .3)
    assert len(out) > 1
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[-1], end)
    assert all(np.all((q >= 0) & (q <= 1)) for q in out)
