from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import push


@pytest.mark.parametrize('hit_body,distance,expected', [(7, .03, .03), (8, .03, .0565685425), (-1, 0, .0565685425)])
def test_contact_radius_uses_subject_surface_only(hit_body, distance, expected):
    u = np.array([1, 1, 0]) / np.sqrt(2)
    centre = np.array([.2, .3])
    origin = np.array([.2, .3, .75])
    def ray(start, end, **kw):
        assert np.allclose(end, origin)
        assert np.dot(origin - start, u) > .056
        return [(hit_body, -1, .5, origin - u * distance, [0, 0, 1])]
    w = SimpleNamespace(aabb=lambda _: (np.array([-.04, -.04, 0]), np.array([.04, .04, 1])),
        bodies={'subject': 7}, cid=0, p=SimpleNamespace(rayTest=ray))
    assert push._contact_radius(w, 'subject', u, centre, .75) == pytest.approx(expected)


def test_push_has_no_unchecked_cartesian_paths():
    import inspect
    assert 'check=False' not in inspect.getsource(push)
