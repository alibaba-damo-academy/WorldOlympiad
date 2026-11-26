from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import push


@pytest.mark.parametrize('outcome', ['success', 'no_path', 'exception'])
def test_contact_separation_is_checked_and_restores_ignore(monkeypatch, outcome):
    world = SimpleNamespace(ee_pose=lambda: ([0, 0, 1], [0, 0, 0, 1]),
                            arm_q=lambda: np.zeros(7))
    calls = []
    attempts = []
    kin = SimpleNamespace(ignore={'existing'}, w=object())

    def cartesian(q, start, end, **kwargs):
        attempts.append(end.p - start.p)
        assert kin.ignore == {'existing', 'subject'}
        assert kwargs == {'gripper': 1.0}  # no disabled collision checks
        if len(attempts) == 1:
            assert np.allclose(end.p - start.p, [0, 0, 0.20])
        if outcome == 'exception':
            raise RuntimeError('IK failed')
        return None if outcome == 'no_path' else [q, q]

    kin.cartesian_path = cartesian
    ex = SimpleNamespace(run=lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(push, 'fine_actions', lambda path, duration: path)
    monkeypatch.setattr(push, 'sync', lambda k, d: calls.append('sync'))
    if outcome == 'exception':
        with pytest.raises(RuntimeError):
            push._separate_pusher(world, kin, ex, 'subject')
    else:
        assert push._separate_pusher(world, kin, ex, 'subject') == (outcome == 'success')
    assert kin.ignore == {'existing'}
    assert bool(calls) == (outcome == 'success')
