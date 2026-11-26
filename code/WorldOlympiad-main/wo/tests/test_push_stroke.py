import numpy as np
import pytest
from wo.expert.kin import Pose
from wo.expert.push import _checked_push_line


class Kin:
    def __init__(self, failure=False):
        self.ignore = {'existing'}
        self.failure = failure

    def cartesian_path(self, q, start, end, **kwargs):
        assert self.ignore == {'existing', 'subject', 'target'}
        assert kwargs['check'] is True
        assert kwargs['gripper'] == 1.0
        if self.failure:
            raise RuntimeError('probe')
        kwargs['diagnostics'].update(reason='collision', body='distractor', progress=0.5)
        return None


def test_stroke_retains_obstacle_evidence_and_exemptions():
    kin = Kin()
    ev = {}
    pose = Pose(np.zeros(3), np.eye(3))
    assert _checked_push_line(kin, np.zeros(7), pose, pose, 'subject', 'target', ev, 2) is None
    assert kin.ignore == {'existing'}
    assert ev['stroke_checks'][0]['body'] == 'distractor'
    assert ev['stroke_checks'][0]['segment'] == 2
    assert ev['stroke_checks'][0]['progress'] == 0.5


def test_stroke_exception_restores_exemptions():
    kin = Kin(True)
    pose = Pose(np.zeros(3), np.eye(3))
    with pytest.raises(RuntimeError):
        _checked_push_line(kin, np.zeros(7), pose, pose, 'subject', 'target', {}, 0)
    assert kin.ignore == {'existing'}
