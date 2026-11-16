"""Path diagnostics must not alter collision/IK acceptance."""
from types import SimpleNamespace
import numpy as np
import pytest
from wo.expert.kin import Kinematics, Pose

@pytest.mark.parametrize('failure', ['ik', 'joint_jump', 'collision', 'success'])
def test_cartesian_failure_evidence(failure):
    calls = []
    def ik(target, **kw):
        if failure == 'ik':
            return None
        return np.ones(7) if failure == 'joint_jump' else np.zeros(7)
    def collision(q, gripper):
        calls.append(gripper)
        return 'obstacle' if failure == 'collision' else None
    kin = SimpleNamespace(ik=ik, in_collision=collision)
    start = Pose(np.zeros(3), np.eye(3))
    info = {}
    path = Kinematics.cartesian_path(kin, np.zeros(7), start,
                                     start.moved([0, 0, .02]), gripper=1., diagnostics=info)
    assert (path is not None) == (failure == 'success')
    assert info['reason'] == failure
    if failure == 'collision':
        assert info['collision'] == 'obstacle'
        assert 0 < info['progress'] <= 1
        assert calls == [1.]
    if failure in ('ik', 'joint_jump'):
        assert calls == []
    # Identical outcome without requesting evidence.
    plain = Kinematics.cartesian_path(kin, np.zeros(7), start, start.moved([0, 0, .02]), gripper=1.)
    assert (plain is not None) == (path is not None)
