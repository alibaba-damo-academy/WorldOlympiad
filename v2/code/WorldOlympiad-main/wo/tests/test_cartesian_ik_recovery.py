import numpy as np
from wo.expert.kin import Kinematics, Pose


class Probe:
    cartesian_path = Kinematics.cartesian_path

    def __init__(self, candidate, collision=None):
        self.candidate = np.full(7, candidate)
        self.collision = collision
        self.restarts = []

    def ik(self, target, seed, restarts, accept_solution=None):
        self.restarts.append(restarts)
        if restarts == 0:
            return None
        return self.candidate if accept_solution(self.candidate) else None

    def in_collision(self, q, gripper):
        assert gripper == 1.0
        return self.collision


def path(probe, check=True):
    start = Pose(np.zeros(3), np.eye(3))
    end = start.moved([0.01, 0, 0])
    diagnostics = {}
    result = probe.cartesian_path(np.zeros(7), start, end, gripper=1.0,
                                  check=check, diagnostics=diagnostics)
    return result, diagnostics


def test_local_ik_failure_recovers_only_continuous_clear_solution():
    result, diagnostics = path(Probe(0.1))
    assert result is not None
    assert diagnostics['ik_recoveries'] == 1
    assert diagnostics['reason'] == 'success'


def test_recovery_rejects_discontinuous_branch():
    result, diagnostics = path(Probe(0.3))
    assert result is None
    assert diagnostics['reason'] == 'ik'


def test_recovery_keeps_collision_checks():
    result, diagnostics = path(Probe(0.1, 'obstacle'))
    assert result is None
    assert diagnostics['reason'] == 'ik'


def test_unchecked_path_does_not_gain_recovery():
    probe = Probe(0.1)
    assert path(probe, check=False)[0] is None
    assert probe.restarts == [0]
