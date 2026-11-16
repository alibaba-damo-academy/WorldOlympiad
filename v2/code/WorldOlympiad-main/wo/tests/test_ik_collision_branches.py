"""Converged but colliding IK branches must not hide free branches."""
from types import SimpleNamespace

import numpy as np

from wo.expert.kin import Kinematics, Pose


def solver(hits):
    pose = Pose.from_pq([0, 0, 1], [0, 0, 0, 1])
    state = {'calls': 0, 'checks': 0}
    def calculate(*args, **kwargs):
        state['calls'] += 1
        return np.zeros(7)
    world = SimpleNamespace(movable=list(range(7)), robot=1, ee=2, cid=0,
                            set_arm=lambda q: None, ee_pose=lambda: (pose.p, pose.q))
    kin = Kinematics.__new__(Kinematics)
    kin.w = world
    kin.p = SimpleNamespace(calculateInverseKinematics=calculate)
    kin.lower, kin.upper, kin.home = -np.ones(7), np.ones(7), np.zeros(7)
    def collision(q):
        i = state['checks']
        state['checks'] += 1
        return hits[min(i, len(hits)-1)]
    kin.in_collision = collision
    return kin, pose, state


def test_downstream_path_acceptance_searches_bounded_branches():
    kin, pose, state = solver([None])
    accepted = []
    def accept(q):
        accepted.append(q.copy())
        return len(accepted) == 2
    assert kin.ik(pose, accept_solution=accept) is not None
    assert len(accepted) == 2 and state['calls'] == 8
    assert state['checks'] == 0


def test_all_downstream_paths_fail_without_unbounded_retry():
    kin, pose, state = solver([None])
    assert kin.ik(pose, restarts=2, accept_solution=lambda q: False) is None
    assert state['calls'] == 12


def test_default_preserves_first_converged_branch():
    kin, pose, state = solver(['table', None])
    assert kin.ik(pose) is not None
    assert state == {'calls': 4, 'checks': 0}


def test_free_branch_after_colliding_branch():
    kin, pose, state = solver(['table', None])
    assert kin.ik(pose, collision_free=True) is not None
    assert state == {'calls': 8, 'checks': 2}


def test_all_branches_colliding_is_failure_with_bounded_work():
    kin, pose, state = solver(['table'])
    assert kin.ik(pose, restarts=2, collision_free=True) is None
    assert state == {'calls': 12, 'checks': 3}
