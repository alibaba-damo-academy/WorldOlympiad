import numpy as np
import pytest
from wo.sim.world import World, GRIPPER_CLOSED_RAD, GRIPPER_MIMIC
from wo.expert.grasps import approach_fraction

class Physics:
    POSITION_CONTROL = 1
    def __init__(self): self.targets = {}
    def setJointMotorControl2(self, robot, joint, mode, **kw):
        self.targets[joint] = kw['targetPosition']

def test_command_preserves_fraction_and_clips():
    w = World.__new__(World)
    w.p, w.cid, w.robot, w.arm = Physics(), 0, 1, []
    w.joint_index = {name: i for i, name in enumerate(GRIPPER_MIMIC)}
    for fraction in [0, 0.23, 0.7, 1, -1, 2]:
        w.command([], fraction)
        for name, mimic in GRIPPER_MIMIC.items():
            assert w.p.targets[w.joint_index[name]] == pytest.approx(
                mimic * np.clip(fraction, 0, 1) * GRIPPER_CLOSED_RAD)

class GapWorld:
    robot, cid, fingers = 1, 0, [2, 3]
    def __init__(self): self.f = 0.15; self.p = self
    def gripper_fraction(self): return self.f
    def set_gripper_state(self, f): self.f = f
    def getClosestPoints(self, *args, **kwargs):
        return [[0]*8 + [0.08 * (1-self.f**2)]]

def test_aperture_measures_nonlinear_geometry_and_restores():
    w = GapWorld()
    f = approach_fraction(w, 0.03)
    assert 0.03 + 0.006 <= 0.08 * (1-f*f) < 0.03602
    assert w.f == 0.15

def test_transport_plan_uses_same_fraction_at_pregrasp_and_descent(monkeypatch):
    from wo.expert import rollout
    from wo.tests.test_transport_candidate_search import setup
    w, kin, step, _ = setup(monkeypatch, set(), ngrasps=1)
    sampler = rollout.sample_grasps
    gs = sampler(w, 'subject')
    gs[0].width = 0.03
    monkeypatch.setattr(rollout, 'sample_grasps', lambda *a, **kw: gs)
    monkeypatch.setattr(rollout, 'approach_fraction', lambda *a: 0.6)
    monkeypatch.setattr(rollout, 'aperture_tcp_offset', lambda *a: np.array([0, 0, 0.162]))
    kin.ik = lambda *a, **kw: np.zeros(7)
    checked = []
    kin.in_collision = lambda *a, **kw: checked.append(kw.get('gripper', 0))
    paths = []
    def path(*a, **kw):
        paths.append(kw['gripper'])
        return [np.zeros(7)]
    kin.cartesian_path = path
    states = []
    w.set_gripper_state = states.append
    plans, _ = rollout.plan_transport(w, kin, w, step, _aperture=True)
    assert plans[0].approach_gripper == 0.6
    assert checked[0] == paths[0] == states[0] == 0.6


def test_no_gap_does_not_close():
    w = GapWorld()
    assert approach_fraction(w, 0.09) == 0
    assert w.f == 0.15
