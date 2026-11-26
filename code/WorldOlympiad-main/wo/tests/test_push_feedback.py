import numpy as np
import pytest
from wo.expert.push import feedback_push, CONTACT_GAP_M

class World:
    cid = 0
    bodies = {'subject': 1}
    def __init__(self, positions):
        self.p = self
        self.positions = iter(positions)
    def getBasePositionAndOrientation(self, *a, **kw):
        return [*next(self.positions), 0], [0, 0, 0, 1]

class Executor:
    def __init__(self): self.commands = []
    def run(self, actions, **kw):
        assert kw == {'phase': 'push'}
        self.commands.extend(actions)

def test_lateral_error_reaims_before_obstacle_envelope():
    # Only 6mm starting clearance: do not finish a sideways-drifting stroke.
    w = World([(0.01, .001), (.02, .002), (.03, .0031), (.04, .007)])
    ex = Executor()
    boxes = [(np.array([-.1, .006]), np.array([.1, .1]))]
    result = feedback_push(ex, w, 'subject', list(range(4)), [0, 0], [1, 0], boxes)
    assert result['executed_actions'] == 3
    assert result['drift_budget_m'] == pytest.approx(.003)
    assert result['reaim']
    assert ex.commands == [0, 1, 2]

def test_forward_motion_not_penalised():
    w = World([(i*.01, 0) for i in range(10)])
    ex = Executor()
    result = feedback_push(ex, w, 'subject', list(range(10)), [0, 0], [1, 0], [])
    assert len(ex.commands) == 10
    assert not result['reaim']
    assert result['lateral_drift_m'] == 0

def test_unconfined_stroke_still_reaims_on_contact_drift():
    w = World([(.01, CONTACT_GAP_M/2), (.02, 0)])
    result = feedback_push(Executor(), w, 'subject', [0, 1], [0, 0], [1, 0], [])
    assert result['executed_actions'] == 1
    assert result['reaim']
