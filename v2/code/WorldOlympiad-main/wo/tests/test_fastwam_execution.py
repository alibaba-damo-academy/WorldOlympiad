"""Native action dispatch must never treat absolute joints as EE deltas."""
import numpy as np
import pytest
from wo.models.lingbot_va_client import execute_policy_action, PhysicsStepClock

class Task:
    def __init__(self):
        self.calls = []
    def take_action(self, action, action_type):
        self.calls.append((np.asarray(action).copy(), action_type))

def test_native_qpos_unchanged():
    task = Task()
    q = np.linspace(-1, 1, 14)
    out = execute_policy_action(task, q, None, fastwam=True)
    assert np.array_equal(out, q)
    assert np.array_equal(task.calls[0][0], q)
    assert task.calls[0][1] == 'qpos'

@pytest.mark.parametrize('raw', [np.zeros(16), np.full(14, np.nan)])
def test_invalid_native_actions_are_infrastructure_errors(raw):
    task = Task()
    with pytest.raises(RuntimeError):
        execute_policy_action(task, raw, None, fastwam=True)
    assert not task.calls

def test_legacy_ee_dispatch_unchanged():
    task = Task()
    init = np.array([.1,.2,.9,0,0,0,1,1, -.1,.2,.9,0,0,0,1,0])
    raw = np.array([.01,0,0,0,0,0,1,.5, 0,.02,0,0,0,0,1,.2])
    out = execute_policy_action(task, raw, init)
    assert task.calls[0][1] == 'ee'
    assert out[:3] == pytest.approx([.11,.2,.9])

class Scene:
    def __init__(self, dt=.004):
        self.dt = dt
        self.count = 0
    def get_timestep(self):
        return self.dt
    def step(self):
        self.count += 1

class SteppingTask:
    def __init__(self, steps):
        self.scene = Scene()
        self.steps = steps
    def take_action(self, action, action_type):
        assert action_type == 'qpos'
        for _ in range(self.steps):
            self.scene.step()

def test_clock_measures_variable_native_duration_and_restores_scene():
    task = SteppingTask(100)
    scene = task.scene
    clock = PhysicsStepClock(scene)
    clock.execute(task, np.zeros(14), None)
    assert task.scene is scene
    assert clock.elapsed_s == pytest.approx(.4)
    task.steps = 25
    clock.execute(task, np.zeros(14), None)
    assert clock.elapsed_s == pytest.approx(.5)
    assert clock.steps == scene.count == 125


def test_clock_fails_closed_if_no_physics_execution():
    task = SteppingTask(0)
    scene = task.scene
    with pytest.raises(RuntimeError, match='no simulator steps'):
        PhysicsStepClock(scene).execute(task, np.zeros(14), None)
    assert task.scene is scene


@pytest.mark.parametrize('dt', [0, -1, float('nan')])
def test_clock_invalid_timestep_restores_scene(dt):
    task = SteppingTask(1)
    scene = task.scene
    scene.dt = dt
    with pytest.raises(RuntimeError, match='invalid simulator timestep'):
        PhysicsStepClock(scene).execute(task, np.zeros(14), None)
    assert task.scene is scene
    assert scene.count == 0


def test_clock_counts_real_ee_execution_instead_of_nominal_action_frames():
    class EE(SteppingTask):
        def take_action(self, action, action_type):
            assert action_type=="ee"
            for _ in range(self.steps):self.scene.step()
    task=EE(50);scene=task.scene;clock=PhysicsStepClock(scene)
    init=np.array([.1,.2,.9,0,0,0,1,1, -.1,.2,.9,0,0,0,1,1])
    raw=np.array([.01,0,0,0,0,0,1,.5, 0,.02,0,0,0,0,1,.2])
    clock.execute(task,raw,init,fastwam=False)
    assert clock.elapsed_s==pytest.approx(.2) and clock.steps==50
    assert task.scene is scene
