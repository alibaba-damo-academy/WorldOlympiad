"""Native qpos rollout consumes measured observations and preserves action type."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wo.models import lingbot_va_native_bridge as bridge


class Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def close(self):
        self.closed = True


class Task:
    step_lim = 4
    eval_success = False

    def __init__(self):
        self.take_action_cnt = 0
        self.calls = []
        self.physics_steps = 0
        self.scene = SimpleNamespace(get_timestep=lambda: 0.01,
                                     step=self._step)

    def _step(self):
        self.physics_steps += 1

    def get_obs(self):
        value = self.take_action_cnt
        return {'observation': {name: {'rgb': np.full((3, 4, 3), value,
                                                      dtype=np.uint8)}
                                for name in ('head_camera', 'left_camera', 'right_camera')},
                'joint_action': {'vector': np.full(14, value, dtype=float)}}

    def take_action(self, action, action_type):
        self.calls.append((np.asarray(action).copy(), action_type))
        self.scene.step()
        self.scene.step()
        self.take_action_cnt += 1


def test_qpos_rollout_records_real_steps_and_all_cameras(tmp_path, monkeypatch):
    writers = []
    def writer(*_args, **_kwargs):
        item = Writer()
        writers.append(item)
        return item
    monkeypatch.setitem(sys.modules, 'imageio', SimpleNamespace(get_writer=writer))
    monkeypatch.setattr(bridge, '_native_trajectory_frame',
                        lambda task: {'frame': task.take_action_cnt})
    task = Task()
    actions = np.stack([np.full(14, 0.1), np.full(14, 0.2)])
    observations = []
    class Client:
        def infer(self, observation, instruction):
            observations.append((observation['joint_action']['vector'].copy(), instruction))
            return actions
    result = bridge._run_qpos_policy(task, Client(), 'stack blocks', tmp_path, 2,
                                     record_native_trajectory=True)
    assert result == {'frames': 3, 'native_actions': 2,
                      'model_requests': 1, 'native_step_limit': 4,
                      'simulation_elapsed_s': pytest.approx(.04)}
    assert [kind for _, kind in task.calls] == ['qpos', 'qpos']
    assert np.array_equal(task.calls[0][0], actions[0])
    assert np.array_equal(task.calls[1][0], actions[1])
    assert np.array_equal(observations[0][0], np.zeros(14))
    assert len(task.wo_native_trajectory_frames) == 3
    assert [frame['time_s'] for frame in task.wo_native_trajectory_frames] == pytest.approx([0, .02, .04])
    assert task.physics_steps == 4
    assert task.scene.step == task._step
    assert len(writers) == 3
    assert all(len(writer.frames) == 3 and writer.closed for writer in writers)


@pytest.mark.parametrize('actions', [np.zeros((0, 14)), np.zeros((1, 16)),
                                     np.full((1, 14), np.nan)])
def test_invalid_qpos_never_reaches_simulator(tmp_path, monkeypatch, actions):
    monkeypatch.setitem(sys.modules, 'imageio', SimpleNamespace(get_writer=lambda *_a, **_k: Writer()))
    task = Task()
    with pytest.raises(RuntimeError, match='invalid joint targets'):
        bridge._run_qpos_policy(task, SimpleNamespace(infer=lambda *_a: actions),
                                'stack blocks', tmp_path, 1)
    assert task.calls == []


def test_qpos_window_uses_physics_time_not_policy_action_count(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'imageio', SimpleNamespace(get_writer=lambda *_a, **_k: Writer()))
    task = Task()
    result = bridge._run_qpos_policy(
        task, SimpleNamespace(infer=lambda *_a: np.zeros((4, 14))),
        'stack blocks', tmp_path, 4, max_sim_time_s=.03)
    assert result['native_actions'] == 2
    assert result['simulation_elapsed_s'] == pytest.approx(.04)
    assert task.scene.step == task._step


def test_qpos_trajectory_samples_inside_long_action(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'imageio', SimpleNamespace(get_writer=lambda *_a, **_k: Writer()))
    monkeypatch.setattr(bridge, '_native_trajectory_frame',
                        lambda task: {'frame': task.take_action_cnt})
    task = Task()
    def long_action(action, action_type):
        task.calls.append((action, action_type))
        task.take_action_cnt += 1
        for _ in range(20):
            task.scene.step()
    task.take_action = long_action
    result = bridge._run_qpos_policy(
        task, SimpleNamespace(infer=lambda *_a: np.zeros((1, 14))),
        'stack blocks', tmp_path, 1, record_native_trajectory=True)
    times = [frame['time_s'] for frame in task.wo_native_trajectory_frames]
    assert result['simulation_elapsed_s'] == pytest.approx(.2)
    assert len(times) >= 4
    assert times[0] == 0 and times[-1] == pytest.approx(.2)
    assert all(a < b for a, b in zip(times, times[1:]))
    assert task.scene.step == task._step
