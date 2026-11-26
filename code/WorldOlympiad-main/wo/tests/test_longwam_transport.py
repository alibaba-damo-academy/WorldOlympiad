import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
import json
from wo.models.fastwam_server import encode_observation
from wo.models.fastwam_client import executable_prefix
from wo.models.longwam_server import infer_payload
from wo.models import longwam
from wo.agent import stages


def test_official_policy_receives_each_raw_observation_without_action_rewriting(monkeypatch):
    observed=[]
    mod=ModuleType('longwam.benchmarks.robotwin.adapter')
    def batch(observation,instruction):
        observed.append((observation,instruction))
        return observation
    mod.policy_batch=batch
    monkeypatch.setitem(sys.modules,mod.__name__,mod)
    class Tensor:
        def __init__(self,x):self.x=x
        def detach(self):return self
        def cpu(self):return self.x
    class Policy:
        def select_action(self,observation):
            return Tensor(np.asarray(observation['joint_action']['vector'])[None,:])
    policy=Policy()
    for i in range(2):
        q=np.arange(14,dtype=np.float32)/10+i
        obs={'observation':{name:{'rgb':np.full((4,6,3),i,dtype=np.uint8)} for name in ['head_camera','left_camera','right_camera']},'joint_action':{'vector':q}}
        result=infer_payload(policy,encode_observation(obs,'put the apple on the plate'))
        assert np.array_equal(executable_prefix(result),q[None,:])
        assert result['replan_steps']==1
    assert len(observed)==2
    assert all(x[1]=='put the apple on the plate' for x in observed)
    assert np.array_equal(observed[1][0]['observation']['head_camera']['rgb'],obs['observation']['head_camera']['rgb'])


def test_longwam_registered_after_reviewed_native_control():
    assert stages.MODELS['longwam']=='wo.models.longwam'
    assert stages.episode_adapter({'model':'longwam'})=='wo.models.longwam'
    stages.STAGES['policy_check'].validate({'run':'r','case':'c','model':'longwam','stock_task':'stack_blocks_two'})


def test_unreviewed_adapters_still_require_explicit_native_control(monkeypatch):
    monkeypatch.setitem(stages.CONTROL_MODELS,'pending_model','wo.models.pending')
    with pytest.raises(ValueError):stages.episode_adapter({'model':'pending_model'})
    assert stages.episode_adapter({'model':'pending_model','control_only':True})=='wo.models.pending'


def test_longwam_reference_path_uses_bound_native_state(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    archive.mkdir()
    task, state = archive / 'task.json', archive / 'state.json'
    task.write_text('{}')
    state.write_text('{}')
    monkeypatch.setattr(longwam.rt, '_reference_bridge_inputs', lambda *_: (task, state))
    def execute(out, args, **kwargs):
        assert args[args.index('--task-spec') + 1] == str(task)
        assert args[args.index('--reference-state') + 1] == str(state)
        assert args[args.index('--out') + 1] == str(out)
        assert args[args.index('--window-s') + 1] == '2'
        (out / 'episode.json').write_text(json.dumps({'model': 'longwam',
                                                       'reference': str(tmp_path / 'ref')}))
        return {'capture': {'native_actions': 3}}
    monkeypatch.setattr(longwam, 'execute', execute)
    result = longwam.run_episode(tmp_path / 'ref', tmp_path / 'out', window_s=2)
    assert result['model'] == 'longwam'
    assert result['action_type'] == 'qpos'
    assert result['native_result']['capture']['native_actions'] == 3


def test_longwam_native_progress_matches_control_steps(tmp_path, monkeypatch):
    out = tmp_path / 'episode'
    (out / 'policy').mkdir(parents=True)
    (out / 'capture').mkdir()
    (out / 'policy/progress.json').write_text(json.dumps({
        'observation_steps': 2, 'inference_chunks': 1}))
    monkeypatch.setattr(longwam.native, 'execute', lambda *_a, **_k: {
        'capture': {'native_actions': 2, 'model_requests': 2}})
    result = longwam.execute(out, ['--out', str(out)])
    assert result['transport_requests'] == 2
    assert result['model_requests'] == 1
    assert json.loads((out / 'capture/capture_metadata.json').read_text()) == result
