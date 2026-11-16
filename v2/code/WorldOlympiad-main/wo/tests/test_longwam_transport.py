import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from wo.models.fastwam_server import encode_observation
from wo.models.fastwam_client import executable_prefix
from wo.models.longwam_server import infer_payload
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
