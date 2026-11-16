import pytest
import json
import hashlib
from autoresearch.articulated_goal_contract import reference_goal, joint_tolerance
from autoresearch.articulated_goal_contract import assess_measured_reference
from autoresearch.robotics_reference_materializer import compiled_execution_steps
from autoresearch.robotics_reference_materializer import _measured_goal_cache_valid


def binding(kind='revolute'):
    return {'joint_type':kind,'closed_position':0.,'open_position':-1.5707963 if kind=='revolute' else .3}


def test_reference_endpoint_tolerance_is_exactly_native_tolerance():
    goal=reference_goal(binding(),{'terminal_joint_fraction':1.,'terminal_operator':'pull'})
    assert (1-goal['required_joint_travel_fraction'])*1.5707963==pytest.approx(joint_tolerance('revolute'))
    assert goal['required_joint_travel_fraction']>.94
    assert goal['release_required'] and goal['measured_release_and_stability_certificate_required']


@pytest.mark.parametrize('fraction',[.6,.9,False])
def test_explicit_fraction_cannot_weaken_the_declared_full_open_goal(fraction):
    with pytest.raises(ValueError):
        reference_goal(binding(),{'terminal_joint_fraction':1.,'terminal_operator':'pull'},requested_fraction=fraction)


def test_close_uses_closed_limit_and_preserves_prismatic_tolerance():
    goal=reference_goal(binding('prismatic'),{'terminal_joint_fraction':1.,'terminal_operator':'close'})
    assert goal['target_position']==0.
    assert goal['position_tolerance']==.02


def test_compilation_does_not_drop_the_declared_joint_target():
    plan={'controller_program':[{'operator':'pull','subjects':['door'],'target':'host','constraints':{'terminal_joint_fraction':1.}}],
          'success_program':[{'subject':'door','object':'host','predicate':'door_open','terminal_joint_fraction':1.}]}
    steps=compiled_execution_steps(plan)
    assert steps[0]['terminal_joint_fraction']==1.


def test_bridge_success_cannot_hide_terminal_rebound_or_closed_gripper():
    b=binding();b.update(joint_name='hinge',lower_limit=-1.5707963,upper_limit=0.)
    goal=reference_goal(b,{'terminal_joint_fraction':1.,'terminal_operator':'pull'})
    frames=[{'time_s':i/15,'articulated':{'joint_name':'hinge','position':-1.55,'velocity':0.,
        'contact_evidence':[{'normal_force_n':2.}] if i==0 else []},
        'robot':{'droid_gripper_position':1. if i==0 else 0.}} for i in range(8)]
    payload={'sample_hz':15,'frames':frames}
    assert assess_measured_reference(payload,goal,b)['passed'] is True
    frames[-1]['articulated']['position']=-1.435
    assert assess_measured_reference(payload,goal,b)['passed'] is False
    frames[-1]['articulated']['position']=-1.55;frames[-1]['robot']['droid_gripper_position']=1.
    assert assess_measured_reference(payload,goal,b)['passed'] is False


def test_reused_cache_rechecks_terminal_measurements_instead_of_success_flag(tmp_path):
    b=binding();b.update(joint_name='hinge',lower_limit=-1.5707963,upper_limit=0.)
    goal=reference_goal(b,{'terminal_joint_fraction':1.,'terminal_operator':'pull'})
    frames=[{'time_s':i/15,'articulated':{'joint_name':'hinge','position':-1.55,'velocity':0.,
        'contact_evidence':[{'normal_force_n':2.}] if i==0 else []},
        'robot':{'droid_gripper_position':1. if i==0 else 0.}} for i in range(8)]
    trajectory={'sample_hz':15,'frames':frames,'binding':b}
    simulation=tmp_path/'simulation';simulation.mkdir()
    certificate=assess_measured_reference(trajectory,goal,b)
    (simulation/'measured_goal_certificate.json').write_text(json.dumps(certificate))
    inputs={'articulation_goal_contract':hashlib.sha256(json.dumps(goal,sort_keys=True).encode()).hexdigest()}
    assert _measured_goal_cache_valid(tmp_path,inputs,trajectory)
    frames[-1]['articulated']['position']=-1.435
    assert not _measured_goal_cache_valid(tmp_path,inputs,trajectory)
