"""Shared measured endpoint contract for articulated reference and native runs."""
from __future__ import annotations
import math
import numpy as np

POLICY = 'articulated_measured_target_release_stability.v1'


def joint_tolerance(joint_type: str) -> float:
    if joint_type == 'revolute':return .08
    if joint_type == 'prismatic':return .02
    raise ValueError('articulated goal requires revolute or prismatic motion')


def reference_goal(binding: dict, step: dict, *, requested_fraction: float | None = None) -> dict:
    declared=step.get('terminal_joint_fraction')
    if isinstance(declared,bool) or not isinstance(declared,(int,float)) or declared != 1.:
        raise ValueError('joint-limit reference needs an explicit full-travel task target')
    span=abs(float(binding['open_position'])-float(binding['closed_position']))
    tolerance=joint_tolerance(binding['joint_type'])
    if not math.isfinite(span) or span <= tolerance:
        raise ValueError('joint travel does not distinguish closed from target at native tolerance')
    native_fraction=1.-tolerance/span
    accepted=native_fraction if requested_fraction is None else float(requested_fraction)
    if not math.isfinite(accepted) or not native_fraction <= accepted <= 1.:
        raise ValueError('reference acceptance may not weaken the frozen native target tolerance')
    target=binding['closed_position'] if step['terminal_operator']=='close' else binding['open_position']
    return {'policy':POLICY,'target_position':float(target),'target_fraction':1.,
            'position_tolerance':tolerance,'required_joint_travel_fraction':accepted,
            'native_required_joint_travel_fraction':native_fraction,'max_speed':.02,
            'hold_steps':5,'control_hz':15,'release_required':True,
            'bridge_endpoint_acceptance_only':True,
            'measured_release_and_stability_certificate_required':True}


def assess_measured_reference(trajectory: dict, goal: dict, binding: dict) -> dict:
    """Certify the measured terminal interval, independently of bridge success."""
    frames=trajectory.get('frames',[])
    if trajectory.get('sample_hz')!=goal['control_hz'] or len(frames)<goal['hold_steps']+1:
        raise ValueError('reference has no native-rate stability interval')
    times=np.array([frame['time_s'] for frame in frames],dtype=float)
    if not np.allclose(np.diff(times),1/goal['control_hz'],atol=1e-6):
        raise ValueError('reference timestamps changed')
    stable=0;first_success=None;records=[];contact_observed=False
    for index,frame in enumerate(frames):
        state=frame['articulated']
        if state['joint_name']!=binding['joint_name']:
            raise ValueError('reference measured another joint')
        position,velocity=float(state['position']),float(state['velocity'])
        gripper=float(frame['robot']['droid_gripper_position'])
        if not all(math.isfinite(v) for v in (position,velocity,gripper)) or not 0<=gripper<=1:
            raise ValueError('reference has invalid measured joint/gripper data')
        if not binding['lower_limit']-.005<=position<=binding['upper_limit']+.005:
            raise ValueError('reference joint leaves native limits')
        contacts=any(float(c.get('normal_force_n',0))>1e-3 for c in state.get('contact_evidence',[]))
        contact_observed |= contacts
        released=not contacts and gripper<.2
        error=abs(position-goal['target_position'])
        at_target=error<=goal['position_tolerance']
        stationary=abs(velocity)<=goal['max_speed']
        stable=stable+1 if released and at_target and stationary else 0
        if stable>=goal['hold_steps'] and first_success is None:first_success=index
        records.append({'frame':index,'position':position,'target_error':error,
                        'released':released,'stationary':stationary,'stable_steps':stable})
    return {'schema_version':'articulated_reference_goal_certificate.v1','policy':goal['policy'],
            'passed':contact_observed and stable>=goal['hold_steps'],
            'contact_observed':contact_observed,'first_goal_frame':first_success,
            'goal_contract':goal,'final':records[-1],
            'terminal_interval':records[-goal['hold_steps']:],
            'measurement_source':'recorded simulator joint position, velocity, gripper and contact force'}
