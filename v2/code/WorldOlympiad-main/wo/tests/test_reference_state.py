import json
from types import SimpleNamespace
import numpy as np
import pytest
from wo.models import reference_state as R
from wo.models import lingbot_va as L
from wo.models.lingbot_va_client import Frame
from wo.tests.test_lingbot_va import make_ref


def test_plan_uses_recorded_link_pose_and_effective_physics_without_rewriting_ref(tmp_path):
    ref = make_ref(tmp_path)
    path = ref / 'trajectory.json'
    data = json.loads(path.read_text())
    data['frames'][0]['objects']['block']['link_position'][0] += .015
    path.write_text(json.dumps(data))
    before = {str(p): R.sha(p) for p in ref.rglob('*') if p.is_file()}
    plan, _ = L.build_plan(ref, tmp_path/'out', window_s=60, seed=0)
    obj = next(o for o in plan['objects'] if o['object_id']=='block')
    assert obj['position'] == data['frames'][0]['objects']['block']['link_position']
    assert obj['measured_dynamics']['lateral_friction'] == 1.
    assert obj['mass'] == pytest.approx(.1)
    assert plan['support_dynamics']['base_link']['lateral_friction'] == .8
    assert {str(p): R.sha(p) for p in ref.rglob('*') if p.is_file()} == before
    R.verify(plan)
    obj['position'][0] += .01
    with pytest.raises(ValueError, match='differs'):
        R.verify(plan)


def test_missing_moving_initial_pose_is_input_error(tmp_path):
    ref = make_ref(tmp_path)
    path = ref/'trajectory.json'; data = json.loads(path.read_text())
    del data['frames'][0]['objects']['block']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='missing canonical reference initial link pose: block'):
        R.collect(ref)


def test_omitted_fixed_support_uses_config_pose(tmp_path):
    ref = make_ref(tmp_path)
    path = ref/'trajectory.json'; data = json.loads(path.read_text())
    del data['frames'][0]['objects']['table']
    path.write_text(json.dumps(data))
    measured = R.collect(ref)
    assert measured['initial_objects']['table']['link_position'] == [0,0,0]
    assert measured['initial_pose_sources']['table'].startswith('fixed scene_inputs')


def test_initial_audit_preserves_nonidentity_quaternion_order():
    # Model the production boundary: SapienWorld receives raw SAPIEN wxyz,
    # then returns xyzw. This catches both a missing and a double conversion.
    q_xyzw = np.array([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    plan = {'objects':[{'object_id':'b','position':[0.,0.,1.], 'orientation_xyzw':q_xyzw.tolist()}]}
    world = SimpleNamespace(object_pose=lambda _: (np.array([0.,0.,1.]), q_xyzw))
    frame = Frame({'yaw':0.,'xy_scale':1.,'xy_offset':[0,0],'z_offset':0})
    result = R.initial_audit(plan, world, frame)
    assert result['status'] == 'ready'
    assert result['objects']['b']['orientation_delta_rad'] == pytest.approx(0.)


def test_initial_audit_detects_settling_drift_in_reference_frame():
    tf = {'yaw':0.,'xy_scale':1.,'xy_offset':[.2,-.1],'z_offset':-.3}
    frame = Frame(tf)
    plan = {'objects':[{'object_id':'b','position':[0.,0.,1.], 'orientation_xyzw':[0,0,0,1]}]}
    world = SimpleNamespace(object_pose=lambda _: (np.array([.2,-.1,.7]),np.array([0.,0.,0.,1.])))
    assert R.initial_audit(plan,world,frame)['status']=='ready'
    world.object_pose = lambda _: (np.array([.25,-.1,.7]),np.array([0.,0.,0.,1.]))
    assert R.initial_audit(plan,world,frame)['status']=='input_mismatch'


def test_initial_audit_also_checks_joint_state():
    plan={'objects':[{'object_id':'cabinet','position':[0,0,0],'orientation_xyzw':[0,0,0,1],
                      'initial_joints':{'hinge':0}}],
          'articulations':{'door':{'host':'cabinet','joint_name':'hinge','joint_type':'revolute'}}}
    world=SimpleNamespace(object_pose=lambda _:(np.zeros(3),np.array([0.,0,0,1.])),joint_q=lambda _: .3)
    frame=Frame({'yaw':0,'xy_scale':1,'xy_offset':[0,0],'z_offset':0})
    result=R.initial_audit(plan,world,frame)
    assert result['objects']['cabinet']['passed']
    assert result['status']=='input_mismatch' and not result['articulations']['door']['passed']
