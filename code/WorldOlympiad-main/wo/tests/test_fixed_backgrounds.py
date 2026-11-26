import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from wo.spec import TaskSpec, TaskObject, Step, Predicate, Articulation
from wo.scene import SceneObject, SceneInputs, fixed_background_ids, resolve_scenecode
from wo.models import lingbot_va as L
from wo.models.lingbot_va_client import Frame
from wo.tests.test_lingbot_va import make_ref, _sdf


def spec_with_background():
    return TaskSpec('background','pick_place','place block on pad',(
        TaskObject('block','block','manipuland',initial_container='container'),
        TaskObject('pad','pad','destination',geometry_constraints={'fixed':True}),
        TaskObject('table','table','support'),
        *[TaskObject(oid,'fixture','distractor',geometry_constraints={'fixed':True})
          for oid in ('background','tool','host','container','visible')],
        TaskObject('movable','box','distractor')),
        (Step('transport','block',Predicate('on_region','block','pad'),target='pad',tool='tool'),),
        (Articulation('door','host','revolute'),))


@pytest.mark.parametrize('value',[None,0,1,'true','false',[],{}])
def test_fixed_authoring_rejects_non_boolean(value):
    with pytest.raises(ValueError,match='fixed must be bool'):
        TaskObject('x','fixture','distractor',geometry_constraints={'fixed':value})


def test_background_selection_preserves_targets_tools_hosts_contents_and_visibility():
    spec=spec_with_background()
    objects={o.object_id:SceneObject(o.object_id,o.role,Path('source.sdf'),(0,0,0),(0,0,0,1),
                                   fixed=o.geometry_constraints.get('fixed',False)) for o in spec.objects}
    assert fixed_background_ids(spec,objects,required_visible=['visible'])=={'background'}
    records=[dict(object_id=o.object_id,role=o.role,fixed=o.fixed) for o in objects.values()]
    assert fixed_background_ids(spec,records,required_visible=['visible'])=={'background'}
    objects['background'].fixed=False
    assert fixed_background_ids(spec,objects,required_visible=['visible'])==set()


def test_no_declared_background_leaves_unrelated_legacy_geometry_uninspected():
    spec = TaskSpec('legacy', 'reorientation', 'upright', (
        TaskObject('block', 'bottle', 'manipuland'), TaskObject('table', 'table', 'support')),
        (Step('reorient', 'block', Predicate('upright', 'block')),))
    assert fixed_background_ids(spec, {'unrelated': object()}) == set()


def test_resolver_binds_fixed_to_the_same_source_geometry_and_pose(tmp_path):
    spec=spec_with_background();asset=tmp_path/'source.sdf';asset.write_text('<sdf/>')
    records={o.object_id:{'sdf_path':str(asset),'transform':{
        'translation':[i*.1,0,.8],'rotation_wxyz':[1,0,0,0]}} for i,o in enumerate(spec.objects)}
    records['table']['transform']['translation']=[0,0,0]
    house=tmp_path/'house.json';house.write_text(json.dumps({'rooms':{'room':{'objects':records}}}))
    (tmp_path/'task_inventory_certificate.json').write_text(json.dumps({'house_state':str(house),
        'matches':[{'task_object_id':o.object_id,'scene_object_id':o.object_id,'status':'matched',
                    'sdf_path':str(asset)} for o in spec.objects]}))
    scene=resolve_scenecode(spec,tmp_path)
    assert scene.objects['background'].fixed is True and scene.objects['block'].fixed is False
    assert scene.objects['background'].position==tuple(records['background']['transform']['translation'])
    assert scene.objects['background'].sdf_path==asset and asset.read_text()=='<sdf/>'


def add_background(ref, tmp_path):
    scene=json.loads((ref/'scene_inputs.json').read_text())
    src=_sdf(tmp_path/'assets/background/background.sdf','background',(.8,.8,.6),mass=2.)
    phys=_sdf(ref/'physics_sdf/background_s1.000/background.sdf','background',(.8,.8,.6),mass=2.)
    tree=ET.parse(phys)
    for link in tree.getroot().iter('link'):
        for visual in list(link.findall('visual')):link.remove(visual)
    tree.write(phys);(phys.parent/'.path').write_text(str(phys))
    row={'object_id':'background','role':'distractor','fixed':True,'scale':1.,'sdf_path':str(src),
         'position':[1.7,1.5,.3],'orientation_xyzw':[0,0,0,1]}
    scene['objects']['background']=row;(ref/'scene_inputs.json').write_text(json.dumps(scene))
    task=tmp_path/'task.json';spec=TaskSpec.load(task)
    spec=replace(spec,objects=(*spec.objects,TaskObject('background','fixture','distractor',
                                                    geometry_constraints={'fixed':True})))
    task.write_text(json.dumps(spec.to_dict()))
    # A stationary fixed root may be absent from legacy trajectories; reference_state
    # already binds its declared source pose instead of inventing a moving state.
    return src


def test_native_plan_keeps_entire_scene_rigid_and_loads_actual_support(tmp_path):
    ref=make_ref(tmp_path);src=add_background(ref,tmp_path);before=src.read_bytes()
    plan,_=L.build_plan(ref,tmp_path/'out',window_s=60,seed=0)
    assert plan['fixed_background_ids']==['background']
    assert plan['support_representation']=='scene_asset' and plan['transform']['xy_scale']==1.
    assert plan['transform']['z_offset']==0.
    assert {o['object_id'] for o in plan['objects']}=={'block','pad','background','table'}
    assert 'background' not in plan['required_visible']
    frame=Frame(plan['transform'])
    for o in plan['objects']:
        assert frame.to_ref(o['pose_rt']['p'])==pytest.approx(o['position'])
        assert o['pose_rt']['p'][2]==o['position'][2]
    assert src.read_bytes()==before
    bg=next(o for o in plan['objects'] if o['object_id']=='background')
    assert bg['fixed'] and len(bg['collisions'])>0 and len(bg['visuals'])>0


def test_background_cannot_enable_compression_of_unreachable_task_layout(tmp_path):
    ref=make_ref(tmp_path,spread=.6);add_background(ref,tmp_path)
    with pytest.raises(L.Unsupported,match='authored rigid task layout'):
        L.build_plan(ref,tmp_path/'out',window_s=60,seed=0)


@pytest.mark.parametrize('blocked',[False,True])
def test_expert_preserves_background_collision_and_authored_layout(tmp_path,monkeypatch,blocked):
    from wo.expert import embodiment as E, kin
    import wo.scene as S
    spec=TaskSpec('x','reorientation','upright',(
        TaskObject('block','bottle','manipuland'),TaskObject('table','table','support'),
        TaskObject('background','cabinet','distractor',geometry_constraints={'fixed':True})),
        (Step('reorient','block',Predicate('upright','block')),))
    obj=lambda oid,role,p,fixed:SceneObject(oid,role,Path('asset'),p,(0,0,0,1),fixed=fixed)
    scene=SceneInputs('x',obj('table','support',(0,0,0),True),{
        'block':obj('block','manipuland',(.2,.1,.8),False),
        'background':obj('background','distractor',(2.,2.,.3),True)},Path('robot'),(0,0,.8),0)
    seen=[]
    class World:
        def __init__(self,actual,*args,**kwargs):
            assert set(actual.objects)=={'block','background'}
            self.support_id='table';self.articulation={};seen.append(actual)
        def aabb(self,oid):
            if oid=='table':return np.array([-.5,-.5,0]),np.array([.5,.5,.8])
            if oid=='background':return np.array([1.5,1.5,0]),np.array([2.5,2.5,.6])
            return np.array([.18,.08,.8]),np.array([.22,.12,.9])
        def close(self):pass
    def candidates(lo,hi,boxes,**kwargs):
        assert len(boxes)==1 and boxes[0][0][0]==pytest.approx(.18)
        return [(0.,(0.,0.,.8),0.)]
    monkeypatch.setattr(E,'World',World);monkeypatch.setattr(E,'prepare_bodies',lambda *a:{})
    monkeypatch.setattr(S,'robot_base_candidates',candidates)
    monkeypatch.setattr(E,'droid_canonical_layout',lambda *a:pytest.fail('authored background was relaid out'))
    monkeypatch.setattr(E,'_reach_poses',lambda *a,**k:[])
    monkeypatch.setattr(kin,'Kinematics',lambda world:SimpleNamespace(home=[],in_collision=lambda q:'background' if blocked else None))
    if blocked:
        with pytest.raises(RuntimeError,match='no collision-free reachable'):
            E.place_robot(scene,tmp_path,spec,layout='droid')
    else:
        actual,info=E.place_robot(scene,tmp_path,spec,layout='droid')
        assert actual.objects==scene.objects and info['layout']['policy']=='preserve_fixed_background_scene'
    assert seen


@pytest.mark.parametrize('visible_background',[False,True])
def test_camera_fit_excludes_only_background_bounds_but_transforms_every_actor(monkeypatch,visible_background):
    import sys
    from wo.models import robotwin_visibility as V, lingbot_va_client as C
    objects=[{'object_id':'block','role':'manipuland','fixed':False,'pose_rt':{'p':[.1,.2,.8],'q_wxyz':[1,0,0,0]}},
             {'object_id':'background','role':'distractor','fixed':True,'pose_rt':{'p':[2.,2.,.3],'q_wxyz':[1,0,0,0]}},
             {'object_id':'table','role':'support','fixed':True,'pose_rt':{'p':[0.,0.,0.],'q_wxyz':[1,0,0,0]}}]
    originals={o['object_id']:list(o['pose_rt']['p']) for o in objects};seen={};bounds_read=[]
    handles={oid:SimpleNamespace(set_pose=lambda p,oid=oid:seen.update({oid:p})) for oid in originals}
    cameras=SimpleNamespace(static_camera_name=['head_camera'],static_camera_list=[
        SimpleNamespace(get_width=lambda:320,get_height=lambda:240)],
        get_config=lambda:{'head_camera':{'intrinsic_cv':np.eye(3),'extrinsic_cv':np.eye(4)}})
    task=SimpleNamespace(wo_handles=handles,cameras=cameras)
    plan={'objects':objects,'support_id':'table','fixed_background_ids':['background'],
          'required_visible':['block']+(['background'] if visible_background else []),
          'transform':{'yaw':0.,'xy_scale':1.,'xy_offset':[0.,0.],'z_offset':0.}}
    monkeypatch.setitem(sys.modules,'sapien',SimpleNamespace(Pose=lambda p,q:SimpleNamespace(p=p,q=q)))
    def vertices(oid):
        bounds_read.append(oid)
        return np.array([[.08,.18,.78],[.12,.22,.82]])
    monkeypatch.setattr(C,'SapienWorld',lambda *a:SimpleNamespace(collision_vertices=vertices))
    yaw=.3;delta=np.array([.03,-.02,0.])
    def fit(*args,**kwargs):
        assert len(args[5])==1 and len(kwargs['geometry'])==1
        return {'status':'ready','yaw_radians':yaw,'shift_xy_m':delta[:2]}
    monkeypatch.setattr(V,'fit_planar',fit)
    if visible_background:
        with pytest.raises(ValueError,match='required-visible'):V.align(task,plan)
        assert bounds_read==[] and seen=={}
    else:
        V.align(task,plan)
        assert bounds_read==['block'] and set(seen)==set(originals)
        for oid in originals:
            np.testing.assert_allclose(seen[oid].p,L._rotz(yaw)@originals[oid]+delta)
            assert seen[oid].p[2]==originals[oid][2]
