import hashlib
import json
from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest
from wo.models.cosmos_articulated import fixed_background_bodies
from wo.spec import TaskSpec,TaskObject,Step,Predicate,Articulation
from wo.tests.test_lingbot_va import _sdf


def fixture(tmp_path, *, fixed=True, tool=False, state=True):
    ref=tmp_path/'ref';ref.mkdir()
    src=_sdf(tmp_path/'source/bg.sdf','bg',(.4,.5,.6),mass=2.,visual='c.obj')
    phys=_sdf(ref/'physics_sdf/bg_s1.000/bg.sdf','bg',(.4,.5,.6),mass=7.,visual='c.obj')
    tree=ET.parse(phys)
    for visual in tree.findall('.//visual'):
        tree.find('.//link').remove(visual)
    ET.SubElement(tree.find('.//inertial'),'pose').text='.012 .023 .034 0 0 0'
    tree.write(phys)
    objects=(TaskObject('host','cabinet','support'),TaskObject('bg','fixture','distractor',
               geometry_constraints={'fixed':fixed}))
    step=Step('articulate','door',Predicate('joint_open','door'),tool='bg' if tool else None)
    spec=TaskSpec('x','articulation','open',objects,(step,),(Articulation('door','host','revolute'),))
    task=tmp_path/'task.json';task.write_text(json.dumps(spec.to_dict()))
    (ref/'certificate.json').write_text(json.dumps({'task':str(task)}))
    scene={'objects':{'bg':{'object_id':'bg','role':'distractor','fixed':fixed,'scale':1.,
        'sdf_path':str(src),'position':[9.,8.,7.],'orientation_xyzw':[0,0,0,1]}}}
    traj={'frames':[{'objects':{'bg':{'link_position':[2.1,1.2,.1],
        'link_orientation_xyzw':[0,0,0,1],'joint_positions':{}}} if state else {}}]}
    return ref,scene,traj,src,phys


def test_articulated_background_uses_bound_prepared_mass_com_and_recorded_pose(tmp_path):
    ref,scene,traj,src,phys=fixture(tmp_path);before={p:p.read_bytes() for p in (src,phys)}
    bodies,binding=fixed_background_bodies(ref,tmp_path/'out',scene,traj,'host')
    assert len(bodies)==1
    body=bodies[0];assert body.name=='bg' and body.static and not body.target
    assert body.pose.xyz==pytest.approx([2.1,1.2,.1])
    assert body.links[0].inertial.mass==pytest.approx(7.)
    assert body.links[0].inertial.com==pytest.approx([.012,.023,.034])
    assert body.links[0].visuals and body.links[0].collisions
    assert binding['pose_sources']['bg']=='reference trajectory frame0 link pose'
    assert binding['source_sdf_sha256']['bg']==hashlib.sha256(src.read_bytes()).hexdigest()
    assert all(p.read_bytes()==content for p,content in before.items())


def test_fixed_background_fallback_is_explicit_and_stays_static(tmp_path):
    ref,scene,traj,_,_=fixture(tmp_path,state=False)
    bodies,binding=fixed_background_bodies(ref,tmp_path/'out',scene,traj,'host')
    assert bodies[0].pose.xyz==pytest.approx([9.,8.,7.])
    assert binding['pose_sources']['bg'].startswith('fixed scene_inputs pose')


def test_missing_fixed_authoring_keeps_legacy_branch_and_creates_no_bundle(tmp_path):
    ref,scene,traj,_,_=fixture(tmp_path,fixed=False)
    assert fixed_background_bodies(ref,tmp_path/'out',scene,traj,'host')==([],None)
    assert not (tmp_path/'out').exists()


def test_fixed_tool_is_not_silently_dropped_as_background(tmp_path):
    ref,scene,traj,_,_=fixture(tmp_path,tool=True)
    with pytest.raises(ValueError,match='cannot omit task-related fixed'):
        fixed_background_bodies(ref,tmp_path/'out',scene,traj,'host')


def test_missing_canonical_physics_never_falls_back_to_raw_collision(tmp_path):
    ref,scene,traj,_,phys=fixture(tmp_path);phys.unlink()
    with pytest.raises(FileNotFoundError,match='missing expert physics asset'):
        fixed_background_bodies(ref,tmp_path/'out',scene,traj,'host')
