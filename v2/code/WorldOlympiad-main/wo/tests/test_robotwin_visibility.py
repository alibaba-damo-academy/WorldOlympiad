from types import SimpleNamespace
import numpy as np
import pytest
from wo.models.robotwin_visibility import fit_shift, pixel_visibility


def test_frustum_fit_uses_one_translation_and_preserves_pair_distance():
    K=np.array([[100,0,50],[0,100,50],[0,0,1]])
    E=np.c_[np.eye(3),np.zeros(3)]
    pts=np.array([[-.7,0,1],[-.5,0,1]])
    result=fit_shift(K,E,100,100,pts,[],[[-1,1],[-1,1]])
    assert result['status']=='ready'
    d=np.r_[result['shift_xy_m'],0]
    assert result['shift_xy_m']==pytest.approx([.23,0])
    assert np.linalg.norm((pts[1]+d)-(pts[0]+d))==pytest.approx(.2)
    for point in pts+d:
        uv=(K@point)[:2]/point[2]
        assert np.all(uv>=3-1e-8) and np.all(uv<=96+1e-8)


def test_impossible_camera_fit_is_not_reported_ready():
    K=np.array([[100,0,50],[0,100,50],[0,0,1]])
    E=np.c_[np.eye(3),np.zeros(3)]
    result=fit_shift(K,E,100,100,[[-3,0,1],[3,0,1]],[],[[-1,1],[-1,1]])
    assert result['status']=='infeasible' and result['shift_xy_m'] is None


def test_actor_center_inside_does_not_pass_occluded_actor_or_operable_region():
    seg=np.zeros((100,100,4),np.uint32)
    seg[5:10,5:10,1]=7
    K=np.array([[100,0,50],[0,100,50],[0,0,1]])
    E=np.c_[np.eye(3),np.zeros(3)]
    camera=SimpleNamespace(take_picture=lambda:None,get_picture=lambda _:seg)
    cameras=SimpleNamespace(static_camera_name=['head_camera'],static_camera_list=[camera],
        get_config=lambda:{'head_camera':{'intrinsic_cv':K,'extrinsic_cv':E}})
    task=SimpleNamespace(cameras=cameras,scene=SimpleNamespace(update_render=lambda:None))
    world=SimpleNamespace(_entities=lambda oid:[SimpleNamespace(get_per_scene_id=lambda:7 if oid=='door' else 8)],
        _root_pose=lambda _:SimpleNamespace(to_transformation_matrix=lambda:np.eye(4)))
    target={'object_id':'door','host':'cabinet','local_position':[0,0,1]}
    result=pixel_visibility(task,world,['door','hidden'],[target])
    assert result['door']['visible_pixels']==25 and result['hidden']['visible_pixels']==0
    assert result['door']['operable_regions'][0]['visible'] is False


def test_planar_fallback_fits_failed_native_bounds_without_scale_or_camera_edits():
    from wo.models.robotwin_visibility import fit_planar, corners
    from wo.models.lingbot_va import _rotz, WORKSPACE_RT
    K=np.array([[358.6421814,0,160],[0,358.6421814,120],[0,0,1]])
    E=np.array([[1,0,0,.032],[0,-.8,-.6,.45],[0,.6,-.8,1.35]])
    bounds=[([-.217420938,-.097049490,.743001082],[-.125146453,-.004893115,.935277846]),
            ([.037499044,-.181535166,.743000861],[.299939408,.081526429,.938418662])]
    points=np.vstack([corners(b) for b in bounds])
    assert fit_shift(K,E,320,240,points,bounds,WORKSPACE_RT)['status']=='infeasible'
    before_K,before_E=K.copy(),E.copy()
    # Round collision profiles with the actual failed native bounds. Acceptance
    # must project each newly axis-aligned box, not rotate the old box corners.
    geometry=[]
    for lo,hi in bounds:
        lo,hi=np.asarray(lo),np.asarray(hi);c=(lo+hi)/2;h=(hi-lo)/2
        vertices=np.array([[c[0]+h[0]*np.cos(a),c[1]+h[1]*np.sin(a),z]
                           for a in np.arange(32)*2*np.pi/32 for z in (lo[2],hi[2])])
        geometry.append({'points':vertices,'visible':True})
    result=fit_planar(K,E,320,240,points,bounds,WORKSPACE_RT,geometry=geometry)
    assert result['status']=='ready' and abs(result['yaw_radians'])>np.deg2rad(15)
    R=_rotz(result['yaw_radians']);delta=np.r_[result['shift_xy_m'],0]
    transformed=[g['points'] @ R.T+delta for g in geometry]
    rotated=np.vstack([corners((v.min(0),v.max(0))) for v in transformed])
    assert np.linalg.norm(transformed[0][0]-transformed[1][-1])==pytest.approx(
        np.linalg.norm(geometry[0]['points'][0]-geometry[1]['points'][-1]))
    camera=(E @ np.c_[rotated,np.ones(len(rotated))].T).T
    uv=(camera @ K.T)[:,:2]/camera[:,2,None]
    assert np.all(uv>=[3-1e-7,3-1e-7]) and np.all(uv<=[316+1e-7,236+1e-7])
    for axis in (0,1):
        assert rotated[:,axis].min()>=WORKSPACE_RT[axis][0]-1e-7
        assert rotated[:,axis].max()<=WORKSPACE_RT[axis][1]+1e-7
    np.testing.assert_array_equal(K,before_K);np.testing.assert_array_equal(E,before_E)


def test_planar_fast_path_is_exact_and_impossible_layout_stays_failed():
    from wo.models.robotwin_visibility import fit_planar
    K=np.array([[100,0,50],[0,100,50],[0,0,1]])
    E=np.c_[np.eye(3),np.zeros(3)]
    args=(K,E,100,100,[[-.7,0,1],[-.5,0,1]],[],[[-1,1],[-1,1]])
    expected=fit_shift(*args)
    result=fit_planar(*args)
    assert result==dict(expected,yaw_radians=0.)
    result=fit_planar(K,E,100,100,[[-3,0,1],[3,0,1]],
                      [([-3,-3,1],[3,3,1])],[[-1,1],[-1,1]],
                      geometry=[{'points':np.array([[-3,-3,1],[3,3,1]]),'visible':True}])
    assert result['status']=='infeasible' and result['shift_xy_m'] is None


def test_alignment_composes_all_actor_roots_and_reference_inverse(monkeypatch):
    import sys
    from copy import deepcopy
    from wo.models import robotwin_visibility as V, lingbot_va as L, lingbot_va_client as C
    yaw=np.deg2rad(-15);delta=np.array([-.065,.065,0.])
    tf={'yaw':.4,'xy_scale':1.,'xy_offset':[.1,-.2],'z_offset':-.3}
    objects=[];seen={}
    for oid,fixed,role,p,q in [('b',False,'manipuland',[.1,.2,1],[0,0,0,1]),
                              ('fixture',True,'support',[-.2,.1,0],[0,0,.3,np.sqrt(.91)])]:
        objects.append(dict(object_id=oid,fixed=fixed,role=role,position=p,orientation_xyzw=q,
                            pose_rt=L.to_rt(tf,p,q),initial_joints={'hinge':.2}))
    before=deepcopy(objects)
    plan={'objects':objects,'support_id':'fixture','required_visible':['b'],'transform':deepcopy(tf)}
    handles={'b':SimpleNamespace(set_pose=lambda p:seen.update(b=p)),
             'fixture':SimpleNamespace(set_root_pose=lambda p:seen.update(fixture=p))}
    cameras=SimpleNamespace(static_camera_name=['head_camera'],static_camera_list=[
        SimpleNamespace(get_width=lambda:320,get_height=lambda:240)],
        get_config=lambda:{'head_camera':{'intrinsic_cv':np.eye(3),'extrinsic_cv':np.eye(4)}})
    task=SimpleNamespace(wo_handles=handles,cameras=cameras)
    monkeypatch.setitem(sys.modules,'sapien',SimpleNamespace(Pose=lambda p,q:SimpleNamespace(p=p,q=q)))
    monkeypatch.setattr(C,'SapienWorld',lambda *a:SimpleNamespace(collision_vertices=lambda oid:np.array([[0,0,0],[1,1,1]])))
    monkeypatch.setattr(V,'fit_planar',lambda *a,**k:dict(status='ready',yaw_radians=yaw,shift_xy_m=delta[:2]))
    V.align(task,plan)
    frame=C.Frame(plan['transform'])
    for obj,original in zip(objects,before):
        pose=obj['pose_rt'];assert obj['initial_joints']==original['initial_joints']
        assert frame.to_ref(pose['p'])==pytest.approx(original['position'])
        np.testing.assert_allclose(L._q_xyzw_mat(frame.rot_to_ref(pose['q_wxyz'])),
                                   L._q_xyzw_mat(original['orientation_xyzw']),atol=1e-12)
        np.testing.assert_allclose(seen[obj['object_id']].p,pose['p'])
        np.testing.assert_allclose(seen[obj['object_id']].q,pose['q_wxyz'])
    assert plan['transform']['xy_scale']==tf['xy_scale'] and plan['transform']['z_offset']==tf['z_offset']


@pytest.mark.parametrize('kind',['inside','covering','on_region','inserted','on','stacked','contact'])
def test_predicate_measurements_preserve_reference_axes_and_contact_heights(monkeypatch,kind):
    import sys
    from wo.models.lingbot_va_client import SapienWorld, Frame, GoalMonitor
    from wo.models.lingbot_va import _rotz, _mat_wxyz
    from wo.predicates import evaluate
    from wo.spec import Predicate
    class Body:
        def get_collision_shapes(self):return [self.shape]
    class Box:
        def get_half_size(self):return self.half
        def get_local_pose(self):return SimpleNamespace(to_transformation_matrix=lambda:np.eye(4))
    class Mesh:pass
    class Sphere:pass
    physx=SimpleNamespace(PhysxRigidBaseComponent=Body,PhysxCollisionShapeBox=Box,
        PhysxCollisionShapeConvexMesh=Mesh,PhysxCollisionShapeSphere=Sphere)
    monkeypatch.setitem(sys.modules,'sapien',SimpleNamespace(physx=physx))
    subject_size=[.16,.08,.1];target_size=[.4,.2,.2];subject_p=[.08,.03,.98]
    if kind=='covering':subject_size=[.5,.3,.05];subject_p=[0,0,1.125]
    if kind in {'on','stacked'}:subject_p=[.08,.03,1.15]
    def world_for(yaw,offset,measure):
        R=_rotz(yaw);offset=np.asarray(offset)
        frame=Frame({'yaw':yaw,'xy_scale':1.,'xy_offset':offset[:2],'z_offset':offset[2]})
        entities={}
        for oid,p,size in [('s',subject_p,subject_size),('t',[0,0,1],target_size)]:
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=R@p+offset
            shape=Box();shape.half=np.asarray(size)/2
            body=Body();body.shape=shape
            pose=SimpleNamespace(p=T[:3,3],q=_mat_wxyz(R),to_transformation_matrix=lambda T=T:T)
            entities[oid]=SimpleNamespace(get_pose=lambda pose=pose:pose,get_components=lambda body=body:[body])
        world=object.__new__(SapienWorld);world.measurement_frame=frame if measure else None
        world._entities=lambda oid:[entities[oid]];world._root_pose=lambda oid:entities[oid].get_pose()
        world.support_id='t';world.handles=entities;world.bodies=entities
        point=R @ np.array([.05,.02,.94])+offset
        world._contacts=[('s','t',[SimpleNamespace(normal=np.array([0,0,1]),position=point,impulse=np.array([0,0,1]))])]
        world.refresh=lambda:None
        return world
    original=world_for(0,[0,0,0],False)
    native=world_for(.71,[.3,-.2,-.5],False)
    recovered=world_for(.71,[.3,-.2,-.5],True)
    pred=Predicate(kind,'s','t',{})
    expected,evidence=evaluate(original,pred)
    actual,recovered_evidence=evaluate(recovered,pred)
    assert actual==expected and expected
    assert recovered.support_below('s')=='t'
    for oid in ('s','t'):
        np.testing.assert_allclose(recovered.aabb(oid),original.aabb(oid),atol=1e-12)
        np.testing.assert_allclose(recovered.object_pose(oid)[0],original.object_pose(oid)[0],atol=1e-12)
    assert native.aabb('s')[0][2]==pytest.approx(original.aabb('s')[0][2]-.5)
    if kind=='inside':
        goal={'kind':kind,'subject':'s','target':'t','params':{},'require_release':False}
        monitor=GoalMonitor(recovered,[goal]);monitor.update(0.);monitor.update(.5)
        assert monitor.success_at==pytest.approx(.5)
        np.testing.assert_allclose(monitor.initial['s'][0],subject_p)
