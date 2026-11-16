from types import SimpleNamespace
import numpy as np
import pytest
from wo.models import lingbot_va as L
from wo.models.lingbot_va_client import SapienWorld


def test_large_fixed_support_uses_rigid_map_without_shrinking():
    fixture={"object_id":"cabinet","role":"support","fixed":True,"position":[0,0,0],
             "local_bounds":[[-1,-1,0],[1,1,1]],"orientation_xyzw":[0,0,0,1]}
    tf=L.fit_layout([-.4,-.17,.7],0,[fixture])
    assert tf["xy_scale"]==1
    assert L.to_rt(tf,[-.4,-.17,.7],[0,0,0,1])["p"][:2]==pytest.approx(L.ROBOT_BASE_RT)
    assert fixture["local_bounds"]==[[-1,-1,0],[1,1,1]]


def test_movable_large_object_still_requires_real_workspace_fit():
    obj={"object_id":"box","role":"manipuland","fixed":False,"position":[0,0,0],
         "local_bounds":[[-1,-1,0],[1,1,1]],"orientation_xyzw":[0,0,0,1]}
    with pytest.raises(L.Unsupported):L.fit_layout([0,0,0],0,[obj])


def test_camera_component_resolves_its_actual_native_link():
    door=SimpleNamespace(get_name=lambda:"door",get_pose=lambda:"door_pose")
    base=SimpleNamespace(get_name=lambda:"base")
    world=object.__new__(SapienWorld)
    world.articulation_spec={"open_part":{"host":"cabinet","child_link":"door"}}
    world.handles={"cabinet":SimpleNamespace(get_links=lambda:[SimpleNamespace(entity=base),SimpleNamespace(entity=door)])}
    assert world._entities("open_part")==[door]
    assert world._root_pose("open_part")=="door_pose"
    world.articulation_spec["open_part"]["child_link"]="missing"
    with pytest.raises(RuntimeError):world._entities("open_part")


def test_fixture_alignment_centres_operable_region_without_scaling_or_z_change():
    fixture={"object_id":"cabinet","role":"support","fixed":True,"position":[0,0,0],
             "local_bounds":[[-1,-1,0],[1,1,1]],"orientation_xyzw":[0,0,0,1]}
    point=[-.2,.1,.3]
    tf=L.fit_layout([-.4,-.17,.7],0,[fixture],[point])
    transformed=L.to_rt(tf,point,[0,0,0,1])["p"]
    assert transformed==pytest.approx([0,L.WORKSPACE_CENTER_Y_RT,.3])
    assert tf["xy_scale"]==1.0 and tf["placement"]=="interaction_region_alignment"


def test_fixture_geometry_probe_uses_physics_asset_not_gltf_visual_urdf(monkeypatch):
    from wo import render_plan
    seen=[]
    class FK:
        def __init__(self,path,fixed):
            seen.append(str(path));self.body=1;self.cid=0;self.link_index={"door":2}
            self.p=SimpleNamespace(getAABB=lambda *a,**k:([0,0,0],[1,1,1]),getNumJoints=lambda *a,**k:1,disconnect=lambda *a:None)
        def pose(self,*a):pass
    monkeypatch.setattr(render_plan,"_FK",FK)
    host={"object_id":"cabinet","role":"support","fixed":True,"position":[0,0,0],
          "orientation_xyzw":[0,0,0,1],"urdf":"rendered_gltf.urdf","physics_urdf":"measured_collision.urdf",
          "initial_joints":{"hinge":0}}
    points=L.fixture_interaction_points([host],{"door":{"host":"cabinet","child_link":"door"}})
    assert seen==["measured_collision.urdf"] and points==[[.5,.5,.5]]


def test_fixture_outward_face_is_oriented_towards_native_arms():
    fixture={"object_id":"support","fixed":True,"role":"support"}
    tf=L.fit_layout([0,0,0],0,[fixture],[[0,-.2,.3]],[[0,-1]])
    assert tf["yaw"]==pytest.approx(0)
    assert L.to_rt(tf,[0,-.2,.3],[0,0,0,1])["p"]==pytest.approx([0,L.WORKSPACE_CENTER_Y_RT,.3])


def test_joint_progress_uses_reference_endpoints_not_observed_start():
    world=object.__new__(SapienWorld)
    world.articulation_spec={"door":{"q_closed":0.,"q_open":-1.5}}
    world.q0={"door":-1.5}
    world._joint=lambda _: (None,0,SimpleNamespace(get_limits=lambda:[[-1.5,0]]))
    world.joint_q=lambda _:0.
    assert world.articulation_fraction("door")==0.
    world.joint_q=lambda _:-1.5
    assert world.articulation_fraction("door")==1.


def test_reference_internal_filter_retains_external_and_other_object_contacts():
    from wo.models.lingbot_va_client import match_reference_internal_collisions
    class Shape:
        def __init__(self, groups): self.groups=list(groups)
        def get_collision_groups(self): return self.groups
        def set_collision_groups(self, groups): self.groups=list(groups)
    robot=Shape([1,1,1,0]);other=Shape([1,1,1,1])
    door=Shape([1,1,4,0x10000]);hinge=Shape([1,1,8,0x10000])
    component=lambda shape:SimpleNamespace(get_collision_shapes=lambda:[shape])
    scene=SimpleNamespace(physx_system=SimpleNamespace(rigid_dynamic_components=[component(other)],
        rigid_static_components=[],articulation_link_components=[component(robot),component(door),component(hinge)]))
    art=SimpleNamespace(get_links=lambda:[component(door),component(hinge)])
    assert match_reference_internal_collisions(scene,art)==2
    def collides(a,b):
        a,b=a.groups,b.groups
        return bool((a[0]&b[1] or a[1]&b[0]) and not (a[2]&b[2] and a[3]&0xffff==b[3]&0xffff))
    assert not collides(door,hinge)
    assert collides(door,robot) and collides(door,other)
    assert door.groups==[1,1,5,0x10002] and hinge.groups==[1,1,9,0x10002]


def test_removed_table_mask_is_synced_in_both_arms_and_batch_planners_only():
    from wo.models.lingbot_va_client import remove_obsolete_table_from_planners
    calls=[]
    def motion(label):
        return SimpleNamespace(world_coll_checker=SimpleNamespace(
            world_model=SimpleNamespace(get_obstacle=lambda name:object() if name in {"table","other_obstacle"} else None),
            get_obstacle_names=lambda:(_ for _ in ()).throw(TypeError("mesh cache is absent")),
            enable_obb=lambda name,enable,env_idx:calls.append((label,name,enable))),
            graph_planner=SimpleNamespace(reset_buffer=lambda:None))
    robot=SimpleNamespace(communication_flag=False)
    for arm in ["left","right"]:
        setattr(robot,arm+"_planner",SimpleNamespace(motion_gen=motion(arm),motion_gen_batch=motion(arm+"_batch")))
    result=remove_obsolete_table_from_planners(robot)
    assert len(calls)==4 and all(name=="table" and enabled is False for _,name,enabled in calls)
    assert result["physical_fixture_collisions"]=="unchanged"
