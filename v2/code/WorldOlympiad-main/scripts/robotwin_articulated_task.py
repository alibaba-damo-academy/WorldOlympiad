"""Project-owned cabinet tasks using native RoboTwin ALOHA actions and SAPIEN contact."""
from copy import deepcopy
from pathlib import Path
import json,math
import numpy as np
import sapien
import transforms3d as t3d
from envs._base_task import Base_Task
from envs.utils import ArmTag,create_box

class FixtureLink:
    """Expose one fixture link to RoboTwin's native reference-framing contract.

    The native ``Actor`` wrapper rewrites the mass of every rigid component it
    wraps, which would change the passive joint physics of the cabinet, so this
    only advertises the link whose measured bounds the framing certificate needs.
    """
    def __init__(self,link,name):
        # A native link wrapper exposes component geometry only through the
        # entity view of that link.
        self.entity=getattr(link,"entity",link)
        self.config={"name":name,"scale":[1.,1.,1.]}
    def get_pose(self):
        return self.entity.get_pose()


class CabinetTask(Base_Task):
    component="drawer"
    def setup_demo(self,is_test=False,**kwags):
        self.fixture=deepcopy(kwags["articulated_fixture"])
        self.trace=[]
        cfg=deepcopy(kwags)
        camera_profile=self.fixture.get("camera_profile","legacy_override")
        if camera_profile not in {"legacy_override","native_translated_mount_v1","native_target_fallback_v2","native_context_fit_v3"}:
            raise ValueError("unknown fixture camera profile")
        for name in ("left_embodiment_config","right_embodiment_config"):
            # Configs can alias the same dictionary. Each side needs its own
            # untouched native mount before applying the base translation.
            robot=deepcopy(kwags[name]);cfg[name]=robot
            native_base=np.asarray(robot["robot_pose"][0][:3],dtype=float)
            actual_base=np.asarray(self.fixture.get("robot_position",[0,-.65,self.fixture.get("robot_lift",0.)]),dtype=float)
            for pose in robot["robot_pose"]:pose[:3]=actual_base.tolist()
            if camera_profile in {"native_translated_mount_v1","native_target_fallback_v2","native_context_fit_v3"}:
                for camera in robot["static_camera_list"]:
                    camera["position"]=(np.asarray(camera["position"])+actual_base-native_base).tolist()
            else:
                eye=np.array(self.fixture["camera_eye"]);target=np.array(self.fixture["camera_target"])
                f=target-eye;f/=np.linalg.norm(f);left=np.cross([0.,0.,1.],f);left/=np.linalg.norm(left)
                robot["static_camera_list"][0].update(position=eye.tolist(),forward=f.tolist(),left=left.tolist())
        self.input_camera_profile=camera_profile
        self._init_task_env_(**cfg)
        self.input_camera_adjustment=None
        if camera_profile=="native_target_fallback_v2":
            camera=self.cameras.static_camera_list[self.cameras.static_camera_name.index("head_camera")]
            point=self.handle_pose()[:3,3]
            xyz=camera.get_extrinsic_matrix()@np.r_[point,1.]
            pixel=camera.get_intrinsic_matrix()@xyz[:3]
            uv=pixel[:2]/pixel[2]/[cfg["head_camera_w"],cfg["head_camera_h"]] if xyz[2]>0 else np.array([-1.,-1.])
            if not (xyz[2]>0 and np.all(uv>=.05) and np.all(uv<=.95)):
                pose=camera.get_pose();eye=np.array(pose.p);forward=point-eye;forward/=np.linalg.norm(forward)
                left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left)
                up=np.cross(forward,left);matrix=np.eye(4);matrix[:3,:3]=np.column_stack([forward,left,up]);matrix[:3,3]=eye
                camera.set_pose(sapien.Pose(matrix))
                self.input_camera_adjustment={"rule":"initial_handle_look_at_if_native_target_out_of_frame",
                    "original_handle_uv":uv.tolist(),"position_unchanged":True,"native_fov_unchanged":True,
                    "target_world":point.tolist(),"original_quaternion_wxyz":pose.q.tolist(),
                    "adjusted_quaternion_wxyz":camera.get_pose().q.tolist()}

        if camera_profile=="native_context_fit_v3":
            from autoresearch.robotics_camera_fit import fit_context_camera
            camera=self.cameras.static_camera_list[self.cameras.static_camera_name.index("head_camera")]
            centers=[self.handle_pose()[:3,3],self.get_arm_pose("left")[:3],self.get_arm_pose("right")[:3]]
            matrix,self.input_camera_adjustment=fit_context_camera(camera.get_pose().p,centers,
                camera.get_intrinsic_matrix(),[cfg["head_camera_w"],cfg["head_camera_h"]])
            camera.set_pose(sapien.Pose(matrix))
        self.step_lim=int(self.fixture["step_limit"])
    def create_table_and_wall(self,table_xy_bias=(0,0),table_height=.74,visual_style=None):
        # The task uses a floor-standing cabinet. A tabletop through its drawer is invalid.
        self.table_xy_bias=list(table_xy_bias);self.wall_texture=None;self.table_texture=None
        self.wall=create_box(self.scene,sapien.Pose([0,2.,1.5]),half_size=[3,.05,1.5],color=[.92,.92,.92],is_static=True,name="wall")
        lift=self.fixture.get("robot_lift",0.)
        if lift:
            self.platform=create_box(self.scene,sapien.Pose(self.fixture.get("robot_position",[0,-.65,lift])[:2]+[lift/2]),half_size=[.4,.3,lift/2],color=[.3,.3,.32],is_static=True,name="robot_platform")
    def load_actors(self):
        loader=self.scene.create_urdf_loader();loader.fix_root_link=True;loader.load_multiple_collisions_from_file=True
        self.cabinet=loader.load(self.fixture["urdf"])
        if self.cabinet is None:raise RuntimeError("cabinet URDF import failed")
        self.cabinet.set_root_pose(sapien.Pose(self.fixture["position"],[0,0,0,1]))
        self.cabinet.set_solver_position_iterations(32)
        self.cabinet.set_solver_velocity_iterations(8)
        self.robot.left_entity.set_solver_position_iterations(32)
        self.robot.left_entity.set_solver_velocity_iterations(8)
        self.cabinet.set_qpos([0.]);self.cabinet.set_qvel([0.])
        self.joint=self.cabinet.get_active_joints()[0]
        self.joint.set_drive_properties(stiffness=0.,damping=.25 if self.component=="drawer" else .18)
        self.joint.set_friction(0.)
        self.child=next(l for l in self.cabinet.get_links() if l.get_name()==("drawer_01" if self.component=="drawer" else "door_right"))
        # Task-owned geometry for the native reference framing certificate: the
        # fixed host and the moving panel must both stay observable.
        self.cabinet_host=FixtureLink(next(l for l in self.cabinet.get_links() if l.get_name()=="base_link"),"cabinet_host")
        self.cabinet_panel=FixtureLink(self.child,"cabinet_panel")
        self.initial_root=self.cabinet.get_root_pose().to_transformation_matrix()
    def handle_pose(self):
        p=np.array([0,.38,.82] if self.component=="drawer" else [-.48,.055,-.25])
        # Local +X is the native RoboTwin approach axis; +Y closes across the bar.
        r=np.array([[0,0,-1],[-1,0,0],[0,1,0]]) if self.component=="drawer" else np.array([[0,1,0],[-1,0,0],[0,0,1]])
        if self.component=="drawer":
            # Approach from above so the native wrist housing clears the lower drawer face.
            r=r@t3d.axangles.axangle2mat([0,0,1],-math.radians(30))
        else:
            r=r@t3d.axangles.axangle2mat([0,1,0],math.radians(20))
        m=np.eye(4);m[:3,:3]=r;m[:3,3]=p
        return self.child.get_pose().to_transformation_matrix()@m
    def target_pose(self,pre=.0):
        m=self.handle_pose();p=m[:3,3]-m[:3,0]*((.15 if self.component=="door" else .12)+pre)
        return p.tolist()+t3d.quaternions.mat2quat(m[:3,:3]).tolist()
    def diagnostics(self):
        if not hasattr(self,"cabinet"):return {}
        q=float(self.cabinet.get_qpos()[0]);v=float(self.cabinet.get_qvel()[0]);lo,hi=self.cabinet.get_qlimits()[0]
        contacts=[]
        for c in self.scene.get_contacts():
            names=[b.entity.name for b in c.bodies]
            if self.child.get_name() in names and any(n in self.robot.gripper_name for n in names):
                contacts.append({"bodies":names,"max_impulse":max([float(np.linalg.norm(p.impulse)) for p in c.points] or [0.])})
        ee=self.robot.get_left_ee_pose()
        return {"component":self.component,"joint_position":q,"joint_velocity":v,"limits":[float(lo),float(hi)],"joint_limit_valid":bool(lo-.01<=q<=hi+.01),"handle_world":self.handle_pose()[:3,3].tolist(),"left_ee":list(ee),"target_pose":self.target_pose(),"robot_qpos":self.robot.left_entity.get_qpos().tolist(),"gripper_contacts":contacts,"history":self.trace.copy()}
    def validate_physics(self):
        q=float(self.cabinet.get_qpos()[0]);v=float(self.cabinet.get_qvel()[0]);lo,hi=self.cabinet.get_qlimits()[0]
        if not np.isfinite([q,v]).all() or q<lo-.01 or q>hi+.01 or abs(v)>5:
            raise RuntimeError(f"invalid passive joint state q={q}, velocity={v}, limits={lo,hi}")
    def mark(self,stage):
        self.validate_physics()
        d=self.diagnostics();d.pop("history",None);d["stage"]=stage;d["plan_success"]=bool(self.plan_success);self.trace.append(d)
        print(json.dumps(d),flush=True)
        if self.fixture.get("diagnostics_dir"):
            out=Path(self.fixture["diagnostics_dir"]);out.mkdir(parents=True,exist_ok=True)
            (out/"contact_trace.json").write_text(json.dumps(self.trace,indent=2))
    def play_once(self):
        arm=ArmTag("left")
        self.mark("initial")
        # Rotate and descend while clear of the cabinet, then approach in small Cartesian steps.
        pre=self.target_pose(.10)
        pre[2] += .18
        self.move(self.move_to_pose(arm,pre));self.mark("rotate_clear")
        self.move(self.move_to_pose(arm,self.target_pose(.10)));self.mark("pregrasp")
        for distance in [.08,.06,.04,.02]:
            self.move(self.move_to_pose(arm,self.target_pose(distance)))
            self.mark("approach_"+str(distance))
        self.move(self.move_to_pose(arm,self.target_pose(0.)));self.mark("grasp_pose")
        self.move(self.close_gripper(arm));self.mark("close")
        for i in range(16 if self.component=="drawer" else 36):
            if not self.plan_success:break
            if self.component=="drawer":
                self.move(self.move_by_displacement(arm,y=-.02))
            else:
                # Move the gripper along the handle's hinge arc; joint remains passive.
                m=self.handle_pose();hinge=self.child.get_pose().p
                theta=-.05;r=t3d.axangles.axangle2mat([0,0,1],theta)
                pos=hinge+r@(m[:3,3]-hinge);rot=r@m[:3,:3]
                target=(pos-rot[:,0]*.15).tolist()+t3d.quaternions.mat2quat(rot).tolist()
                self.move(self.move_to_pose(arm,target))
            self.mark("pull_"+str(i))
            if self.check_success():break
        for _ in range(100):self.scene.step()
        self.mark("settled")
        self.info["info"]={"{a}":"left","component":self.component}
        return self.info
    # The instruction asks for the panel to be fully open, so the expert is only
    # credited for the operator acceptance fraction of the fixture's declared
    # joint travel, measured on the passive joint instead of a hard-coded angle.
    required_joint_travel_fraction=.9
    def check_success(self):
        if not hasattr(self,"cabinet"):return False
        q=float(self.cabinet.get_qpos()[0]);v=float(self.cabinet.get_qvel()[0])
        lo,hi=self.cabinet.get_qlimits()[0]
        closed=lo if self.component=="drawer" else hi
        return bool(abs(q-closed)>=self.required_joint_travel_fraction*abs(hi-lo) and abs(v)<.1)
