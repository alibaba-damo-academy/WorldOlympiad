import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autoresearch.articulated_native_task import JointTask, JointTermination

def task(kind="prismatic"):
    r=Rotation.from_euler("xyz",[.2,.4,-.7])
    return JointTask(kind,tuple(r.apply([0,0,1])),(.2,.3,.4),tuple(r.as_quat()),
                     (.1,.1,.2),0,.32 if kind=="prismatic" else -1.2,
                     0 if kind=="prismatic" else -1.2,.32 if kind=="prismatic" else 0,
                     .02,.02,3,12)

@pytest.mark.parametrize("kind",["prismatic","revolute"])
def test_measures_nontrivial_world_axis_and_signed_progress(kind):
    t=task(kind)
    for coordinate in [0,t.target_position/2,t.target_position]:
        xyz,quat=t.pose_at(coordinate)
        m=t.measure(xyz,quat,[0,0,0],[0,0,0])
        assert m["position"] == pytest.approx(coordinate)
        assert m["translation_residual_m"] < 1e-10
        assert m["rotation_residual_rad"] < 1e-10

def test_rejects_link_leaving_joint_axis():
    t=task()
    xyz,quat=t.pose_at(.32)
    axis=np.array(t.axis_world)
    off=np.cross(axis,[1,0,0]);off=off/np.linalg.norm(off)*.02
    m=t.measure(np.array(xyz)+off,quat,[0,0,0],[0,0,0])
    with pytest.raises(RuntimeError,match="constraint"):
        JointTermination(t).step(m,contacts=False,gripper=0,step=1)

def test_requires_release_and_consecutive_stability_then_independent_timeout():
    t=task(); checker=JointTermination(t)
    xyz,quat=t.pose_at(.32); m=t.measure(xyz,quat,[0,0,0],[0,0,0])
    assert not checker.step(m,contacts=True,gripper=0,step=1)["done"]
    assert not checker.step(m,contacts=False,gripper=1,step=2)["done"]
    assert not checker.step(m,contacts=False,gripper=0,step=3)["done"]
    checker.step(m,contacts=True,gripper=0,step=4)
    for step in (5,6): assert not checker.step(m,contacts=False,gripper=0,step=step)["done"]
    assert checker.step(m,contacts=False,gripper=0,step=7)["task_success"]
    xyz,quat=t.pose_at(0);closed=t.measure(xyz,quat,[0,0,0],[0,0,0])
    end=JointTermination(t).step(closed,contacts=False,gripper=0,step=12)
    assert end["reason"]=="project_task_timeout" and not end["task_success"]


def test_shared_window_records_success_but_still_measures_final_state():
    t=task();checker=JointTermination(t,evaluation_max_steps=12)
    xyz,quat=t.pose_at(t.target_position);opened=t.measure(xyz,quat,[0,0,0],[0,0,0])
    xyz,quat=t.pose_at(0);closed=t.measure(xyz,quat,[0,0,0],[0,0,0])
    for step in range(1,7):
        result=checker.step(opened,contacts=False,gripper=0,step=step)
        assert not result['done']
    assert result['task_success'] and result['completion_frame']==3
    for step in range(7,13):
        result=checker.step(closed,contacts=False,gripper=0,step=step)
    assert result['done'] and not result['task_success']
    assert result['completion_frame']==3 and result['reason']=='evaluation_window_complete'
