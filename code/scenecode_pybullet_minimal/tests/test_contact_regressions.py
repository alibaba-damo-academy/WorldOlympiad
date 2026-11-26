from dataclasses import replace
from pathlib import Path
import pytest
from scenecode_pybullet_bridge.articulated import (
    ArticulationBinding, ArticulatedSceneInputs, configure_contact_dynamics,
)

def binding():
    return ArticulationBinding(
        case_id="test", component_id="door", host_object_id="host", host_sdf="fixture.sdf",
        host_sdf_sha256="a"*64, joint_name="hinge", joint_type="revolute",
        parent_link="base", child_link="door", axis_xyz=(0,0,1), lower_limit=-1.2,
        upper_limit=0, closed_position=0, open_position=-1.2, travel=1.2,
        open_delta_xyz=(.3,.4,0), contact_target_local_xyz=(-.48,.055,-.25),
        contact_normal_local_xyz=(0,1,0), handle_axis_local_xyz=(0,0,1),
    )

def test_multibody_contact_friction_is_applied():
    bullet = pytest.importorskip("pybullet")
    from pybullet_utils.bullet_client import BulletClient
    p = BulletClient(connection_mode=bullet.DIRECT)
    try:
        shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=(.1,.1,.1))
        body=p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=shape,
            linkMasses=[1], linkCollisionShapeIndices=[shape],
            linkVisualShapeIndices=[-1], linkPositions=[(0,0,.3)],
            linkOrientations=[(0,0,0,1)], linkInertialFramePositions=[(0,0,0)],
            linkInertialFrameOrientations=[(0,0,0,1)], linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE], linkJointAxis=[(0,0,1)],
        )
        for friction in (.9,4.0):
            configure_contact_dynamics(p,body,0,friction)
            for _ in range(3):p.stepSimulation()
            assert p.getDynamicsInfo(body,0)[1] == pytest.approx(friction)
    finally:p.disconnect()

def test_partial_or_degenerate_handle_frame_is_rejected():
    with pytest.raises(ValueError,match="requires target"):
        replace(binding(),contact_normal_local_xyz=None).validate()
    with pytest.raises(ValueError,match="parallel"):
        replace(binding(),handle_axis_local_xyz=(0,1,0)).validate()
    with pytest.raises(ValueError,match="three-vector"):
        replace(binding(),contact_target_local_xyz=(1,2)).validate()

@pytest.mark.parametrize("override", [
    {"contact_pull_force":45}, {"reference_joint_actuation":True},
    {"gripper_geometry_scale":.5}, {"handle_collision_scale":2},
])
def test_contact_frame_cannot_enable_assisted_or_scaled_execution(override):
    with pytest.raises(ValueError,match="unassisted native geometry"):
        ArticulatedSceneInputs(table_sdf=None,host_sdf=Path("a"),panda_urdf=Path("b"),
            output_dir=Path("c"),binding=binding(),**override).validate()

@pytest.mark.parametrize("approach,contact",[(.01,.02),(.1,-.01)])
def test_invalid_clearances_fail(approach,contact):
    with pytest.raises(ValueError,match="clearances"):
        ArticulatedSceneInputs(table_sdf=None,host_sdf=Path("a"),panda_urdf=Path("b"),
            output_dir=Path("c"),binding=binding(),approach_clearance_m=approach,
            contact_clearance_m=contact).validate()
