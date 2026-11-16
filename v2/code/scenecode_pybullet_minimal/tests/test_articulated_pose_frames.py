import numpy as np
import pytest
from scenecode_pybullet_bridge.articulated import PyBulletDroidArticulatedScene

def test_robot_root_export_does_not_include_inertial_offset_twice(tmp_path):
    bullet=pytest.importorskip("pybullet")
    from pybullet_utils.bullet_client import BulletClient
    p=BulletClient(connection_mode=bullet.DIRECT)
    try:
        path=tmp_path/"robot.urdf"
        path.write_text("""<robot name="fixture"><link name="base"><inertial>
        <origin xyz=".02 -.03 .05" rpy=".1 .2 .3"/><mass value="1"/>
        <inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/>
        </inertial></link></robot>""")
        position=(.6,.8,.35);orientation=p.getQuaternionFromEuler((.1,-.2,-1.57))
        robot=p.loadURDF(str(path),basePosition=position,baseOrientation=orientation,useFixedBase=True)
        scene=object.__new__(PyBulletDroidArticulatedScene);scene.p=p;scene.panda=robot
        xyz,quat=scene._robot_base_link_pose()
        assert not np.allclose(p.getBasePositionAndOrientation(robot)[0],position,atol=1e-3)
        np.testing.assert_allclose(xyz,position,atol=1e-6)
        assert abs(float(np.dot(quat,orientation)))==pytest.approx(1,abs=1e-6)
    finally:p.disconnect()
