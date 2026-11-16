import numpy as np
from scipy.spatial.transform import Rotation
from autoresearch.native_joint_reference import bounded_ik_step,pose_matrix,jacobian_at_link_origin,contact_balance_step


def test_contact_pose_step_is_bounded_and_uses_joint_limits():
    current=pose_matrix([0,0,0],[0,0,0,1]);desired=pose_matrix([.1,0,0],Rotation.from_euler('z',.1).as_quat())
    jacobian=np.c_[np.eye(6),np.zeros(6)];limits=np.array([[-.01,.01]]*7)
    q=bounded_ik_step(np.zeros(7),jacobian,current,desired,damping=.05,max_delta=.025,limits=limits)
    assert np.max(np.abs(q))<=.01
    assert q[0]>0 and q[5]>0


def test_com_jacobian_shift_predicts_velocity_at_controlled_link_origin():
    # A rigid link rotates about its COM; its origin is 10 cm along +x.
    jacobian=np.zeros((6,1));jacobian[5,0]=1
    shifted=jacobian_at_link_origin(jacobian,[.1,0,0],[0,0,0])
    np.testing.assert_allclose(shifted[:,0],[0,.1,0,0,0,1])
    np.testing.assert_allclose(jacobian[:,0],[0,0,0,0,0,1])
    # The actual pose derivative gives the same result.
    epsilon=1e-6
    moved=Rotation.from_rotvec([0,0,epsilon]).apply([.1,0,0])
    np.testing.assert_allclose((moved-np.array([.1,0,0]))/epsilon,shifted[:3,0],atol=1e-7)


def test_force_centering_moves_toward_stronger_jaw_without_unbounded_offset():
    # Upper jaw has weak contact; moving the gripper downward seats that jaw
    # and reduces lower-jaw penetration. Equal contacts cause no translation.
    result=contact_balance_step([[0,0,1],[0,0,-5]],[0,0,0],gain=.00015,limit=.006)
    assert result[2]<0
    np.testing.assert_allclose(result[:2],[0,0])
    np.testing.assert_array_equal(contact_balance_step([[0,0,2],[0,0,-2]],result,gain=.00015,limit=.006),result)
    result=contact_balance_step([[0,0,0],[0,0,-5]],[0,0,-.006],gain=.00015,limit=.006)
    assert np.linalg.norm(result)<=.006
