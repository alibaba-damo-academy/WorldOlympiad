import numpy as np
import pytest
from autoresearch.robotics_camera_fit import fit_context_camera

def test_context_fit_contains_grippers_and_target_without_lens_change():
    k=np.array([[210,0,160],[0,210,128],[0,0,1.]])
    points=np.array([[0,.1,1.2],[-.3,-.45,1.2],[.3,-.45,1.2]])
    mat,r=fit_context_camera([0,-.6,1.6],points,k,[320,256])
    np.testing.assert_allclose(mat[:3,:3].T@mat[:3,:3],np.eye(3),atol=1e-12)
    assert np.linalg.det(mat[:3,:3])==pytest.approx(1.)
    assert r["backward_translation_m"]>0
    assert min(r["support_bbox_uv"][:2])>=.08-1e-12
    assert max(r["support_bbox_uv"][2:])<=.92+1e-12
    assert not r["lens_changed"] and not r["future_reference_state_used"]

def test_shared_world_translation_only_translates_camera():
    eye=np.array([0,-.8,1.5]);p=np.array([[0,0,1.],[-.3,-.5,1.],[.3,-.5,1.]])
    k=np.array([[210,0,160],[0,210,128],[0,0,1.]])
    a,ra=fit_context_camera(eye,p,k,[320,256]);b,rb=fit_context_camera(eye+7,p+7,k,[320,256])
    np.testing.assert_allclose(a[:3,:3],b[:3,:3],atol=1e-12)
    np.testing.assert_allclose(a[:3,3]+7,b[:3,3],atol=1e-12)
    assert ra["backward_translation_m"]==pytest.approx(rb["backward_translation_m"])
