import numpy as np
import pytest
from autoresearch.lingbot_world_physics import camera_inputs,native_dimensions,complete_chunk_frame_count

def calibration():
    a=.37;r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    c=np.eye(4);c[:3,:3]=r;c[:3,3]=[17.3,-2.1,8.2]
    return {"width":960,"height":720,"intrinsics":[[772,0,480],[0,772,360],[0,0,1]],"camera_to_world_opencv":c.tolist()}

def test_original_pixel_rays_preserved_after_upstream_K_transform():
    c=calibration();poses,ks,out=camera_inputs(c,[960,720],77)
    assert native_dimensions(960,720)==(720,544)
    np.testing.assert_allclose(out,[772*720/960,772*544/720,360,272],rtol=1e-6)
    xyz=np.array([[.2,.1,1.],[1,-.3,4.],[-.5,.2,3.]])
    source=xyz[:,:2]/xyz[:,2:]*[772,772]+[480,360]
    target=xyz[:,:2]/xyz[:,2:]*out[:2]+out[2:]
    np.testing.assert_allclose(target,source*[720/960,544/720],atol=1e-4)
    np.testing.assert_array_equal(poses,np.repeat(np.eye(4)[None],77,axis=0))
    assert ks.shape==(77,4)

def test_complete_chunk_budget_covers_reference_without_shortening():
    for duration in [.5,1,4,10]:
        n=complete_chunk_frame_count(duration)
        assert (n-1)%4==0
        assert ((n-1)//4+1)%4==0
        assert (n-1)/16>=duration
        assert (n-17)/16<duration
    assert complete_chunk_frame_count(4)==77
    with pytest.raises(ValueError):complete_chunk_frame_count(float("nan"))

@pytest.mark.parametrize("defect",["dimensions","skew","rotation","homogeneous","frames"])
def test_invalid_input_contract_rejected(defect):
    c=calibration();n=77;wh=[960,720]
    if defect=="dimensions":wh=[832,480]
    if defect=="skew":c["intrinsics"][0][1]=12
    if defect=="rotation":c["camera_to_world_opencv"][0][0]=2
    if defect=="homogeneous":c["camera_to_world_opencv"][3][0]=1
    if defect=="frames":n=81
    with pytest.raises(ValueError):camera_inputs(c,wh,n)


def test_direct_render_uses_stable_native_dimensions():
    assert native_dimensions(832,464)==(832,464)
    # Match the released float/floor calculation, including its nominal-480P edge.
    assert native_dimensions(832,480)==(832,464)

def test_source_requiring_image_resize_is_rejected_before_generation(tmp_path):
    import json
    from PIL import Image
    from autoresearch.lingbot_world_physics import prepare_case
    src=tmp_path/"source";ref=src/"reference";ref.mkdir(parents=True)
    (src/"simulation.json").write_text(json.dumps({"duration_s":4.0}))
    (ref/"calibration.json").write_text(json.dumps(calibration()))
    (ref/"scene_data.json").write_text("{}")
    Image.new("RGB",(960,720)).save(ref/"camera_rgb.png")
    with pytest.raises(ValueError,match="resizing is forbidden"):
        prepare_case(src,tmp_path/"inputs")
    assert not (tmp_path/"inputs").exists()
