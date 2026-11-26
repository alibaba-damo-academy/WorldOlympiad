"""Fit a fixed-lens context camera to initial manipulation geometry."""
import itertools
import numpy as np

def fit_context_camera(eye, points, intrinsics, image_size_wh, *, padding_m=.10, margin=.08):
    """Preserve lens; choose the smallest backward translation of a look-at camera.

    The support points are task affordance and measured initial gripper centers.
    Padding is a declared input-context policy, never an evaluation score parameter.
    All geometry must come from the initial state, not future reference motion.
    """
    eye=np.asarray(eye,dtype=float);points=np.asarray(points,dtype=float);k=np.asarray(intrinsics,dtype=float)
    w,h=image_size_wh
    if eye.shape!=(3,) or points.ndim!=2 or points.shape[1]!=3 or not len(points):
        raise ValueError("invalid camera fit geometry")
    if not np.isfinite(eye).all() or not np.isfinite(points).all() or not np.isfinite(k).all():
        raise ValueError("nonfinite camera fit input")
    if padding_m<0 or not 0<margin<.5 or w<=0 or h<=0:raise ValueError("invalid camera fit bounds")
    center=points.mean(axis=0);forward=center-eye
    if np.linalg.norm(forward)<1e-8:raise ValueError("camera coincides with context center")
    forward/=np.linalg.norm(forward);left=np.cross([0.,0.,1.],forward)
    if np.linalg.norm(left)<1e-8:raise ValueError("vertical camera fit needs an explicit up axis")
    left/=np.linalg.norm(left);up=np.cross(forward,left)
    offsets=np.asarray(list(itertools.product([-padding_m,padding_m],repeat=3)))
    support=(points[:,None,:]+offsets[None,:,:]).reshape(-1,3)
    def project(distance):
        relative=support-(eye-distance*forward);depth=relative@forward
        if np.any(depth<=.02):return False,None
        uv=np.column_stack([(k[0,2]-k[0,0]*(relative@left)/depth)/w,
                            (k[1,2]-k[1,1]*(relative@up)/depth)/h])
        return bool(np.all((uv>=margin)&(uv<=1-margin))),uv
    lower=0.;upper=0.;ok,uv=project(0.)
    if not ok:
        upper=.1
        while upper<=4. and not project(upper)[0]:upper*=2
        if upper>4.:raise ValueError("no bounded context camera fits the initial geometry")
        for _ in range(48):
            mid=(lower+upper)/2
            if project(mid)[0]:upper=mid
            else:lower=mid
        _,uv=project(upper)
    matrix=np.eye(4);matrix[:3,:3]=np.column_stack([forward,left,up]);matrix[:3,3]=eye-upper*forward
    return matrix,{"rule":"initial_handle_and_dual_gripper_context_fit_v1",
        "native_origin":eye.tolist(),"position_world":matrix[:3,3].tolist(),
        "backward_translation_m":float(upper),"initial_support_centers":points.tolist(),
        "padding_m":padding_m,"frame_margin":margin,
        "support_bbox_uv":[*uv.min(0).tolist(),*uv.max(0).tolist()],
        "lens_changed":False,"future_reference_state_used":False}
