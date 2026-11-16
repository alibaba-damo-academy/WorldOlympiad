"""Render directly at LingBot size while preserving upstream D435 resize rays."""
import math


def upstream_resize_intrinsics(source_width,source_height,target_width,target_height,fovy_degrees):
    if min(source_width,source_height,target_width,target_height)<=0 or not 0<fovy_degrees<180:
        raise ValueError('invalid camera dimensions or field of view')
    focal=source_height/(2*math.tan(math.radians(fovy_degrees)/2))
    sx=target_width/source_width;sy=target_height/source_height
    # PyTorch bilinear align_corners=False maps centers using (x+.5)*s-.5.
    return {'fx':focal*sx,'fy':focal*sy,
            'cx':(source_width/2+.5)*sx-.5,'cy':(source_height/2+.5)*sy-.5,'skew':0.}


def apply_upstream_camera_intrinsics(camera,config,near,far):
    source=config.get('upstream_resize_source_wh')
    if source is None:return
    k=upstream_resize_intrinsics(source[0],source[1],config['w'],config['h'],config['fovy'])
    camera.set_perspective_parameters(near,far,k['fx'],k['fy'],k['cx'],k['cy'],k['skew'])
