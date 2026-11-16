"""Direct native-size SceneCode/PyBullet replay with actual camera calibration audit."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

path=Path(__file__).with_name("blender_pybullet_replay.py")
spec=importlib.util.spec_from_file_location("canonical_blender_replay",path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module._setup_camera

def setup_camera(calibration,width,height):
    bpy.context.scene.render.resolution_x=width
    bpy.context.scene.render.resolution_y=height
    bpy.context.scene.render.resolution_percentage=100
    camera=original(calibration,width,height)
    bpy.context.view_layer.update()
    transform=np.asarray(calibration["camera_to_world_opencv"])
    k=np.asarray(calibration["intrinsics"])
    probes=np.array([[0,0,1],[.1,0,1],[0,.1,1],[-.2,.15,2]])
    actual=[];expected=[]
    for p in probes:
        world=transform@np.r_[p,1]
        projected=world_to_camera_view(bpy.context.scene,camera,Vector(world[:3]))
        actual.append([projected.x*width,(1-projected.y)*height])
        v=k@p;expected.append((v[:2]/v[2]).tolist())
    error=float(np.abs(np.asarray(actual)-expected).max())
    receipt={"width":width,"height":height,"pixel_projection_max_abs_error":error,
        "intrinsics":calibration["intrinsics"],"probes_actual":actual,"probes_expected":expected,
        "direct_render":True,"rgb_resized":False,"renderer_source":str(path)}
    target=Path(calibration["camera_audit_path"])
    target.write_text(json.dumps(receipt,indent=2)+"\n")
    if error>1e-3:raise ValueError("Blender camera does not match declared native intrinsics")
    return camera

module._setup_camera=setup_camera
module.main()
