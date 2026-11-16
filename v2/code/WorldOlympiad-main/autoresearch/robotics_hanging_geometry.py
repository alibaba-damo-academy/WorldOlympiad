"""Measure mug apertures and hook geometry from the declared source meshes."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def visual_mesh(path: Path):
    root = ET.parse(path).getroot()
    visual = root.find(".//visual")
    uri = Path(visual.findtext("geometry/mesh/uri"))
    uri = uri if uri.is_absolute() else path.parent / uri
    mesh = trimesh.load(uri, force="scene").to_geometry()
    if uri.suffix in {".gltf", ".glb"}:
        mesh.apply_transform([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
    mesh.apply_scale(np.fromstring(visual.findtext("geometry/mesh/scale", "1 1 1"), sep=" "))
    pose = np.fromstring(visual.findtext("pose", "0 0 0 0 0 0"), sep=" ")
    transform = np.eye(4);transform[:3,:3] = Rotation.from_euler("xyz",pose[3:]).as_matrix()
    transform[:3,3] = pose[:3];mesh.apply_transform(transform)
    mesh.merge_vertices()
    return mesh


def measure_assets(mug: Path, rack: Path) -> dict:
    """Extract a vertical ring and a horizontal hook from the actual meshes."""
    return measure_meshes(visual_mesh(mug), visual_mesh(rack))


def measure_meshes(cup, hook) -> dict:
    """Measure already transformed meshes in each body's local frame."""
    low, high = cup.bounds
    # Mesh exporters may split a torus along UV/normal seams. Measure the
    # complete protruding ring instead of selecting one disconnected fragment.
    top = cup.vertices[cup.vertices[:,2] > high[2] - .02*(high[2]-low[2])]
    body_center=(top.min(axis=0)+top.max(axis=0))/2
    body_radius=(high[1]-low[1])/2
    handle_side=-1. if body_center[0]-low[0] > high[0]-body_center[0] else 1.
    wall=body_center[0]+handle_side*body_radius
    protruding=cup.vertices[(cup.vertices[:,0]-wall)*handle_side>1e-5]
    if len(protruding)<12:raise ValueError("no measurable side handle outside the mug body")
    a=protruding.min(axis=0);b=protruding.max(axis=0)
    outer_radius=(b[2]-a[2])/2
    centre=np.array([a[0]+outer_radius if handle_side<0 else b[0]-outer_radius,
                     (a[1]+b[1])/2,(a[2]+b[2])/2])
    far=protruding[(protruding[:,0]-centre[0])*handle_side>0]
    radius=float(np.linalg.norm((far-centre)[:,[0,2]],axis=1).min())
    if radius<=1e-4 or radius>=outer_radius:raise ValueError("ring aperture cannot be measured")
    ring_centre=centre.copy()
    a[0]=centre[0]-outer_radius;b[0]=centre[0]+outer_radius
    left=centre[0]-radius;right=centre[0]+radius
    if handle_side<0:right=min(right,wall)
    else:left=max(left,wall)
    hole_x=(left+right)/2
    usable_radius=min((right-left)/2,radius-abs(hole_x-centre[0]))
    centre[0]=hole_x
    rl,rh=hook.bounds; verts=hook.vertices
    near_base=verts[verts[:,2]<rl[2]+.1*(rh[2]-rl[2])]
    post_x=float((near_base[:,0].min()+near_base[:,0].max())/2)
    top_band=verts[verts[:,2]>rl[2]+.35*(rh[2]-rl[2])]
    # Extension direction is measured relative to the freestanding base.
    left=post_x-top_band[:,0].min();right=top_band[:,0].max()-post_x
    direction=-1. if left>right else 1.
    tip=float(top_band[:,0].min() if direction<0 else top_band[:,0].max())
    length=abs(tip-post_x)
    tip_points=top_band[np.abs(top_band[:,0]-tip)<max(.15*length,1e-5)]
    tip_top=float(tip_points[:,2].max())
    section=hook.section(plane_origin=((tip+post_x)/2,0,0),plane_normal=(1,0,0))
    if section is None:raise ValueError("horizontal hook cross-section is absent")
    middle=np.asarray(section.vertices)
    middle=middle[middle[:,2]>rl[2]+.35*(rh[2]-rl[2])]
    if len(middle)<3: raise ValueError("horizontal hook segment cannot be measured")
    bar_z=float((middle[:,2].min()+middle[:,2].max())/2)
    bar_y=float((middle[:,1].min()+middle[:,1].max())/2)
    rod_radius=float((middle[:,1].max()-middle[:,1].min())/2)
    return {"mug_bounds":cup.bounds.tolist(),"body_center":body_center.tolist(),
            "body_radius":float(body_radius),"handle_center":centre.tolist(),
            "ring_center":ring_centre.tolist(),"body_wall_x":float(wall),"handle_side":handle_side,
            "handle_bounds":[a.tolist(),b.tolist()],"handle_inner_radius":float(radius),
            "usable_handle_radius":float(usable_radius),
            "rack_bounds":hook.bounds.tolist(),"post_x":post_x,
            "hook_tip_x":tip,"hook_tip_z":tip_top,"hook_bar_z":bar_z,
            "hook_bar_y":bar_y,"hook_radius":rod_radius,"hook_direction":direction,
            "hook_length":length}
