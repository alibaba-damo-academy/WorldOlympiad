"""Select an observable, closed and contact-free initial pose from a verified native reference."""
from __future__ import annotations
import itertools
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
import trimesh

from .articulated_link_frame_rebase import _relative
from .math3d import compose, pose_from_sdf
from .models import Pose

def rotation_wxyz(q):
    return Rotation.from_quat([*q[1:], q[0]])

def wrist_mount(calibration):
    ee_r = rotation_wxyz(calibration["ee_quaternion_wxyz"])
    camera = calibration["cameras"]["wrist"]
    position = ee_r.inv().apply(np.asarray(camera["position_world"]) - calibration["ee_position"])
    rotation = ee_r.inv() * rotation_wxyz(camera["quaternion_opengl_wxyz"])
    return position, rotation

def target_corners(source_sdf, binding):
    source_sdf = Path(source_sdf).resolve()
    root = ET.parse(source_sdf).getroot()
    links = {x.attrib["name"]: x for x in root.findall(".//link")}
    frames = {x.attrib["name"]: x for x in root.findall(".//frame")}
    child = links[binding["child_link"]]
    handle_name = (binding.get("contact_frame_evidence") or {}).get("handle_link")
    link = links[handle_name] if handle_name else child
    resolved = {"__model__": Pose()}
    visiting = set()
    def model_pose(name):
        if name in resolved:
            return resolved[name]
        if name in visiting or name not in {**links, **frames}:
            raise ValueError("articulated handle has cyclic or missing SDF frame")
        visiting.add(name)
        element = links[name] if name in links else frames[name]
        pose_element = element.find("pose")
        default_parent = element.get("attached_to", "__model__") if name in frames else "__model__"
        parent = pose_element.get("relative_to", default_parent) if pose_element is not None else default_parent
        result = compose(model_pose(parent), pose_from_sdf(
            pose_element.text if pose_element is not None else None))
        visiting.remove(name)
        resolved[name] = result
        return result
    child_pose = model_pose(binding["child_link"])
    link_pose = model_pose(link.attrib["name"])
    link_to_child = _relative(child_pose, link_pose)
    target = np.asarray(binding["contact_target_local_xyz"])
    candidates = []
    for collision in link.findall("collision"):
        pose = compose(link_to_child, pose_from_sdf(collision.findtext("pose")))
        geometry = collision.find("geometry")
        if geometry.find("box") is not None:
            half = np.fromstring(geometry.findtext("box/size"), sep=" ") / 2
            corners = np.array(list(itertools.product([-1, 1], repeat=3))) * half
        elif geometry.find("cylinder") is not None:
            radius = float(geometry.findtext("cylinder/radius"))
            half = np.array([radius, radius, float(geometry.findtext("cylinder/length")) / 2])
            corners = np.array(list(itertools.product([-1, 1], repeat=3))) * half
        elif geometry.find("mesh") is not None:
            uri = geometry.findtext("mesh/uri")
            mesh_path = (source_sdf.parent / uri).resolve()
            mesh_path.relative_to(source_sdf.parent)
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            evidence = (binding.get("contact_frame_evidence") or {}).get(
                "input_file_sha256") or {}
            expected = evidence.get(str(mesh_path))
            if expected is None and handle_name:
                relative = mesh_path.relative_to(source_sdf.parent).parts
                matches = {digest for declared, digest in evidence.items()
                           if Path(declared).parts[-len(relative):] == relative}
                if len(matches) == 1:
                    expected = matches.pop()
            if handle_name and expected != hashlib.sha256(mesh_path.read_bytes()).hexdigest():
                raise ValueError("certified handle collision mesh changed")
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=float)
            scale = np.fromstring(geometry.findtext("mesh/scale", "1 1 1"), sep=" ")
            if vertices.ndim != 2 or vertices.shape[1] != 3 or scale.shape != (3,):
                raise ValueError("handle collision mesh has invalid geometry")
            scaled = vertices * scale
            low, high = scaled.min(axis=0), scaled.max(axis=0)
            corners = np.asarray(list(itertools.product(*zip(low, high))))
        else:
            continue
        corners = Rotation.from_quat(pose.quat_xyzw).apply(corners) + pose.xyz
        center = corners.mean(axis=0)
        candidates.append((float(np.linalg.norm(center - target)), corners))
    if not candidates:
        raise ValueError("The selected handle needs certified collision geometry")
    return min(candidates, key=lambda x: x[0])[1]

def project(frame, binding, corners, mount, intrinsics, image_shape):
    robot = frame["robot"]
    ee_r = rotation_wxyz(robot["ee_quaternion_wxyz"])
    eye = np.asarray(robot["ee_position"]) + ee_r.apply(mount[0])
    camera_r = ee_r * mount[1]
    child_r = Rotation.from_quat(frame["child_link_xyzw"])
    local = np.concatenate([np.asarray(binding["contact_target_local_xyz"])[None], corners])
    world = child_r.apply(local) + frame["child_link_position"]
    camera = camera_r.inv().apply(world - eye)
    depth = -camera[:, 2]
    if np.any(depth <= 0.02):
        return None
    k = np.asarray(intrinsics)
    uv = np.stack([k[0, 0] * camera[:, 0] / depth + k[0, 2],
                   k[1, 2] - k[1, 1] * camera[:, 1] / depth], axis=-1)
    h, w = image_shape
    uv = uv / [w, h]
    lo, hi = uv[1:].min(0), uv[1:].max(0)
    area = np.prod(np.maximum(hi - lo, 0))
    visible = np.prod(np.maximum(np.minimum(hi, 1) - np.maximum(lo, 0), 0)) / max(area, 1e-12)
    return {
        "center_uv": uv[0].tolist(), "handle_bbox_uv": [*lo.tolist(), *hi.tolist()],
        "bbox_visible_fraction": float(visible), "depth_m": float(depth[0]),
        "ee_to_handle_m": float(np.linalg.norm(world[0] - robot["ee_position"])),
    }

def select_initial_view(reference, binding, calibration, source_sdf):
    corners = target_corners(source_sdf, binding)
    mount = wrist_mount(calibration)
    camera = calibration["cameras"]["wrist"]
    candidates = []
    for index, frame in enumerate(reference["frames"]):
        if (abs(frame["articulated"]["position"] - binding["closed_position"]) > 0.002
                or frame["robot"]["gripper"] > 0.05 or any(frame["finger_contacts"])):
            break
        if not frame.get("physical_valid", False):
            continue
        view = project(frame, binding, corners, mount, camera["intrinsics"], camera["image_shape"])
        if view is None:
            continue
        u, v = view["center_uv"]
        if not (0.1 <= u <= 0.9 and 0.15 <= v <= 0.6
                and view["bbox_visible_fraction"] >= 0.45 and view["ee_to_handle_m"] >= 0.22):
            continue
        score = float(np.linalg.norm(np.array([u, v]) - [.5, .4])
                      + .3 * (1 - view["bbox_visible_fraction"]))
        candidates.append({"reference_index": index, "reference_frame": frame["frame"],
                           "score": score, **view})
    if not candidates:
        raise ValueError("No observable closed, open-gripper, contact-free start exists in the reference prefix")
    selected = min(candidates, key=lambda x: (x["score"], x["reference_index"]))
    return selected, candidates
