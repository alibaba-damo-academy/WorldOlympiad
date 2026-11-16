"""Prepare a replay render plan from a wo reference (or model) trajectory.

The plan lists every visual mesh once (with its fixed transform inside a node)
and, per frame, the world transform of every node and every camera.  A single
Blender process (wo/blender_replay.py) then imports the meshes once and only
moves nodes and cameras, instead of re-exporting the whole scene per frame.

Cameras reproduce the registered RoboLab DROID rig used by the native model:
exterior_left/right (robot-root OpenGL poses), the robot-frame top look-at, and
the wrist camera on the Robotiq base link.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .sim.sdf_urdf import sdf_visuals

# RoboLab DROID rig (autoresearch/robotics_native_cameras.py, render_scenecode_robotics_multiview.py)
EXTERIOR = {
    "exterior_left": ((0.05, 0.57, 0.66), (-0.393, -0.195, 0.399, 0.805)),
    "exterior_right": ((0.05, -0.57, 0.66), (0.805, 0.399, -0.195, -0.393)),
}  # position, OpenGL quaternion wxyz, in the robot root frame
TOP_EYE, TOP_TARGET = (1.16, -0.72, 1.12), (0.44, 0.0, 0.17)
WRIST_IN_BASE_LINK_DRAKE = (
    np.array([[0.9997098581, 0.0214786747, 0.0109025789],
              [-0.0238843654, 0.9425649207, 0.3331679867],
              [-0.0031203816, -0.3333317219, 0.9428044476]]),
    np.array([-0.031, -0.074, 0.011]),
)
VIEWS = {  # width, height, vertical fov
    "exterior_left": (320, 180, 71.5077745089),
    "exterior_right": (320, 180, 71.5077745089),
    "top": (640, 360, 71.5077745089),
    "wrist": (640, 360, 56.74),
}
DRAKE_TO_GL = np.diag([1.0, -1.0, -1.0])


def _T(p, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _quat_xyzw(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return _quat_xyzw((x / n, y / n, z / n, w / n))


def _rpy(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _look_at_gl(eye, target) -> np.ndarray:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    back = eye - target
    back /= np.linalg.norm(back)  # camera +z points backwards (OpenGL / Blender)
    right = np.cross([0, 0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return _T(eye, np.stack([right, up, back], axis=1))



def _fit_showcase_camera(points, orientation, view=(1280, 720, 55.0)):
    """One fixed presentation camera covering the complete recorded task motion."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("showcase requires finite recorded task bounds")
    target = (points.min(axis=0) + points.max(axis=0)) / 2
    local = (points - target) @ orientation
    width, height, fov = view
    tan_v = math.tan(math.radians(fov) / 2)
    distance = max(float(np.max(local[:, 2] + np.maximum(
        np.abs(local[:, 0]) / (tan_v * width / height), np.abs(local[:, 1]) / tan_v))), 0.3)
    return _T(target + orientation[:, 2] * (distance * 1.2 + 0.06), orientation)


def _showcase_points(scene, meshes, poses, frames, articulated_hosts=()):
    import itertools
    import trimesh

    points, bounds = [], {}
    active = set(scene["objects"]) | set(articulated_hosts)
    for mesh in meshes:
        if mesh["node"].split("/", 1)[0] not in active:
            continue
        path = mesh["file"]
        if path not in bounds:
            box = trimesh.load(path, force="scene").bounds
            if box is None:
                raise ValueError(f"no visual bounds for {path}")
            if Path(path).suffix.lower() in (".gltf", ".glb"):
                box = box[:, [0, 2, 1]] * [1, -1, 1]
                box = np.array([box.min(axis=0), box.max(axis=0)])
            bounds[path] = np.array([(*xyz, 1.0) for xyz in itertools.product(*box.T)])
        corners = bounds[path].copy()
        corners[:, :3] *= np.asarray(mesh.get("scale", [1, 1, 1]))
        visual = np.asarray(mesh["link_T_visual"])
        for frame in poses:
            transform = np.asarray(frame[mesh["node"]]).reshape(4, 4) @ visual
            points.extend((corners @ transform.T)[:, :3])
    for frame in frames:
        ee = np.asarray(frame["robot"]["end_effector_position"])
        points.extend([ee - 0.06, ee + 0.06])
    return points


def _urdf_visuals(urdf: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):
        vis = []
        for v in link.findall("visual"):
            mesh = v.find("geometry/mesh")
            if mesh is None:
                continue
            org = v.find("origin")
            xyz = [float(x) for x in (org.get("xyz", "0 0 0") if org is not None else "0 0 0").split()]
            rpy = [float(x) for x in (org.get("rpy", "0 0 0") if org is not None else "0 0 0").split()]
            fn = mesh.get("filename")
            path = Path(fn) if Path(fn).is_absolute() else (urdf.parent / fn).resolve()
            scale = [float(x) for x in mesh.get("scale", "1 1 1").split()]
            vis.append({"file": str(path), "scale": scale, "link_T_visual": _T(xyz, _rpy(*rpy)).tolist()})
        out[link.get("name")] = vis
    return out


class _FK:
    """PyBullet forward kinematics for a URDF body (link frames in world)."""

    def __init__(self, urdf: Path, fixed: bool) -> None:
        import pybullet as p

        self.p = p
        self.cid = p.connect(p.DIRECT)
        self.body = p.loadURDF(str(urdf), useFixedBase=fixed, physicsClientId=self.cid)
        self.joint = {}
        self.link_index = {}
        for j in range(p.getNumJoints(self.body, physicsClientId=self.cid)):
            info = p.getJointInfo(self.body, j, physicsClientId=self.cid)
            self.joint[info[1].decode()] = j
            self.link_index[info[12].decode()] = j
        self.base_name = p.getBodyInfo(self.body, physicsClientId=self.cid)[0].decode()

    def pose(self, base_T: np.ndarray, joints: dict[str, float]) -> dict[str, np.ndarray]:
        p, c, b = self.p, self.cid, self.body
        ipos, iorn = p.getDynamicsInfo(b, -1, physicsClientId=c)[3:5]
        com = base_T @ _T(ipos, _quat_xyzw(iorn))
        q = _mat_quat(com[:3, :3])
        p.resetBasePositionAndOrientation(b, com[:3, 3].tolist(), q, physicsClientId=c)
        for name, v in joints.items():
            if name in self.joint:
                p.resetJointState(b, self.joint[name], float(v), physicsClientId=c)
        out = {self.base_name: base_T}
        for name, li in self.link_index.items():
            s = p.getLinkState(b, li, computeForwardKinematics=True, physicsClientId=c)
            out[name] = _T(s[4], _quat_xyzw(s[5]))
        return out


def _mat_quat(R: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat().tolist()


def _prepared_urdf(ref: Path, oid: str, obj: dict) -> Path:
    """Resolve the exact physics artifact, never an older glob match."""
    source = Path(obj["sdf_path"])
    directory = ref / "physics_sdf" / f"{oid}_s{obj.get('scale', 1.0):.3f}"
    binding, marker = directory / ".path", directory / ".ok"
    if binding.is_file() and marker.is_file() and marker.read_text() == str(source):
        path = Path(binding.read_text())
    else:
        path = directory / source.with_suffix(".urdf").name
    if path.suffix != ".urdf" or not path.is_file():
        raise FileNotFoundError(f"Current articulated physics URDF missing: {path}")
    return path


def build_plan(ref: Path, views=tuple(VIEWS), traj_name: str = "trajectory.json") -> dict:
    scene = json.loads((ref / "scene_inputs.json").read_text())
    cert = json.loads((ref / "certificate.json").read_text())
    traj = json.loads((ref / traj_name).read_text())
    frames = traj["frames"]
    view_specs = {**VIEWS, "showcase": (1280, 720, 55.0)}
    if not views or set(views) - set(view_specs):
        raise ValueError("unknown or empty render views")
    arts = cert.get("articulations") or {}
    meshes, nodes = [], []

    def add_node(name):
        nodes.append(name)
        return name

    # objects (support included) -----------------------------------------
    objs = {scene["support"]["object_id"]: scene["support"], **scene["objects"]}
    obj_fk, obj_vis = {}, {}
    for oid, o in objs.items():
        sdf = Path(o["sdf_path"])
        vis = sdf_visuals(sdf)
        obj_vis[oid] = vis
        for link, info in vis["links"].items():
            node = add_node(f"{oid}/{link}")
            for k, v in enumerate(info["visuals"]):
                meshes.append({"node": node, **v})
        if len(vis["links"]) > 1:
            urdf = _prepared_urdf(ref, oid, o)
            obj_fk[oid] = _FK(urdf, fixed=True)
    # robot ----------------------------------------------------------------
    robot_urdf = Path(scene["robot_urdf"])
    rvis = _urdf_visuals(robot_urdf)
    for link, vs in rvis.items():
        node = add_node(f"robot/{link}")
        for v in vs:
            meshes.append({"node": node, **v})
    rfk = _FK(robot_urdf, fixed=True)

    per_frame, cams = [], []
    for fr in frames:
        T = {}
        for oid, o in objs.items():
            vis = obj_vis[oid]
            if oid in fr.get("objects", {}):
                st = fr["objects"][oid]
                root_T = _T(st["link_position"], _quat_xyzw(st["link_orientation_xyzw"]))
            else:
                root_T = _T(o["position"], _quat_xyzw(o["orientation_xyzw"]))
            if oid in obj_fk:
                joints = {a["joint_name"]: fr.get("articulation_positions", {}).get(c, a["q_closed"])
                          for c, a in arts.items() if a["host"] == oid}
                urdf_T = obj_fk[oid].pose(root_T, joints)
                for link, info in vis["links"].items():
                    T[f"{oid}/{link}"] = urdf_T[link] @ np.asarray(info["urdf_T_link"])
            else:
                T[f"{oid}/{vis['root']}"] = root_T
        rb = fr["robot"]
        base_T = _T(rb["base_position"], _quat_xyzw(rb["base_orientation_xyzw"]))
        joints = dict(zip(rb["joint_names"], rb["joint_positions"]))
        for link, M in rfk.pose(base_T, joints).items():
            T[f"robot/{link}"] = M
        per_frame.append({k: np.asarray(v).reshape(-1).tolist() for k, v in T.items() if k in nodes})
        cam = {}
        for v in views:
            if v == "showcase":
                continue
            if v in EXTERIOR:
                pos, q = EXTERIOR[v]
                M = base_T @ _T(pos, _quat_wxyz(q))
            elif v == "top":
                eye = (base_T @ np.array([*TOP_EYE, 1.0]))[:3]
                tgt = (base_T @ np.array([*TOP_TARGET, 1.0]))[:3]
                M = _look_at_gl(eye, tgt)
            else:  # wrist on the Robotiq base link (Drake/OpenCV camera axes -> OpenGL)
                R, t = WRIST_IN_BASE_LINK_DRAKE
                M = T["robot/base_link"] @ _T(t, R) @ _T(np.zeros(3), DRAKE_TO_GL)
            cam[v] = M.reshape(-1).tolist()
        cams.append(cam)
    if "showcase" in views:
        first = frames[0]["robot"]
        root = _T(first["base_position"], _quat_xyzw(first["base_orientation_xyzw"]))
        orientation = (root @ _T(EXTERIOR["exterior_left"][0],
                                 _quat_wxyz(EXTERIOR["exterior_left"][1])))[:3, :3]
        camera = _fit_showcase_camera(_showcase_points(
            scene, meshes, per_frame, frames, articulated_hosts=[a["host"] for a in arts.values()]), orientation)
        for camera_frame in cams:
            camera_frame["showcase"] = camera.reshape(-1).tolist()
    primitives = []
    mount = traj.get("robot_mount")
    if mount and mount.get("height_m", 0) > 0:
        height = float(mount["height_m"])
        node = add_node("robot_mount/base")
        primitives.append({"node": node, "kind": "box",
                           "size": [*mount["size_xy_m"], height], "rgba": [.28, .3, .32, 1]})
        base = frames[0]["robot"]["base_position"]
        transform = _T([*base[:2], height / 2], np.eye(3)).reshape(-1).tolist()
        for frame in per_frame:
            frame[node] = transform
    plan = {"schema_version": "wo_replay_render_plan.v1", "views": {v: view_specs[v] for v in views},
            "fps": 15, "nodes": nodes, "meshes": meshes, "primitives": primitives,
            "frames": per_frame, "cameras": cams}
    if "showcase" in views:
        plan["presentation"] = {"source": str(ref / traj_name), "trajectory_unchanged": True,
                                "evaluation_views_unchanged": True, "camera": "fixed over full recorded task motion"}
    return plan


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trajectory", default="trajectory.json")
    a = ap.parse_args()
    plan = build_plan(a.ref, traj_name=a.trajectory)
    a.out.write_text(json.dumps(plan))
    print(a.out, len(plan["frames"]), "frames", len(plan["meshes"]), "meshes")


if __name__ == "__main__":
    main()
