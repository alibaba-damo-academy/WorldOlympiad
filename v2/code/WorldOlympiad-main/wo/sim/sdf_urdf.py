"""Convert a (multi-link) SceneCode SDF model to a collision-only URDF.

PyBullet's SDF importer mis-places links that hang below a moving link (e.g. a
handle fixed to a door is rebased in the wrong frame).  URDF has unambiguous
semantics in PyBullet, so articulated assets are converted:

  * every SDF frame is resolved into the model frame (``relative_to``,
    ``attached_to``, implicit defaults, joint frames, axis ``expressed_in``),
  * each URDF child link frame is placed at its SDF joint frame,
  * collisions and inertials are re-expressed in that frame.

Visuals are dropped (rendering uses the original SDF in Drake, which applies SDF
semantics; only the joint values are transferred).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MIN_LINK_MASS_KG = 0.005


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _rpy_mat(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _mat_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sp = -R[2, 0]
    if abs(sp) < 0.999999:
        p = math.asin(sp)
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        p = math.copysign(math.pi / 2, sp)
        r = 0.0
        y = math.atan2(-R[0, 1], R[1, 1])
    return r, p, y


def _T(pose_el) -> tuple[np.ndarray, str | None]:
    if pose_el is None or not (pose_el.text or "").strip():
        return np.eye(4), (pose_el.get("relative_to") if pose_el is not None else None)
    v = [float(x) for x in pose_el.text.split()]
    v += [0.0] * (6 - len(v))
    T = np.eye(4)
    T[:3, :3] = _rpy_mat(*v[3:6])
    T[:3, 3] = v[:3]
    return T, pose_el.get("relative_to")


def _fmt(T: np.ndarray) -> dict[str, str]:
    r, p, y = _mat_rpy(T[:3, :3])
    return {"xyz": " ".join(f"{v:.9g}" for v in T[:3, 3]), "rpy": f"{r:.9g} {p:.9g} {y:.9g}"}


class _Frames:
    """Resolve named SDF frames to model-frame transforms."""

    def __init__(self, model) -> None:
        self.defs: dict[str, tuple[np.ndarray, str]] = {"__model__": (np.eye(4), "")}
        for link in _children(model, "link"):
            T, rel = _T(_child(link, "pose"))
            self.defs[link.get("name")] = (T, rel or "__model__")
        for fr in _children(model, "frame"):
            T, rel = _T(_child(fr, "pose"))
            self.defs[fr.get("name")] = (T, rel or fr.get("attached_to") or "__model__")
        for j in _children(model, "joint"):
            T, rel = _T(_child(j, "pose"))
            self.defs[j.get("name")] = (T, rel or _child(j, "child").text.strip())
        self.cache: dict[str, np.ndarray] = {"__model__": np.eye(4)}

    def model(self, name: str, depth: int = 0) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        if depth > 64 or name not in self.defs:
            raise ValueError(f"unresolvable SDF frame {name!r}")
        T, rel = self.defs[name]
        out = self.model(rel, depth + 1) @ T
        self.cache[name] = out
        return out


def link_count(sdf: Path) -> int:
    model = _child(ET.parse(sdf).getroot(), "model")
    return len(_children(model, "link")) if model is not None else 0


def sdf_to_urdf(sdf: Path, urdf: Path) -> Path:
    root = ET.parse(sdf).getroot()
    model = _child(root, "model")
    if model is None:
        raise ValueError("SDF has no model")
    frames = _Frames(model)
    links = {l.get("name"): l for l in _children(model, "link")}
    joints = _children(model, "joint")
    child_of = {_child(j, "child").text.strip(): j for j in joints}
    roots = [n for n in links if n not in child_of]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, got {roots}")
    # URDF link frame: root keeps its SDF link frame; children use their joint frame
    urdf_frame = {}
    for name in links:
        j = child_of.get(name)
        urdf_frame[name] = frames.model(j.get("name")) if j is not None else frames.model(name)

    robot = ET.Element("robot", name=model.get("name", "asset"))
    base_dir = sdf.parent
    for name, link in links.items():
        L = ET.SubElement(robot, "link", name=name)
        X_U_L = np.linalg.inv(urdf_frame[name]) @ frames.model(name)  # SDF link in URDF link frame
        inertial = _child(link, "inertial")
        if inertial is None:
            I = ET.SubElement(L, "inertial")
            ET.SubElement(I, "mass", value=f"{MIN_LINK_MASS_KG}")
            ET.SubElement(I, "inertia", ixx="1e-6", iyy="1e-6", izz="1e-6", ixy="0", ixz="0", iyz="0")
        else:
            Ti, rel = _T(_child(inertial, "pose"))
            X = (np.linalg.inv(urdf_frame[name]) @ frames.model(rel) @ Ti) if rel else X_U_L @ Ti
            I = ET.SubElement(L, "inertial")
            ET.SubElement(I, "origin", **_fmt(X))
            mass = _child(inertial, "mass")
            m = float(mass.text) if mass is not None and mass.text else 0.0
            # SceneCode code-articulated assets export massless root links; PyBullet
            # would treat a massless URDF root as a fixed base, so give it a token mass
            ET.SubElement(I, "mass", value=f"{max(m, MIN_LINK_MASS_KG):.9g}")
            ine = _child(inertial, "inertia")
            vals = {k: "0" for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
            vals.update(ixx="1e-4", iyy="1e-4", izz="1e-4")
            if ine is not None:
                for k in vals:
                    node = _child(ine, k)
                    if node is not None and node.text:
                        vals[k] = node.text.strip()
            ET.SubElement(I, "inertia", **vals)
        for col in _children(link, "collision"):
            Tc, rel = _T(_child(col, "pose"))
            X = (np.linalg.inv(urdf_frame[name]) @ frames.model(rel) @ Tc) if rel else X_U_L @ Tc
            geom = _child(col, "geometry")
            C = ET.SubElement(L, "collision", name=col.get("name", "collision"))
            ET.SubElement(C, "origin", **_fmt(X))
            G = ET.SubElement(C, "geometry")
            g = list(geom)[0]
            kind = _local(g.tag)
            if kind == "mesh":
                uri = _child(g, "uri").text.strip()
                path = Path(uri) if Path(uri).is_absolute() else (base_dir / uri).resolve()
                scale = _child(g, "scale")
                attrs = {"filename": str(path)}
                if scale is not None and scale.text:
                    attrs["scale"] = scale.text.strip()
                ET.SubElement(G, "mesh", **attrs)
            elif kind == "box":
                ET.SubElement(G, "box", size=_child(g, "size").text.strip())
            elif kind == "cylinder":
                ET.SubElement(G, "cylinder", radius=_child(g, "radius").text.strip(),
                              length=_child(g, "length").text.strip())
            elif kind == "sphere":
                ET.SubElement(G, "sphere", radius=_child(g, "radius").text.strip())
            else:
                raise ValueError(f"unsupported collision geometry {kind}")
    for j in joints:
        jtype = j.get("type")
        utype = {"revolute": "revolute", "prismatic": "prismatic", "fixed": "fixed",
                 "continuous": "continuous"}.get(jtype)
        if utype is None:
            raise ValueError(f"unsupported joint type {jtype}")
        parent = _child(j, "parent").text.strip()
        child = _child(j, "child").text.strip()
        if parent == "world":
            continue
        J = ET.SubElement(robot, "joint", name=j.get("name"), type=utype)
        ET.SubElement(J, "parent", link=parent)
        ET.SubElement(J, "child", link=child)
        ET.SubElement(J, "origin", **_fmt(np.linalg.inv(urdf_frame[parent]) @ urdf_frame[child]))
        axis = _child(j, "axis")
        if axis is not None and utype != "fixed":
            xyz_el = _child(axis, "xyz")
            v = np.array([float(x) for x in xyz_el.text.split()])
            expressed = xyz_el.get("expressed_in")
            if expressed:  # rotate into the joint frame
                Rj = urdf_frame[child][:3, :3]
                v = Rj.T @ frames.model(expressed)[:3, :3] @ v
            ET.SubElement(J, "axis", xyz=" ".join(f"{x:.9g}" for x in v))
            lim = _child(axis, "limit")
            if lim is not None and utype in {"revolute", "prismatic"}:
                lo = _child(lim, "lower")
                hi = _child(lim, "upper")
                ET.SubElement(J, "limit", lower=lo.text.strip() if lo is not None else "-3.14",
                              upper=hi.text.strip() if hi is not None else "3.14",
                              effort="100", velocity="2")
            dyn = _child(axis, "dynamics")
            if dyn is not None:
                damp = _child(dyn, "damping")
                fric = _child(dyn, "friction")
                ET.SubElement(J, "dynamics",
                              damping=damp.text.strip() if damp is not None else "0",
                              friction=fric.text.strip() if fric is not None else "0")
    urdf.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space=" ")
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    return urdf


def sdf_visuals(sdf: Path) -> dict:
    """Per-link visual meshes of an SDF model, plus each link's offset from the
    corresponding URDF link frame written by ``sdf_to_urdf``.

    Returns {"root": name, "links": {link: {"urdf_T_link": 4x4,
             "visuals": [{"file", "scale", "link_T_visual"}]}}}.
    """
    model = _child(ET.parse(sdf).getroot(), "model")
    frames = _Frames(model)
    links = {l.get("name"): l for l in _children(model, "link")}
    joints = _children(model, "joint")
    child_of = {_child(j, "child").text.strip(): j for j in joints}
    roots = [n for n in links if n not in child_of]
    out = {"root": roots[0] if roots else next(iter(links)), "links": {}}
    for name, link in links.items():
        j = child_of.get(name)
        urdf_frame = frames.model(j.get("name")) if j is not None else frames.model(name)
        link_model = frames.model(name)
        vis = []
        for v in _children(link, "visual"):
            Tv, rel = _T(_child(v, "pose"))
            link_T_v = (np.linalg.inv(link_model) @ frames.model(rel) @ Tv) if rel else Tv
            g = list(_child(v, "geometry"))[0]
            if _local(g.tag) != "mesh":
                continue
            uri = _child(g, "uri").text.strip()
            path = Path(uri) if Path(uri).is_absolute() else (sdf.parent / uri).resolve()
            sc = _child(g, "scale")
            scale = [float(x) for x in sc.text.split()] if sc is not None and sc.text else [1.0] * 3
            vis.append({"file": str(path), "scale": scale, "link_T_visual": link_T_v.tolist()})
        out["links"][name] = {"urdf_T_link": (np.linalg.inv(urdf_frame) @ link_model).tolist(),
                              "visuals": vis}
    return out
