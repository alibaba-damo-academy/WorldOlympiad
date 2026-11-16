"""Stateful native terminal measurements selected by typed goal semantics."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from .io import sha256_file
from .robotics_hanging_geometry import measure_meshes

SUPPORTED = frozenset({"inside", "in_container", "on_region", "on", "stacked", "contact", "hung", "inserted", "covering"})
PROTOCOL_PATH = Path(__file__).parents[1] / "deployment/seetacloud/cosmos_task_graph_protocol.json"


def protocol() -> dict:
    return {**json.loads(PROTOCOL_PATH.read_text()), "sha256": sha256_file(PROTOCOL_PATH)}


def local_mesh(body: dict, root: Path, *, role="visuals", state=None):
    # Floating mechanisms must use measured link frames, not the authored
    # zero-joint assembly. Return geometry in the measured object model frame
    # so the existing predicate transforms remain valid.
    live = None
    if state is not None and body.get("joints") and not body.get("static"):
        live = state.get("link_states")
        if not isinstance(live, dict) or set(live) != {link['name'] for link in body['links']}:
            raise ValueError("floating predicate geometry requires every live link pose")
        model_p, model_r = pose(state)
        if not np.isfinite(model_p).all():
            raise ValueError("nonfinite floating object model pose")
    meshes = []
    for link in body["links"]:
        for geometry in link[role]:
            if geometry["kind"] == "mesh":
                mesh = trimesh.load(root / geometry["asset_path"], force="scene").to_geometry()
            elif geometry["kind"] == "box":
                mesh = trimesh.creation.box(geometry["size"])
            elif geometry["kind"] == "cylinder":
                mesh = trimesh.creation.cylinder(geometry["radius"], geometry["length"])
            elif geometry["kind"] == "sphere":
                mesh = trimesh.creation.icosphere(radius=geometry["radius"])
            else:
                raise ValueError("unsupported native predicate geometry")
            mesh.apply_scale(geometry.get("scale", [1, 1, 1]))
            link_pose = link.get("pose", {})
            if live is not None:
                link_p, link_r = pose(live[link['name']])
                if not np.isfinite(link_p).all():
                    raise ValueError("nonfinite floating link pose")
                link_pose = {'xyz': model_r.inv().apply(link_p-model_p),
                             'quat_xyzw': (model_r.inv()*link_r).as_quat()}
            for geometry_pose in (geometry.get("pose", {}), link_pose):
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat(geometry_pose.get("quat_xyzw", [0, 0, 0, 1])).as_matrix()
                matrix[:3, 3] = geometry_pose.get("xyz", [0, 0, 0])
                mesh.apply_transform(matrix)
            meshes.append(mesh)
    if not meshes:
        raise ValueError("native goal measurement needs declared visual geometry")
    return trimesh.util.concatenate(meshes)


def pose(state: dict):
    p = np.asarray(state.get("link_position", state.get("position")), dtype=float)
    q = state.get("link_orientation_xyzw", state.get("orientation_xyzw"))
    return p, Rotation.from_quat(q)


class NativeTaskGraphPredicates:
    def __init__(self, graph: dict, plan: dict, plan_root: Path):
        self.goals = graph.get("success_program") or []
        self.release_subjects = {s for op in graph.get("controller_program", [])
                                 if op.get("operator") in {"place", "stack", "insert", "cover"} for s in op.get("subjects", [])}
        unknown = {goal["predicate"] for goal in self.goals} - SUPPORTED
        if not self.goals or unknown:
            raise ValueError(f"missing native terminal measurement: {sorted(unknown)}")
        self.config = protocol()
        self.bodies = {body["name"]: body for body in plan["bodies"] if body["name"] != "robot"}
        self.meshes = {name: local_mesh(body, plan_root) for name, body in self.bodies.items()}
        self.by_id = {}
        for name, body in self.bodies.items():
            identity = (body.get("metadata") or {}).get("task_object_id", name)
            self.by_id.setdefault(identity, []).append(name)
        self.hanging = {}
        self.hulls = {}
        self.regions = {}
        self.support_meshes = {}
        for goal in self.goals:
            target = self.names(goal["object"])[0]
            if goal["predicate"] in {"inside", "in_container"}:
                self.hulls[target] = ConvexHull(self.meshes[target].vertices).equations
                if goal.get("pour_required"):
                    source = self.names(goal["source_container"])[0]
                    self.hulls[source] = ConvexHull(self.meshes[source].vertices).equations
            if goal["predicate"] in {"inserted", "covering"}:
                for name in [target, *self.names(goal["subject"])]:
                    self.support_meshes[name] = local_mesh(self.bodies[name], plan_root, role="collisions")
            if goal["predicate"] in {"on_region", "on", "stacked"}:
                for name in [target, *self.names(goal["subject"])]:
                    self.support_meshes[name] = local_mesh(self.bodies[name], plan_root, role="collisions")
                self.regions[target] = ConvexHull(self.support_meshes[target].vertices[:, :2]).equations
                if self.names(goal["subject"]) != ["cup"]:
                    raise ValueError("on-region release lacks a bound native subject sensor")
            if goal["predicate"] in {"contact", "hung"}:
                if self.names(goal["subject"]) != ["cup"] or self.names(goal["object"]) != ["tray"]:
                    raise ValueError("goal contact pair is not bound to the native contact sensors")
            if goal["subject"] in self.release_subjects:
                if any(self.bodies[name]["static"] for name in self.names(goal["subject"])):
                    raise ValueError("placement release subject must be a dynamic native body")
            if goal["predicate"] == "hung":
                subject, target = self.names(goal["subject"])[0], self.names(goal["object"])[0]
                self.hanging[subject] = measure_meshes(self.meshes[subject], self.meshes[target])
        self.plan_root = plan_root
        self.initial = None
        self.previous = None
        self.grasped = False
        self.lifted = set()
        self.tilted = set()
        self.impacts = set()
        self.stable_since = None
        self.last_evidence = {}

    def names(self, identity):
        if identity not in self.by_id:
            raise ValueError(f"goal object is not bound to native bodies: {identity}")
        return self.by_id[identity]

    def states(self, frame):
        objects = frame.get("objects") or {}
        states = {}
        for name, body in self.bodies.items():
            original = (body.get("metadata") or {}).get("trajectory_object_name", name)
            if original in objects:
                states[name] = objects[original]
            elif name in {"cup", "tray"} and name in frame:
                states[name] = frame[name]
            elif body["static"]:
                states[name] = {"position": body["pose"]["xyz"],
                                "orientation_xyzw": body["pose"]["quat_xyzw"]}
            else:
                raise ValueError(f"missing live state for native goal: {name}")
        return states

    def inside(self, subject, target, states):
        p, r = pose(states[subject]); tp, tr = pose(states[target])
        center = self.meshes[subject].centroid
        relative = tr.inv().apply(r.apply(center)+p-tp)
        margin = self.config["geometry_tolerance_m"]
        hull = self.hulls[target]
        return bool(np.all(hull[:, :3] @ relative + hull[:, 3] <= margin))

    def hung(self, subject, target, states, frame):
        geometry = self.hanging[subject]
        p, r = pose(states[subject]); tp, tr = pose(states[target])
        shaft = np.array([[geometry["post_x"], geometry["hook_bar_y"], geometry["hook_bar_z"]],
                          [geometry["hook_tip_x"], geometry["hook_bar_y"], geometry["hook_bar_z"]]])
        shaft = r.inv().apply(tr.apply(shaft)+tp-p)
        center = np.asarray(geometry["ring_center"])
        delta = shaft[1]-shaft[0]
        if abs(delta[1]) < 1e-8:
            return False
        fraction = (center[1]-shaft[0, 1])/delta[1]
        cross = shaft[0]+fraction*delta
        engaged = (0 <= fraction <= 1 and np.linalg.norm((cross-center)[[0, 2]])
                   + geometry["hook_radius"] < geometry["handle_inner_radius"])
        corners = np.array(list(itertools.product(*zip(*self.meshes[subject].bounds))))
        bottom = float((r.apply(corners)+p)[:, 2].min())
        supports = [name for name, body in self.bodies.items()
                    if (body.get("metadata") or {}).get("task_role") == "support"]
        top = max((float((pose(states[name])[1].apply(self.meshes[name].vertices)
                          + pose(states[name])[0])[:, 2].max()) for name in supports), default=-np.inf)
        return bool(engaged and bottom > top+self.config["geometry_tolerance_m"]
                    and frame.get("object_container_contact", frame.get("subject_target_contact"))
                    and not frame.get("robot_cup_contact_links") and subject in self.lifted)

    def on_region(self, subject, target, states, frame):
        """Measured support in the declared footprint, not a sample's center."""
        p, r = pose(states[subject]); tp, tr = pose(states[target])
        center = tr.inv().apply(r.apply(self.support_meshes[subject].centroid)+p-tp)
        region = self.regions[target]
        inside = np.all(region[:, :2] @ center[:2]+region[:, 2] <= self.config["geometry_tolerance_m"])
        subject_bottom = (r.apply(self.support_meshes[subject].vertices)+p)[:, 2].min()
        target_top = (tr.apply(self.support_meshes[target].vertices)+tp)[:, 2].max()
        return bool(inside and abs(subject_bottom-target_top) <= self.config["geometry_tolerance_m"]
                    and not frame.get("robot_cup_contact_links"))

    def inserted(self, subject, target, states):
        """Shared insertion predicate on transformed live collision vertices.

        Native geometry has no Bullet collision-margin padding. No convex-hull
        containment test is substituted for the declared depth/XY criterion.
        """
        from wo.predicates import THRESHOLDS
        bounds = []
        for name in (subject, target):
            p, r = pose(states[name])
            vertices = r.apply(self.support_meshes[name].vertices) + p
            bounds.append((vertices.min(axis=0), vertices.max(axis=0)))
        (s_lo, s_hi), (t_lo, t_hi) = bounds
        center = (s_lo[:2] + s_hi[:2]) / 2
        depth = float(t_hi[2] - s_lo[2])
        need = float(THRESHOLDS["inserted_depth_fraction"] * (s_hi[2] - s_lo[2]))
        inside = bool(np.all(center >= t_lo[:2]) and np.all(center <= t_hi[:2]))
        return bool(inside and depth >= need), {
            "insertion_depth_m": depth, "required_depth_m": need,
            "center_inside_target_xy": inside,
            "subject_collision_aabb": [s_lo.tolist(), s_hi.tolist()],
            "target_collision_aabb": [t_lo.tolist(), t_hi.tolist()]}

    def covering(self, subject, target, states, frame):
        from wo.predicates import THRESHOLDS
        bounds = []
        for name in (subject, target):
            p, r = pose(states[name])
            vertices = r.apply(self.support_meshes[name].vertices) + p
            bounds.append((vertices.min(axis=0), vertices.max(axis=0)))
        (s_lo, s_hi), (t_lo, t_hi) = bounds
        overlap = np.maximum(0., np.minimum(s_hi[:2], t_hi[:2]) - np.maximum(s_lo[:2], t_lo[:2]))
        area = float(np.prod(t_hi[:2] - t_lo[:2]))
        fraction = float(np.prod(overlap) / area) if area > 0 else 0.
        evidence = frame.get('contact_support', {}).get(subject, {})
        measured = evidence.get('authority') == 'physx_per_contact_normal_force'
        touching = measured and target in evidence.get('touching_bodies', [])
        supported = measured and evidence.get('supported_by') == target
        return bool(fraction >= THRESHOLDS['xy_overlap_fraction'] and touching and supported), {
            'target_covered_fraction': fraction, 'contact_support': evidence,
            'subject_collision_aabb': [s_lo.tolist(), s_hi.tolist()],
            'target_collision_aabb': [t_lo.tolist(), t_hi.tolist()]}

    def update(self, frame: dict) -> bool:
        states = self.states(frame)
        # Rebuild only movable mechanisms; rigid geometry retains its cached
        # assembly. Convex containment/support regions must move with links too.
        for name, body in self.bodies.items():
            if body.get('joints') and not body.get('static'):
                self.meshes[name] = local_mesh(body, self.plan_root, state=states[name])
                if name in self.support_meshes:
                    self.support_meshes[name] = local_mesh(body, self.plan_root,
                        role='collisions', state=states[name])
                if name in self.hulls:
                    self.hulls[name] = ConvexHull(self.meshes[name].vertices).equations
                if name in self.regions:
                    self.regions[name] = ConvexHull(self.support_meshes[name].vertices[:, :2]).equations
        time = float(frame["time_s"])
        if self.initial is None:
            self.initial = {name: pose(state)[0].copy() for name, state in states.items()}
        self.grasped |= bool(frame.get("robot_cup_contact_links"))
        velocities = {}
        angular_speeds = {}
        for name, state in states.items():
            p, r = pose(state)
            if p[2]-self.initial[name][2] > self.config["minimum_lift_m"] and self.grasped:
                self.lifted.add(name)
            if np.arccos(np.clip(r.apply([0, 0, 1])[2], -1, 1)) > self.config["minimum_tilt_rad"]:
                self.tilted.add(name)
            velocity = state.get("linear_velocity")
            if velocity is None:
                velocity = ((p-pose(self.previous["states"][name])[0])/(time-self.previous["time"])) if self.previous and time > self.previous["time"] else np.zeros(3)
            velocities[name] = float(np.linalg.norm(velocity))
            angular = state.get("angular_velocity")
            if angular is None:
                angular = ((r*pose(self.previous["states"][name])[1].inv()).as_rotvec()/(time-self.previous["time"])) if self.previous and time > self.previous["time"] else np.zeros(3)
            angular_speeds[name] = float(np.linalg.norm(angular))
        target_contact = bool(frame.get("object_container_contact", frame.get("subject_target_contact")))
        for idx, goal in enumerate(self.goals):
            subjects = self.names(goal["subject"])
            if (goal["predicate"] == "contact" and target_contact and self.grasped
                    and self.previous and not self.previous["contact"]
                    and any(self.previous["velocities"][s] > self.config["minimum_impact_speed_m_s"] for s in subjects)):
                self.impacts.add(idx)
        checks = {}
        geometry_evidence = {}
        stationary_names = set()
        for idx, goal in enumerate(self.goals):
            subjects, target = self.names(goal["subject"]), self.names(goal["object"])[0]
            kind = goal["predicate"]
            if kind in {"inside", "in_container"}:
                checks[str(idx)] = all(self.inside(s, target, states) for s in subjects)
                stationary_names.update(subjects)
                if goal.get("pour_required"):
                    source = self.names(goal["source_container"])[0]
                    checks[str(idx)] &= source in self.lifted and source in self.tilted and all(not self.inside(s, source, states) for s in subjects)
            elif kind in {"inserted", "covering"}:
                measurements = [self.inserted(s, target, states) if kind == 'inserted'
                                else self.covering(s, target, states, frame) for s in subjects]
                checks[str(idx)] = all(passed for passed, _ in measurements)
                geometry_evidence[str(idx)] = dict(zip(subjects, [ev for _, ev in measurements]))
                stationary_names.update(subjects)
            elif kind == "contact":
                checks[str(idx)] = idx in self.impacts if goal.get("impact_required") else target_contact
            elif kind in {"on_region", "on", "stacked"}:
                checks[str(idx)] = self.on_region(subjects[0], target, states, frame)
                stationary_names.update(subjects)
                if goal.get("reorientation_required"):
                    checks[str(idx)] &= subjects[0] in self.lifted
            else:
                checks[str(idx)] = self.hung(subjects[0], target, states, frame)
                stationary_names.update(subjects)
            if goal.get("nonprehensile_required"):
                # A pushing finger necessarily contacts the subject; contact
                # alone is not a grasp. Reject lift-and-place histories instead.
                checks[str(idx)] &= all(s not in self.lifted for s in subjects)
            if goal.get("terminal_upright"):
                # The semantic axis is measured in the authored object frame;
                # applying local +Z here silently accepts side-lying assets
                # whose base-to-cap axis is local Y. Keep the world target
                # explicit, but transform the declared local axis at runtime.
                local_axis = np.asarray(goal.get("terminal_upright_axis_local", [0., 0., 1.]), dtype=float)
                desired = np.asarray(goal.get("terminal_upright_axis_world", [0., 0., 1.]), dtype=float)
                if local_axis.shape != (3,) or desired.shape != (3,):
                    raise ValueError("terminal upright axes must be 3-vectors")
                local_axis /= np.linalg.norm(local_axis)
                desired /= np.linalg.norm(desired)
                checks[str(idx)] &= all(np.dot(pose(states[s])[1].apply(local_axis), desired)
                                      >= goal.get("terminal_upright_min_cosine", .99) for s in subjects)
            if goal["subject"] in self.release_subjects:
                # Require measured release for EVERY expanded subject. Missing
                # evidence is not evidence of no contact. Preserve the legacy
                # cup slot while multi-object placements use per-body sensors.
                contacts = frame.get("robot_subject_contacts", {})
                checks[str(idx)] &= all(
                    not bool(contacts[s]) if s in contacts else
                    (not bool(frame["robot_cup_contact_links"])
                     if s == "cup" and "robot_cup_contact_links" in frame else False)
                    for s in subjects)
        stationary = all(velocities[name] <= self.config["stationary_speed_m_s"] and angular_speeds[name] <= self.config["stationary_angular_speed_rad_s"] for name in stationary_names)
        passed = all(checks.values()) and stationary
        self.stable_since = (time if self.stable_since is None else self.stable_since) if passed else None
        success = bool(passed and self.stable_since is not None and time-self.stable_since >= self.config["stable_duration_s"])
        self.last_evidence = {"native_predicates": checks, "geometry": geometry_evidence, "stationary": stationary,
            "stable_since_s": self.stable_since, "measured_at_s": time,
            "protocol_sha256": self.config["sha256"], "grasp_observed": self.grasped,
            "lifted_bodies": sorted(self.lifted), "tilted_bodies": sorted(self.tilted),
            "impact_goals": sorted(self.impacts), "success": success}
        self.previous = {"states": states, "time": time, "contact": target_contact, "velocities": velocities}
        return success
