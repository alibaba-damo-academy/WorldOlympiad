"""Embodiment fit: make a manipuland graspable by the registered gripper.

If the grasp sampler finds no candidate at native scale, the asset is uniformly
scaled (visual + collision + inertia, via the existing SceneCode scale-variant
derivation) by the largest factor from a fixed ladder that yields grasps.
The variant is recorded in the reference certificate; it is never silent.
"""

from __future__ import annotations

import sys

import numpy as np
from dataclasses import replace
from pathlib import Path

from ..scene import SceneInputs, WORKSPACE
from ..sim.world import World, prepare_bodies
from .grasps import sample_grasps

SCALE_LADDER = (1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35)
# Each sampled grasp already passes geometric and holding-torque filters.
# Candidate multiplicity is not a feasibility criterion (thin lids may have
# only one antipodal pair). Planning and physical rollout certify that pair.
MIN_GRASPS = 1

_REPO = WORKSPACE / "code/WorldOlympiad-main"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def scaled_scene(scene: SceneInputs, object_id: str, scale: float, work: Path) -> SceneInputs:
    if scale == 1.0:
        return scene
    from scripts.derive_scenecode_uniform_scale_variant import derive_uniform_scale_variant

    obj = scene.objects[object_id]
    # name by the cumulative scale from the SceneCode asset, so chained scalings never collide
    out_dir = work / "variants" / f"{object_id}_s{obj.scale * scale:.3f}"
    sdf = out_dir / obj.sdf_path.name
    if not sdf.is_file():
        sdf = derive_uniform_scale_variant(
            obj.sdf_path, out_dir, variant_id=f"{object_id}_gripper_fit_{int(scale * 1000)}",
            scale=scale,
        )
    # keep the asset's measured bottom on the same support
    bottom = _aabb_bottom(scene, work, object_id)
    new_obj = replace(obj, sdf_path=sdf, scale=obj.scale * scale, bbox_local=None)
    trial = dict(scene.objects)
    trial[object_id] = new_obj
    new_bottom = _aabb_bottom(replace(scene, objects=trial), work, object_id)
    new_obj = replace(new_obj, position=(obj.position[0], obj.position[1],
                                         obj.position[2] + bottom - new_bottom))
    objects = dict(scene.objects)
    objects[object_id] = new_obj
    return replace(scene, objects=objects)


def _reach_poses(world: World, spec, base: tuple[float, float, float], *, grasp_fallback=False) -> list[list[Pose]]:
    """For each task-relevant place, alternative TCP poses of which one must be reachable."""
    from .kin import Pose

    down = np.diag([1.0, -1.0, -1.0])
    b = np.asarray(base[:2])
    out: list[list[Pose]] = []
    subjects, targets = set(), set()
    for st in spec.steps:
        if st.primitive == "articulate":
            continue
        subjects.add(st.subject)
        for t in (st.target, st.goal.target):
            if t is not None and t != world.support_id and t in world.bodies:
                targets.add(t)
    for oid in subjects:
        lo, hi = world.aabb(oid)
        out.append([Pose(np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 0.10]), down)])
    for oid in targets - subjects:
        lo, hi = world.aabb(oid)
        near = np.clip(b, lo[:2] + 0.05, hi[:2] - 0.05) if np.all(hi[:2] - lo[:2] > 0.1) \
            else (lo[:2] + hi[:2]) / 2
        out.append([Pose(np.array([near[0], near[1], hi[2] + 0.10]), down)])
    from .articulate import rigid_links, link_pose

    for comp, art in world.articulation.items():
        body, joint = art["body"], art["joint"]
        attached = rigid_links(world, body, joint) - {joint}
        feature = sorted(attached)[0] if attached else joint  # handle if present
        q0 = world.joint_q(comp)
        grasps = []
        if grasp_fallback:
            # Use the executor's actual moving-link grasp frames, rather than
            # assuming that a radial handle-center orientation is attainable.
            # Sampling is bounded identically to the articulation planner.
            world.set_joint_q(comp, art["q_closed"])
            grasps = sample_grasps(world, art["host"], links=attached | {joint},
                                   frame=link_pose(world, body, joint),
                                   prefer_links=attached)[:24]
        for frac in (0.0, 0.5, 0.85):
            world.set_joint_q(comp, art["q_closed"] + frac * (art["q_open"] - art["q_closed"]))
            lo, hi = world.p.getAABB(body, feature, physicsClientId=world.cid)
            c = (np.asarray(lo) + np.asarray(hi)) / 2
            toward = c[:2] - b
            toward = toward / (np.linalg.norm(toward) + 1e-9)
            a = np.array([toward[0], toward[1], 0.0])
            alts = []
            for tilt in (0.0, 0.5):
                aa = a * np.cos(tilt) - np.array([0, 0, np.sin(tilt)])
                for x in (np.array([-a[1], a[0], 0.0]), np.array([0, 0, 1.0])):
                    xx = x - aa * (x @ aa)
                    xx /= np.linalg.norm(xx)
                    R = np.stack([xx, np.cross(aa, xx), aa], axis=1)
                    alts.append(Pose(c - aa * 0.02, R))
            if grasps:
                moving_frame = link_pose(world, body, joint)
                alts.extend(moving_frame @ g.tcp_in_obj for g in grasps)
            out.append(alts)
        world.set_joint_q(comp, q0)
    return out


def placement_candidates(canonical, fallback=(), budget=40, lower_mounts=()):
    """Exhaust the existing layout before a separately bounded preserve search."""
    for preserve, candidates in ((False, canonical), (True, fallback), (True, lower_mounts)):
        seen = set()
        for cost, base, yaw in candidates:
            key = tuple(round(float(v), 3) for v in (*base, yaw))
            if key in seen:
                continue
            if len(seen) >= budget:
                break
            seen.add(key)
            yield preserve, cost, base, yaw


def place_robot(scene: SceneInputs, work: Path, spec, *, layout: str = "scene") -> tuple[SceneInputs, dict]:
    """First base candidate whose home pose is collision-free and reaches the task."""
    from ..scene import robot_base_candidates
    from .kin import Kinematics, tcp_to_ee

    from ..scene import fixed_background_ids
    backgrounds = fixed_background_ids(spec, scene.objects,
                                       hosts=[a['host'] for a in scene.articulations.values()])
    prepared = prepare_bodies(scene, work)
    probe = World(scene, prepared, mode="objects")
    try:
        s_lo, s_hi = probe.aabb(probe.support_id)
        boxes = [probe.aabb(oid) for oid in scene.objects if oid not in backgrounds]
        hosts = {a['host'] for a in scene.articulations.values()}
        host_boxes = [probe.aabb(oid) for oid in hosts if oid in scene.objects]
        for art in probe.articulation.values():
            lo, hi = probe.p.getAABB(art["body"], art["joint"], physicsClientId=probe.cid)
            boxes.append((np.asarray(lo), np.asarray(hi)))
    finally:
        probe.close()
    if not boxes:
        raise RuntimeError("scene has no task object besides the support")
    # never mount the arm on an articulated host: stand on a pedestal outside it
    host_support = scene.support.object_id in {a["host"] for a in scene.articulations.values()}
    candidates = robot_base_candidates(s_lo, s_hi, boxes, outside_only=host_support)
    fallback = []
    if layout == 'droid' and host_boxes and not host_support:
        # Moving a tall linkage into the tabletop workspace may intersect the
        # home arm for every canonical base. Preserve the entire authored scene
        # and search outside its actual host footprint at the same mount height.
        # No root/joint/support transform or collision rule changes here.
        lo = np.min([b[0] for b in host_boxes], axis=0)
        hi = np.max([b[1] for b in host_boxes], axis=0)
        hi[2] = s_hi[2]
        fallback = robot_base_candidates(lo, hi, boxes, outside_only=True)
    lower_mounts = []
    if layout == 'droid' and host_boxes:
        # Compact tabletop linkages can be below the arm's horizontal-grasp
        # workspace when its root is at tabletop height. Search lower pedestals
        # outside BOTH the support and host footprints, without moving either.
        perimeter_lo = np.minimum(s_lo, np.min([b[0] for b in host_boxes], axis=0))
        perimeter_hi = np.maximum(s_hi, np.max([b[1] for b in host_boxes], axis=0))
        perimeter_hi[2] = s_hi[2]
        perimeter = robot_base_candidates(perimeter_lo, perimeter_hi, boxes, outside_only=True)
        for cost, base, yaw in perimeter:
            for drop in (0.15, 0.30, 0.45):
                if base[2] - drop > 0:
                    lower_mounts.append((cost, (base[0], base[1], base[2] - drop), yaw))
    tried = []
    for preserve, cost, base, yaw in placement_candidates(candidates, fallback, lower_mounts=lower_mounts):
        cand = replace(scene, robot_base_position=tuple(base), robot_base_yaw=float(yaw))
        layout_info = None
        if layout == "droid":
            # The host SDF contains the complete linkage: translating its root
            # moves all child links and joint frames together. Only a host that
            # IS the support must remain fixed in the support/world frame.
            if backgrounds:
                layout_info = {"policy":"preserve_fixed_background_scene",
                               "background_ids":sorted(backgrounds)}
            elif host_support or preserve:
                layout_info = {"policy": "preserve_articulated_scene",
                               "mount_drop_m": max(0.0, float(s_hi[2] + 0.002 - base[2])),
                               "reason": "canonical placement exhausted; measured perimeter/pedestal fallback"
                               if preserve else "articulated host is the support/world frame"}
            else:
                try:
                    cand, layout_info = droid_canonical_layout(cand, work)
                except RuntimeError as exc:
                    # A layout can fail for this base/yaw while fitting from
                    # another table edge. Continue the embodiment search.
                    tried.append({"base": list(base), "yaw": yaw,
                                  "layout_error": str(exc), "reach": None,
                                  "reach_status": "not_tested_layout_failed"})
                    continue
        world = World(cand, prepared, mode="kinematic")
        try:
            kin = Kinematics(world)
            hit = kin.in_collision(kin.home)
            # A colliding home rejects this candidate before IK. Do not report
            # the unmeasured reach test as False in failure diagnostics.
            reach_ok = None if hit is not None else all(
                any(kin.ik(tcp_to_ee(p)) is not None for p in alts)
                for alts in _reach_poses(world, spec, base))
            reach_grasp_fallback = False
            if hit is None and not reach_ok and world.articulation:
                reach_grasp_fallback = True
                reach_ok = all(
                    any(kin.ik(tcp_to_ee(p)) is not None for p in alts)
                    for alts in _reach_poses(world, spec, base, grasp_fallback=True))
        finally:
            world.close()
        tried.append({"base": list(base), "yaw": yaw, "home_collision": hit, "reach": reach_ok,
                      "reach_status": "not_tested_home_collision" if hit is not None else "tested",
                      "reach_grasp_fallback": reach_grasp_fallback, "layout": layout_info})
        if hit is None and reach_ok:
            on_support = bool(s_lo[0] <= base[0] <= s_hi[0] and s_lo[1] <= base[1] <= s_hi[1])
            top = float(s_hi[2])
            mount = None if on_support else {"height_m": float(base[2] - 0.002), "size_xy_m": [0.30, 0.30]}
            return cand, {"base": list(base), "yaw": yaw, "tried": len(tried), "mount": mount,
                          "support_top_z": top, "layout": layout_info}
    raise RuntimeError(f"no collision-free reachable robot base ({len(tried)} candidates): {tried}")


def _aabb_bottom(scene: SceneInputs, work: Path, object_id: str) -> float:
    w = World(scene, prepare_bodies(scene, work), mode="objects", include=[object_id])
    try:
        return float(w.aabb(object_id)[0][2])
    finally:
        w.close()


def fit_gripper(scene: SceneInputs, object_id: str, work: Path) -> tuple[SceneInputs, dict]:
    tried = []
    for scale in SCALE_LADDER:
        candidate = scaled_scene(scene, object_id, scale, work)
        # Sizing tests intrinsic compatibility, not accessibility in the initial
        # clutter. Retain the support, but exclude other task bodies: shrinking
        # a subject cannot legitimately resolve a neighbour blocking its grasp.
        # Full-scene sampling, planning and rollout still certify accessibility.
        world = World(candidate, prepare_bodies(candidate, work), mode="objects",
                      include=[object_id])
        try:
            n = len(sample_grasps(world, object_id))
        finally:
            world.close()
        tried.append({"scale": scale, "grasps": n})
        if n >= MIN_GRASPS:
            return candidate, {"object_id": object_id, "scale": scale, "ladder": tried}
    raise RuntimeError(f"no gripper-compatible scale for {object_id}: {tried}")


# DROID tabletop workspace (robot root frame): where RoboLab/DROID episodes put
# task objects.  Official RoboLab BananaInBowl: banana (0.51,-0.12), bowl (0.56,0.15).
DROID_WORKSPACE_CENTER = (0.53, 0.0)
DROID_MAX_LATERAL = 0.17
DROID_MAX_DEPTH = 0.12
DROID_MIN_GAP_M = 0.04


def containment_groups(objects: dict) -> dict[str, list[str]]:
    """Keep declared initial contents with their source container."""
    groups = {}
    for oid in objects:
        root = oid
        seen = set()
        while True:
            if root in seen:
                raise ValueError("initial container relations form a cycle")
            seen.add(root)
            parent = getattr(objects[root], "initial_container", None)
            if parent not in objects:
                break
            root = parent
        groups.setdefault(root, []).append(oid)
    return groups


def droid_canonical_layout(scene: SceneInputs, work: Path) -> tuple[SceneInputs, dict]:
    """Move the task objects (rigidly, keeping their arrangement) into the DROID
    workspace in front of the chosen robot base, compressing the spread when the
    objects lie wider than DROID's usual tabletop region.  Orientations and
    heights are unchanged; the minimum footprint gap is preserved."""
    import math

    yaw = scene.robot_base_yaw
    c, s_ = math.cos(yaw), math.sin(yaw)
    base = np.asarray(scene.robot_base_position[:2])
    to_local = lambda p: np.array([c * (p[0] - base[0]) + s_ * (p[1] - base[1]),
                                   -s_ * (p[0] - base[0]) + c * (p[1] - base[1])])
    to_world = lambda q: base + np.array([c * q[0] - s_ * q[1], s_ * q[0] + c * q[1]])
    probe = World(scene, prepare_bodies(scene, work), mode="objects")
    try:
        body_boxes = {oid: probe.aabb(oid) for oid in scene.objects}
        support_box = probe.aabb(probe.support_id)
    finally:
        probe.close()
    groups = containment_groups(scene.objects)
    boxes = {root: (np.min([body_boxes[o][0] for o in members], axis=0),
                    np.max([body_boxes[o][1] for o in members], axis=0))
             for root, members in groups.items()}
    centres = {oid: to_local((lo[:2] + hi[:2]) / 2) for oid, (lo, hi) in boxes.items()}
    half = {oid: (hi[:2] - lo[:2]) / 2 for oid, (lo, hi) in boxes.items()}
    mean = np.mean(list(centres.values()), axis=0)
    rel = {oid: v - mean for oid, v in centres.items()}
    spread = np.max(np.abs(np.array(list(rel.values()))), axis=0) if len(rel) > 1 else np.zeros(2)
    k = min(1.0, DROID_MAX_DEPTH / max(spread[0], 1e-6), DROID_MAX_LATERAL / max(spread[1], 1e-6))
    ids = list(rel)
    # never compress so far that footprints overlap
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = np.abs(rel[ids[i]] - rel[ids[j]])
            need = half[ids[i]] + half[ids[j]] + DROID_MIN_GAP_M
            sep_ok = [need[a] / d[a] for a in range(2) if d[a] > 1e-6]
            if sep_ok:
                k = max(k, min(1.0, min(sep_ok)))
    target = np.asarray(DROID_WORKSPACE_CENTER, dtype=float)
    # every footprint must stay on the support top (a deep workspace target can push
    # objects past a shallow table edge, where they would fall)
    s_lo, s_hi = support_box
    margin = 0.03
    for _ in range(3):
        corners = []
        for oid in ids:
            c_w = to_world(target + k * rel[oid])
            corners += [c_w - half[oid], c_w + half[oid]]
        lo_w, hi_w = np.min(corners, axis=0), np.max(corners, axis=0)
        shift_w = np.maximum(s_lo[:2] + margin - lo_w, 0) - np.maximum(hi_w - (s_hi[:2] - margin), 0)
        if np.allclose(shift_w, 0):
            break
        # world-frame shift -> robot frame
        target = target + np.array([c * shift_w[0] + s_ * shift_w[1], -s_ * shift_w[0] + c * shift_w[1]])
    else:
        raise RuntimeError("task objects do not fit on the support inside the DROID workspace")
    objects = dict(scene.objects)
    moves = {}
    for oid in ids:
        new_centre = to_world(target + k * rel[oid])
        old_centre = to_world(centres[oid])
        delta = new_centre - old_centre
        # Translate the whole containment group, including its vertical lift.
        lift = max(0.0, float(s_hi[2] - boxes[oid][0][2]))
        for member in groups[oid]:
            o = objects[member]
            objects[member] = replace(o, position=(o.position[0] + delta[0],
                                                   o.position[1] + delta[1],
                                                   o.position[2] + lift))
            moves[member] = {"from_robot_xy": centres[oid].round(3).tolist(),
                             "to_robot_xy": (target + k * rel[oid]).round(3).tolist()}
            if len(groups[oid]) > 1:
                moves[member]["containment_group"] = oid
    return replace(scene, objects=objects), {"policy": "droid_canonical_v1", "compression": k,
                                             "moves": moves}


# DROID-typical size envelope for destinations/containers (bowls, bins, trays, pads)
DEST_MAX_FOOTPRINT_M = 0.30
DEST_MAX_HEIGHT_M = 0.20


def destination_scale(scene: SceneInputs, object_id: str, work: Path) -> float:
    """Scale bringing a destination into the DROID size envelope (<= 1)."""
    from ..scene import _quat_mat
    from ..tasks.geometry_contract import collision_bounds

    obj = scene.objects[object_id]
    # SDF geometry already includes every encoded scale; obj.scale is lineage.
    # Bullet broadphase bounds also include margins and cannot measure asset size.
    lo, hi = collision_bounds(obj.sdf_path, rotation=_quat_mat(obj.orientation_xyzw))
    size = hi - lo
    return min(1.0, DEST_MAX_FOOTPRINT_M / max(size[0], size[1]), DEST_MAX_HEIGHT_M / size[2])


def fit_destination(scene: SceneInputs, object_id: str, work: Path) -> tuple[SceneInputs, dict]:
    """Uniformly shrink an oversized destination into the DROID size envelope."""
    probe = World(scene, prepare_bodies(scene, work), mode="objects", include=[object_id])
    try:
        lo, hi = probe.aabb(object_id)
    finally:
        probe.close()
    size = hi - lo
    k = min(1.0, DEST_MAX_FOOTPRINT_M / max(size[0], size[1]), DEST_MAX_HEIGHT_M / size[2])
    info = {"object_id": object_id, "size_m": size.round(3).tolist(), "scale": round(k, 3)}
    if k >= 0.999:
        return scene, info
    return scaled_scene(scene, object_id, round(k, 3), work), info
