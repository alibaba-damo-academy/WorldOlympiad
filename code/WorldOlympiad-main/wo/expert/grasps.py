"""Antipodal parallel-jaw grasp sampling by ray casting the collision geometry.

A candidate is a TCP frame (columns: closing axis x, y = z cross x, approach z).
For each candidate we cast 3x3 pad rays along the closing axis from both sides.
Pads must hit the subject first (no other body in between), the jaw width must
fit the open gripper, and the hit normals must oppose the closing axis.  Surface
contact (many consistent pad rays) scores higher than line contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.world import World
from .kin import MAX_OPENING_M, Pose

PAD_HALF_OPEN_M = 0.0415
# A parallel-jaw pinch (line contact on curved parts) cannot resist much torque
# about the closing axis: gravity times the horizontal COM lever arm must stay
# below this, otherwise the object pivots out of the grasp when lifted.
MAX_HOLD_TORQUE_NM = 0.30
NORMAL_COS = 0.70
PAD_OFFSETS = [(u, v) for u in (-0.010, 0.0, 0.010) for v in (-0.008, 0.0, 0.008)]


@dataclass
class Grasp:
    tcp_in_obj: Pose  # TCP pose in the subject link frame
    width: float
    score: float
    kind: str  # top | side

    def world(self, obj_pose: Pose) -> Pose:
        return obj_pose @ self.tcp_in_obj


def approach_fraction(w: World, width: float, clearance: float = 0.006) -> float:
    """Largest closure retaining a measured collision-geometry jaw gap.

    Use the actual URDF finger surfaces, not a linear width/angle assumption.
    Conservative minimum surface distance also accounts for finger thickness.
    Restore the planning state after the bounded monotone search.
    """
    previous = w.gripper_fraction()
    def gap(fraction):
        w.set_gripper_state(fraction)
        points = w.p.getClosestPoints(w.robot, w.robot, 0.2,
                    linkIndexA=w.fingers[0], linkIndexB=w.fingers[1],
                    physicsClientId=w.cid)
        return min((pt[8] for pt in points), default=0.0)
    try:
        required = width + clearance
        if gap(0.0) < required:
            return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(14):
            mid = (lo + hi) / 2
            if gap(mid) >= required:
                lo = mid
            else:
                hi = mid
        return lo
    finally:
        w.set_gripper_state(previous)


def _frame(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    x = x / np.linalg.norm(x)
    a = a - x * (a @ x)
    a = a / np.linalg.norm(a)
    y = np.cross(a, x)
    return np.stack([x, y, a], axis=1)


REACH_M = 0.10  # ray start, outside the open pads (+-0.0415) and finger thickness


def _finger_sweep(base: np.ndarray, approach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the collision-check segment for one finger.

    A jaw approaches the TCP, not the support below a thin object.  Keeping
    this as a small helper makes the distinction explicit and testable.
    """
    return base - 0.14 * approach, base


def _antipodal(w: World, body: int, c: np.ndarray, R: np.ndarray, links=None,
               *, contact_reach: float = REACH_M, jaw_half: float = PAD_HALF_OPEN_M):
    x, y, a = R[:, 0], R[:, 1], R[:, 2]
    starts, ends = [], []
    for u, v in PAD_OFFSETS:
        o = c + u * y + v * a
        starts += [o + contact_reach * x, o - contact_reach * x]
        ends += [o - contact_reach * x, o + contact_reach * x]
    # the centre-line start points must lie outside the subject, otherwise the
    # finger would sit inside material (rays starting inside a hull see no face)
    o = c
    starts += [o + contact_reach * x, o - contact_reach * x]
    ends += [o + 4 * REACH_M * x, o - 4 * REACH_M * x]
    # open fingers sweep along the approach axis: nothing may lie in their path
    sweep = []
    for side in (1.0, -1.0):
        for u in (-0.012, 0.0, 0.012):
            base = c + side * (jaw_half + 0.006) * x + u * y
            # The fingers approach the grasp centre along ``a``.  Extending the
            # sweep past the TCP is incorrect for thin objects resting on a
            # support: it reports the support/table as an obstruction even
            # though the real parallel jaws stop at the object mid-plane.
            # Check the approach only up to the TCP; the pad rays below still
            # certify the actual antipodal contact and jaw clearance.
            sweep.append(_finger_sweep(base, a))
    starts += [s for s, _ in sweep]
    ends += [e for _, e in sweep]
    hits = w.p.rayTestBatch(starts, ends, physicsClientId=w.cid)
    if any(h[0] >= 0 for h in hits[-len(sweep):]):
        return None
    hits = hits[: -len(sweep)]
    if hits[-1][0] == body or hits[-2][0] == body:
        return None
    widths, mids, ok_normals = [], [], []
    center_ok = False
    for k in range(len(PAD_OFFSETS)):
        hp, hm = hits[2 * k], hits[2 * k + 1]
        if hp[0] != body or hm[0] != body or (
            links is not None and (hp[1] not in links or hm[1] not in links)
        ):
            if k == 4:
                return None  # centre pad blocked or missing the subject
            continue
        pp, pm = np.asarray(hp[3]), np.asarray(hm[3])
        # both surfaces must lie inside the open jaw around the TCP line
        if (pp - c) @ x > jaw_half - 0.003 or (c - pm) @ x > jaw_half - 0.003:
            if k == 4:
                return None
            continue
        width = float((pp - pm) @ x)
        n_ok = (np.asarray(hp[4]) @ x >= NORMAL_COS) and (np.asarray(hm[4]) @ -x >= NORMAL_COS)
        widths.append(width)
        mids.append(float(((pp + pm) / 2 - c) @ x))
        ok_normals.append(n_ok)
        if k == 4:
            center_ok = n_ok and 0.004 < width <= MAX_OPENING_M - 0.004
            cw, cm = width, mids[-1]
    if not center_ok:
        return None
    # enveloping: no subject material on the closing line outside the jaw width
    # (rejects pinching an internal part or a wall while the outer shell is wider)
    far = w.p.rayTestBatch([(c + 0.4 * x).tolist(), (c - 0.4 * x).tolist()],
                           [(c - 0.4 * x).tolist(), (c + 0.4 * x).tolist()],
                           physicsClientId=w.cid)
    if far[0][0] == body and far[1][0] == body:
        outer = float((np.asarray(far[0][3]) - np.asarray(far[1][3])) @ x)
        if outer > cw + 0.006:
            return None
    consistent = sum(1 for wd, n in zip(widths, ok_normals) if n and abs(wd - cw) < 0.012)
    return cw, cm, consistent / len(PAD_OFFSETS)


def _candidates(w: World, body: int, lo: np.ndarray, hi: np.ndarray, links,
                *, tilted_top: bool = False, adaptive_depths: bool = False):
    size = hi - lo
    # Fixed centimetre penetrations can pass entirely through slender handles.
    # Supplement them from the sampled region's measured collision thickness.
    shallow = (float(np.min(size)) / 4, float(np.min(size)) / 2) if adaptive_depths else ()
    cands: list[tuple[np.ndarray, np.ndarray, str]] = []
    # top-down: grid over the footprint, several depths under the local top surface
    step = max(0.01, min(size[0], size[1]) / 6)
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    pts = [(x, y) for x in xs for y in ys]
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    pts += [(cx + dx, cy + dy) for dx in (-0.01, 0, 0.01) for dy in (-0.01, 0, 0.01)]
    starts = [(x, y, hi[2] + 0.05) for x, y in pts]
    ends = [(x, y, lo[2] - 0.01) for x, y in pts]
    tops = w.p.rayTestBatch(starts, ends, physicsClientId=w.cid) if starts else []
    for h in tops:
        if h[0] != body or (links is not None and h[1] not in links):
            continue
        top = np.asarray(h[3])
        for depth in (0.012, 0.025, 0.040) + shallow:
            c = top - np.array([0, 0, depth])
            if c[2] < lo[2] + 0.004:
                continue
            for k in range(12):
                psi = math.pi * k / 12
                R = _frame(np.array([math.cos(psi), math.sin(psi), 0.0]), np.array([0, 0, -1.0]))
                if not tilted_top:
                    cands.append((c, R, "top"))
                else:
                    # Keep the closing axis horizontal so the same antipodal
                    # surface contacts remain available. Tilt about that axis
                    # in both directions to move the wrist clear of a rim.
                    # Full pad/finger sweeps and planner checks still certify
                    # each candidate; a clear ray is not a robot clearance proof.
                    x = R[:, 0]
                    transverse = np.cross(x, np.array([0., 0., -1.]))
                    for degrees in (-45., -25., 25., 45.):
                        theta = math.radians(degrees)
                        a = np.array([0., 0., -math.cos(theta)]) + transverse * math.sin(theta)
                        cands.append((c, _frame(x, a), "tilted_top"))
    if tilted_top:
        return cands
    # side and angled approaches: azimuth x elevation, lines through the region's axis
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(size)) / 2 + 0.05
    for elev_deg in (0.0, 30.0, 50.0, 70.0):
        e = math.radians(elev_deg)
        for frac in np.linspace(0.15, 0.95, 9):
            z = lo[2] + frac * size[2]
            p0 = np.array([centre[0], centre[1], z])
            for k in range(12):
                phi = 2 * math.pi * k / 12
                a = np.array([-math.cos(phi) * math.cos(e), -math.sin(phi) * math.cos(e), -math.sin(e)])
                hit = w.p.rayTest((p0 - radius * a).tolist(), (p0 + radius * a).tolist(),
                                  physicsClientId=w.cid)[0]
                if hit[0] != body or (links is not None and hit[1] not in links):
                    continue
                entry = np.asarray(hit[3])
                kind = "side" if elev_deg == 0.0 else "angled"
                horiz = np.array([-math.sin(phi), math.cos(phi), 0.0])
                closings = [horiz] if elev_deg > 0 else [horiz, np.array([0.0, 0.0, 1.0])]
                for x in closings:
                    for depth in (0.015, 0.03, 0.045) + shallow:
                        cands.append((entry + a * depth, _frame(x, a), kind))
    return cands


def _mass_centre(w: World, body: int) -> np.ndarray:
    """Mass-weighted centre of all links (world frame)."""
    p, c = w.p, w.cid
    total, acc = 0.0, np.zeros(3)
    m0 = p.getDynamicsInfo(body, -1, physicsClientId=c)[0]
    acc += m0 * np.asarray(p.getBasePositionAndOrientation(body, physicsClientId=c)[0]); total += m0
    for li in range(p.getNumJoints(body, physicsClientId=c)):
        m = p.getDynamicsInfo(body, li, physicsClientId=c)[0]
        acc += m * np.asarray(p.getLinkState(body, li, physicsClientId=c)[0]); total += m
    return acc / max(total, 1e-9)


def sample_grasps(w: World, object_id: str, *, max_out: int = 64, links=None,
                  frame: Pose | None = None, prefer_links=None,
                  _contact_reach: float = REACH_M,
                  tilted_top: bool = False, aperture_sampling: bool = False,
                  _jaw_half: float = PAD_HALF_OPEN_M,
                  _adaptive_depths: bool = False) -> list[Grasp]:
    """Grasps on ``object_id`` (optionally only on ``links``), expressed in ``frame``
    (default: the object's link frame).  With ``links`` every link is sampled around
    its own bounding box, so small attached features (handles) are found;
    ``prefer_links`` get a score bonus."""
    if aperture_sampling:
        # A narrowed approach needs narrowed pad origins AND swept fingers.
        # This is opt-in after failed ordinary planning; never change legacy
        # rankings. These rays certify contacts only: the planner still checks
        # the actual URDF at its measured width+clearance aperture.
        candidates = []
        for half in np.linspace(PAD_HALF_OPEN_M / 2, PAD_HALF_OPEN_M, 5):
            candidates.extend(sample_grasps(
                w, object_id, max_out=max_out, links=links, frame=frame,
                prefer_links=prefer_links, tilted_top=tilted_top,
                _contact_reach=float(half), _jaw_half=float(half),
                _adaptive_depths=_adaptive_depths))
        candidates.sort(key=lambda g: -g.score)
        kept = []
        for g in candidates:
            if all(np.linalg.norm(g.tcp_in_obj.p - other.tcp_in_obj.p) > 0.006
                   or abs(g.tcp_in_obj.R[:, 2] @ other.tcp_in_obj.R[:, 2]) < 0.95
                   or abs(g.tcp_in_obj.R[:, 0] @ other.tcp_in_obj.R[:, 0]) < 0.95
                   for other in kept):
                kept.append(g)
            if len(kept) >= max_out:
                break
        return kept
    body = w.bodies[object_id]
    regions = []
    if links is None:
        regions.append(w.aabb(object_id))
    else:
        for li in sorted(links):
            b = w.p.getAABB(body, li, physicsClientId=w.cid)
            regions.append((np.asarray(b[0]), np.asarray(b[1])))
    lo = np.min([r[0] for r in regions], axis=0)
    hi = np.max([r[1] for r in regions], axis=0)
    size = hi - lo
    obj_pose = frame if frame is not None else Pose.from_pq(*w.object_pose(object_id))
    com = (lo + hi) / 2 if links is not None else np.asarray(
        w.p.getBasePositionAndOrientation(body, physicsClientId=w.cid)[0])
    cands = []
    for r_lo, r_hi in regions:
        cands += _candidates(w, body, r_lo, r_hi, links, tilted_top=tilted_top,
                             adaptive_depths=_adaptive_depths)
    # free objects only: an articulated host's moving part is carried by its joint
    mass = None
    if links is None:
        mass = sum(w.p.getDynamicsInfo(body, li, physicsClientId=w.cid)[0]
                   for li in range(-1, w.p.getNumJoints(body, physicsClientId=w.cid)))
        com_xy = _mass_centre(w, body)[:2]

    out: list[Grasp] = []
    span = float(np.linalg.norm(size)) + 1e-6
    for c, R, kind in cands:
        res = _antipodal(w, body, c, R, links, contact_reach=_contact_reach, jaw_half=_jaw_half)
        if res is None:
            continue
        width, mid, contact = res
        c2 = c + R[:, 0] * mid  # centre the jaws on the object
        if width / 2 + 0.003 > PAD_HALF_OPEN_M:
            continue
        if mass is not None:
            lever = float(np.linalg.norm(c2[:2] - com_xy))
            if mass * 9.81 * lever > MAX_HOLD_TORQUE_NM:
                continue
        com_dist = float(np.linalg.norm(c2 - com)) / span
        score = 1.0 * contact + 0.6 * (1 - com_dist) + (0.15 if kind == "top" else 0.0)
        if prefer_links:
            h = w.p.rayTest((c2 + 0.1 * R[:, 0]).tolist(), c2.tolist(), physicsClientId=w.cid)[0]
            if h[0] == body and h[1] in prefer_links:
                score += 1.0
        tcp_world = Pose(c2, R)
        out.append(Grasp(obj_pose.inv() @ tcp_world, width, score, kind))
    if not out and _contact_reach == REACH_M:
        # Long probes are a conservative first pass. In confined spaces they
        # can hit a container outside the actual open pads. Retry from the pad
        # origins only when the conservative pass found nothing; do not change
        # existing feasible grasp rankings. All finger sweeps, envelope, normal,
        # link, width and torque checks remain identical.
        retry = sample_grasps(w, object_id, max_out=max_out, links=links,
                             frame=frame, prefer_links=prefer_links,
                             _contact_reach=PAD_HALF_OPEN_M, tilted_top=tilted_top,
                             _adaptive_depths=_adaptive_depths)
        if retry:
            return retry
    if not out and not _adaptive_depths:
        return sample_grasps(w, object_id, max_out=max_out, links=links,
                             frame=frame, prefer_links=prefer_links,
                             _contact_reach=_contact_reach, _jaw_half=_jaw_half,
                             tilted_top=tilted_top, _adaptive_depths=True)
    out.sort(key=lambda g: -g.score)
    # de-duplicate nearby candidates
    kept: list[Grasp] = []
    for g in out:
        if all(np.linalg.norm(g.tcp_in_obj.p - k.tcp_in_obj.p) > 0.006
               or abs(g.tcp_in_obj.R[:, 2] @ k.tcp_in_obj.R[:, 2]) < 0.95
               or abs(g.tcp_in_obj.R[:, 0] @ k.tcp_in_obj.R[:, 0]) < 0.95 for k in kept):
            kept.append(g)
        if len(kept) >= max_out:
            break
    return kept
