"""Confined-space pad rays preserve full approach checks and legacy ranking."""
import numpy as np
import pybullet as p
from wo.expert import grasps as G


class World:
    def __init__(self, half_width=.015):
        self.cid = p.connect(p.DIRECT)
        self.p = p
        self.body = self.box([half_width, .02, .02], [0, 0, .2])
        self.bodies = {'subject': self.body}

    def box(self, half, pos):
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        return p.createMultiBody(.05, shape, basePosition=pos, physicsClientId=self.cid)

    def aabb(self, oid):
        return tuple(np.array(v) for v in p.getAABB(self.bodies[oid], physicsClientId=self.cid))

    def object_pose(self, oid):
        return p.getBasePositionAndOrientation(self.bodies[oid], physicsClientId=self.cid)

    def grasp(self, reach=G.PAD_HALF_OPEN_M):
        return G._antipodal(self, self.body, np.array([0., 0., .2]),
                           G._frame(np.array([1., 0., 0.]), np.array([0., 0., -1.])),
                           contact_reach=reach)


def test_container_outside_fingers_does_not_occlude_pad_contact():
    w = World()
    try:
        for sign in (-1, 1):
            w.box([.004, .1, .08], [sign * .065, 0, .23])
        assert w.grasp(G.REACH_M) is None
        assert abs(w.grasp()[0] - .03) < 1e-6
        assert G.sample_grasps(w, 'subject')
    finally:
        p.disconnect(w.cid)


def test_wall_in_finger_approach_rejects():
    w = World()
    try:
        w.box([.004, .1, .08], [G.PAD_HALF_OPEN_M + .006, 0, .23])
        assert w.grasp() is None
    finally:
        p.disconnect(w.cid)


def test_obstacle_between_pad_and_subject_rejects():
    w = World()
    try:
        w.box([.003, .1, .02], [.03, 0, .2])
        assert w.grasp() is None
    finally:
        p.disconnect(w.cid)


def test_subject_wider_than_open_jaws_rejects():
    w = World(half_width=.05)
    try:
        assert w.grasp() is None
    finally:
        p.disconnect(w.cid)


def test_narrow_aperture_certifies_contacts_without_open_finger_sweep():
    w = World(half_width=.015)
    try:
        # Full-open swept fingers intersect these walls. Narrow fingers fit.
        for sign in (-1, 1):
            w.box([.004, .1, .08], [sign * .0475, 0, .23])
        assert w.grasp() is None
        assert G.sample_grasps(w, 'subject', aperture_sampling=True)
    finally:
        p.disconnect(w.cid)


def test_narrow_aperture_keeps_wall_between_pad_and_subject_blocking():
    w = World(half_width=.015)
    try:
        w.box([.004, .1, .08], [.019, 0, .23])
        R = G._frame(np.array([1., 0., 0.]), np.array([0., 0., -1.]))
        assert G._antipodal(w, w.body, np.array([0., 0., .2]), R,
                            contact_reach=.026, jaw_half=.026) is None
    finally:
        p.disconnect(w.cid)


def test_existing_grasps_do_not_use_bounded_fallback(monkeypatch):
    w = World()
    calls = []
    original = G._antipodal
    def record(*args, **kwargs):
        calls.append(kwargs['contact_reach'])
        return original(*args, **kwargs)
    monkeypatch.setattr(G, '_antipodal', record)
    try:
        assert G.sample_grasps(w, 'subject')
        assert set(calls) == {G.REACH_M}
    finally:
        p.disconnect(w.cid)
