"""Collision-derived penetration depths for slender grasp features."""
import math
import numpy as np
import pybullet as p
import pytest
from wo.expert import grasps as G


class World:
    def __init__(self, yaw):
        self.p = p
        self.cid = p.connect(p.DIRECT)
        shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=.004,
                    height=.08, collisionFrameOrientation=p.getQuaternionFromEuler([0, math.pi/2, 0]),
                    physicsClientId=self.cid)
        self.body = p.createMultiBody(.05, shape, basePosition=[0, 0, .2],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=self.cid)
        self.bodies = {'subject': self.body}

    def aabb(self, oid):
        return tuple(np.array(v) for v in p.getAABB(self.body, physicsClientId=self.cid))

    def object_pose(self, oid):
        return p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)


@pytest.mark.parametrize('yaw', [0., math.pi-.039, math.pi/2])
def test_slender_bar_has_certified_contacts(yaw):
    w = World(yaw)
    try:
        assert G.sample_grasps(w, 'subject')
    finally:
        p.disconnect(w.cid)


def test_existing_success_never_requests_adaptive_depths(monkeypatch):
    w = World(0.)
    try:
        # A thicker bar already has fixed-depth candidates.
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.015,.02,.02], physicsClientId=w.cid)
        p.removeBody(w.body, physicsClientId=w.cid)
        w.body = p.createMultiBody(.05, shape, basePosition=[0,0,.2], physicsClientId=w.cid)
        w.bodies['subject'] = w.body
        original = G._candidates
        def checked(*args, **kwargs):
            assert not kwargs.get('adaptive_depths')
            return original(*args, **kwargs)
        monkeypatch.setattr(G, '_candidates', checked)
        assert G.sample_grasps(w, 'subject')
    finally:
        p.disconnect(w.cid)
