import numpy as np
import pybullet as p

from wo.expert import grasps as G


class _W:
    """Minimal world: one free box body on a plane."""
    def __init__(self, half, mass):
        self.cid = p.connect(p.DIRECT)
        self.p = p
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        self.body = p.createMultiBody(mass, col, basePosition=[0, 0, half[2]], physicsClientId=self.cid)
        self.bodies = {"o": self.body}
    def aabb(self, oid):
        lo, hi = p.getAABB(self.body, physicsClientId=self.cid)
        return np.asarray(lo), np.asarray(hi)
    def object_pose(self, oid):
        return p.getBasePositionAndOrientation(self.body, physicsClientId=self.cid)


def _levers(w, grasps):
    com = np.asarray(p.getBasePositionAndOrientation(w.body, physicsClientId=w.cid)[0])
    pose = G.Pose.from_pq(*w.object_pose("o"))
    return [float(np.linalg.norm(g.world(pose).p[:2] - com[:2])) for g in grasps]


def test_heavy_long_object_only_grasped_near_com():
    w = _W([0.03, 0.20, 0.03], mass=0.8)  # 40 cm long bar, 6 cm thick
    gs = G.sample_grasps(w, "o")
    assert gs, "a heavy bar must still be graspable near its middle"
    assert max(_levers(w, gs)) * 0.8 * 9.81 <= G.MAX_HOLD_TORQUE_NM + 1e-6
    p.disconnect(w.cid)


def test_light_object_allows_offset_grasps():
    w = _W([0.03, 0.20, 0.03], mass=0.05)
    assert max(_levers(w, G.sample_grasps(w, "o"))) > 0.1
    p.disconnect(w.cid)
