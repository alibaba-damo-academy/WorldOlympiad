from types import SimpleNamespace
import numpy as np
import pytest
from wo.expert.goals import _touching_any, sweep_down


@pytest.mark.parametrize("normal,gap,blocked", [
    ((1, 0, 0), .0005, False),
    ((1, 0, 0), -.0001, True),
    ((0, 0, 1), .001, True),
    ((0, 0, -1), .001, False),
    ((.8, 0, .6), .001, False),
    ((.8, 0, .6), .0008, True),
])
def test_downward_guard_uses_projected_clearance_and_keeps_collisions(normal, gap, blocked):
    cp = (None,)*7 + (normal, gap)
    p = SimpleNamespace(performCollisionDetection=lambda **kw: None,
                        getClosestPoints=lambda *a, **kw: [cp])
    world = SimpleNamespace(p=p, cid=0, bodies={"object":1,"target":2}, plane=3)
    assert _touching_any(world, 1, set()) is blocked


def test_actual_bullet_narrow_slot_reaches_floor_without_wall_penetration():
    import pybullet as p
    cid = p.connect(p.DIRECT)
    try:
        def box(half, position):
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
            return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                     basePosition=position, physicsClientId=cid)
        floor = box([.2,.2,.01], [0,0,-.01])
        left = box([.01,.1,.1], [-.036,0,.1])
        right = box([.01,.1,.1], [.036,0,.1])
        obj = box([.025]*3, [0,0,.3])
        world = SimpleNamespace(p=p,cid=cid,bodies={"object":obj,"left":left,"right":right,"floor":floor},plane=floor,
             set_link_pose=lambda body, pos, orn: p.resetBasePositionAndOrientation(body,pos,orn,physicsClientId=cid))
        # Side clearance is 1 mm: smaller than the unchanged 1.5 mm forward guard.
        z = sweep_down(world,"object",[0,0],np.eye(3),.3,-.05)
        assert z == pytest.approx(.0265, abs=.00005)
        world.set_link_pose(obj,[0,0,z],[0,0,0,1])
        assert p.getClosestPoints(obj,left,.0015,physicsClientId=cid)
        assert not p.getClosestPoints(obj,left,0,physicsClientId=cid)
        assert not p.getClosestPoints(obj,right,0,physicsClientId=cid)
        assert not p.getClosestPoints(obj,floor,0,physicsClientId=cid)
    finally:
        p.disconnect(cid)
