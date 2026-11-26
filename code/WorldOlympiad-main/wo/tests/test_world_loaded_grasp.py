from types import SimpleNamespace

from wo.sim.world import World


def _cp(force, distance):
    # Bullet contact tuple fields used by World.load_bearing_grasped.
    row = [0] * 10
    row[8], row[9] = distance, force
    return tuple(row)


def test_loaded_grasp_rejects_positive_distance_proximity_contacts():
    w = World.__new__(World)
    w.finger_contacts = lambda _: {1: [_cp(0.0, 0.001)], 2: [_cp(2.0, -0.001)]}
    assert not w.load_bearing_grasped('cube')


def test_loaded_grasp_requires_both_loaded_fingers():
    w = World.__new__(World)
    w.finger_contacts = lambda _: {1: [_cp(2.0, -0.001)], 2: [_cp(1.0, 0.0)]}
    assert w.load_bearing_grasped('cube')
