"""Placement must check the exact layout used by rollout."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from wo.expert import embodiment as e
from wo.expert.run import _droid_destinations
from wo.expert import kin
from wo import scene as scene_module


@dataclass
class Scene:
    support: object
    objects: dict
    articulations: dict
    robot_base_position: tuple = (0, 0, 0)
    robot_base_yaw: float = 0
    canonical: bool = False


def test_droid_scaling_excludes_articulated_hosts():
    step = SimpleNamespace(primitive="transport", goal=SimpleNamespace(target="cabinet"))
    spec = SimpleNamespace(steps=[step], objects=[], articulations=[])
    scene = SimpleNamespace(objects={"cabinet": object()}, articulations={"door": {"host": "cabinet"}})
    assert _droid_destinations(spec, scene) == []


def test_placement_candidates_preserves_first_search_and_separate_budget():
    canonical = [(i, (i, 0, 1), 0) for i in range(45)]
    fallback = [(i, (i, 0, 1), 0) for i in range(45)]
    result = list(e.placement_candidates(canonical, fallback))
    assert len(result) == 80
    assert all(not x[0] for x in result[:40])
    assert all(x[0] for x in result[40:])
    assert list(e.placement_candidates([canonical[0]] * 100, fallback[:1])) == [
        (False, *canonical[0]), (True, *fallback[0])]


def test_preserved_assembly_fallback_after_canonical_collision(monkeypatch, tmp_path):
    scene = Scene(SimpleNamespace(object_id='table'), {'cabinet': object()},
                  {'door': {'host': 'cabinet'}})
    layouts = []
    searches = []
    class World:
        def __init__(self, scene, *args, **kwargs):
            self.scene = scene
            self.support_id = 'table'
            self.articulation = {}
        def aabb(self, oid):
            return (np.array([2, 3, 0]), np.array([4, 5, 2])) if oid == 'cabinet' else (np.zeros(3), np.ones(3))
        def close(self):
            pass
    class Kinematics:
        home = None
        def __init__(self, world):
            self.world = world
        def in_collision(self, q):
            return 'cabinet' if self.world.scene.canonical else None
        def ik(self, pose):
            return [0]
    def candidates(lo, hi, boxes, **kwargs):
        searches.append((lo.copy(), hi.copy(), kwargs))
        return [(0, (6, 3, 1.002), 0)]
    def layout(s, w):
        layouts.append(s)
        return replace(s, canonical=True), {'policy': 'canonical'}
    monkeypatch.setattr(e, 'World', World)
    monkeypatch.setattr(e, 'prepare_bodies', lambda *args: {})
    monkeypatch.setattr(kin, 'Kinematics', Kinematics)
    monkeypatch.setattr(e, '_reach_poses', lambda *args: [])
    monkeypatch.setattr(scene_module, 'robot_base_candidates', candidates)
    monkeypatch.setattr(e, 'droid_canonical_layout', layout)
    result, info = e.place_robot(scene, tmp_path, SimpleNamespace(steps=[], objects=[], articulations=[]), layout='droid')
    assert not result.canonical and result.objects is scene.objects
    assert len(layouts) == 1 and info['tried'] == 2
    assert info['layout']['policy'] == 'preserve_articulated_scene'
    assert np.array_equal(searches[1][0], [2, 3, 0])
    assert np.array_equal(searches[1][1], [4, 5, 1])
    assert searches[1][2]['outside_only']
    assert info['mount']['height_m'] == 1


@pytest.mark.parametrize("host", [None, "table", "cube"])
def test_layout_precedes_reachability(monkeypatch, tmp_path, host):
    articulated = host == 'table'
    scene = Scene(SimpleNamespace(object_id="table"), {"cube": object()},
                  {"hinge": {"host": host}} if host else {})
    seen = []
    class World:
        def __init__(self, scene, prepared, mode):
            self.scene = scene
            self.support_id = "table"
            self.articulation = {}
        def aabb(self, oid):
            return np.zeros(3), np.ones(3)
        def close(self):
            pass
    class Kinematics:
        home = None
        def __init__(self, world):
            self.world = world
        def in_collision(self, q):
            seen.append(self.world.scene.canonical)
            return None
        def ik(self, pose):
            return [0]
    monkeypatch.setattr(e, "World", World)
    monkeypatch.setattr(e, "prepare_bodies", lambda *args: {})
    monkeypatch.setattr(kin, "Kinematics", Kinematics)
    monkeypatch.setattr(e, "_reach_poses", lambda *args: [])
    monkeypatch.setattr(scene_module, "robot_base_candidates",
                        lambda *args, **kwargs: [(0, (0, 0, 1), 0)])
    monkeypatch.setattr(e, "droid_canonical_layout",
                        lambda s, w: (replace(s, canonical=True), {"policy": "canonical"}))
    result, info = e.place_robot(scene, tmp_path, SimpleNamespace(steps=[], objects=[], articulations=[]), layout="droid")
    assert seen == [not articulated]
    assert result.canonical == (not articulated)
    assert info["layout"]["policy"] == ("preserve_articulated_scene" if articulated else "canonical")


def test_failed_layout_does_not_abort_base_search(monkeypatch, tmp_path):
    scene = Scene(SimpleNamespace(object_id="table"), {"cube": object()}, {})
    class World:
        def __init__(self, *args, **kwargs):
            self.support_id = "table"
            self.articulation = {}
        def aabb(self, oid):
            return np.zeros(3), np.ones(3)
        def close(self):
            pass
    class Kinematics:
        home = None
        def __init__(self, world):
            pass
        def in_collision(self, q):
            return None
        def ik(self, pose):
            return [0]
    def layout(s, w):
        if s.robot_base_yaw == 0:
            raise RuntimeError('does not fit')
        return s, {'policy': 'canonical'}
    monkeypatch.setattr(e, 'World', World)
    monkeypatch.setattr(e, 'prepare_bodies', lambda *args: {})
    monkeypatch.setattr(kin, 'Kinematics', Kinematics)
    monkeypatch.setattr(e, '_reach_poses', lambda *args: [])
    monkeypatch.setattr(scene_module, 'robot_base_candidates',
                        lambda *args, **kwargs: [(0, (0, 0, 1), 0), (1, (0, 0, 1), 1)])
    monkeypatch.setattr(e, 'droid_canonical_layout', layout)
    result, info = e.place_robot(scene, tmp_path, SimpleNamespace(steps=[], objects=[], articulations=[]), layout='droid')
    assert result.robot_base_yaw == 1
    assert info['tried'] == 2


@pytest.mark.parametrize('failure', ['collision', 'layout'])
def test_failure_reports_all_candidates_and_untested_reach(monkeypatch, tmp_path, failure):
    import ast
    scene = Scene(SimpleNamespace(object_id='table'), {'cube': object()}, {})
    class World:
        def __init__(self, *args, **kwargs):
            self.support_id = 'table'
            self.articulation = {}
        def aabb(self, oid):
            return np.zeros(3), np.ones(3)
        def close(self):
            pass
    class Kinematics:
        home = None
        def __init__(self, world):
            pass
        def in_collision(self, q):
            return 'assembly'
        def ik(self, pose):
            raise AssertionError('IK must not run from colliding home')
    def layout(s, w):
        if failure == 'layout':
            raise RuntimeError('does not fit')
        return s, {'policy': 'canonical'}
    monkeypatch.setattr(e, 'World', World)
    monkeypatch.setattr(e, 'prepare_bodies', lambda *args: {})
    monkeypatch.setattr(kin, 'Kinematics', Kinematics)
    monkeypatch.setattr(e, '_reach_poses', lambda *args: pytest.fail('reach was not measured'))
    monkeypatch.setattr(scene_module, 'robot_base_candidates',
                        lambda *args, **kwargs: [(i, (i, 0, 1), 0) for i in range(8)])
    monkeypatch.setattr(e, 'droid_canonical_layout', layout)
    with pytest.raises(RuntimeError) as exc:
        e.place_robot(scene, tmp_path, SimpleNamespace(steps=[], objects=[], articulations=[]), layout='droid')
    message = str(exc.value)
    assert '(8 candidates)' in message
    candidates = ast.literal_eval(message.split(': ', 1)[1])
    assert len(candidates) == 8
    assert candidates[-1]['base'] == [7, 0, 1]
    assert all(c['reach'] is None for c in candidates)
    status = 'not_tested_home_collision' if failure == 'collision' else 'not_tested_layout_failed'
    assert all(c['reach_status'] == status for c in candidates)
    if failure == 'collision':
        assert all(c['layout'] == {'policy': 'canonical'} for c in candidates)
