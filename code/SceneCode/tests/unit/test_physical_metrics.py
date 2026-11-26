"""Tests for COL/STB physical metric helpers."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from pydrake.all import RigidTransform

from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenecode.metrics.COL import compute_scene_collision_rate
from scenecode.metrics.STB import compute_scene_static_equilibrium


TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
BOX_SDF = TEST_DATA_DIR / "simple_box.sdf"
ROOM_SDF = TEST_DATA_DIR / "simple_room_geometry.sdf"


def _box(
    object_id: str,
    xyz: tuple[float, float, float],
    object_type: ObjectType = ObjectType.FURNITURE,
) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=object_type,
        name=object_id,
        description=object_id,
        transform=RigidTransform(np.array(xyz)),
        sdf_path=BOX_SDF,
    )


def _scene(room_geometry: bool = False) -> RoomScene:
    if not room_geometry:
        return RoomScene(room_geometry=None, scene_dir=TEST_DATA_DIR)
    geometry = RoomGeometry(sdf_tree=ET.parse(ROOM_SDF), sdf_path=ROOM_SDF)
    return RoomScene(room_geometry=geometry, scene_dir=TEST_DATA_DIR)


def test_col_no_collision_for_separated_boxes():
    scene = _scene()
    scene.add_object(_box("box_a", (0.0, 0.0, 0.25)))
    scene.add_object(_box("box_b", (2.0, 0.0, 0.25)))

    colliding_ids, pairs, num_evaluated, skipped = compute_scene_collision_rate(scene)

    assert num_evaluated == 2
    assert colliding_ids == set()
    assert pairs == []
    assert skipped == []


def test_col_counts_each_object_once_for_deep_collision():
    scene = _scene()
    scene.add_object(_box("box_a", (0.0, 0.0, 0.25)))
    scene.add_object(_box("box_b", (0.25, 0.0, 0.25)))

    colliding_ids, pairs, num_evaluated, _ = compute_scene_collision_rate(scene)

    assert num_evaluated == 2
    assert colliding_ids == {"box_a", "box_b"}
    assert len(pairs) == 1
    assert pairs[0].penetration_m > 0.001


def test_col_ignores_penetration_not_exceeding_one_mm():
    scene = _scene()
    scene.add_object(_box("box_a", (0.0, 0.0, 0.25)))
    # Box side length is 0.5m, so this overlaps by 0.5mm.
    scene.add_object(_box("box_b", (0.4995, 0.0, 0.25)))

    colliding_ids, pairs, num_evaluated, _ = compute_scene_collision_rate(scene)

    assert num_evaluated == 2
    assert colliding_ids == set()
    assert pairs == []


def test_stb_ground_box_is_stable_on_floor():
    scene = _scene(room_geometry=True)
    scene.add_object(_box("box_a", (0.0, 0.0, 0.25)))

    details, skipped = compute_scene_static_equilibrium(
        scene,
        support_labels={},
        simulation_time_s=0.2,
        time_step_s=0.001,
    )

    assert skipped == []
    assert len(details) == 1
    assert details[0].stable
    assert details[0].support_type == "ground"


def test_stb_suspended_ground_box_is_unstable():
    scene = _scene(room_geometry=True)
    scene.add_object(_box("box_a", (0.0, 0.0, 2.0)))

    details, _ = compute_scene_static_equilibrium(
        scene,
        support_labels={},
        simulation_time_s=0.5,
        time_step_s=0.001,
    )

    assert len(details) == 1
    assert not details[0].stable
    assert details[0].displacement_m > 0.01


def test_stb_wall_and_ceiling_objects_are_welded_stable():
    scene = _scene(room_geometry=True)
    scene.add_object(_box("wall_art", (0.0, 4.8, 1.5), ObjectType.WALL_MOUNTED))
    scene.add_object(_box("ceiling_light", (0.0, 0.0, 2.5), ObjectType.CEILING_MOUNTED))

    details, skipped = compute_scene_static_equilibrium(
        scene,
        support_labels={},
        simulation_time_s=0.2,
        time_step_s=0.001,
    )

    assert skipped == []
    assert len(details) == 2
    assert all(detail.stable for detail in details)
    assert all(detail.welded for detail in details)
    assert {detail.support_type for detail in details} == {"wall", "ceiling"}
