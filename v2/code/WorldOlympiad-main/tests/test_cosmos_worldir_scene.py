from __future__ import annotations

from autoresearch.cosmos_worldir_scene import resolve_worldir_scene


def _state(role: str, x: float) -> dict:
    return {
        "role": role,
        "link_position": [x, 0.0, 0.5],
        "link_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_legacy_body_names_bind_by_role_and_keep_semantic_ids() -> None:
    graph = {
        "scene_generation": {"objects": [
            {"object_id": "mug", "role": "manipuland", "category": "mug"},
            {"object_id": "rack_hook", "role": "destination", "category": "hook"},
            {"object_id": "wooden_table", "role": "support", "category": "table"},
        ]},
    }
    trajectory = {
        "prepared_assets": {"cup": {}, "tray": {}, "table": {}},
        "frames": [{"objects": {
            "cup": _state("manipuland", -0.2),
            "tray": _state("destination", 0.2),
        }}],
    }

    scene = resolve_worldir_scene(task_graph=graph, trajectory=trajectory)

    assert [(item.body_name, item.task_object_id, item.asset_key) for item in scene] == [
        ("cup", "mug", "cup"),
        ("tray", "rack_hook", "tray"),
        ("wooden_table", "wooden_table", "table"),
    ]
    assert [item.dynamic for item in scene] == [True, False, False]


def test_multibody_scene_retains_every_object_and_maps_static_support() -> None:
    graph = {
        "scene_generation": {"objects": [
            {"object_id": "small_container", "role": "support", "category": "container"},
            {"object_id": "rigid_contents", "role": "manipuland", "category": "cubes"},
            {"object_id": "large_bin", "role": "destination", "category": "bin"},
            {"object_id": "table_support", "role": "support", "category": "table"},
        ]},
        "controller_program": [
            {"operator": "pick", "subjects": ["small_container"]},
            {"operator": "pour", "subjects": ["small_container"], "target": "large_bin"},
        ],
    }
    trajectory = {
        "prepared_assets": {
            "small_container": {}, "rigid_contents": {},
            "large_bin": {}, "table": {},
        },
        "frames": [{"objects": {
            "small_container": _state("manipuland", -0.2),
            "rigid_contents": _state("manipuland", -0.1),
            "large_bin": _state("destination", 0.2),
        }}],
    }

    scene = resolve_worldir_scene(task_graph=graph, trajectory=trajectory)

    assert [item.body_name for item in scene] == [
        "cup", "rigid_contents", "tray", "table_support",
    ]
    assert [item.asset_key for item in scene] == [
        "small_container", "rigid_contents", "large_bin", "table",
    ]
    assert all(item.pose is not None for item in scene[:3])
    assert scene[-1].pose is None


def test_scene_resolution_rejects_ambiguous_role_alias() -> None:
    graph = {
        "scene_generation": {"objects": [
            {"object_id": "first", "role": "manipuland"},
            {"object_id": "second", "role": "manipuland"},
        ]}
    }
    trajectory = {
        "prepared_assets": {"cup": {}},
        "frames": [{"objects": {"cup": _state("manipuland", 0.0)}}],
    }

    try:
        resolve_worldir_scene(task_graph=graph, trajectory=trajectory)
    except ValueError as error:
        assert "no task-graph identity" in str(error)
    else:
        raise AssertionError("ambiguous semantic aliases must be rejected")


def test_reused_asset_bytes_do_not_merge_distinct_declared_objects():
    asset = {"source_sha256": "same-visual", "physics_sha256": "same-physics"}
    graph = {"scene_generation": {"objects": [
        {"object_id": "item", "category": "bottle", "role": "manipuland"},
        {"object_id": "new_surface", "category": "table_region", "role": "destination"},
        {"object_id": "new_host", "category": "table", "role": "support"}]}}
    trajectory = {"prepared_assets": {"cup": {}, "tray": asset, "table": asset},
                  "frames": [{"objects": {"cup": _state("manipuland", 0), "tray": _state("destination", 1)}}]}
    scene = resolve_worldir_scene(task_graph=graph, trajectory=trajectory)
    assert len(scene) == 3
    assert scene[2].task_object_id == "new_host"
    assert scene[1].pose["position"] == [1, 0., .5]
