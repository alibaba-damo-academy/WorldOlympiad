import json
from pathlib import Path

import pytest

from autoresearch.robotics_scene_objects import (
    carry_evidence_frame_ids,
    load_scene_object_sources,
)


def test_parallel_workers_use_independent_blender_server_locks() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "render_scenecode_robotics_multiview_parallel.py"
    ).read_text(encoding="utf-8")
    assert 'worker_env["SCENECODE_BLENDER_GLOBAL_LOCK"]' in source
    assert 'lock_root / f"{view}.lock"' in source
    assert "env=worker_env" in source
    assert "one_lock_per_dedicated_blender_server_and_physical_view" in source


def test_trajectory_renderer_uses_audited_eevee_sample_profile() -> None:
    root = Path(__file__).parents[1]
    settings = (
        root / "deployment" / "ziplab5090" / "scenecode_robotics_render_settings.py"
    ).read_text(encoding="utf-8")
    renderer = (
        root / "scripts" / "render_scenecode_robotics_multiview.py"
    ).read_text(encoding="utf-8")
    assert "scene.eevee.taa_render_samples = 16" in settings
    assert "bpy_settings_file=settings_file" in renderer


def test_scene_object_table_resolves_arbitrary_ids_and_relative_sdfs(tmp_path) -> None:
    for name in ("first.sdf", "second.sdf"):
        (tmp_path / name).write_text("<sdf version='1.7'/>", encoding="utf-8")
    manifest = tmp_path / "objects.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "robotics_scene_object_table.v1",
                "objects": [
                    {"object_id": "manipuland_alpha", "sdf_path": "first.sdf"},
                    {"object_id": "destination_beta", "sdf_path": "second.sdf"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_scene_object_sources(manifest) == {
        "manipuland_alpha": (tmp_path / "first.sdf").resolve(),
        "destination_beta": (tmp_path / "second.sdf").resolve(),
    }


def test_scene_object_table_rejects_duplicate_ids(tmp_path) -> None:
    source = tmp_path / "asset.sdf"
    source.write_text("<sdf version='1.7'/>", encoding="utf-8")
    manifest = tmp_path / "objects.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "robotics_scene_object_table.v1",
                "objects": [
                    {"object_id": "same", "sdf_path": str(source)},
                    {"object_id": "same", "sdf_path": str(source)},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_scene_object_sources(manifest)


def test_carry_evidence_samples_every_operator_step() -> None:
    frames = [
        {
            "frame": step * 100 + index,
            "operator_step_index": step,
            "phase": ("lift", "transfer", "lower")[index % 3],
        }
        for step in range(2)
        for index in range(12)
    ]
    selected = carry_evidence_frame_ids(frames, max_samples=12)
    assert len(selected) == 12
    assert {frame_id // 100 for frame_id in selected} == {0, 1}


def test_view_specific_frame_parser_supports_conditioning_only_wrist() -> None:
    import importlib.util
    module_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "render_scenecode_robotics_multiview_parallel.py"
    )
    spec = importlib.util.spec_from_file_location("parallel_renderer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._view_frames("wrist=0") == ("wrist", "0")
    assert module._view_frames("top=0,93,246") == ("top", "0,93,246")
    with pytest.raises(Exception):
        module._view_frames("wrist=bad")
