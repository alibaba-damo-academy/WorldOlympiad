"""Unit tests for furniture checkpoint branching workflow."""

import json
import shutil
import tempfile
import unittest

from pathlib import Path

from scenecode.experiments.indoor_scene_generation import (
    _copy_checkpoint_for_stage,
    _fix_paths_in_json_file,
)


class TestPathFixing(unittest.TestCase):
    """Test path fixing in JSON files."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_fix_paths_in_json_file_simple(self):
        """Test fixing simple absolute paths."""
        # Create a JSON file with absolute paths.
        json_path = self.temp_path / "test.json"
        old_base = "/old/path/scene_000/room_main"
        data = {
            "sdf_path": f"{old_base}/generated_assets/furniture/table.sdf",
            "geometry_path": f"{old_base}/generated_assets/furniture/table.glb",
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Fix paths to new base.
        new_base = self.temp_path / "room_main"
        new_base.mkdir(parents=True)
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)

        # Verify paths were fixed.
        with open(json_path) as f:
            fixed_data = json.load(f)

        self.assertEqual(
            fixed_data["sdf_path"],
            str(new_base / "generated_assets/furniture/table.sdf"),
        )
        self.assertEqual(
            fixed_data["geometry_path"],
            str(new_base / "generated_assets/furniture/table.glb"),
        )

    def test_fix_paths_in_json_file_nested(self):
        """Test fixing paths in nested structures."""
        json_path = self.temp_path / "test.json"
        old_base = "/old/experiment/scene_000/room_main"
        data = {
            "objects": [
                {
                    "name": "table",
                    "sdf_path": f"{old_base}/generated_assets/furniture/table.sdf",
                },
                {
                    "name": "chair",
                    "sdf_path": f"{old_base}/generated_assets/furniture/chair.sdf",
                },
            ],
            "metadata": {
                "nested": {
                    "path": f"{old_base}/generated_assets/textures/wood.png",
                }
            },
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        new_base = self.temp_path / "room_main"
        new_base.mkdir(parents=True)
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)

        with open(json_path) as f:
            fixed_data = json.load(f)

        self.assertEqual(
            fixed_data["objects"][0]["sdf_path"],
            str(new_base / "generated_assets/furniture/table.sdf"),
        )
        self.assertEqual(
            fixed_data["objects"][1]["sdf_path"],
            str(new_base / "generated_assets/furniture/chair.sdf"),
        )
        self.assertEqual(
            fixed_data["metadata"]["nested"]["path"],
            str(new_base / "generated_assets/textures/wood.png"),
        )

    def test_fix_paths_preserves_relative_paths(self):
        """Test that relative paths are preserved."""
        json_path = self.temp_path / "test.json"
        data = {
            "relative_path": "generated_assets/furniture/table.sdf",
            "other_path": "../floor_plans/floor.sdf",
            "name": "test object",
            "count": 42,
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        new_base = self.temp_path / "room_main"
        new_base.mkdir(parents=True)
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)

        with open(json_path) as f:
            fixed_data = json.load(f)

        # Relative paths should be unchanged.
        self.assertEqual(
            fixed_data["relative_path"], "generated_assets/furniture/table.sdf"
        )
        self.assertEqual(fixed_data["other_path"], "../floor_plans/floor.sdf")
        self.assertEqual(fixed_data["name"], "test object")
        self.assertEqual(fixed_data["count"], 42)

    def test_fix_paths_scene_level(self):
        """Test fixing scene-level paths (room_geometry/, floor_plans/)."""
        # Create directory structure: scene_dir/room_dir.
        scene_dir = self.temp_path / "scene_000"
        room_dir = scene_dir / "room_main"
        room_dir.mkdir(parents=True)

        json_path = room_dir / "test.json"
        old_scene = "/old/experiment/scene_000"
        data = {
            # Room-level path.
            "furniture_sdf": f"{old_scene}/room_main/generated_assets/furniture/table.sdf",
            # Scene-level paths.
            "wall_sdf": f"{old_scene}/room_geometry/room_geometry_main.sdf",
            "floor_sdf": f"{old_scene}/floor_plans/floor.sdf",
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        _fix_paths_in_json_file(
            json_path=json_path,
            new_room_dir=room_dir,
            new_scene_dir=scene_dir,
        )

        with open(json_path) as f:
            fixed_data = json.load(f)

        # Room-level path should point to room_dir.
        self.assertEqual(
            fixed_data["furniture_sdf"],
            str(room_dir / "generated_assets/furniture/table.sdf"),
        )
        # Scene-level paths should point to scene_dir.
        self.assertEqual(
            fixed_data["wall_sdf"],
            str(scene_dir / "room_geometry/room_geometry_main.sdf"),
        )
        self.assertEqual(
            fixed_data["floor_sdf"],
            str(scene_dir / "floor_plans/floor.sdf"),
        )

    def test_fix_paths_nonexistent_file(self):
        """Test that nonexistent files are handled gracefully."""
        json_path = self.temp_path / "does_not_exist.json"
        new_base = self.temp_path / "room_main"

        # Should not raise an error.
        _fix_paths_in_json_file(json_path=json_path, new_room_dir=new_base)


class TestCopyCheckpointForStage(unittest.TestCase):
    """Test selective checkpoint copy for stage resumption."""

    def setUp(self):
        """Set up test fixtures."""
        self.source_dir = Path(tempfile.mkdtemp())
        self.target_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.source_dir)
        shutil.rmtree(self.target_dir)

    def _create_source_scene(self, with_furniture_assets: bool = True):
        """Create a source scene directory structure for testing."""
        scene_dir = self.source_dir / "scene_000"
        room_dir = scene_dir / "room_main"
        room_dir.mkdir(parents=True)

        # Create house_layout.json (no paths to fix).
        house_layout = scene_dir / "house_layout.json"
        house_layout.write_text(json.dumps({"rooms": ["room_main"], "width": 5.0}))

        # Create floor_plans directory.
        floor_plans = scene_dir / "floor_plans"
        floor_plans.mkdir()
        (floor_plans / "floor.sdf").write_text("<sdf>floor</sdf>")

        # Create room_geometry at scene level.
        room_geometry = scene_dir / "room_geometry"
        room_geometry.mkdir()
        (room_geometry / "room_geometry_main.sdf").write_text("<sdf>room</sdf>")

        if with_furniture_assets:
            # Create generated_assets with absolute paths.
            assets_dir = room_dir / "generated_assets" / "furniture"
            assets_dir.mkdir(parents=True)
            (assets_dir / "table.sdf").write_text("<sdf>table</sdf>")
            (assets_dir / "table.glb").write_text("glb content")

            # Create asset_registry.json with absolute paths.
            registry = {
                "table_01": {
                    "sdf_path": str(room_dir / "generated_assets/furniture/table.sdf"),
                    "geometry_path": str(
                        room_dir / "generated_assets/furniture/table.glb"
                    ),
                }
            }
            registry_path = assets_dir / "asset_registry.json"
            registry_path.write_text(json.dumps(registry))

            # Create scene_states with absolute paths.
            scene_states = room_dir / "scene_states" / "scene_after_furniture"
            scene_states.mkdir(parents=True)
            scene_state = {
                "objects": [
                    {
                        "id": "table_01",
                        "sdf_path": str(
                            room_dir / "generated_assets/furniture/table.sdf"
                        ),
                    }
                ]
            }
            (scene_states / "scene_state.json").write_text(json.dumps(scene_state))

            # Create scene.dmd.yaml with file:// URIs.
            dmd_content = f"""directives:
- add_model:
    name: table_01
    file: file://{room_dir}/generated_assets/furniture/table.sdf
"""
            (scene_states / "scene.dmd.yaml").write_text(dmd_content)

            # Create session DB (should NOT be copied).
            (room_dir / "furniture_agent.db").write_text("session data")

            # Create render directory (should NOT be copied).
            renders_dir = room_dir / "scene_renders" / "furniture" / "renders_001"
            renders_dir.mkdir(parents=True)
            (renders_dir / "render.png").write_text("render content")

        return scene_dir

    def test_copy_scene_for_furniture_stage(self):
        """Test copying scene for furniture stage (no checkpoint needed)."""
        source_scene = self._create_source_scene(with_furniture_assets=False)
        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)

        _copy_checkpoint_for_stage(
            source_scene_dir=source_scene,
            target_scene_dir=target_scene,
            start_stage="furniture",
        )

        # Verify scene-level directories were copied.
        self.assertTrue((target_scene / "house_layout.json").exists())
        self.assertTrue((target_scene / "floor_plans" / "floor.sdf").exists())
        self.assertTrue(
            (target_scene / "room_geometry" / "room_geometry_main.sdf").exists()
        )

        # Verify room directory was created but no checkpoint copied.
        target_room = target_scene / "room_main"
        self.assertTrue(target_room.exists())
        self.assertFalse((target_room / "scene_states").exists())

    def test_copy_scene_for_wall_mounted_stage(self):
        """Test copying scene for wall_mounted stage copies furniture checkpoint."""
        source_scene = self._create_source_scene(with_furniture_assets=True)
        source_room = source_scene / "room_main"
        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)

        _copy_checkpoint_for_stage(
            source_scene_dir=source_scene,
            target_scene_dir=target_scene,
            start_stage="wall_mounted",
        )

        target_room = target_scene / "room_main"

        # Verify checkpoint was copied.
        checkpoint_dir = target_room / "scene_states" / "scene_after_furniture"
        self.assertTrue(checkpoint_dir.exists())
        self.assertTrue((checkpoint_dir / "scene_state.json").exists())
        self.assertTrue((checkpoint_dir / "scene.dmd.yaml").exists())

        # Verify furniture assets were copied.
        self.assertTrue(
            (target_room / "generated_assets" / "furniture" / "table.sdf").exists()
        )

        # Verify paths were fixed in scene_state.json.
        with open(checkpoint_dir / "scene_state.json") as f:
            state = json.load(f)
        table_sdf = state["objects"][0]["sdf_path"]
        self.assertIn(str(target_room), table_sdf)
        self.assertNotIn(str(source_room), table_sdf)

        # Verify paths were fixed in scene.dmd.yaml.
        dmd_content = (checkpoint_dir / "scene.dmd.yaml").read_text()
        self.assertIn(str(target_room), dmd_content)
        self.assertNotIn(str(source_room), dmd_content)

        # Verify session DB was NOT copied.
        self.assertFalse((target_room / "furniture_agent.db").exists())

        # Verify render directories were NOT copied.
        self.assertFalse((target_room / "scene_renders").exists())

    def test_copy_scene_for_manipuland_stage(self):
        """Test copying scene for manipuland stage copies all previous assets."""
        source_scene = self._create_source_scene(with_furniture_assets=True)

        # Add wall_mounted and ceiling_mounted assets.
        source_room = source_scene / "room_main"
        wall_dir = source_room / "generated_assets" / "wall_mounted"
        wall_dir.mkdir(parents=True)
        (wall_dir / "picture.sdf").write_text("<sdf>picture</sdf>")

        ceiling_dir = source_room / "generated_assets" / "ceiling_mounted"
        ceiling_dir.mkdir(parents=True)
        (ceiling_dir / "light.sdf").write_text("<sdf>light</sdf>")

        # Create ceiling checkpoint.
        ceiling_state = source_room / "scene_states" / "scene_after_ceiling_objects"
        ceiling_state.mkdir(parents=True)
        (ceiling_state / "scene_state.json").write_text("{}")

        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)

        _copy_checkpoint_for_stage(
            source_scene_dir=source_scene,
            target_scene_dir=target_scene,
            start_stage="manipuland",
        )

        target_room = target_scene / "room_main"

        # Verify all asset directories were copied.
        self.assertTrue(
            (target_room / "generated_assets" / "furniture" / "table.sdf").exists()
        )
        self.assertTrue(
            (target_room / "generated_assets" / "wall_mounted" / "picture.sdf").exists()
        )
        self.assertTrue(
            (
                target_room / "generated_assets" / "ceiling_mounted" / "light.sdf"
            ).exists()
        )

        # Verify ceiling checkpoint was copied (not furniture checkpoint).
        self.assertTrue(
            (target_room / "scene_states" / "scene_after_ceiling_objects").exists()
        )
        self.assertFalse(
            (target_room / "scene_states" / "scene_after_furniture").exists()
        )

    def test_copy_scene_nonexistent_source(self):
        """Test error when source scene doesn't exist."""
        target_scene = self.target_dir / "scene_000"
        target_scene.mkdir(parents=True)
        nonexistent_source = self.source_dir / "scene_000"

        with self.assertRaises(FileNotFoundError):
            _copy_checkpoint_for_stage(
                source_scene_dir=nonexistent_source,
                target_scene_dir=target_scene,
                start_stage="furniture",
            )


if __name__ == "__main__":
    unittest.main()
