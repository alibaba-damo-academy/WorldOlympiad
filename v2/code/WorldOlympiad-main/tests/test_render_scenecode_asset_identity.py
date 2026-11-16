from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "render_scenecode_asset_identity.py"
SPEC = importlib.util.spec_from_file_location("render_scenecode_asset_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SceneCodeAssetIdentityBoundsTest(unittest.TestCase):
    def test_blender_server_uses_a_concrete_loopback_port(self) -> None:
        port = MODULE._free_loopback_port()
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_sha256_binds_render_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(len(MODULE._sha256(path)), 64)

    def test_gltf_y_up_bounds_are_rotated_into_sdf_z_up(self) -> None:
        bounds = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
        points = MODULE._mesh_points_in_sdf_frame(bounds, Path("asset.glb"))
        np.testing.assert_allclose(points.min(axis=0), [-1.0, -3.0, -2.0])
        np.testing.assert_allclose(points.max(axis=0), [1.0, 3.0, 2.0])

    def test_render_coverage_uses_actual_background_and_detects_edges(self) -> None:
        framed = np.full((100, 100, 3), [204, 229, 255], dtype=np.uint8)
        framed[30:70, 35:65] = [140, 30, 30]
        fraction, border_fraction = MODULE._render_coverage(framed)
        self.assertAlmostEqual(fraction, 0.12)
        self.assertEqual(border_fraction, 0.0)

        cropped = framed.copy()
        cropped[:20, 40:60] = [140, 30, 30]
        _, cropped_border = MODULE._render_coverage(cropped)
        self.assertGreater(cropped_border, 0.0)

    def test_bounds_include_link_and_visual_pose(self) -> None:
        sdf = """\
<sdf version="1.7">
  <model name="asset">
    <link name="body">
      <pose>1 2 3 0 0 0</pose>
      <visual name="shape">
        <pose>0.5 0 0.25 0 0 0</pose>
        <geometry><box><size>2 4 0.5</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.sdf"
            path.write_text(sdf, encoding="utf-8")
            low, high = MODULE.sdf_visual_bounds(path)
        np.testing.assert_allclose(low, [0.5, 0.0, 3.0])
        np.testing.assert_allclose(high, [2.5, 4.0, 3.5])

    def test_collision_bounds_prefer_physical_support_geometry(self) -> None:
        sdf = """\
<sdf version="1.7">
  <model name="asset">
    <link name="body">
      <visual name="visible"><geometry><box><size>0.02 0.02 0.02</size></box></geometry></visual>
      <collision name="support"><geometry><box><size>0.03 0.04 0.025</size></box></geometry></collision>
    </link>
  </model>
</sdf>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.sdf"
            path.write_text(sdf, encoding="utf-8")
            low, high = MODULE.sdf_collision_bounds(path)
        np.testing.assert_allclose(high - low, [0.03, 0.04, 0.025])

    def test_named_relative_frame_is_resolved_for_articulated_assets(self) -> None:
        sdf = """\
<sdf version="1.7">
  <model name="asset">
    <link name="body"><pose>1 0 0 0 0 0</pose></link>
    <link name="moving">
      <pose relative_to="body">0 2 0 0 0 0</pose>
      <visual name="shape">
        <geometry><box><size>2 2 2</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.sdf"
            path.write_text(sdf, encoding="utf-8")
            low, high = MODULE.sdf_visual_bounds(path)
        np.testing.assert_allclose(low, [0.0, 1.0, -1.0])
        np.testing.assert_allclose(high, [2.0, 3.0, 1.0])

    def test_unknown_relative_frame_fails_closed(self) -> None:
        sdf = """\
<sdf version="1.7">
  <model name="asset">
    <link name="body">
      <visual name="shape">
        <pose relative_to="other">0 0 0 0 0 0</pose>
        <geometry><sphere><radius>0.2</radius></sphere></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.sdf"
            path.write_text(sdf, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown SDF .*relative_to"):
                MODULE.sdf_visual_bounds(path)


if __name__ == "__main__":
    unittest.main()
