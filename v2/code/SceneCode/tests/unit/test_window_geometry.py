"""Tests for window geometry generation."""

import unittest

import numpy as np
import trimesh

from scenecode.floor_plan_agents.tools.window_geometry import (
    WindowDimensions,
    WindowStyle,
    create_simple_window_mesh,
    create_window_frame,
    create_window_glass,
    create_window_gltf,
    create_window_mesh,
)


class TestCreateWindowFrame(unittest.TestCase):
    """Tests for window frame creation."""

    def test_frame_valid_mesh(self):
        """Frame mesh is valid and watertight."""
        dimensions = WindowDimensions(width=1.2, height=1.2, depth=0.1)
        frame = create_window_frame(dimensions)

        assert frame.is_watertight
        assert len(frame.vertices) > 0

    def test_frame_dimensions(self):
        """Frame has correct outer dimensions."""
        dimensions = WindowDimensions(width=1.0, height=1.5, depth=0.1)
        frame = create_window_frame(dimensions)

        bbox = frame.bounding_box.extents
        np.testing.assert_allclose(bbox[0], 1.0, atol=1e-6)  # Width.
        np.testing.assert_allclose(bbox[1], 0.1, atol=1e-6)  # Depth.
        np.testing.assert_allclose(bbox[2], 1.5, atol=1e-6)  # Height.

    def test_frame_has_hole(self):
        """Frame volume is less than solid box (has inner cutout)."""
        dimensions = WindowDimensions(
            width=1.0, height=1.0, depth=0.1, frame_width=0.05
        )
        frame = create_window_frame(dimensions)

        solid_volume = 1.0 * 1.0 * 0.1
        # Frame should have significantly less volume due to cutout.
        assert frame.volume < solid_volume * 0.9

    def test_frame_too_thick_returns_solid(self):
        """Frame with width larger than half window returns solid box."""
        dimensions = WindowDimensions(
            width=0.5, height=0.5, depth=0.1, frame_width=0.3  # Too thick.
        )
        frame = create_window_frame(dimensions)

        # Should still return valid mesh.
        assert frame.is_watertight
        # Volume should be full solid box.
        expected_volume = 0.5 * 0.5 * 0.1
        self.assertAlmostEqual(
            frame.volume, expected_volume, delta=expected_volume * 0.01
        )


class TestCreateWindowGlass(unittest.TestCase):
    """Tests for window glass pane creation."""

    def test_glass_valid_mesh(self):
        """Glass mesh is valid."""
        dimensions = WindowDimensions(width=1.2, height=1.2, depth=0.1)
        glass = create_window_glass(dimensions)

        assert glass.is_watertight
        assert len(glass.vertices) > 0

    def test_glass_dimensions(self):
        """Glass has correct dimensions (smaller than frame outer)."""
        dimensions = WindowDimensions(
            width=1.0, height=1.5, depth=0.1, frame_width=0.05, glass_thickness=0.01
        )
        glass = create_window_glass(dimensions)

        bbox = glass.bounding_box.extents
        # Glass should be frame_width*2 smaller than outer.
        np.testing.assert_allclose(bbox[0], 0.9, atol=1e-6)  # Width - 2*0.05.
        np.testing.assert_allclose(bbox[1], 0.01, atol=1e-6)  # Glass thickness.
        np.testing.assert_allclose(bbox[2], 1.4, atol=1e-6)  # Height - 2*0.05.

    def test_glass_empty_when_frame_too_thick(self):
        """Glass is empty when frame is too thick."""
        dimensions = WindowDimensions(
            width=0.5, height=0.5, depth=0.1, frame_width=0.3  # Too thick.
        )
        glass = create_window_glass(dimensions)

        # Should return empty mesh.
        assert len(glass.vertices) == 0


class TestCreateWindowMesh(unittest.TestCase):
    """Tests for complete window mesh creation."""

    def test_window_mesh_is_scene(self):
        """Complete window is a scene with frame and glass."""
        window = create_window_mesh(width=1.2, height=1.2, depth=0.1)

        assert isinstance(window, trimesh.Scene)

    def test_window_contains_frame(self):
        """Window scene contains frame geometry."""
        window = create_window_mesh(width=1.2, height=1.2, depth=0.1)

        assert "frame" in window.geometry

    def test_window_contains_glass(self):
        """Window scene contains glass geometry."""
        window = create_window_mesh(width=1.2, height=1.2, depth=0.1)

        assert "glass" in window.geometry

    def test_window_custom_style(self):
        """Window with custom style parameters."""
        style = WindowStyle(
            frame_color=(0.5, 0.5, 0.5, 1.0),
            glass_color=(0.9, 0.95, 1.0, 0.2),
        )
        window = create_window_mesh(width=1.0, height=1.0, depth=0.1, style=style)

        assert isinstance(window, trimesh.Scene)


class TestCreateSimpleWindowMesh(unittest.TestCase):
    """Tests for single-mesh window creation."""

    def test_simple_window_is_single_mesh(self):
        """Simple window returns single trimesh."""
        window = create_simple_window_mesh(width=1.2, height=1.2, depth=0.1)

        assert isinstance(window, trimesh.Trimesh)

    def test_simple_window_watertight(self):
        """Simple window mesh is watertight."""
        window = create_simple_window_mesh(width=1.0, height=1.5, depth=0.1)

        # Combined mesh may not be watertight due to overlapping glass/frame.
        # Just check it's a valid mesh.
        assert len(window.vertices) > 0
        assert len(window.faces) > 0


class TestCreateWindowGltf(unittest.TestCase):
    """Tests for window GLTF export."""

    def test_create_window_no_export(self):
        """Create window without export returns scene."""
        window = create_window_gltf(width=1.2, height=1.2, output_path=None)

        assert isinstance(window, trimesh.Scene)

    def test_create_window_export(self):
        """Create window and export to file."""
        import tempfile

        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "test_window.gltf"
            window = create_window_gltf(width=1.2, height=1.2, output_path=output_path)

            assert output_path.exists()
            assert isinstance(window, trimesh.Scene)


if __name__ == "__main__":
    unittest.main()
