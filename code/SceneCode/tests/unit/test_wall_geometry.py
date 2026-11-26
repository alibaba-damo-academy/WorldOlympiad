"""Tests for wall geometry generation with door/window cutouts."""

import unittest

import numpy as np
import trimesh

from scenecode.floor_plan_agents.tools.wall_geometry import (
    WallDimensions,
    WallOpening,
    apply_box_uv_projection,
    create_wall_gltf,
    create_wall_mesh,
    create_wall_with_openings,
    validate_openings,
)


class TestCreateWallMesh(unittest.TestCase):
    """Tests for basic wall mesh creation."""

    def test_wall_without_openings(self):
        """Plain wall mesh is valid box with correct dimensions."""
        wall = create_wall_mesh(width=5.0, height=2.5, thickness=0.05)

        assert wall.is_watertight
        bbox = wall.bounding_box.extents
        np.testing.assert_allclose(bbox, [5.0, 0.05, 2.5], atol=1e-6)

    def test_wall_bottom_at_floor(self):
        """Wall should have its bottom at z=0."""
        wall = create_wall_mesh(width=3.0, height=2.0, thickness=0.05)

        # Bottom of bounding box should be at z=0.
        self.assertAlmostEqual(wall.bounds[0, 2], 0.0, places=6)
        # Top should be at wall height.
        self.assertAlmostEqual(wall.bounds[1, 2], 2.0, places=6)

    def test_wall_centered_on_xy(self):
        """Wall should be centered on X and Y axes."""
        wall = create_wall_mesh(width=4.0, height=2.5, thickness=0.1)

        # X bounds should be symmetric around 0.
        self.assertAlmostEqual(wall.bounds[0, 0], -2.0, places=6)
        self.assertAlmostEqual(wall.bounds[1, 0], 2.0, places=6)

        # Y bounds should be symmetric around 0.
        self.assertAlmostEqual(wall.bounds[0, 1], -0.05, places=6)
        self.assertAlmostEqual(wall.bounds[1, 1], 0.05, places=6)


class TestCreateWallWithOpenings(unittest.TestCase):
    """Tests for wall mesh creation with door/window cutouts."""

    def test_wall_with_door_cutout(self):
        """Wall with door has valid geometry and reduced volume."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=2.5,  # Center of wall.
                width=1.0,
                height=2.1,
                sill_height=0.0,  # Door starts at floor.
            )
        ]

        wall = create_wall_with_openings(dimensions, openings)

        assert wall.is_watertight
        # Volume should be less than solid wall.
        solid_volume = 5.0 * 0.05 * 2.5
        door_volume = 1.0 * 0.05 * 2.1
        expected_volume = solid_volume - door_volume
        self.assertAlmostEqual(
            wall.volume, expected_volume, delta=expected_volume * 0.01
        )

    def test_wall_with_window_cutout(self):
        """Wall with window has valid geometry."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=2.5,
                width=1.2,
                height=1.2,
                sill_height=0.9,  # Window above floor.
            )
        ]

        wall = create_wall_with_openings(dimensions, openings)

        assert wall.is_watertight
        # Volume should be reduced by window opening.
        solid_volume = 5.0 * 0.05 * 2.5
        window_volume = 1.2 * 0.05 * 1.2
        expected_volume = solid_volume - window_volume
        self.assertAlmostEqual(
            wall.volume, expected_volume, delta=expected_volume * 0.01
        )

    def test_wall_with_multiple_openings(self):
        """Wall with both door and window."""
        dimensions = WallDimensions(width=6.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=1.5,
                width=1.0,
                height=2.1,
                sill_height=0.0,
            ),
            WallOpening(
                position_along_wall=4.5,
                width=1.2,
                height=1.2,
                sill_height=0.9,
            ),
        ]

        wall = create_wall_with_openings(dimensions, openings)

        assert wall.is_watertight
        solid_volume = 6.0 * 0.05 * 2.5
        door_volume = 1.0 * 0.05 * 2.1
        window_volume = 1.2 * 0.05 * 1.2
        expected_volume = solid_volume - door_volume - window_volume
        self.assertAlmostEqual(
            wall.volume, expected_volume, delta=expected_volume * 0.01
        )

    def test_wall_no_openings_returns_solid(self):
        """Empty openings list returns solid wall."""
        dimensions = WallDimensions(width=4.0, height=2.5, thickness=0.05)

        wall = create_wall_with_openings(dimensions, [])

        assert wall.is_watertight
        expected_volume = 4.0 * 0.05 * 2.5
        self.assertAlmostEqual(
            wall.volume, expected_volume, delta=expected_volume * 0.01
        )


class TestValidateOpenings(unittest.TestCase):
    """Tests for opening validation."""

    def test_valid_door_in_center(self):
        """Door in center of wall passes validation."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=2.5,
                width=1.0,
                height=2.1,
                sill_height=0.0,
            )
        ]

        errors = validate_openings(dimensions, openings)
        assert errors == []

    def test_door_too_close_to_left_edge(self):
        """Door too close to wall start fails validation."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=0.05,  # Left edge at 0.05m, less than margin.
                width=1.0,
                height=2.1,
                sill_height=0.0,
            )
        ]

        errors = validate_openings(dimensions, openings, margin=0.1)
        assert len(errors) == 1
        assert "left edge" in errors[0].lower()

    def test_door_too_close_to_right_edge(self):
        """Door too close to wall end fails validation."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=4.0,  # Right edge at 5.0m, no margin from wall end.
                width=1.0,
                height=2.1,
                sill_height=0.0,
            )
        ]

        errors = validate_openings(dimensions, openings, margin=0.1)
        assert len(errors) == 1
        assert "right edge" in errors[0].lower()

    def test_window_too_tall(self):
        """Window extending above wall top fails validation."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=2.5,
                width=1.2,
                height=1.5,
                sill_height=1.2,  # Top at 2.7m, wall is 2.5m.
            )
        ]

        errors = validate_openings(dimensions, openings, margin=0.1)
        assert len(errors) == 1
        assert "top edge" in errors[0].lower()

    def test_negative_sill_height(self):
        """Negative sill height fails validation."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=2.5,
                width=1.0,
                height=2.0,
                sill_height=-0.1,
            )
        ]

        errors = validate_openings(dimensions, openings)
        assert len(errors) == 1
        assert "negative" in errors[0].lower()

    def test_overlapping_openings(self):
        """Overlapping openings fail validation."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=2.0,
                width=1.0,
                height=2.1,
                sill_height=0.0,
            ),
            WallOpening(
                position_along_wall=2.3,  # Overlaps with first.
                width=1.0,
                height=2.1,
                sill_height=0.0,
            ),
        ]

        errors = validate_openings(dimensions, openings)
        assert len(errors) == 1
        assert "overlap" in errors[0].lower()

    def test_non_overlapping_side_by_side(self):
        """Non-overlapping side-by-side openings pass validation."""
        dimensions = WallDimensions(width=6.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=1.5,
                width=1.0,
                height=2.1,
                sill_height=0.0,
            ),
            WallOpening(
                position_along_wall=4.5,
                width=1.0,
                height=2.1,
                sill_height=0.0,
            ),
        ]

        errors = validate_openings(dimensions, openings, margin=0.1)
        assert errors == []


class TestApplyBoxUVProjection(unittest.TestCase):
    """Tests for UV coordinate generation."""

    def test_uv_projection_creates_uvs(self):
        """UV projection creates UV coordinates."""
        wall = create_wall_mesh(width=2.0, height=2.0, thickness=0.05)
        wall_with_uvs = apply_box_uv_projection(wall, scale=0.5)

        # Check that visual has UV coordinates.
        assert wall_with_uvs.visual is not None
        assert hasattr(wall_with_uvs.visual, "uv")

    def test_uv_scale_affects_tiling(self):
        """Different UV scales produce different UV ranges."""
        wall = create_wall_mesh(width=2.0, height=2.0, thickness=0.05)

        wall_small = apply_box_uv_projection(wall.copy(), scale=0.5)
        wall_large = apply_box_uv_projection(wall.copy(), scale=1.0)

        # Smaller scale means more tiles, so larger UV range.
        uv_range_small = np.ptp(wall_small.visual.uv, axis=0)
        uv_range_large = np.ptp(wall_large.visual.uv, axis=0)

        # Small scale should have ~2x the UV range of large scale.
        assert uv_range_small[0] > uv_range_large[0]


class TestCreateWallGltf(unittest.TestCase):
    """Tests for GLTF export functionality."""

    def test_create_wall_gltf_no_export(self):
        """Create wall GLTF without exporting returns mesh."""
        dimensions = WallDimensions(width=4.0, height=2.5, thickness=0.05)

        wall = create_wall_gltf(dimensions, openings=None, output_path=None)

        assert isinstance(wall, trimesh.Trimesh)
        # Mesh has UV coordinates for texturing. Note: mesh is not watertight
        # after UV projection because vertices are unmerged for correct UVs.
        # Watertightness is only relevant for collision geometry, not visual.
        assert wall.visual is not None
        assert hasattr(wall.visual, "uv")
        assert wall.visual.uv is not None

    def test_create_wall_gltf_with_openings(self):
        """Create wall GLTF with openings."""
        dimensions = WallDimensions(width=5.0, height=2.5, thickness=0.05)
        openings = [
            WallOpening(
                position_along_wall=2.5,
                width=1.0,
                height=2.1,
                sill_height=0.0,
            )
        ]

        wall = create_wall_gltf(dimensions, openings=openings, output_path=None)

        assert isinstance(wall, trimesh.Trimesh)
        # Mesh has UV coordinates for texturing.
        assert wall.visual is not None
        assert hasattr(wall.visual, "uv")
        assert wall.visual.uv is not None

    def test_create_wall_gltf_export(self):
        """Create wall GLTF and export to file."""
        import tempfile

        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dimensions = WallDimensions(width=3.0, height=2.5, thickness=0.05)
            output_path = tmp_path / "test_wall.gltf"

            wall = create_wall_gltf(dimensions, openings=None, output_path=output_path)

            assert output_path.exists()
            assert isinstance(wall, trimesh.Trimesh)


if __name__ == "__main__":
    unittest.main()
