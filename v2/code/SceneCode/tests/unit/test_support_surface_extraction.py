import logging
import tempfile
import unittest

from pathlib import Path

import numpy as np
import trimesh

from omegaconf import OmegaConf

from scenecode.agent_utils.room import SupportSurface
from scenecode.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
    extract_support_surfaces_from_mesh,
)

console_logger = logging.getLogger(__name__)


class TestSupportSurfaceExtraction(unittest.TestCase):
    """Test suite for support surface extraction algorithm."""

    @classmethod
    def setUpClass(cls):
        """Load test meshes once for all tests."""
        test_data_dir = (
            Path(__file__).parent.parent / "test_data/support_surface_algorithm"
        )
        cls.shelf_3x3_path = test_data_dir / "shelf_3x3_canonical.gltf"
        cls.artistic_shelf_path = test_data_dir / "artistic_shelf_canonical.gltf"
        cls.round_table_path = test_data_dir / "round_table_canonical.gltf"

        # Verify test files exist.
        assert cls.shelf_3x3_path.exists(), f"Missing: {cls.shelf_3x3_path}"
        assert cls.artistic_shelf_path.exists(), f"Missing: {cls.artistic_shelf_path}"
        assert cls.round_table_path.exists(), f"Missing: {cls.round_table_path}"

    def test_round_table_single_surface(self):
        """Round table should have exactly 2 horizontal surfaces (tabletop + base)."""
        surfaces = extract_support_surfaces_from_mesh(mesh_path=self.round_table_path)

        self.assertEqual(
            len(surfaces),
            2,
            f"Round table should have exactly 2 surfaces (tabletop + base), got {len(surfaces)}",
        )

        # First surface should be the largest (tabletop).
        table_surface = surfaces[0]
        self.assertGreater(
            table_surface.area,
            3.0,
            f"Table surface should be large (> 3.0 m² for tabletop), got {table_surface.area:.3f}",
        )

        # Tabletop should be elevated (not on ground).
        tabletop_height = table_surface.transform.translation()[2]
        self.assertGreater(
            tabletop_height,
            0.5,
            f"Tabletop should be elevated (> 0.5m), got {tabletop_height:.3f}m",
        )

    def test_shelf_3x3_multiple_surfaces(self):
        """3x3 shelf should have exactly 13 horizontal surfaces."""
        surfaces = extract_support_surfaces_from_mesh(mesh_path=self.shelf_3x3_path)

        self.assertEqual(
            len(surfaces),
            13,
            f"Shelf should have exactly 13 surfaces, got {len(surfaces)}",
        )

        # Surfaces should be sorted by area (largest first).
        # Allow small tolerance for floating point precision (1mm tolerance → ~0.01 m²).
        for i in range(len(surfaces) - 1):
            # Either surfaces[i] is larger, or they're within tolerance.
            area_diff = surfaces[i].area - surfaces[i + 1].area
            self.assertGreaterEqual(
                area_diff,
                -0.01,  # Allow up to 1cm² ordering violations due to precision.
                f"Surface {i} area ({surfaces[i].area:.4f} m²) should be >= "
                f"surface {i+1} area ({surfaces[i+1].area:.4f} m²) within tolerance",
            )

    def test_artistic_shelf_at_least_one_surface(self):
        """Artistic shelf should have exactly 17 horizontal surfaces."""
        surfaces = extract_support_surfaces_from_mesh(
            mesh_path=self.artistic_shelf_path
        )

        self.assertEqual(
            len(surfaces),
            17,
            f"Artistic shelf should have exactly 17 surfaces, got {len(surfaces)}",
        )

    def test_surface_normals_point_upward(self):
        """All surface normals should point upward (Z+ direction)."""
        for mesh_path, name in [
            (self.shelf_3x3_path, "shelf_3x3"),
            (self.round_table_path, "round_table"),
            (self.artistic_shelf_path, "artistic_shelf"),
        ]:
            with self.subTest(mesh=name):
                surfaces = extract_support_surfaces_from_mesh(mesh_path=mesh_path)

                for i, surface in enumerate(surfaces):
                    # Extract Z-axis from transform rotation.
                    z_axis = surface.transform.rotation().matrix()[:, 2]

                    # Z component should be close to 1 (upward-pointing).
                    self.assertGreater(
                        z_axis[2],
                        0.95,
                        f"{name} surface {i}: Z-axis should point upward "
                        f"(z_component > 0.95), got {z_axis[2]:.3f}",
                    )

    def test_surface_bounds_valid(self):
        """Surface bounds should be geometrically valid."""
        surfaces = extract_support_surfaces_from_mesh(mesh_path=self.shelf_3x3_path)

        for i, surface in enumerate(surfaces):
            # Min < Max in all dimensions.
            self.assertTrue(
                np.all(surface.bounding_box_min < surface.bounding_box_max),
                f"Surface {i}: Min should be < Max in all dimensions",
            )

            # Reasonable dimensions (not degenerate).
            dims = surface.bounding_box_max - surface.bounding_box_min
            self.assertGreater(
                dims[0], 0.01, f"Surface {i}: Width should be > 1cm, got {dims[0]}m"
            )
            self.assertGreater(
                dims[1], 0.01, f"Surface {i}: Depth should be > 1cm, got {dims[1]}m"
            )

    def test_surface_areas_reasonable(self):
        """Surface areas should be within reasonable bounds."""
        surfaces = extract_support_surfaces_from_mesh(mesh_path=self.shelf_3x3_path)

        for i, surface in enumerate(surfaces):
            # Should meet minimum area threshold.
            self.assertGreaterEqual(
                surface.area,
                0.01,
                f"Surface {i}: Area should be >= 0.01 m² (100 cm²), "
                f"got {surface.area:.4f} m²",
            )

            # Should not be unreasonably large (sanity check).
            self.assertLess(
                surface.area,
                10.0,
                f"Surface {i}: Area should be < 10 m² (sanity check), "
                f"got {surface.area:.4f} m²",
            )

    def test_algorithm_deterministic(self):
        """Running extraction twice should give identical results."""
        surfaces1 = extract_support_surfaces_from_mesh(mesh_path=self.shelf_3x3_path)
        surfaces2 = extract_support_surfaces_from_mesh(mesh_path=self.shelf_3x3_path)

        self.assertEqual(len(surfaces1), len(surfaces2), "Should extract same count")

        for i, (s1, s2) in enumerate(zip(surfaces1, surfaces2)):
            np.testing.assert_array_almost_equal(
                s1.bounding_box_min,
                s2.bounding_box_min,
                decimal=5,
                err_msg=f"Surface {i}: bounding_box_min should match",
            )
            np.testing.assert_array_almost_equal(
                s1.bounding_box_max,
                s2.bounding_box_max,
                decimal=5,
                err_msg=f"Surface {i}: bounding_box_max should match",
            )
            self.assertAlmostEqual(
                s1.area, s2.area, places=5, msg=f"Surface {i}: area should match"
            )

    def test_empty_result_for_vertical_mesh(self):
        """Mesh with mostly vertical surfaces should return few/no surfaces."""
        # Create vertical wall mesh (XZ plane, minimal horizontal surfaces).
        wall_mesh = trimesh.creation.box(extents=[2.0, 0.1, 2.0])
        # Rotate to vertical (standing up in XZ plane).
        wall_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            wall_mesh.export(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            surfaces = extract_support_surfaces_from_mesh(mesh_path=tmp_path)
            # Box has 2 small horizontal ends - accept 0-2 surfaces.
            self.assertLessEqual(
                len(surfaces),
                2,
                "Vertical mesh should have at most 2 small horizontal surfaces",
            )
        finally:
            tmp_path.unlink()

    def test_surface_local_coordinates_centered(self):
        """Surface bounding box should be centered around origin in local frame."""
        surfaces = extract_support_surfaces_from_mesh(mesh_path=self.round_table_path)

        for i, surface in enumerate(surfaces):
            # Check that min and max are symmetric around origin.
            center = (surface.bounding_box_min + surface.bounding_box_max) / 2

            # Center should be close to origin in XY plane.
            # Z-coordinate may have offset (placement volume height).
            self.assertAlmostEqual(
                center[0],
                0.0,
                places=2,
                msg=f"Surface {i}: X-center should be ~0 in local frame",
            )
            self.assertAlmostEqual(
                center[1],
                0.0,
                places=2,
                msg=f"Surface {i}: Y-center should be ~0 in local frame",
            )

    def test_surface_transform_has_valid_rotation(self):
        """Surface transform should have valid rotation matrix."""
        surfaces = extract_support_surfaces_from_mesh(mesh_path=self.round_table_path)

        for i, surface in enumerate(surfaces):
            rotation_matrix = surface.transform.rotation().matrix()

            # Check orthonormality (R^T * R = I).
            identity = rotation_matrix.T @ rotation_matrix
            np.testing.assert_array_almost_equal(
                identity,
                np.eye(3),
                decimal=5,
                err_msg=f"Surface {i}: Rotation matrix should be orthonormal",
            )

            # Check determinant = 1 (proper rotation, not reflection).
            det = np.linalg.det(rotation_matrix)
            self.assertAlmostEqual(
                det,
                1.0,
                places=5,
                msg=f"Surface {i}: Rotation determinant should be 1 "
                f"(proper rotation)",
            )

    def test_custom_config_applied(self):
        """Custom configuration should be respected."""
        # Use very strict area threshold to filter out small surfaces.
        config = SupportSurfaceExtractionConfig(min_surface_area_m2=0.5)

        surfaces_default = extract_support_surfaces_from_mesh(
            mesh_path=self.shelf_3x3_path
        )
        surfaces_strict = extract_support_surfaces_from_mesh(
            mesh_path=self.shelf_3x3_path, config=config
        )

        # Strict config should return fewer or equal surfaces.
        self.assertLessEqual(
            len(surfaces_strict),
            len(surfaces_default),
            "Strict area threshold should filter out more surfaces",
        )

        # All strict surfaces should meet the threshold.
        for surface in surfaces_strict:
            self.assertGreaterEqual(
                surface.area, 0.5, "Surface should meet strict area threshold"
            )

    def test_recompute_hssd_surfaces_config_default(self):
        """Test that recompute_hssd_surfaces defaults to False."""
        config = SupportSurfaceExtractionConfig()
        self.assertFalse(
            config.recompute_hssd_surfaces,
            "recompute_hssd_surfaces should default to False",
        )

    def test_recompute_hssd_surfaces_config_can_be_set(self):
        """Test that recompute_hssd_surfaces can be set to True."""
        config = SupportSurfaceExtractionConfig(recompute_hssd_surfaces=True)
        self.assertTrue(
            config.recompute_hssd_surfaces,
            "recompute_hssd_surfaces should be True when explicitly set",
        )

    def test_from_config_reads_recompute_hssd_surfaces(self):
        """Test that from_config correctly reads recompute_hssd_surfaces."""
        # Create a mock OmegaConf config matching the YAML structure.
        cfg_dict = {
            "hssd": {"recompute_surfaces": True},
            "face_clustering": {
                "normal_cluster_threshold": 0.9,
                "normal_adjacent_threshold": 0.95,
                "horizontal_normal_z_min": 0.95,
                "vertical_normal_z_max": 0.05,
            },
            "filtering": {
                "min_surface_area_m2": 0.1,
                "min_area_ratio": 0.20,
                "min_inscribed_radius_m": 0.10,
            },
            "clearance": {
                "min_clearance_m": 0.1,
                "max_measured_clearance_m": 5.0,
                "top_surface_clearance_m": 0.5,
                "self_intersection_threshold_m": 0.001,
                "clearance_percentile": 10.0,
            },
            "height": {
                "surface_offset_m": 0.01,
                "use_max_z_for_surface_height": True,
                "max_z_percentile": 99.5,
                "height_tolerance_m": 0.05,
            },
        }
        cfg = OmegaConf.create(cfg_dict)

        config = SupportSurfaceExtractionConfig.from_config(cfg)

        self.assertTrue(
            config.recompute_hssd_surfaces,
            "from_config should read recompute_surfaces as recompute_hssd_surfaces",
        )

    def test_from_config_reads_recompute_hssd_surfaces_false(self):
        """Test from_config with recompute_surfaces=False."""
        # Create a mock OmegaConf config with recompute_surfaces=False.
        cfg_dict = {
            "hssd": {"recompute_surfaces": False},
            "face_clustering": {
                "normal_cluster_threshold": 0.9,
                "normal_adjacent_threshold": 0.95,
                "horizontal_normal_z_min": 0.95,
                "vertical_normal_z_max": 0.05,
            },
            "filtering": {
                "min_surface_area_m2": 0.1,
                "min_area_ratio": 0.20,
                "min_inscribed_radius_m": 0.10,
            },
            "clearance": {
                "min_clearance_m": 0.1,
                "max_measured_clearance_m": 5.0,
                "top_surface_clearance_m": 0.5,
                "self_intersection_threshold_m": 0.001,
                "clearance_percentile": 10.0,
            },
            "height": {
                "surface_offset_m": 0.01,
                "use_max_z_for_surface_height": True,
                "max_z_percentile": 99.5,
                "height_tolerance_m": 0.05,
            },
        }
        cfg = OmegaConf.create(cfg_dict)

        config = SupportSurfaceExtractionConfig.from_config(cfg)

        self.assertFalse(
            config.recompute_hssd_surfaces,
            "from_config should correctly read False value",
        )

    def test_handles_nonexistent_file(self):
        """Should raise FileNotFoundError for nonexistent mesh."""
        nonexistent = Path("/tmp/nonexistent_mesh.gltf")

        with self.assertRaises(FileNotFoundError):
            extract_support_surfaces_from_mesh(mesh_path=nonexistent)

    def test_returns_list_type(self):
        """Should always return a list of SupportSurface objects."""
        surfaces = extract_support_surfaces_from_mesh(mesh_path=self.round_table_path)

        self.assertIsInstance(surfaces, list, "Should return a list")
        for surface in surfaces:
            self.assertIsInstance(
                surface, SupportSurface, "Each element should be SupportSurface"
            )


if __name__ == "__main__":
    unittest.main()
