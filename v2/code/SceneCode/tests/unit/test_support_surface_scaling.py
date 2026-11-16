"""Tests for support surface mesh scaling."""

import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import trimesh

from pydrake.all import RigidTransform

from scenecode.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
    extract_and_propagate_support_surfaces,
)
from scenecode.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)


class TestSupportSurfaceMeshScaling(unittest.TestCase):
    """Tests for support surface mesh scaling with furniture scale_factor."""

    def setUp(self):
        """Create a test mesh file with known dimensions."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create a simple 1m x 1m table mesh with a flat top surface.
        # Top surface at Z=0.75m, dimensions 1m x 1m.
        vertices = np.array(
            [
                # Top surface (Z=0.75).
                [-0.5, -0.5, 0.75],
                [0.5, -0.5, 0.75],
                [0.5, 0.5, 0.75],
                [-0.5, 0.5, 0.75],
                # Bottom (Z=0).
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        faces = np.array(
            [
                # Top.
                [0, 1, 2],
                [0, 2, 3],
                # Bottom.
                [4, 6, 5],
                [4, 7, 6],
                # Sides.
                [0, 4, 1],
                [1, 4, 5],
                [1, 5, 2],
                [2, 5, 6],
                [2, 6, 3],
                [3, 6, 7],
                [3, 7, 0],
                [0, 7, 4],
            ]
        )
        self.test_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.mesh_path = self.temp_dir / "test_table.glb"
        self.test_mesh.export(str(self.mesh_path))

        # Create mock scene.
        self.scene = Mock(spec=RoomScene)
        self.scene.objects = {}
        self.scene.generate_surface_id = Mock(
            side_effect=lambda: UniqueID(
                f"S_{len(self.scene.generate_surface_id.call_args_list)}"
            )
        )

        # Create config.
        self.config = SupportSurfaceExtractionConfig()

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extracted_surface_mesh_is_scaled_with_furniture(self):
        """Surface mesh should be scaled by furniture's scale_factor."""
        scale_factor = 0.8

        # Create furniture with scale_factor=0.8.
        furniture = SceneObject(
            object_id=UniqueID("table_001"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(),
            geometry_path=self.mesh_path,
            sdf_path=None,
            support_surfaces=[],
            metadata={},
            bbox_min=np.array([-0.5, -0.5, 0.0]),
            bbox_max=np.array([0.5, 0.5, 0.75]),
            scale_factor=scale_factor,
        )
        self.scene.objects = {furniture.object_id: furniture}

        # Extract surfaces.
        surfaces = extract_and_propagate_support_surfaces(
            scene=self.scene, furniture_object=furniture, config=self.config
        )

        self.assertGreater(len(surfaces), 0, "Should extract at least one surface")

        # Get the top surface (should be the largest).
        top_surface = surfaces[0]
        self.assertIsNotNone(top_surface.mesh, "Surface should have a mesh")

        # The original mesh had vertices at +/-0.5.
        # With scale_factor=0.8, mesh vertices should be at +/-0.4.
        expected_max = 0.5 * scale_factor  # 0.4
        actual_max = np.abs(top_surface.mesh.vertices[:, :2]).max()

        self.assertAlmostEqual(
            actual_max,
            expected_max,
            places=2,
            msg=f"Mesh vertices should be scaled by {scale_factor}. "
            f"Expected max ~{expected_max:.2f}, got {actual_max:.2f}. "
            f"This indicates mesh is not being scaled.",
        )

    def test_propagated_surface_mesh_is_scaled(self):
        """Propagated surfaces should have mesh scaled by target's scale_factor."""
        # Create two furniture pieces with same geometry but different scales.
        furniture_1 = SceneObject(
            object_id=UniqueID("table_001"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(),
            geometry_path=self.mesh_path,
            sdf_path=None,
            support_surfaces=[],
            metadata={},
            scale_factor=1.0,
        )

        furniture_2 = SceneObject(
            object_id=UniqueID("table_002"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(p=[2.0, 0.0, 0.0]),  # Different position.
            geometry_path=self.mesh_path,  # Same geometry.
            sdf_path=None,
            support_surfaces=[],
            metadata={},
            scale_factor=0.7,  # Different scale.
        )

        self.scene.objects = {
            furniture_1.object_id: furniture_1,
            furniture_2.object_id: furniture_2,
        }

        # Extract surfaces for furniture_1 (should propagate to furniture_2).
        surfaces_1 = extract_and_propagate_support_surfaces(
            scene=self.scene, furniture_object=furniture_1, config=self.config
        )

        self.assertGreater(
            len(surfaces_1), 0, "Should extract surfaces for furniture_1"
        )
        self.assertGreater(
            len(furniture_2.support_surfaces),
            0,
            "Surfaces should propagate to furniture_2",
        )

        # Get top surfaces.
        surface_1 = surfaces_1[0]
        surface_2 = furniture_2.support_surfaces[0]

        self.assertIsNotNone(surface_1.mesh, "Surface 1 should have mesh")
        self.assertIsNotNone(surface_2.mesh, "Surface 2 should have mesh")

        # Furniture_1 scale=1.0, furniture_2 scale=0.7.
        # Mesh max coords should be proportionally different.
        max_1 = np.abs(surface_1.mesh.vertices[:, :2]).max()
        max_2 = np.abs(surface_2.mesh.vertices[:, :2]).max()

        expected_ratio = 0.7 / 1.0
        actual_ratio = max_2 / max_1

        self.assertAlmostEqual(
            actual_ratio,
            expected_ratio,
            places=2,
            msg=f"Propagated mesh should be scaled by target's scale_factor. "
            f"Expected ratio {expected_ratio:.2f}, got {actual_ratio:.2f}.",
        )


class TestSupportSurfaceBehavior(unittest.TestCase):
    """Tests for SupportSurface behavior with scaled meshes."""

    def test_contains_point_2d_respects_scaled_mesh(self):
        """Point at edge of unscaled surface should be outside when mesh is scaled down."""
        vertices = np.array(
            [
                [-0.5, -0.5, 0],
                [0.5, -0.5, 0],
                [0.5, 0.5, 0],
                [-0.5, 0.5, 0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh_1x = trimesh.Trimesh(vertices=vertices, faces=faces)

        surface_1x = SupportSurface(
            surface_id=UniqueID("test_surface"),
            bounding_box_min=np.array([-0.5, -0.5, 0]),
            bounding_box_max=np.array([0.5, 0.5, 0.01]),
            transform=RigidTransform(),
            mesh=mesh_1x,
        )

        # Point at (0.4, 0.0) should be INSIDE 1x surface.
        self.assertTrue(surface_1x.contains_point_2d(np.array([0.4, 0.0])))

        # Scaled surface.
        mesh_0_5x = trimesh.Trimesh(vertices=vertices * 0.5, faces=faces)
        surface_0_5x = SupportSurface(
            surface_id=UniqueID("test_surface_scaled"),
            bounding_box_min=np.array([-0.25, -0.25, 0]),
            bounding_box_max=np.array([0.25, 0.25, 0.01]),
            transform=RigidTransform(),
            mesh=mesh_0_5x,
        )

        # Point at (0.4, 0.0) should be OUTSIDE 0.5x surface.
        self.assertFalse(surface_0_5x.contains_point_2d(np.array([0.4, 0.0])))

        # Point at (0.2, 0.0) should be INSIDE 0.5x surface.
        self.assertTrue(surface_0_5x.contains_point_2d(np.array([0.2, 0.0])))


if __name__ == "__main__":
    unittest.main()
