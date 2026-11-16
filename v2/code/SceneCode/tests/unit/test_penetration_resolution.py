"""Unit tests for penetration resolution tools and utilities."""

import json
import unittest

from pathlib import Path

import numpy as np
import trimesh

from pydrake.all import (
    MathematicalProgram,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Solve,
)
from pydrake.geometry.optimization import HPolyhedron, VPolytope

from scenecode.agent_utils.physical_feasibility import get_object_xy_footprint
from scenecode.agent_utils.room import PlacementInfo, SupportSurface, UniqueID
from scenecode.manipuland_agents.tools.response_dataclasses import (
    ManipulandErrorType,
    PenetrationResolutionResult,
)

# Path to test data.
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


class TestGetXYConvexHull(unittest.TestCase):
    """Tests for SupportSurface.get_xy_convex_hull()."""

    def test_rectangular_surface_from_bounding_box(self) -> None:
        """Test VPolytope from axis-aligned bounding box (no mesh)."""
        # Create surface with no mesh, just bounding box.
        surface = SupportSurface(
            surface_id=UniqueID("test_surface"),
            bounding_box_min=np.array([-0.5, -0.3, 0.0]),
            bounding_box_max=np.array([0.5, 0.3, 0.1]),
            transform=RigidTransform(),
            mesh=None,  # No mesh provided.
        )

        vpoly = surface.get_xy_convex_hull()

        # Verify it's a VPolytope.
        self.assertIsInstance(vpoly, VPolytope)

        # Verify ambient dimension is 2 (XY).
        self.assertEqual(vpoly.ambient_dimension(), 2)

        # Check that the bounds are correct.
        # VPolytope vertices should form the rectangle corners.
        vertices = vpoly.vertices()  # 2xN array
        self.assertEqual(vertices.shape[0], 2)  # 2D

        # For a box, there should be 4 vertices.
        self.assertEqual(vertices.shape[1], 4)

        # Check min/max X and Y are correct.
        x_min, x_max = vertices[0].min(), vertices[0].max()
        y_min, y_max = vertices[1].min(), vertices[1].max()
        self.assertAlmostEqual(x_min, -0.5)
        self.assertAlmostEqual(x_max, 0.5)
        self.assertAlmostEqual(y_min, -0.3)
        self.assertAlmostEqual(y_max, 0.3)

    def test_circular_surface_from_mesh(self) -> None:
        """Test VPolytope from a circular mesh (round table)."""
        # Create a circular mesh for round table.
        n_points = 32
        radius = 0.6
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        vertices_2d = np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
            ]
        )
        # Add Z=0 for 3D mesh.
        vertices_3d = np.column_stack([vertices_2d, np.zeros(n_points)])

        # Create simple triangulation from center to edges.
        center = np.array([[0.0, 0.0, 0.0]])
        all_vertices = np.vstack([center, vertices_3d])
        faces = []
        for i in range(n_points):
            faces.append([0, i + 1, (i % n_points) + 1 + 1])
        # Fix last face.
        faces[-1] = [0, n_points, 1]
        faces = np.array(faces)

        mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces)

        surface = SupportSurface(
            surface_id=UniqueID("round_table"),
            bounding_box_min=np.array([-0.6, -0.6, 0.0]),
            bounding_box_max=np.array([0.6, 0.6, 0.1]),
            transform=RigidTransform(),
            mesh=mesh,
        )

        vpoly = surface.get_xy_convex_hull()

        # Verify it's a VPolytope with 2D.
        self.assertIsInstance(vpoly, VPolytope)
        self.assertEqual(vpoly.ambient_dimension(), 2)

        # Should have multiple vertices for circular hull.
        vertices = vpoly.vertices()
        self.assertGreaterEqual(vertices.shape[1], 8)

        # Points should be approximately on circle of radius 0.6.
        distances = np.linalg.norm(vertices, axis=0)
        np.testing.assert_array_almost_equal(
            distances,
            np.full(distances.shape, radius),
            decimal=2,
        )

    def test_degenerate_mesh_falls_back_to_bounding_box(self) -> None:
        """Test that degenerate mesh (collinear points) uses bounding box."""
        # Create a degenerate mesh (line, not area).
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],  # Collinear with first two.
            ]
        )
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        surface = SupportSurface(
            surface_id=UniqueID("degenerate"),
            bounding_box_min=np.array([-0.5, -0.3, 0.0]),
            bounding_box_max=np.array([0.5, 0.3, 0.1]),
            transform=RigidTransform(),
            mesh=mesh,
        )

        vpoly = surface.get_xy_convex_hull()

        # Should fall back to bounding box (4 vertices).
        self.assertIsInstance(vpoly, VPolytope)
        vertices = vpoly.vertices()
        self.assertEqual(vertices.shape[1], 4)

    def test_rotated_surface_returns_world_frame_hull(self) -> None:
        """Test that rotated surface returns hull in world frame, not local frame.

        This test catches the frame mismatch bug where get_xy_convex_hull()
        returned local-frame coordinates but IK constraints need world-frame.
        """
        # Create a 1x0.5 surface (local X is 1m, local Y is 0.5m).
        # Rotated 90 degrees around Z, so world X becomes local Y and vice versa.
        yaw_90 = np.pi / 2
        surface = SupportSurface(
            surface_id=UniqueID("rotated_table"),
            bounding_box_min=np.array([-0.5, -0.25, 0.0]),
            bounding_box_max=np.array([0.5, 0.25, 0.1]),
            transform=RigidTransform(
                p=[2.0, 3.0, 0.75],  # Offset from origin.
                rpy=RollPitchYaw([0.0, 0.0, yaw_90]),
            ),
            mesh=None,
        )

        vpoly = surface.get_xy_convex_hull()
        vertices = vpoly.vertices()  # 2xN array

        # With 90-degree rotation:
        # Local bbox: X in [-0.5, 0.5], Y in [-0.25, 0.25]
        # World frame: rotated local X becomes world Y, local Y becomes world -X
        # World bbox after rotation: X in [-0.25+2, 0.25+2], Y in [-0.5+3, 0.5+3]
        #                          = X in [1.75, 2.25], Y in [2.5, 3.5]

        x_min, x_max = vertices[0].min(), vertices[0].max()
        y_min, y_max = vertices[1].min(), vertices[1].max()

        # Verify world-frame coordinates (with rotation + translation).
        self.assertAlmostEqual(x_min, 1.75, places=4)
        self.assertAlmostEqual(x_max, 2.25, places=4)
        self.assertAlmostEqual(y_min, 2.5, places=4)
        self.assertAlmostEqual(y_max, 3.5, places=4)

        # Verify hull dimensions are swapped due to rotation.
        x_extent = x_max - x_min  # Should be ~0.5 (was local Y extent).
        y_extent = y_max - y_min  # Should be ~1.0 (was local X extent).
        self.assertAlmostEqual(x_extent, 0.5, places=4)
        self.assertAlmostEqual(y_extent, 1.0, places=4)

    def test_rotated_surface_mesh_returns_world_frame_hull(self) -> None:
        """Test that rotated surface with mesh returns world-frame hull."""
        # Create a simple rectangular mesh in local frame.
        vertices = np.array(
            [
                [-0.5, -0.25, 0.0],
                [0.5, -0.25, 0.0],
                [0.5, 0.25, 0.0],
                [-0.5, 0.25, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 45-degree rotation.
        yaw_45 = np.pi / 4
        surface = SupportSurface(
            surface_id=UniqueID("rotated_table_mesh"),
            bounding_box_min=np.array([-0.5, -0.25, 0.0]),
            bounding_box_max=np.array([0.5, 0.25, 0.1]),
            transform=RigidTransform(
                p=[0.0, 0.0, 0.5],
                rpy=RollPitchYaw([0.0, 0.0, yaw_45]),
            ),
            mesh=mesh,
        )

        vpoly = surface.get_xy_convex_hull()
        vertices_hull = vpoly.vertices()

        # With 45-degree rotation, the rectangular surface becomes diamond-shaped.
        # The 4 corner points of the local bbox rotate around origin.
        # Local corners: (-0.5, -0.25), (0.5, -0.25), (0.5, 0.25), (-0.5, 0.25)
        # After rotation: each corner is R @ corner, where R is yaw rotation.
        cos45, sin45 = np.cos(yaw_45), np.sin(yaw_45)
        local_corners = np.array(
            [
                [-0.5, -0.25],
                [0.5, -0.25],
                [0.5, 0.25],
                [-0.5, 0.25],
            ]
        )
        # Rotation matrix for 2D.
        R_2d = np.array([[cos45, -sin45], [sin45, cos45]])
        expected_corners = (R_2d @ local_corners.T).T  # No translation at origin.

        # Verify the hull contains approximately these corners.
        for corner in expected_corners:
            # Check if this corner is approximately in the hull vertices.
            distances = np.linalg.norm(vertices_hull.T - corner, axis=1)
            min_dist = distances.min()
            self.assertLess(min_dist, 0.01, f"Expected corner {corner} not found")


class TestFromWorldPose(unittest.TestCase):
    """Tests for SupportSurface.from_world_pose()."""

    def test_from_world_pose_is_inverse_of_to_world_pose(self) -> None:
        """Test that from_world_pose inverts to_world_pose."""
        # Create surface at non-trivial position and rotation.
        surface_transform = RigidTransform(
            p=[1.0, 2.0, 0.8],  # Offset from origin.
            rpy=RollPitchYaw([0.0, 0.0, 0.0]),  # No rotation for simplicity.
        )
        surface = SupportSurface(
            surface_id=UniqueID("test_surface"),
            bounding_box_min=np.array([-0.5, -0.3, 0.0]),
            bounding_box_max=np.array([0.5, 0.3, 0.1]),
            transform=surface_transform,
            mesh=None,
        )

        # Original surface-relative pose.
        original_position_2d = np.array([0.2, -0.1])
        original_rotation_2d = 0.5  # radians

        # Convert to world and back.
        world_pose = surface.to_world_pose(original_position_2d, original_rotation_2d)
        recovered_position_2d, recovered_rotation_2d = surface.from_world_pose(
            world_pose
        )

        # Should recover original values.
        np.testing.assert_array_almost_equal(
            recovered_position_2d, original_position_2d, decimal=6
        )
        self.assertAlmostEqual(recovered_rotation_2d, original_rotation_2d, places=6)

    def test_from_world_pose_with_rotated_surface(self) -> None:
        """Test from_world_pose with a rotated surface (e.g., tilted table)."""
        # Surface with yaw rotation.
        yaw_angle = np.pi / 4  # 45 degrees.
        surface_transform = RigidTransform(
            p=[0.0, 0.0, 0.5], rpy=RollPitchYaw([0.0, 0.0, yaw_angle])
        )
        surface = SupportSurface(
            surface_id=UniqueID("rotated_surface"),
            bounding_box_min=np.array([-0.5, -0.5, 0.0]),
            bounding_box_max=np.array([0.5, 0.5, 0.1]),
            transform=surface_transform,
            mesh=None,
        )

        # Test with various surface-relative poses.
        test_cases = [
            (np.array([0.0, 0.0]), 0.0),
            (np.array([0.3, 0.2]), 0.0),
            (np.array([-0.1, 0.1]), 1.0),
            (np.array([0.0, 0.0]), -0.5),
        ]

        for original_pos_2d, original_rot_2d in test_cases:
            world_pose = surface.to_world_pose(original_pos_2d, original_rot_2d)
            recovered_pos_2d, recovered_rot_2d = surface.from_world_pose(world_pose)

            np.testing.assert_array_almost_equal(
                recovered_pos_2d, original_pos_2d, decimal=6
            )
            self.assertAlmostEqual(recovered_rot_2d, original_rot_2d, places=6)


class TestGetObjectXYFootprint(unittest.TestCase):
    """Tests for get_object_xy_footprint()."""

    def test_box_footprint_no_rotation(self) -> None:
        """Test footprint of a box with no rotation."""
        # Create a 0.3 x 0.2 x 0.1 box centered at origin.
        box = trimesh.creation.box(extents=[0.3, 0.2, 0.1])

        rotation = RotationMatrix()  # Identity.
        vpoly = get_object_xy_footprint(box, rotation)

        # Should be 2D.
        self.assertEqual(vpoly.ambient_dimension(), 2)

        # Footprint should be approximately 0.3 x 0.2.
        vertices = vpoly.vertices()
        x_extent = vertices[0].max() - vertices[0].min()
        y_extent = vertices[1].max() - vertices[1].min()
        self.assertAlmostEqual(x_extent, 0.3, places=2)
        self.assertAlmostEqual(y_extent, 0.2, places=2)

    def test_long_object_rotation_changes_footprint(self) -> None:
        """Test that rotating a long object changes its XY footprint.

        A knife-like object (long in X) when rotated 90 degrees becomes
        long in Y, affecting placement feasibility near edges.
        """
        # Create elongated object (knife-like): 0.3m x 0.02m x 0.01m.
        knife = trimesh.creation.box(extents=[0.3, 0.02, 0.01])

        # No rotation - long in X.
        rot_0 = RotationMatrix()
        footprint_0 = get_object_xy_footprint(knife, rot_0)
        verts_0 = footprint_0.vertices()
        x_extent_0 = verts_0[0].max() - verts_0[0].min()
        y_extent_0 = verts_0[1].max() - verts_0[1].min()

        # 90 degree rotation around Z - now long in Y.
        rot_90 = RotationMatrix.MakeZRotation(np.pi / 2)
        footprint_90 = get_object_xy_footprint(knife, rot_90)
        verts_90 = footprint_90.vertices()
        x_extent_90 = verts_90[0].max() - verts_90[0].min()
        y_extent_90 = verts_90[1].max() - verts_90[1].min()

        # After 90 rotation, X and Y extents should swap.
        self.assertAlmostEqual(x_extent_0, y_extent_90, places=2)
        self.assertAlmostEqual(y_extent_0, x_extent_90, places=2)


class TestPontryaginDifference(unittest.TestCase):
    """Tests for Pontryagin difference feasible region computation."""

    def test_rectangular_surface_with_small_object(self) -> None:
        """Test feasible region for small object on rectangular surface."""
        # Surface: 1.0 x 0.5 meters.
        surface_vpoly = VPolytope.MakeBox(
            lb=np.array([-0.5, -0.25]), ub=np.array([0.5, 0.25])
        )
        surface_hpoly = HPolyhedron(vpoly=surface_vpoly)

        # Object: 0.1 x 0.1 meters (small box).
        object_vpoly = VPolytope.MakeBox(
            lb=np.array([-0.05, -0.05]), ub=np.array([0.05, 0.05])
        )
        object_hpoly = HPolyhedron(vpoly=object_vpoly)

        # Compute feasible region.
        feasible = surface_hpoly.PontryaginDifference(object_hpoly)

        # Feasible region should be smaller than surface.
        self.assertFalse(feasible.IsEmpty())

        # Center should be feasible.
        self.assertTrue(feasible.PointInSet(np.array([0.0, 0.0])))

        # Corner of feasible region should be ~(0.45, 0.20) - object fits.
        self.assertTrue(feasible.PointInSet(np.array([0.44, 0.19])))

        # Edge of surface should NOT be feasible - object would go outside.
        self.assertFalse(feasible.PointInSet(np.array([0.5, 0.25])))

    def test_long_object_orientation_affects_feasibility(self) -> None:
        """Test that long object orientation affects where it can be placed.

        A knife parallel to an edge can be placed closer to that edge
        than a knife perpendicular to it.
        """
        # Surface: 0.4 x 0.4 meters.
        surface_vpoly = VPolytope.MakeBox(
            lb=np.array([-0.2, -0.2]), ub=np.array([0.2, 0.2])
        )
        surface_hpoly = HPolyhedron(vpoly=surface_vpoly)

        # Knife horizontal (long in X): 0.3 x 0.02 meters.
        knife_horizontal = VPolytope.MakeBox(
            lb=np.array([-0.15, -0.01]), ub=np.array([0.15, 0.01])
        )
        knife_h_hpoly = HPolyhedron(vpoly=knife_horizontal)
        feasible_h = surface_hpoly.PontryaginDifference(knife_h_hpoly)

        # Knife vertical (long in Y): 0.02 x 0.3 meters.
        knife_vertical = VPolytope.MakeBox(
            lb=np.array([-0.01, -0.15]), ub=np.array([0.01, 0.15])
        )
        knife_v_hpoly = HPolyhedron(vpoly=knife_vertical)
        feasible_v = surface_hpoly.PontryaginDifference(knife_v_hpoly)

        # Horizontal knife can be placed closer to top/bottom edges.
        # At y=0.18, horizontal knife extends 0.01 in Y (stays in bounds).
        self.assertTrue(feasible_h.PointInSet(np.array([0.0, 0.18])))

        # Vertical knife cannot be at y=0.18 (would extend to 0.33).
        self.assertFalse(feasible_v.PointInSet(np.array([0.0, 0.18])))

        # Vertical knife can be placed closer to left/right edges.
        self.assertTrue(feasible_v.PointInSet(np.array([0.18, 0.0])))
        self.assertFalse(feasible_h.PointInSet(np.array([0.18, 0.0])))


class TestPenetrationResolutionResultDTO(unittest.TestCase):
    """Tests for PenetrationResolutionResult serialization."""

    def test_successful_result_serializes_to_json(self) -> None:
        """Test that successful result serializes correctly."""
        result = PenetrationResolutionResult(
            success=True,
            message="Resolved penetrations. Moved 2 objects.",
            num_objects_considered=3,
            num_objects_moved=2,
            moved_object_ids=["obj_a", "obj_b"],
            max_displacement_m=0.045,
            error_type=None,
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["num_objects_considered"], 3)
        self.assertEqual(parsed["num_objects_moved"], 2)
        self.assertEqual(parsed["moved_object_ids"], ["obj_a", "obj_b"])
        self.assertAlmostEqual(parsed["max_displacement_m"], 0.045)
        self.assertIsNone(parsed["error_type"])

    def test_failure_result_serializes_with_error_type(self) -> None:
        """Test that failure result includes error type."""
        result = PenetrationResolutionResult(
            success=False,
            message="Objects on different surfaces",
            num_objects_considered=2,
            num_objects_moved=0,
            moved_object_ids=[],
            max_displacement_m=0.0,
            error_type=ManipulandErrorType.OBJECTS_ON_DIFFERENT_SURFACES,
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        self.assertFalse(parsed["success"])
        self.assertEqual(parsed["num_objects_moved"], 0)
        self.assertEqual(parsed["moved_object_ids"], [])
        self.assertEqual(parsed["error_type"], "objects_on_different_surfaces")


class TestPlacementInfoSynchronization(unittest.TestCase):
    """Tests for placement_info synchronization after physics resolution."""

    def test_placement_info_updated_after_transform_change(self) -> None:
        """Test that placement_info can be correctly updated after world transform changes.

        This verifies the logic used in apply_surface_projection() to keep
        placement_info.position_2d in sync with obj.transform after resolution.
        """
        # Create a surface at known position.
        surface_transform = RigidTransform(
            p=[2.0, 3.0, 0.75],  # Table at (2, 3) at height 0.75.
            rpy=RollPitchYaw([0.0, 0.0, 0.0]),
        )
        surface = SupportSurface(
            surface_id=UniqueID("table_surface"),
            bounding_box_min=np.array([-0.5, -0.3, 0.0]),
            bounding_box_max=np.array([0.5, 0.3, 0.05]),
            transform=surface_transform,
            mesh=None,
        )

        # Create object with initial placement_info at (0.1, 0.05) on surface.
        initial_pos_2d = np.array([0.1, 0.05])
        initial_rot_2d = 0.3  # radians

        placement_info = PlacementInfo(
            position_2d=initial_pos_2d.copy(),
            rotation_2d=initial_rot_2d,
            parent_surface_id=surface.surface_id,
        )

        # Initial world transform from placement_info.
        initial_world_transform = surface.to_world_pose(initial_pos_2d, initial_rot_2d)

        # Simulate physics resolution moving the object to a new world position.
        # Say physics moved it by (0.05, 0.02) in world coordinates.
        delta_world = np.array([0.05, 0.02, 0.0])
        new_world_translation = initial_world_transform.translation() + delta_world
        new_world_transform = RigidTransform(
            R=initial_world_transform.rotation(), p=new_world_translation
        )

        # Use from_world_pose to compute new placement_info values.
        # This is exactly what apply_surface_projection() does after IK solve.
        new_pos_2d, new_rot_2d = surface.from_world_pose(new_world_transform)

        # Update placement_info (simulating what apply_surface_projection does).
        placement_info.position_2d = new_pos_2d.copy()
        placement_info.rotation_2d = new_rot_2d

        # Verify the new placement_info converts back to the new world transform.
        reconstructed_world = surface.to_world_pose(
            placement_info.position_2d, placement_info.rotation_2d
        )

        np.testing.assert_array_almost_equal(
            reconstructed_world.translation(),
            new_world_transform.translation(),
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            reconstructed_world.rotation().matrix(),
            new_world_transform.rotation().matrix(),
            decimal=6,
        )

        # Also verify that the position_2d changed by expected amount.
        # For non-rotated surface, world delta equals surface delta.
        pos_2d_delta = placement_info.position_2d - initial_pos_2d
        self.assertAlmostEqual(pos_2d_delta[0], delta_world[0], places=5)
        self.assertAlmostEqual(pos_2d_delta[1], delta_world[1], places=5)

    def test_placement_info_sync_with_rotated_surface(self) -> None:
        """Test placement_info sync with a rotated surface (45 degree yaw)."""
        # Surface rotated 45 degrees around Z.
        yaw = np.pi / 4
        surface_transform = RigidTransform(
            p=[1.0, 1.0, 0.5],
            rpy=RollPitchYaw([0.0, 0.0, yaw]),
        )
        surface = SupportSurface(
            surface_id=UniqueID("rotated_table"),
            bounding_box_min=np.array([-0.4, -0.4, 0.0]),
            bounding_box_max=np.array([0.4, 0.4, 0.05]),
            transform=surface_transform,
            mesh=None,
        )

        # Initial position at center of surface.
        initial_pos_2d = np.array([0.0, 0.0])
        initial_rot_2d = 0.0

        placement_info = PlacementInfo(
            position_2d=initial_pos_2d.copy(),
            rotation_2d=initial_rot_2d,
            parent_surface_id=surface.surface_id,
        )

        initial_world = surface.to_world_pose(initial_pos_2d, initial_rot_2d)

        # Physics moves object 0.1m in world +X direction.
        delta_world = np.array([0.1, 0.0, 0.0])
        new_world = RigidTransform(
            R=initial_world.rotation(),
            p=initial_world.translation() + delta_world,
        )

        # Update placement_info via from_world_pose.
        new_pos_2d, new_rot_2d = surface.from_world_pose(new_world)
        placement_info.position_2d = new_pos_2d.copy()
        placement_info.rotation_2d = new_rot_2d

        # Verify round-trip.
        reconstructed = surface.to_world_pose(
            placement_info.position_2d, placement_info.rotation_2d
        )
        np.testing.assert_array_almost_equal(
            reconstructed.translation(), new_world.translation(), decimal=6
        )

        # For 45-degree rotated surface, a world +X move projects onto surface
        # as diagonal (positive X and negative Y in surface frame).
        # cos(45) = sin(45) = sqrt(2)/2 ≈ 0.707.
        sqrt2_2 = np.sqrt(2) / 2
        expected_surface_delta = np.array([0.1 * sqrt2_2, -0.1 * sqrt2_2])
        actual_delta = placement_info.position_2d - initial_pos_2d
        np.testing.assert_array_almost_equal(
            actual_delta, expected_surface_delta, decimal=5
        )


class TestXYRegionConstraintIntegration(unittest.TestCase):
    """Integration test for XY region constraints in solve_non_penetration_ik."""

    def test_constraint_keeps_objects_in_region(self) -> None:
        """Test that XY region constraint keeps objects within bounds.

        This is a focused test verifying that AddPointInSetConstraints
        correctly restricts object positions.
        """
        # Create a simple 2D optimization with XY region constraint.
        prog = MathematicalProgram()

        # 2 decision variables for x, y position.
        xy = prog.NewContinuousVariables(2, "xy")

        # Define allowed region: square from (-0.5, -0.5) to (0.5, 0.5).
        region = HPolyhedron.MakeBox(
            lb=np.array([-0.5, -0.5]),
            ub=np.array([0.5, 0.5]),
        )

        # Add point-in-set constraint.
        region.AddPointInSetConstraints(prog, xy)

        # Add cost to push toward (2.0, 2.0) - outside the region.
        prog.AddQuadraticCost(
            np.eye(2),  # Q
            -2 * np.array([2.0, 2.0]),  # b (push toward (2, 2))
            0.0,  # c
            xy,
        )

        # Solve.
        result = Solve(prog)
        self.assertTrue(result.is_success())

        # Solution should be at corner of region, not (2, 2).
        solution = result.GetSolution(xy)
        self.assertAlmostEqual(solution[0], 0.5, places=3)
        self.assertAlmostEqual(solution[1], 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
