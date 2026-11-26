"""Unit tests for fill container utilities."""

import unittest

from pathlib import Path

import numpy as np
import trimesh

from pydrake.all import RigidTransform
from pydrake.math import RotationMatrix

from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenecode.manipuland_agents.tools.fill_container import (
    ContainerInteriorBounds,
    _compute_fill_object_rotation,
    _should_flip_for_thick_end_up,
    compute_container_interior_bounds,
    compute_fill_spawn_transforms,
    simulate_fill_physics,
)

# Path to test data directory.
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "stacking_assets"


class TestContainerInteriorBounds(unittest.TestCase):
    """Test container interior bounds computation."""

    def _create_bowl_mesh(
        self, radius: float = 0.1, depth: float = 0.05
    ) -> trimesh.Trimesh:
        """Create a simple bowl-like mesh (hemisphere)."""
        # Create a sphere and cut off the bottom half.
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        # Keep only vertices with z > -depth (top portion).
        bowl = sphere.slice_plane([0, 0, -depth], [0, 0, 1])
        return bowl

    def _create_box_container(
        self, width: float = 0.2, depth: float = 0.15, height: float = 0.1
    ) -> trimesh.Trimesh:
        """Create a box-shaped container (open top)."""
        # Create outer box.
        outer = trimesh.creation.box([width, depth, height])
        # Offset to have bottom at z=0.
        outer.apply_translation([0, 0, height / 2])
        return outer

    def test_interior_bounds_has_valid_hull(self):
        """Test that interior bounds has a valid convex hull."""
        mesh = self._create_box_container(width=0.2, depth=0.15, height=0.1)
        interior = compute_container_interior_bounds(
            collision_meshes=[mesh],
            top_rim_height_fraction=0.15,
            interior_scale=0.95,
        )

        self.assertIsInstance(interior, ContainerInteriorBounds)
        self.assertGreater(len(interior.hull_vertices_2d), 2)
        self.assertEqual(interior.hull_vertices_2d.shape[1], 2)

    def test_interior_bounds_centroid_near_origin(self):
        """Test that centered container has centroid near origin."""
        mesh = self._create_box_container(width=0.2, depth=0.15, height=0.1)
        interior = compute_container_interior_bounds(
            collision_meshes=[mesh],
            top_rim_height_fraction=0.15,
            interior_scale=1.0,  # No scaling for this test.
        )

        # Centroid should be near origin for centered container.
        self.assertAlmostEqual(interior.centroid_2d[0], 0.0, places=2)
        self.assertAlmostEqual(interior.centroid_2d[1], 0.0, places=2)

    def test_interior_bounds_top_z_at_top(self):
        """Test that top_z is at or near the top of the container."""
        height = 0.1
        mesh = self._create_box_container(width=0.2, depth=0.15, height=height)
        interior = compute_container_interior_bounds(
            collision_meshes=[mesh],
            top_rim_height_fraction=0.15,
            interior_scale=1.0,
        )

        # top_z should be near the top of the container.
        # Box container has bottom at z=0, top at z=height.
        # The top_z is computed from top 15% of vertices, so it should be
        # in the upper portion of the container.
        self.assertGreater(interior.top_z, height * 0.5)
        self.assertLessEqual(interior.top_z, height + 0.01)  # Small tolerance.
        # bottom_z should be less than top_z.
        self.assertLess(interior.bottom_z, interior.top_z)

    def test_interior_scale_reduces_area(self):
        """Test that interior_scale < 1.0 reduces the spawn area."""
        mesh = self._create_box_container(width=0.2, depth=0.15, height=0.1)

        interior_full = compute_container_interior_bounds(
            collision_meshes=[mesh],
            top_rim_height_fraction=0.15,
            interior_scale=1.0,
        )
        interior_scaled = compute_container_interior_bounds(
            collision_meshes=[mesh],
            top_rim_height_fraction=0.15,
            interior_scale=0.8,
        )

        # Scaled interior should have smaller bounding box.
        full_range_x = (
            interior_full.hull_vertices_2d[:, 0].max()
            - interior_full.hull_vertices_2d[:, 0].min()
        )
        scaled_range_x = (
            interior_scaled.hull_vertices_2d[:, 0].max()
            - interior_scaled.hull_vertices_2d[:, 0].min()
        )
        self.assertLess(scaled_range_x, full_range_x)


def _create_box_mesh(size_x: float, size_y: float, size_z: float) -> trimesh.Trimesh:
    """Create a simple box mesh with given dimensions, centered at origin."""
    return trimesh.creation.box([size_x, size_y, size_z])


def _create_meshes_with_height(height: float) -> list[trimesh.Trimesh]:
    """Create collision meshes with specified height (z_min=0, z_max=height)."""
    box = _create_box_mesh(0.02, 0.02, height)
    # Translate so bottom is at z=0.
    box.apply_translation([0, 0, height / 2])
    return [box]


class TestFillSpawnTransforms(unittest.TestCase):
    """Test fill spawn transform computation."""

    def test_spawn_transforms_count_matches_input(self):
        """Test that spawn transforms count matches fill objects count."""
        # Create 3 meshes with different heights.
        meshes_list = [
            _create_meshes_with_height(0.05),
            _create_meshes_with_height(0.03),
            _create_meshes_with_height(0.04),
        ]
        interior = ContainerInteriorBounds(
            hull_vertices_2d=np.array(
                [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]
            ),
            centroid_2d=np.array([0.0, 0.0]),
            top_z=0.1,
            bottom_z=0.0,
        )
        container_transform = RigidTransform([0.5, 0.5, 0.0])
        rng = np.random.default_rng(42)

        transforms = compute_fill_spawn_transforms(
            fill_collision_meshes=meshes_list,
            container_interior=interior,
            container_transform=container_transform,
            spawn_height_above_rim=0.05,
            rng=rng,
        )

        self.assertEqual(len(transforms), 3)

    def test_spawn_transforms_above_rim(self):
        """Test that spawn positions are above container rim."""
        meshes_list = [_create_meshes_with_height(0.05)]
        top_z = 0.1
        spawn_height = 0.05
        interior = ContainerInteriorBounds(
            hull_vertices_2d=np.array(
                [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]
            ),
            centroid_2d=np.array([0.0, 0.0]),
            top_z=top_z,
            bottom_z=0.0,
        )
        container_transform = RigidTransform([0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)

        transforms = compute_fill_spawn_transforms(
            fill_collision_meshes=meshes_list,
            container_interior=interior,
            container_transform=container_transform,
            spawn_height_above_rim=spawn_height,
            rng=rng,
        )

        # Spawn Z should be >= top_z + spawn_height (+ half object height).
        expected_min_z = top_z + spawn_height
        actual_z = transforms[0].translation()[2]
        self.assertGreaterEqual(actual_z, expected_min_z - 0.001)  # Small tolerance.

    def test_spawn_transforms_within_interior_bounds(self):
        """Test that spawn XY positions are within interior bounds."""
        meshes_list = [_create_meshes_with_height(0.02) for _ in range(5)]  # 5 objects.
        interior = ContainerInteriorBounds(
            hull_vertices_2d=np.array(
                [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]
            ),
            centroid_2d=np.array([0.0, 0.0]),
            top_z=0.1,
            bottom_z=0.0,
        )
        container_transform = RigidTransform([0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)

        transforms = compute_fill_spawn_transforms(
            fill_collision_meshes=meshes_list,
            container_interior=interior,
            container_transform=container_transform,
            spawn_height_above_rim=0.05,
            rng=rng,
        )

        # All XY positions should be within the hull bounds.
        for t in transforms:
            x, y = t.translation()[:2]
            self.assertGreaterEqual(x, -0.1)
            self.assertLessEqual(x, 0.1)
            self.assertGreaterEqual(y, -0.1)
            self.assertLessEqual(y, 0.1)


class TestFillSimulationEndToEnd(unittest.TestCase):
    """End-to-end tests for fill simulation with real assets."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures using real bread plate asset as container/fill."""
        cls.bread_plate_sdf = TEST_DATA_DIR / "bread_plate" / "bread_plate_2.sdf"
        if not cls.bread_plate_sdf.exists():
            raise unittest.SkipTest(
                f"Test asset not found: {cls.bread_plate_sdf}. "
                "Run test data setup script first."
            )

    def _create_scene_object(self, name: str, index: int) -> SceneObject:
        """Create a SceneObject for testing."""
        return SceneObject(
            object_id=UniqueID(f"{name}_{index}"),
            object_type=ObjectType.MANIPULAND,
            name=f"{name}_{index}",
            description=f"Test {name} {index}",
            transform=RigidTransform(),
            sdf_path=self.bread_plate_sdf,
        )

    def test_simulate_fill_returns_valid_result(self):
        """Test that fill simulation returns valid FillSimulationResult."""
        container = self._create_scene_object("container", 0)
        fills = [self._create_scene_object("fill", i) for i in range(2)]

        container_transform = RigidTransform([0.0, 0.0, 0.0])
        # Simple initial transforms above container.
        fill_transforms = [
            RigidTransform([0.0, 0.0, 0.2]),
            RigidTransform([0.02, 0.0, 0.25]),
        ]

        result = simulate_fill_physics(
            container_scene_object=container,
            container_transform=container_transform,
            new_fill_objects=fills,
            new_fill_transforms=fill_transforms,
            catch_floor_z=-5.0,
            inside_z_threshold=-2.0,
            simulation_time=1.0,
            simulation_time_step=0.001,
        )

        # Should have valid result structure.
        self.assertIsNotNone(result)
        self.assertEqual(len(result.final_transforms), 2)
        # Inside + outside should equal total fills.
        self.assertEqual(len(result.inside_indices) + len(result.outside_indices), 2)

    def test_simulate_fill_detects_fallen_objects(self):
        """Test that objects spawned outside container are detected as outside."""
        container = self._create_scene_object("container", 0)
        fills = [self._create_scene_object("fill", i) for i in range(2)]

        container_transform = RigidTransform([0.0, 0.0, 0.0])
        # One object above container, one far to the side (will fall).
        fill_transforms = [
            RigidTransform([0.0, 0.0, 0.2]),  # Above container.
            RigidTransform([1.0, 0.0, 0.2]),  # Far to the side - will fall.
        ]

        result = simulate_fill_physics(
            container_scene_object=container,
            container_transform=container_transform,
            new_fill_objects=fills,
            new_fill_transforms=fill_transforms,
            catch_floor_z=-5.0,
            inside_z_threshold=-2.0,
            simulation_time=2.0,
            simulation_time_step=0.001,
        )

        # The second object should fall and be detected as outside.
        self.assertIn(1, result.outside_indices)

    def test_simulate_fill_preserves_objects_count(self):
        """Test that simulation preserves all object transforms."""
        container = self._create_scene_object("container", 0)
        n_fills = 3
        fills = [self._create_scene_object("fill", i) for i in range(n_fills)]

        container_transform = RigidTransform([0.0, 0.0, 0.0])
        fill_transforms = [
            RigidTransform([0.01 * i, 0.0, 0.2 + 0.05 * i]) for i in range(n_fills)
        ]

        result = simulate_fill_physics(
            container_scene_object=container,
            container_transform=container_transform,
            new_fill_objects=fills,
            new_fill_transforms=fill_transforms,
            catch_floor_z=-5.0,
            inside_z_threshold=-2.0,
            simulation_time=1.0,
            simulation_time_step=0.001,
        )

        # Should have transforms for all fills.
        self.assertEqual(len(result.final_transforms), n_fills)

    def test_simulate_fill_with_settled_objects(self):
        """Test that settled objects from previous iterations are included."""
        container = self._create_scene_object("container", 0)

        # First batch: 2 objects that settle inside.
        settled_fills = [self._create_scene_object("settled", i) for i in range(2)]
        settled_transforms = [
            RigidTransform([0.0, 0.0, 0.05]),  # Resting on container floor.
            RigidTransform([0.02, 0.0, 0.08]),  # Stacked on first.
        ]

        # New object to simulate.
        new_fills = [self._create_scene_object("new", 0)]
        new_transforms = [RigidTransform([0.0, 0.0, 0.2])]  # Spawned above.

        container_transform = RigidTransform([0.0, 0.0, 0.0])

        result = simulate_fill_physics(
            container_scene_object=container,
            container_transform=container_transform,
            new_fill_objects=new_fills,
            new_fill_transforms=new_transforms,
            settled_fill_objects=settled_fills,
            settled_fill_transforms=settled_transforms,
            catch_floor_z=-5.0,
            inside_z_threshold=-2.0,
            simulation_time=1.0,
            simulation_time_step=0.001,
        )

        # Result should only have transforms for NEW objects.
        self.assertEqual(len(result.final_transforms), 1)
        # The new object should be classified.
        self.assertEqual(len(result.inside_indices) + len(result.outside_indices), 1)


class TestThickEndUpOrientation(unittest.TestCase):
    """Test thick-end-up orientation logic for asymmetric objects."""

    def _create_utensil_mesh(self) -> np.ndarray:
        """Create vertices for a utensil-like asymmetric object.

        Shape: thin handle (cylinder) with thick head (disk).
        - Handle: thin cylinder along Z (radius 0.01, height 0.15)
        - Head: thick disk at top (radius 0.03, height 0.02)

        This simulates a spatula or pan flipper where the business end
        is thicker than the handle.
        """
        # Handle vertices (cylinder approximated by 8 points per ring, 2 rings).
        handle_radius = 0.01
        handle_height = 0.15
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        handle_bottom = np.column_stack(
            [
                handle_radius * np.cos(angles),
                handle_radius * np.sin(angles),
                np.zeros(8),
            ]
        )
        handle_top = np.column_stack(
            [
                handle_radius * np.cos(angles),
                handle_radius * np.sin(angles),
                np.full(8, handle_height),
            ]
        )

        # Head vertices (disk at top).
        head_radius = 0.03
        head_bottom_z = handle_height
        head_top_z = handle_height + 0.02
        head_bottom = np.column_stack(
            [
                head_radius * np.cos(angles),
                head_radius * np.sin(angles),
                np.full(8, head_bottom_z),
            ]
        )
        head_top = np.column_stack(
            [
                head_radius * np.cos(angles),
                head_radius * np.sin(angles),
                np.full(8, head_top_z),
            ]
        )

        return np.vstack([handle_bottom, handle_top, head_bottom, head_top])

    def _create_symmetric_cylinder_mesh(self) -> np.ndarray:
        """Create vertices for a symmetric cylinder (no thick end)."""
        radius = 0.02
        height = 0.15
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        bottom = np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros(16),
            ]
        )
        top = np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full(16, height),
            ]
        )
        return np.vstack([bottom, top])

    def _create_test_container_interior(
        self, width: float = 0.3, depth: float = 0.2
    ) -> ContainerInteriorBounds:
        """Create a simple rectangular container interior for testing."""
        # Create rectangular hull (width along X, depth along Y).
        half_w, half_d = width / 2, depth / 2
        hull_vertices_2d = np.array(
            [
                [-half_w, -half_d],
                [half_w, -half_d],
                [half_w, half_d],
                [-half_w, half_d],
            ]
        )
        return ContainerInteriorBounds(
            hull_vertices_2d=hull_vertices_2d,
            centroid_2d=np.array([0.0, 0.0]),
            top_z=0.1,
            bottom_z=0.0,
        )

    def test_should_flip_for_thick_end_up_with_utensil(self):
        """Test that a utensil with thick end on bottom triggers flip."""
        # Create utensil with thick end at top.
        vertices = self._create_utensil_mesh()

        # Create identity rotation (no rotation applied yet).
        rotation = RotationMatrix()

        # With thick end on top (as created), should NOT flip.
        self.assertFalse(_should_flip_for_thick_end_up(vertices, rotation))

        # Now flip the utensil upside down (thick end on bottom).
        flipped_vertices = vertices.copy()
        flipped_vertices[:, 2] = -flipped_vertices[:, 2]  # Negate Z.
        flipped_vertices[:, 2] += 0.17  # Shift back to positive Z.

        # With thick end on bottom, SHOULD flip.
        self.assertTrue(_should_flip_for_thick_end_up(flipped_vertices, rotation))

    def test_should_not_flip_symmetric_object(self):
        """Test that a symmetric cylinder does not trigger flip."""
        vertices = self._create_symmetric_cylinder_mesh()
        rotation = RotationMatrix()

        # Symmetric object should not flip.
        self.assertFalse(_should_flip_for_thick_end_up(vertices, rotation))

    def test_compute_orientation_with_vertices_flips_thick_end_up(self):
        """Test that orientation function flips asymmetric objects correctly."""
        # Create utensil-like mesh with thick end at bottom (wrong orientation).
        vertices = self._create_utensil_mesh()
        # Flip upside down.
        vertices[:, 2] = -vertices[:, 2]
        vertices[:, 2] += 0.17

        # Object is elongated along Z (already vertical) but thick end is down.
        extents = (0.06, 0.06, 0.17)  # dx, dy, dz (dz is longest).
        container_interior = self._create_test_container_interior()

        rotation = _compute_fill_object_rotation(
            extents=extents,
            container_interior=container_interior,
            vertices=vertices,
        )

        # Apply rotation to vertices.
        rotated = (rotation.matrix() @ vertices.T).T

        # After rotation, the thick end (larger XY extent) should be on top.
        z_min, z_max = rotated[:, 2].min(), rotated[:, 2].max()
        center_z = (z_min + z_max) / 2

        top_verts = rotated[rotated[:, 2] > center_z]
        bottom_verts = rotated[rotated[:, 2] <= center_z]

        def xy_extent(verts):
            return (
                verts[:, 0].max() - verts[:, 0].min(),
                verts[:, 1].max() - verts[:, 1].min(),
            )

        top_extent = xy_extent(top_verts)
        bottom_extent = xy_extent(bottom_verts)
        top_area = top_extent[0] * top_extent[1]
        bottom_area = bottom_extent[0] * bottom_extent[1]

        # Top should have larger XY extent (thick end up).
        self.assertGreater(top_area, bottom_area * 1.1)

    def test_compute_orientation_preserves_symmetric_object(self):
        """Test that symmetric objects are not unnecessarily flipped."""
        vertices = self._create_symmetric_cylinder_mesh()
        extents = (0.04, 0.04, 0.15)  # dx, dy, dz (dz is longest).
        container_interior = self._create_test_container_interior()

        rotation = _compute_fill_object_rotation(
            extents=extents,
            container_interior=container_interior,
            vertices=vertices,
        )

        # Apply rotation to vertices.
        rotated = (rotation.matrix() @ vertices.T).T

        # For symmetric object, top and bottom should have similar XY extent.
        z_min, z_max = rotated[:, 2].min(), rotated[:, 2].max()
        center_z = (z_min + z_max) / 2

        top_verts = rotated[rotated[:, 2] > center_z]
        bottom_verts = rotated[rotated[:, 2] <= center_z]

        def xy_area(verts):
            x_extent = verts[:, 0].max() - verts[:, 0].min()
            y_extent = verts[:, 1].max() - verts[:, 1].min()
            return x_extent * y_extent

        top_area = xy_area(top_verts)
        bottom_area = xy_area(bottom_verts)

        # Areas should be within 10% (not flipped unnecessarily).
        ratio = max(top_area, bottom_area) / min(top_area, bottom_area)
        self.assertLess(ratio, 1.2)  # Within 20% is close enough.


if __name__ == "__main__":
    unittest.main()
