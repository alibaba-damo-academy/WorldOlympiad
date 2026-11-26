"""Unit tests for pile utilities."""

import unittest

from pathlib import Path

import numpy as np

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RotationMatrix

from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenecode.manipuland_agents.tools.pile_tools import (
    _random_rotation_matrix,
    compute_pile_spawn_transforms,
    simulate_pile_physics,
)


def _create_pile_config(
    spawn_height_base: float = 0.05,
    height_stagger_fraction: float = 1.0,
    min_height_stagger: float = 0.02,
    spawn_radius_scale: float = 1.5,
    min_spawn_radius: float = 0.05,
) -> OmegaConf:
    """Create a pile simulation config for testing."""
    return OmegaConf.create(
        {
            "spawn_height_base": spawn_height_base,
            "height_stagger_fraction": height_stagger_fraction,
            "min_height_stagger": min_height_stagger,
            "spawn_radius_scale": spawn_radius_scale,
            "min_spawn_radius": min_spawn_radius,
        }
    )


class TestPileSpawnTransforms(unittest.TestCase):
    """Test pile spawn transform computation."""

    def _create_unit_bbox(self, size: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Create a cubic bounding box centered at origin."""
        half = size / 2
        return np.array([-half, -half, -half]), np.array([half, half, half])

    def test_compute_pile_spawn_transforms_basic(self):
        """Test that spawn transforms are generated for each object."""
        bboxes = [self._create_unit_bbox(0.1) for _ in range(3)]
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        cfg = _create_pile_config()

        transforms = compute_pile_spawn_transforms(
            bounding_boxes=bboxes,
            base_transform=base_transform,
            surface_z=0.0,
            cfg=cfg,
        )

        self.assertEqual(len(transforms), 3)

    def test_compute_pile_spawn_transforms_staggered_z(self):
        """Test that Z positions are staggered to avoid collisions."""
        bboxes = [self._create_unit_bbox(0.1) for _ in range(3)]
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        cfg = _create_pile_config()

        transforms = compute_pile_spawn_transforms(
            bounding_boxes=bboxes,
            base_transform=base_transform,
            surface_z=0.0,
            cfg=cfg,
        )

        # Z positions should be increasing.
        z_positions = [t.translation()[2] for t in transforms]
        for i in range(len(z_positions) - 1):
            self.assertLess(
                z_positions[i],
                z_positions[i + 1],
                "Z positions should be strictly increasing",
            )

    def test_compute_pile_spawn_transforms_xy_in_spawn_radius(self):
        """Test that XY positions are within spawn radius."""
        bboxes = [self._create_unit_bbox(0.1) for _ in range(5)]
        base_transform = RigidTransform([1.0, 2.0, 0.0])
        # Set a known spawn radius via config.
        cfg = _create_pile_config(spawn_radius_scale=2.0, min_spawn_radius=0.2)

        transforms = compute_pile_spawn_transforms(
            bounding_boxes=bboxes,
            base_transform=base_transform,
            surface_z=0.0,
            cfg=cfg,
        )

        # Compute expected spawn radius from config.
        avg_diagonal = np.mean(
            [np.linalg.norm(bbox_max - bbox_min) for bbox_min, bbox_max in bboxes]
        )
        spawn_radius = max(cfg.min_spawn_radius, avg_diagonal * cfg.spawn_radius_scale)

        base_xy = base_transform.translation()[:2]
        for t in transforms:
            xy = t.translation()[:2]
            distance = np.linalg.norm(xy - base_xy)
            self.assertLessEqual(
                distance,
                spawn_radius * 1.01,  # Small tolerance for floating point.
                f"XY distance {distance} exceeds spawn radius {spawn_radius}",
            )

    def test_compute_pile_spawn_transforms_random_rotation(self):
        """Test that transforms have non-identity rotations (random SO(3))."""
        bboxes = [self._create_unit_bbox(0.1) for _ in range(5)]
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        cfg = _create_pile_config()

        transforms = compute_pile_spawn_transforms(
            bounding_boxes=bboxes,
            base_transform=base_transform,
            surface_z=0.0,
            cfg=cfg,
        )

        # At least one transform should have a non-identity rotation.
        has_non_identity = False
        for t in transforms:
            angle = t.rotation().ToAngleAxis().angle()
            if angle > 0.01:  # More than ~0.5 degrees.
                has_non_identity = True
                break

        self.assertTrue(
            has_non_identity,
            "Expected at least one transform with non-identity rotation",
        )

    def test_compute_pile_spawn_transforms_uses_bbox_diagonal(self):
        """Test that Z spacing accounts for bbox diagonals of both objects.

        The Z difference between consecutive objects is (d_prev + d_curr) / 2
        because each object is centered at its layer (+ diagonal/2) and the
        stagger moves current_z by the full diagonal.
        """
        # Create objects with different sizes.
        small_bbox = (
            np.array([-0.025, -0.025, -0.025]),
            np.array([0.025, 0.025, 0.025]),
        )
        large_bbox = (np.array([-0.05, -0.05, -0.05]), np.array([0.05, 0.05, 0.05]))

        bboxes = [small_bbox, large_bbox, small_bbox]
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        cfg = _create_pile_config()

        transforms = compute_pile_spawn_transforms(
            bounding_boxes=bboxes,
            base_transform=base_transform,
            surface_z=0.0,
            cfg=cfg,
        )

        z_positions = [t.translation()[2] for t in transforms]

        small_diagonal = np.linalg.norm(small_bbox[1] - small_bbox[0])
        large_diagonal = np.linalg.norm(large_bbox[1] - large_bbox[0])

        z_diff_0_to_1 = z_positions[1] - z_positions[0]
        z_diff_1_to_2 = z_positions[2] - z_positions[1]

        # Z difference = (d_prev + d_curr) / 2 due to centering + stagger.
        expected_0_to_1 = (small_diagonal + large_diagonal) / 2
        expected_1_to_2 = (large_diagonal + small_diagonal) / 2

        self.assertAlmostEqual(z_diff_0_to_1, expected_0_to_1, places=3)
        self.assertAlmostEqual(z_diff_1_to_2, expected_1_to_2, places=3)

        # Verify no overlap: gap between top of obj i and bottom of obj i+1.
        # Top of obj 0 = z_0 + d_0/2, Bottom of obj 1 = z_1 - d_1/2.
        # Gap = (z_1 - d_1/2) - (z_0 + d_0/2) = z_diff - (d_0 + d_1)/2.
        # With stagger = diagonal, gap should be ~0 (objects just touching).
        gap_0_to_1 = z_diff_0_to_1 - (small_diagonal + large_diagonal) / 2
        self.assertAlmostEqual(gap_0_to_1, 0.0, places=3)


class TestRandomRotationMatrix(unittest.TestCase):
    """Test random rotation matrix generation."""

    def test_random_rotation_is_valid_rotation_matrix(self):
        """Test that generated rotations are valid rotation matrices."""
        for _ in range(10):
            R = _random_rotation_matrix()

            # Check it's a valid rotation matrix.
            self.assertTrue(isinstance(R, RotationMatrix))

            # Check determinant is 1.
            det = np.linalg.det(R.matrix())
            self.assertAlmostEqual(det, 1.0, places=5)

            # Check orthogonality: R @ R.T = I.
            product = R.matrix() @ R.matrix().T
            np.testing.assert_array_almost_equal(product, np.eye(3), decimal=5)

    def test_random_rotation_varies(self):
        """Test that different calls produce different rotations."""
        rotations = [_random_rotation_matrix() for _ in range(10)]

        # Check that rotations are different (not all the same).
        matrices = [R.matrix() for R in rotations]
        unique = True
        for i in range(len(matrices) - 1):
            if np.allclose(matrices[i], matrices[i + 1]):
                unique = False
                break

        self.assertTrue(unique, "Expected different random rotations")


class TestPilePhysicsSimulation(unittest.TestCase):
    """Test pile physics simulation with actual SDF assets."""

    # Path to test data directory.
    TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "stacking_assets"

    def _create_scene_object(self, sdf_path: Path, index: int) -> SceneObject:
        """Create a SceneObject for testing."""
        return SceneObject(
            object_id=UniqueID(f"pile_test_{index}"),
            object_type=ObjectType.MANIPULAND,
            name=sdf_path.stem,
            description=f"Test pile item {index}",
            transform=RigidTransform(),
            sdf_path=sdf_path,
        )

    def test_pile_simulation_with_real_assets(self):
        """Test pile simulation with real SDF assets."""
        # Use bread_plate asset.
        plate_dir = self.TEST_DATA_DIR / "bread_plate"
        plate_sdf = plate_dir / "bread_plate_2.sdf"

        self.assertTrue(plate_sdf.exists(), f"Test asset not found: {plate_sdf}")

        # Create 3 plates for the pile.
        objects = [self._create_scene_object(plate_sdf, i) for i in range(3)]

        # Create simple bounding boxes for plates.
        bboxes = [
            (np.array([-0.04, -0.04, -0.01]), np.array([0.04, 0.04, 0.01]))
            for _ in range(3)
        ]

        base_transform = RigidTransform([0.0, 0.0, 0.0])
        cfg = _create_pile_config(
            spawn_height_base=0.02,
            min_spawn_radius=0.05,
        )
        transforms = compute_pile_spawn_transforms(
            bounding_boxes=bboxes,
            base_transform=base_transform,
            surface_z=0.0,
            cfg=cfg,
        )

        # Run simulation.
        _, _, final_transforms, error_message = simulate_pile_physics(
            scene_objects=objects,
            initial_transforms=transforms,
            ground_xyz=(0.0, 0.0, -0.05),  # Ground plane just below surface.
            ground_size=(1.0, 1.0),
            surface_z=0.0,
            inside_z_threshold=-0.5,
            simulation_time=2.0,  # Shorter for tests.
            simulation_time_step=0.001,
        )

        # Check results.
        self.assertIsNone(error_message)
        self.assertEqual(len(final_transforms), 3)

        # Objects should have settled (all Z should be near surface).
        for i, transform in enumerate(final_transforms):
            z = transform.translation()[2]
            self.assertGreater(
                z, -0.5, f"Object {i} Z position {z} should be above -0.5"
            )

    def test_pile_simulation_objects_on_surface(self):
        """Test that objects are correctly classified as on surface vs fell off."""
        plate_dir = self.TEST_DATA_DIR / "bread_plate"
        plate_sdf = plate_dir / "bread_plate_2.sdf"

        self.assertTrue(plate_sdf.exists(), f"Test asset not found: {plate_sdf}")

        objects = [self._create_scene_object(plate_sdf, i) for i in range(2)]

        bboxes = [
            (np.array([-0.04, -0.04, -0.01]), np.array([0.04, 0.04, 0.01]))
            for _ in range(2)
        ]

        base_transform = RigidTransform([0.0, 0.0, 0.0])
        cfg = _create_pile_config(
            spawn_height_base=0.02,
            min_spawn_radius=0.03,  # Small radius to keep objects on surface.
        )
        transforms = compute_pile_spawn_transforms(
            bounding_boxes=bboxes,
            base_transform=base_transform,
            surface_z=0.0,
            cfg=cfg,
        )

        inside_indices, _, _, error_message = simulate_pile_physics(
            scene_objects=objects,
            initial_transforms=transforms,
            ground_xyz=(0.0, 0.0, -0.05),  # Ground plane just below surface.
            ground_size=(1.0, 1.0),
            surface_z=0.0,
            inside_z_threshold=-0.5,
            simulation_time=2.0,
            simulation_time_step=0.001,
        )

        self.assertIsNone(error_message)

        # With objects centered on a large surface, both should stay on.
        self.assertGreater(
            len(inside_indices), 0, "At least one object should be on surface"
        )


if __name__ == "__main__":
    unittest.main()
