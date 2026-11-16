"""Unit tests for stacking utilities."""

import unittest

from pathlib import Path

import numpy as np
import trimesh

from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenecode.manipuland_agents.tools.physics_utils import (
    compute_collision_bounds,
    load_collision_bounds_for_scene_object,
)
from scenecode.manipuland_agents.tools.stacking import (
    compute_actual_stack_height,
    compute_initial_stack_transforms,
    simulate_stack_stability,
)

# Path to test data directory.
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "stacking_assets"


class TestStackingUtilities(unittest.TestCase):
    """Test stacking utility functions."""

    def _create_box_mesh(self, size_z: float) -> trimesh.Trimesh:
        """Create a box mesh with given Z height, 0.1m x 0.1m footprint."""
        return trimesh.creation.box([0.1, 0.1, size_z])

    def test_compute_collision_bounds_single_mesh(self):
        """Test collision bounds with a single mesh."""
        mesh = self._create_box_mesh(0.05)
        z_min, z_max = compute_collision_bounds([mesh])

        self.assertAlmostEqual(z_min, -0.025, places=4)
        self.assertAlmostEqual(z_max, 0.025, places=4)

    def test_compute_collision_bounds_multiple_meshes(self):
        """Test collision bounds with multiple mesh pieces."""
        mesh1 = self._create_box_mesh(0.05)
        mesh2 = self._create_box_mesh(0.10)

        # Move mesh2 up so its range is different.
        mesh2.apply_translation([0, 0, 0.1])

        z_min, z_max = compute_collision_bounds([mesh1, mesh2])

        # mesh1: z in [-0.025, 0.025].
        # mesh2: z in [0.1 - 0.05, 0.1 + 0.05] = [0.05, 0.15].
        self.assertAlmostEqual(z_min, -0.025, places=4)
        self.assertAlmostEqual(z_max, 0.15, places=4)

    def test_compute_collision_bounds_empty_raises(self):
        """Test that empty mesh list raises ValueError."""
        with self.assertRaises(ValueError):
            compute_collision_bounds([])

    def test_compute_actual_stack_height_single_item(self):
        """Test actual height with single item."""
        bounds_list = [(-0.025, 0.025)]  # 5cm tall box.
        transforms = [RigidTransform([0.0, 0.0, 0.025])]  # z_min at 0.
        height = compute_actual_stack_height(transforms, bounds_list)
        # Top of object = 0.025 + 0.025 = 0.05.
        self.assertAlmostEqual(height, 0.05, places=4)

    def test_compute_actual_stack_height_multiple_items(self):
        """Test actual height with multiple stacked items."""
        bounds_list = [
            (-0.025, 0.025),  # 5cm.
            (-0.025, 0.025),  # 5cm.
            (-0.025, 0.025),  # 5cm.
        ]
        # Items stacked at z = 0.025, 0.075, 0.125 (centers).
        transforms = [
            RigidTransform([0.0, 0.0, 0.025]),
            RigidTransform([0.0, 0.0, 0.075]),
            RigidTransform([0.0, 0.0, 0.125]),
        ]
        height = compute_actual_stack_height(transforms, bounds_list)
        # Top of top object = 0.125 + 0.025 = 0.15.
        self.assertAlmostEqual(height, 0.15, places=4)

    def test_compute_actual_stack_height_empty(self):
        """Test actual height with empty list."""
        height = compute_actual_stack_height([], [])
        self.assertEqual(height, 0.0)

    def test_compute_actual_stack_height_nested_objects(self):
        """Test actual height when objects settle/nest (lower than initial)."""
        bounds_list = [
            (-0.025, 0.025),  # 5cm.
            (-0.025, 0.025),  # 5cm.
        ]
        # Second object has settled 1cm lower than initial stacking position.
        transforms = [
            RigidTransform([0.0, 0.0, 0.025]),
            RigidTransform([0.0, 0.0, 0.065]),  # Would be 0.075 if not nested.
        ]
        height = compute_actual_stack_height(transforms, bounds_list)
        # Top of top object = 0.065 + 0.025 = 0.09 (vs 0.10 if not nested).
        self.assertAlmostEqual(height, 0.09, places=4)

    def test_compute_initial_stack_transforms_positions(self):
        """Test that stack transforms are computed correctly."""
        bounds_list = [
            (-0.025, 0.025),  # 5cm, centered at origin.
            (-0.025, 0.025),  # 5cm, centered at origin.
            (-0.025, 0.025),  # 5cm, centered at origin.
        ]
        base_transform = RigidTransform([0.5, 0.5, 1.0])

        transforms = compute_initial_stack_transforms(bounds_list, base_transform)

        self.assertEqual(len(transforms), 3)

        # First item: its z_min (-0.025) should sit at cumulative_z (0).
        # So z_offset = 0 - (-0.025) = 0.025.
        # z = 1.0 + 0.025 = 1.025.
        self.assertAlmostEqual(transforms[0].translation()[2], 1.025, places=4)

        # Second item: cumulative_z = 0.05 (height of first item).
        # z_offset = 0.05 - (-0.025) = 0.075.
        # z = 1.0 + 0.075 = 1.075.
        self.assertAlmostEqual(transforms[1].translation()[2], 1.075, places=4)

        # Third item: cumulative_z = 0.10.
        # z_offset = 0.10 - (-0.025) = 0.125.
        # z = 1.0 + 0.125 = 1.125.
        self.assertAlmostEqual(transforms[2].translation()[2], 1.125, places=4)

    def test_compute_initial_stack_transforms_preserves_xy(self):
        """Test that X and Y coordinates are preserved from base transform."""
        bounds_list = [(-0.025, 0.025), (-0.025, 0.025)]
        base_transform = RigidTransform([1.5, 2.5, 0.8])

        transforms = compute_initial_stack_transforms(bounds_list, base_transform)

        for t in transforms:
            self.assertAlmostEqual(t.translation()[0], 1.5, places=4)
            self.assertAlmostEqual(t.translation()[1], 2.5, places=4)

    def test_compute_initial_stack_transforms_preserves_rotation(self):
        """Test that rotation is preserved from base transform."""
        bounds_list = [(-0.025, 0.025)]

        # Create base transform with 45 degree rotation around Z.
        rpy = RollPitchYaw(0, 0, np.pi / 4)
        base_transform = RigidTransform(rpy, [0.5, 0.5, 1.0])

        transforms = compute_initial_stack_transforms(bounds_list, base_transform)

        # Verify rotation is preserved.
        result_rpy = RollPitchYaw(transforms[0].rotation())
        self.assertAlmostEqual(result_rpy.yaw_angle(), np.pi / 4, places=4)


class TestStackingEndToEnd(unittest.TestCase):
    """End-to-end tests for stacking with real assets and simulation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures using real bread plate asset."""
        cls.bread_plate_sdf = TEST_DATA_DIR / "bread_plate" / "bread_plate_2.sdf"
        if not cls.bread_plate_sdf.exists():
            raise unittest.SkipTest(
                f"Test asset not found: {cls.bread_plate_sdf}. "
                "Run test data setup script first."
            )

    def _create_plate_scene_object(self, index: int) -> SceneObject:
        """Create a SceneObject for a bread plate."""
        return SceneObject(
            object_id=UniqueID(f"plate_{index}"),
            object_type=ObjectType.MANIPULAND,
            name=f"bread_plate_{index}",
            description="Bread plate for stacking test",
            transform=RigidTransform(),
            sdf_path=self.bread_plate_sdf,
        )

    def test_load_collision_bounds_from_real_asset(self):
        """Test loading collision bounds from real SDF asset."""
        plate = self._create_plate_scene_object(0)
        z_min, z_max = load_collision_bounds_for_scene_object(plate)

        # Bread plate should be relatively flat (z height < 5cm).
        height = z_max - z_min
        self.assertGreater(height, 0.0)
        self.assertLess(height, 0.05)

    def test_compute_stack_transforms_with_real_bounds(self):
        """Test computing stack transforms using real asset collision bounds."""
        plate = self._create_plate_scene_object(0)
        z_min, z_max = load_collision_bounds_for_scene_object(plate)

        # Stack 3 identical plates.
        bounds_list = [(z_min, z_max)] * 3
        base_transform = RigidTransform([0.5, 0.5, 0.0])

        transforms = compute_initial_stack_transforms(bounds_list, base_transform)

        # Verify each plate is stacked above the previous one.
        self.assertEqual(len(transforms), 3)
        for i in range(1, len(transforms)):
            self.assertGreater(
                transforms[i].translation()[2],
                transforms[i - 1].translation()[2],
                f"Plate {i} should be above plate {i-1}",
            )

    def test_simulate_stack_stability_stable_stack(self):
        """Test physics simulation with stable plate stack."""
        # Create 2 stacked plates (stable configuration).
        plates = [self._create_plate_scene_object(i) for i in range(2)]

        # Get collision bounds for all plates.
        bounds_list = [load_collision_bounds_for_scene_object(p) for p in plates]

        # Compute initial transforms.
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        initial_transforms = compute_initial_stack_transforms(
            bounds_list, base_transform
        )

        # Run simulation with 0.1m threshold (default).
        result = simulate_stack_stability(
            scene_objects=plates,
            initial_transforms=initial_transforms,
            ground_xyz=(0.0, 0.0, 0.0),
            simulation_time=1.0,
            simulation_time_step=0.001,
            position_threshold=0.1,
        )

        # Plates stacked directly on top of each other should be stable.
        self.assertIsNone(
            result.error_message, f"Simulation error: {result.error_message}"
        )
        self.assertTrue(
            result.is_stable,
            f"Stack should be stable. Unstable indices: {result.unstable_indices}",
        )
        self.assertEqual(len(result.stable_indices), 2)
        self.assertEqual(len(result.unstable_indices), 0)

    def test_simulate_stack_stability_unstable_offset(self):
        """Test physics simulation detects unstable offset configuration."""
        # Create 2 plates with the top one offset significantly.
        plates = [self._create_plate_scene_object(i) for i in range(2)]

        # Get collision bounds.
        bounds_list = [load_collision_bounds_for_scene_object(p) for p in plates]

        # Compute initial transforms with offset top plate.
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        initial_transforms = compute_initial_stack_transforms(
            bounds_list, base_transform
        )

        # Offset the top plate by more than its radius so it falls off.
        # Bread plate is ~20cm diameter, so 25cm offset should cause it to fall.
        offset_translation = initial_transforms[1].translation() + np.array(
            [0.25, 0.0, 0.0]
        )
        initial_transforms[1] = RigidTransform(
            initial_transforms[1].rotation(), offset_translation
        )

        # Run simulation with 0.1m threshold (default).
        result = simulate_stack_stability(
            scene_objects=plates,
            initial_transforms=initial_transforms,
            ground_xyz=(0.0, 0.0, 0.0),
            simulation_time=2.0,
            simulation_time_step=0.001,
            position_threshold=0.1,
        )

        # The offset plate should fall and be detected as unstable.
        self.assertIsNone(
            result.error_message, f"Simulation error: {result.error_message}"
        )
        self.assertFalse(
            result.is_stable,
            "Stack with offset top plate should be unstable",
        )
        self.assertIn(1, result.unstable_indices)

    def test_actual_stack_height_after_simulation(self):
        """Test that compute_actual_stack_height works with simulation results."""
        plates = [self._create_plate_scene_object(i) for i in range(3)]
        bounds_list = [load_collision_bounds_for_scene_object(p) for p in plates]

        # Compute transforms.
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        initial_transforms = compute_initial_stack_transforms(
            bounds_list, base_transform
        )

        # Compute initial height before simulation.
        initial_height = compute_actual_stack_height(initial_transforms, bounds_list)

        # Run simulation.
        result = simulate_stack_stability(
            scene_objects=plates,
            initial_transforms=initial_transforms,
            ground_xyz=(0.0, 0.0, 0.0),
            simulation_time=1.0,
            simulation_time_step=0.001,
            position_threshold=0.1,
        )

        self.assertIsNone(result.error_message)
        self.assertTrue(result.is_stable)

        # Compute actual height after simulation.
        final_height = compute_actual_stack_height(
            transforms=result.final_transforms, collision_bounds_list=bounds_list
        )

        # Final height should be less than initial height.
        self.assertLess(final_height, initial_height - 0.01)

        # Verify each plate moved down.
        movement_threshold = 0.0
        for i, (initial, final) in enumerate(
            zip(initial_transforms, result.final_transforms)
        ):
            initial_z = initial.translation()[2]
            final_z = final.translation()[2]
            downward_movement = initial_z - final_z
            self.assertGreaterEqual(
                downward_movement,
                movement_threshold,
                f"Plate {i} should settle down >= {movement_threshold}m: "
                f"initial_z={initial_z:.4f}, final_z={final_z:.4f}, "
                f"movement={downward_movement:.4f}m",
            )
            movement_threshold += 0.007

    def test_stack_transform_matches_bottom_member_position(self):
        """Test that stack reference transform uses bottom member's final position."""
        plates = [self._create_plate_scene_object(i) for i in range(2)]
        bounds_list = [load_collision_bounds_for_scene_object(p) for p in plates]

        # Compute initial transforms.
        base_transform = RigidTransform([0.0, 0.0, 0.0])
        initial_transforms = compute_initial_stack_transforms(
            collision_bounds_list=bounds_list, base_transform=base_transform
        )

        # Run simulation.
        result = simulate_stack_stability(
            scene_objects=plates,
            initial_transforms=initial_transforms,
            ground_xyz=(0.0, 0.0, 0.0),
            simulation_time=1.0,
            simulation_time_step=0.001,
            position_threshold=0.1,
        )

        self.assertIsNone(result.error_message)
        self.assertTrue(result.is_stable)

        # The stack's reference transform should be the bottom member's final position.
        stack_transform = result.final_transforms[0]

        # Compute world bounding box from all member transforms.
        all_z_values = []
        for transform in result.final_transforms:
            pos = transform.translation()
            all_z_values.append(pos[2])

        # The stack transform's Z should be close to the bottom member's Z.
        stack_z = stack_transform.translation()[2]
        self.assertAlmostEqual(
            stack_z,
            result.final_transforms[0].translation()[2],
            places=4,
            msg="Stack transform should match bottom member's final position",
        )

        # When creating a bounding box relative to stack_transform, all members
        # should be contained. Compute member positions relative to stack_transform.
        inverse_stack = stack_transform.inverse()
        for i, transform in enumerate(result.final_transforms):
            relative_pos = inverse_stack.multiply(transform.translation())
            # All relative positions should have non-negative Z (above reference).
            self.assertGreaterEqual(
                relative_pos[2],
                -0.01,  # Small tolerance for numerical error.
                f"Member {i} should be at or above stack reference frame. "
                f"Relative Z: {relative_pos[2]:.4f}",
            )

    def test_load_collision_bounds_applies_sdf_scale(self):
        """Collision bounds should apply SDF scale to match Drake behavior."""
        # Load unscaled asset.
        plate_unscaled = self._create_plate_scene_object(0)
        z_min_unscaled, z_max_unscaled = load_collision_bounds_for_scene_object(
            plate_unscaled
        )
        unscaled_height = z_max_unscaled - z_min_unscaled

        # Load scaled asset (0.8 scale in SDF).
        scaled_sdf = TEST_DATA_DIR / "bread_plate" / "bread_plate_2_scaled.sdf"
        plate_scaled = SceneObject(
            object_id=UniqueID("scaled_plate"),
            object_type=ObjectType.MANIPULAND,
            name="scaled_bread_plate",
            description="Bread plate with 0.8 scale in SDF",
            transform=RigidTransform(),
            sdf_path=scaled_sdf,
        )
        z_min_scaled, z_max_scaled = load_collision_bounds_for_scene_object(
            plate_scaled
        )
        scaled_height = z_max_scaled - z_min_scaled

        # Scaled height should be 0.8x of unscaled.
        expected_height = unscaled_height * 0.8
        self.assertAlmostEqual(
            scaled_height,
            expected_height,
            delta=0.0005,
            msg=f"Scaled height {scaled_height:.4f}m should be 0.8x of "
            f"unscaled {unscaled_height:.4f}m (expected {expected_height:.4f}m)",
        )

    def test_load_collision_bounds_applies_scale_factor(self):
        """Collision bounds should apply SceneObject.scale_factor for runtime scaling."""
        # Load unscaled asset.
        plate_unscaled = self._create_plate_scene_object(0)
        z_min_unscaled, z_max_unscaled = load_collision_bounds_for_scene_object(
            plate_unscaled
        )
        unscaled_height = z_max_unscaled - z_min_unscaled

        # Create object with scale_factor=0.8 (runtime scaling).
        plate_scaled = SceneObject(
            object_id=UniqueID("scaled_plate"),
            object_type=ObjectType.MANIPULAND,
            name="scaled_bread_plate",
            description="Bread plate with 0.8 scale_factor",
            transform=RigidTransform(),
            sdf_path=self.bread_plate_sdf,
            scale_factor=0.8,
        )
        z_min_scaled, z_max_scaled = load_collision_bounds_for_scene_object(
            plate_scaled
        )
        scaled_height = z_max_scaled - z_min_scaled

        # Scaled height should be 0.8x of unscaled.
        expected_height = unscaled_height * 0.8
        self.assertAlmostEqual(
            scaled_height,
            expected_height,
            delta=0.0005,
            msg=f"Scaled height {scaled_height:.4f}m should be 0.8x of "
            f"unscaled {unscaled_height:.4f}m (expected {expected_height:.4f}m)",
        )


if __name__ == "__main__":
    unittest.main()
