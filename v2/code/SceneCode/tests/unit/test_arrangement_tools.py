"""Unit tests for arrangement tools."""

import unittest

from unittest.mock import patch

import numpy as np

from pydrake.math import RigidTransform

from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenecode.manipuland_agents.tools.arrangement_tools import (
    _get_container_bounds_info,
    _validate_item_within_container_bounds,
)


class TestValidateItemWithinContainerBounds(unittest.TestCase):
    """Test item position validation within container bounds."""

    def _create_mock_item(self, name: str = "test_item") -> SceneObject:
        """Create a minimal SceneObject for testing (only name used in errors)."""
        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.MANIPULAND,
            name=name,
            description="Test item",
            transform=RigidTransform(),
            sdf_path=None,
        )

    # Rectangular container tests.

    def test_center_valid_rectangular(self):
        """Item at center (0, 0) is valid for rectangular container."""
        bounds = {"shape": "rectangular", "x": [-0.15, 0.15], "y": [-0.10, 0.10]}
        item = self._create_mock_item("plate")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.0, y_offset=0.0, bounds_info=bounds
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_inside_bounds_valid_rectangular(self):
        """Item inside bounds is valid."""
        bounds = {"shape": "rectangular", "x": [-0.15, 0.15], "y": [-0.10, 0.10]}
        item = self._create_mock_item("knife")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.10, y_offset=0.05, bounds_info=bounds
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_at_edge_valid_rectangular(self):
        """Item exactly at edge is valid (center check only)."""
        bounds = {"shape": "rectangular", "x": [-0.15, 0.15], "y": [-0.10, 0.10]}
        item = self._create_mock_item("fork")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.15, y_offset=0.0, bounds_info=bounds
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_outside_x_bound_fails_rectangular(self):
        """Item outside X bounds fails with error message."""
        bounds = {"shape": "rectangular", "x": [-0.15, 0.15], "y": [-0.10, 0.10]}
        item = self._create_mock_item("plate")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.20, y_offset=0.0, bounds_info=bounds
        )

        self.assertFalse(is_valid)
        self.assertIn("plate", error)
        self.assertIn("0.20", error)
        self.assertIn("-0.15", error)
        self.assertIn("0.15", error)

    def test_outside_y_bound_fails_rectangular(self):
        """Item outside Y bounds fails with error message."""
        bounds = {"shape": "rectangular", "x": [-0.15, 0.15], "y": [-0.10, 0.10]}
        item = self._create_mock_item("mug")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.0, y_offset=0.15, bounds_info=bounds
        )

        self.assertFalse(is_valid)
        self.assertIn("mug", error)
        self.assertIn("-0.10", error)
        self.assertIn("0.10", error)

    def test_negative_position_inside_valid_rectangular(self):
        """Negative positions inside bounds are valid."""
        bounds = {"shape": "rectangular", "x": [-0.15, 0.15], "y": [-0.10, 0.10]}
        item = self._create_mock_item("napkin")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=-0.12, y_offset=-0.08, bounds_info=bounds
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    # Circular container tests.

    def test_center_valid_circular(self):
        """Item at center (0, 0) is valid for circular container."""
        bounds = {"shape": "circular", "radius": 0.15}
        item = self._create_mock_item("cheese")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.0, y_offset=0.0, bounds_info=bounds
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_inside_radius_valid_circular(self):
        """Item inside radius is valid."""
        bounds = {"shape": "circular", "radius": 0.15}
        item = self._create_mock_item("cracker")

        # Distance = sqrt(0.08^2 + 0.06^2) = 0.10 < 0.15.
        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.08, y_offset=0.06, bounds_info=bounds
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_at_radius_valid_circular(self):
        """Item exactly at radius is valid."""
        bounds = {"shape": "circular", "radius": 0.15}
        item = self._create_mock_item("olive")

        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.15, y_offset=0.0, bounds_info=bounds
        )

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_outside_radius_fails_circular(self):
        """Item outside radius fails with error message."""
        bounds = {"shape": "circular", "radius": 0.15}
        item = self._create_mock_item("grape")

        # Distance = sqrt(0.12^2 + 0.12^2) = 0.17 > 0.15.
        is_valid, error = _validate_item_within_container_bounds(
            item_asset=item, x_offset=0.12, y_offset=0.12, bounds_info=bounds
        )

        self.assertFalse(is_valid)
        self.assertIn("grape", error)
        self.assertIn("0.17", error)  # Distance.
        self.assertIn("0.15", error)  # Radius.


class TestGetContainerBoundsInfo(unittest.TestCase):
    """Test container bounds info computation."""

    def _create_container(
        self,
        name: str,
        width: float,
        depth: float,
        height: float = 0.05,
    ) -> SceneObject:
        """Create a SceneObject with bounding box for testing."""
        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.MANIPULAND,
            name=name,
            description=f"Test container {name}",
            transform=RigidTransform(),
            sdf_path=None,
            bbox_min=np.array([-width / 2, -depth / 2, 0.0]),
            bbox_max=np.array([width / 2, depth / 2, height]),
        )

    @patch("scenecode.manipuland_agents.tools.arrangement_tools.is_circular_object")
    def test_rectangular_container_returns_xy_bounds(self, mock_is_circular):
        """Rectangular container returns x and y bound ranges."""
        mock_is_circular.return_value = False
        container = self._create_container("tray", width=0.40, depth=0.25)

        bounds = _get_container_bounds_info(container_asset=container, cfg=None)

        self.assertEqual(bounds["shape"], "rectangular")
        self.assertAlmostEqual(bounds["x"][0], -0.20, places=4)
        self.assertAlmostEqual(bounds["x"][1], 0.20, places=4)
        self.assertAlmostEqual(bounds["y"][0], -0.125, places=4)
        self.assertAlmostEqual(bounds["y"][1], 0.125, places=4)

    @patch("scenecode.manipuland_agents.tools.arrangement_tools.is_circular_object")
    def test_circular_container_returns_radius(self, mock_is_circular):
        """Circular container returns radius (min dimension / 2)."""
        mock_is_circular.return_value = True
        # Width 0.30, depth 0.28 -> radius = min(0.30, 0.28) / 2 = 0.14.
        container = self._create_container("platter", width=0.30, depth=0.28)

        bounds = _get_container_bounds_info(container_asset=container, cfg=None)

        self.assertEqual(bounds["shape"], "circular")
        self.assertAlmostEqual(bounds["radius"], 0.14, places=4)

    def test_missing_bbox_raises_error(self):
        """Container without bounding box raises ValueError."""
        container = SceneObject(
            object_id=UniqueID("no_bbox"),
            object_type=ObjectType.MANIPULAND,
            name="no_bbox",
            description="Container without bbox",
            transform=RigidTransform(),
            sdf_path=None,
            bbox_min=None,
            bbox_max=None,
        )

        with self.assertRaises(ValueError) as ctx:
            _get_container_bounds_info(container_asset=container, cfg=None)

        self.assertIn("no bounding box", str(ctx.exception))
        self.assertIn("no_bbox", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
