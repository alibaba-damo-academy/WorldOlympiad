import unittest

from unittest.mock import Mock

from scenecode.agent_utils.asset_registry import AssetRegistry
from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID


class TestAssetRegistry(unittest.TestCase):
    """Test cases for AssetRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = AssetRegistry()

        # Create mock asset.
        self.mock_asset = Mock(spec=SceneObject)
        self.mock_asset.object_id = UniqueID("test_asset_0")
        self.mock_asset.name = "Test Asset"
        self.mock_asset.object_type = ObjectType.FURNITURE

    def test_initialization(self):
        """Test registry initialization."""
        self.assertEqual(self.registry.size(), 0)

    def test_register_and_get(self):
        """Test registering and retrieving assets."""
        # Register asset.
        self.registry.register(self.mock_asset)
        self.assertEqual(self.registry.size(), 1)

        # Retrieve asset.
        retrieved = self.registry.get(self.mock_asset.object_id)
        self.assertEqual(retrieved, self.mock_asset)

    def test_get_nonexistent(self):
        """Test getting non-existent asset."""
        fake_id = UniqueID("nonexistent_0")
        result = self.registry.get(fake_id)
        self.assertIsNone(result)

    def test_exists(self):
        """Test checking if asset exists."""
        self.assertFalse(self.registry.exists(self.mock_asset.object_id))

        self.registry.register(self.mock_asset)
        self.assertTrue(self.registry.exists(self.mock_asset.object_id))

    def test_list_all(self):
        """Test listing all assets."""
        self.assertEqual(len(self.registry.list_all()), 0)

        self.registry.register(self.mock_asset)
        assets = self.registry.list_all()
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0], self.mock_asset)

    def test_clear(self):
        """Test clearing registry."""
        self.registry.register(self.mock_asset)
        self.assertEqual(self.registry.size(), 1)

        self.registry.clear()
        self.assertEqual(self.registry.size(), 0)

    def test_register_duplicate_raises_error(self):
        """Test that registering same ID raises ValueError."""
        # Register original asset.
        self.registry.register(self.mock_asset)

        # Create new asset with same ID.
        new_asset = Mock(spec=SceneObject)
        new_asset.object_id = self.mock_asset.object_id
        new_asset.name = "New Asset"

        # Register should raise ValueError.
        with self.assertRaises(ValueError) as cm:
            self.registry.register(new_asset)
        self.assertIn("already registered", str(cm.exception))

        # Original should still be there.
        self.assertEqual(self.registry.size(), 1)
        retrieved = self.registry.get(self.mock_asset.object_id)
        self.assertEqual(retrieved, self.mock_asset)

    def test_generate_unique_id(self):
        """Test generating unique IDs for assets."""
        # Generate ID with empty registry.
        id1 = self.registry.generate_unique_id("chair")
        self.assertEqual(str(id1), "chair_0")

        # Register an asset with that ID.
        asset1 = Mock(spec=SceneObject)
        asset1.object_id = id1
        asset1.name = "Test Chair"
        self.registry.register(asset1)

        # Next ID should be incremented.
        id2 = self.registry.generate_unique_id("chair")
        self.assertEqual(str(id2), "chair_1")


if __name__ == "__main__":
    unittest.main()
