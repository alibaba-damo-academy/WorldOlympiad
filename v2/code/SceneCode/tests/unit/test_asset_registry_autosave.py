"""Tests for AssetRegistry auto-save functionality."""

import json
import tempfile

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenecode.agent_utils.asset_registry import AssetRegistry
from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID


def test_auto_save_on_register():
    """Test that registry auto-saves after each registration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.json"

        # Create registry with auto-save enabled.
        registry = AssetRegistry(auto_save_path=registry_path)

        # Initially, registry file should not exist.
        assert not registry_path.exists()

        # Register first asset.
        asset1 = SceneObject(
            object_id=UniqueID("test_asset_1"),
            object_type=ObjectType.FURNITURE,
            name="Test Asset 1",
            description="A test asset",
            transform=RigidTransform(),
            geometry_path=None,
            sdf_path=None,
            image_path=None,
            support_surfaces=[],
            metadata={},
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 1.0, 1.0]),
        )
        registry.register(asset=asset1)

        # Registry should have been auto-saved.
        assert registry_path.exists()

        # Load and verify content.
        with open(registry_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "test_asset_1" in data
        assert data["test_asset_1"]["name"] == "Test Asset 1"

        # Register second asset.
        asset2 = SceneObject(
            object_id=UniqueID("test_asset_2"),
            object_type=ObjectType.MANIPULAND,
            name="Test Asset 2",
            description="Another test asset",
            transform=RigidTransform(),
            geometry_path=None,
            sdf_path=None,
            image_path=None,
            support_surfaces=[],
            metadata={},
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 1.0, 1.0]),
        )
        registry.register(asset=asset2)

        # Registry should have both assets saved.
        with open(registry_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "test_asset_1" in data
        assert "test_asset_2" in data
        assert len(data) == 2


def test_no_auto_save_when_path_not_set():
    """Test that registry doesn't try to save when auto_save_path is None."""
    # Create registry without auto-save path.
    registry = AssetRegistry(auto_save_path=None)

    # Register asset (should not crash even though no save path).
    asset = SceneObject(
        object_id=UniqueID("test_asset"),
        object_type=ObjectType.FURNITURE,
        name="Test Asset",
        description="A test asset",
        transform=RigidTransform(),
        geometry_path=None,
        sdf_path=None,
        image_path=None,
        support_surfaces=[],
        metadata={},
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
    )
    registry.register(asset=asset)

    # Should complete without error.
    assert registry.size() == 1


def test_load_from_file_and_continue_auto_save():
    """Test that loading a registry and then registering new assets continues to auto-save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.json"

        # Create and save first registry.
        registry1 = AssetRegistry(auto_save_path=registry_path)
        asset1 = SceneObject(
            object_id=UniqueID("asset_1"),
            object_type=ObjectType.FURNITURE,
            name="Asset 1",
            description="First asset",
            transform=RigidTransform(),
            geometry_path=None,
            sdf_path=None,
            image_path=None,
            support_surfaces=[],
            metadata={},
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 1.0, 1.0]),
        )
        registry1.register(asset=asset1)
        assert registry_path.exists()

        # Create new registry and load from file.
        registry2 = AssetRegistry(auto_save_path=registry_path)
        registry2.load_from_file(file_path=registry_path)
        assert registry2.size() == 1

        # Register new asset - should auto-save.
        asset2 = SceneObject(
            object_id=UniqueID("asset_2"),
            object_type=ObjectType.FURNITURE,
            name="Asset 2",
            description="Second asset",
            transform=RigidTransform(),
            geometry_path=None,
            sdf_path=None,
            image_path=None,
            support_surfaces=[],
            metadata={},
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 1.0, 1.0]),
        )
        registry2.register(asset=asset2)

        # Load file and verify both assets are present.
        with open(registry_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert "asset_1" in data
        assert "asset_2" in data
