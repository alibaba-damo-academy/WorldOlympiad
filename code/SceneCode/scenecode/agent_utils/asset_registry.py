"""Asset registry for tracking generated assets during a session."""

import json
import logging

from pathlib import Path

import numpy as np

from pydrake.all import Quaternion, RigidTransform, RotationMatrix

from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID

console_logger = logging.getLogger(__name__)


class AssetRegistry:
    """Registry for tracking generated assets within a session (tracked in memory)."""

    def __init__(self, auto_save_path: Path | None = None) -> None:
        """Initialize empty registry.

        Args:
            auto_save_path: If provided, registry will auto-save to this path
                after each registration. This ensures the registry is saved
                incrementally, even if scene generation fails partway through.
        """
        self._assets: dict[UniqueID, SceneObject] = {}
        self.auto_save_path = auto_save_path
        console_logger.info(
            f"Initialized AssetRegistry"
            f"{' with auto-save to ' + str(auto_save_path) if auto_save_path else ''}"
        )

    def register(self, asset: SceneObject) -> None:
        """Register a generated asset.

        Args:
            asset: SceneObject to register for reuse.

        Raises:
            ValueError: If asset_id already exists in registry.
        """
        asset_id = asset.object_id
        if asset_id in self._assets:
            raise ValueError(
                f"Asset {asset_id} already registered. Use generate_unique_id() "
                f"to generate collision-free IDs."
            )

        self._assets[asset_id] = asset
        console_logger.info(f"Registered asset {asset_id} ({asset.name})")

        # Auto-save if path is configured (fail-fast on errors).
        if self.auto_save_path:
            self.save_to_file(file_path=self.auto_save_path)
            console_logger.debug(f"Auto-saved registry to {self.auto_save_path}")

    def generate_unique_id(self, name: str) -> UniqueID:
        """Generate unique ID that doesn't conflict with registered assets.

        Args:
            name: Human-readable name for the asset.

        Returns:
            UniqueID guaranteed not to conflict with existing assets.
        """
        return UniqueID.generate_unique(name, self._assets)

    def get(self, asset_id: UniqueID) -> SceneObject | None:
        """Get asset by ID.

        Args:
            asset_id: Unique identifier of the asset.

        Returns:
            SceneObject if found, None otherwise.
        """
        asset = self._assets.get(asset_id)
        if asset:
            console_logger.debug(f"Retrieved asset {asset_id}")
        else:
            console_logger.debug(f"Asset {asset_id} not found in registry")
        return asset

    def list_all(self) -> list[SceneObject]:
        """List all registered assets.

        Returns:
            List of all registered SceneObjects.
        """
        assets = list(self._assets.values())
        console_logger.debug(f"Listed {len(assets)} available assets")
        return assets

    def exists(self, asset_id: UniqueID) -> bool:
        """Check if asset exists in registry.

        Args:
            asset_id: Unique identifier to check.

        Returns:
            True if asset exists, False otherwise.
        """
        return asset_id in self._assets

    def clear(self) -> None:
        """Clear all registered assets."""
        count = len(self._assets)
        self._assets.clear()
        console_logger.info(f"Cleared {count} assets from registry")

    def size(self) -> int:
        """Get number of registered assets."""
        return len(self._assets)

    def apply_scale_by_sdf_path(self, sdf_path: Path, scale_factor: float) -> int:
        """Apply scale to all registry entries with matching sdf_path.

        Updates bbox_min, bbox_max, and scale_factor for all assets that share
        the given SDF file. This keeps the registry in sync after rescaling.

        Args:
            sdf_path: Path to the SDF file that was rescaled.
            scale_factor: Scale multiplier that was applied.

        Returns:
            Number of registry entries updated.
        """
        updated_count = 0
        for asset in self._assets.values():
            if asset.sdf_path == sdf_path:
                asset.apply_scale(scale_factor)
                updated_count += 1

        if updated_count > 0:
            console_logger.info(
                f"Updated {updated_count} registry entries for rescaled asset "
                f"{sdf_path.name}"
            )

            # Auto-save if path is configured.
            if self.auto_save_path:
                self.save_to_file(file_path=self.auto_save_path)
                console_logger.debug(f"Auto-saved registry after rescale")

        return updated_count

    def save_to_file(self, file_path: Path) -> None:
        """Save registry to JSON file for persistence.

        Args:
            file_path: Path where registry JSON will be saved.
        """
        registry_data = {}

        for asset_id, asset in self._assets.items():
            # Serialize transform.
            translation = asset.transform.translation()
            rotation_quaternion = asset.transform.rotation().ToQuaternion()

            asset_data = {
                "object_id": str(asset.object_id),
                "object_type": asset.object_type.value,
                "name": asset.name,
                "description": asset.description,
                "transform": {
                    "translation": translation.tolist(),
                    "rotation_wxyz": [
                        rotation_quaternion.w(),
                        rotation_quaternion.x(),
                        rotation_quaternion.y(),
                        rotation_quaternion.z(),
                    ],
                },
                "geometry_path": (
                    str(asset.geometry_path) if asset.geometry_path else None
                ),
                "sdf_path": str(asset.sdf_path) if asset.sdf_path else None,
                "image_path": str(asset.image_path) if asset.image_path else None,
                "metadata": asset.metadata,
                "bbox_min": (
                    asset.bbox_min.tolist() if asset.bbox_min is not None else None
                ),
                "bbox_max": (
                    asset.bbox_max.tolist() if asset.bbox_max is not None else None
                ),
                "scale_factor": asset.scale_factor,
            }

            registry_data[str(asset_id)] = asset_data

        # Ensure parent directory exists.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file.
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)

        console_logger.info(f"Saved {len(registry_data)} assets to {file_path}")

    def load_from_file(self, file_path: Path) -> None:
        """Load registry from JSON file.

        Args:
            file_path: Path to registry JSON file.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Registry file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            registry_data = json.load(f)

        # Clear existing registry.
        self._assets.clear()

        # Reconstruct SceneObject instances from registry data.
        for asset_id_str, asset_data in registry_data.items():
            # Parse transform.
            transform_data = asset_data["transform"]
            translation = np.array(transform_data["translation"])
            rotation_wxyz = transform_data["rotation_wxyz"]
            quaternion = Quaternion(wxyz=rotation_wxyz)
            rotation_matrix = RotationMatrix(quaternion)
            transform = RigidTransform(rotation_matrix, translation)

            # Create SceneObject.
            scene_object = SceneObject(
                object_id=UniqueID(asset_data["object_id"]),
                object_type=ObjectType(asset_data["object_type"]),
                name=asset_data["name"],
                description=asset_data["description"],
                transform=transform,
                geometry_path=(
                    Path(asset_data["geometry_path"])
                    if asset_data["geometry_path"]
                    else None
                ),
                sdf_path=(
                    Path(asset_data["sdf_path"]) if asset_data["sdf_path"] else None
                ),
                image_path=(
                    Path(asset_data["image_path"]) if asset_data["image_path"] else None
                ),
                support_surfaces=[],
                metadata=asset_data.get("metadata", {}),
                bbox_min=(
                    np.array(asset_data["bbox_min"])
                    if asset_data.get("bbox_min")
                    else None
                ),
                bbox_max=(
                    np.array(asset_data["bbox_max"])
                    if asset_data.get("bbox_max")
                    else None
                ),
                scale_factor=asset_data.get("scale_factor", 1.0),
            )

            # Register the asset.
            self._assets[scene_object.object_id] = scene_object

        console_logger.info(f"Loaded {len(self._assets)} assets from {file_path}")
