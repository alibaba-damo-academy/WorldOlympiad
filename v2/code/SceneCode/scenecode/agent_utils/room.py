import hashlib
import json
import logging
import os
import uuid

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scenecode.agent_utils.house import RoomGeometry

import numpy as np
import trimesh

from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.geometry.optimization import VPolytope

from scenecode.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
    extract_support_surfaces_articulated,
    extract_support_surfaces_from_mesh,
)
from scenecode.utils.geometry_utils import compute_aabb_corners, safe_convex_hull_2d
from scenecode.utils.path_utils import safe_relative_path
from scenecode.utils.sdf_utils import (
    deserialize_rigid_transform,
    extract_base_link_name_from_sdf,
    is_static_sdf_model,
    serialize_rigid_transform,
)

console_logger = logging.getLogger(__name__)


def _int_to_base36(num: int) -> str:
    """Convert integer to base-36 (0-9, a-z) representation.

    Args:
        num: Non-negative integer to convert.

    Returns:
        Base-36 string representation.

    Examples:
        >>> _int_to_base36(0)
        '0'
        >>> _int_to_base36(10)
        'a'
        >>> _int_to_base36(36)
        '10'
    """
    if num < 10:
        return str(num)
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while num:
        result = chars[num % 36] + result
        num //= 36
    return result


class UniqueID(str):
    """Type-safe unique identifier string."""

    @classmethod
    def generate(cls) -> "UniqueID":
        """Generate a new unique ID using UUID4."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def generate_unique(
        cls, name: str, existing_ids: set["UniqueID"] | dict[str, Any]
    ) -> "UniqueID":
        """Generate unique ID that doesn't conflict with existing IDs.

        Uses base-36 sequential numbering (0-9, a-z) for compact IDs:
        - First occurrence: single digit suffix (e.g., "chair_0")
        - 11th: single letter (e.g., "chair_a")
        - 37th+: two chars (e.g., "chair_10")

        Args:
            name: Human-readable name for the object.
            existing_ids: Set or dict of IDs to check against for uniqueness.

        Returns:
            UniqueID guaranteed not to be in existing_ids.
        """
        base_name = name.lower().replace(" ", "_")
        index = 0
        while True:
            suffix = _int_to_base36(index)
            candidate = cls(f"{base_name}_{suffix}")
            if candidate not in existing_ids:
                return candidate
            index += 1


class ObjectType(Enum):
    """Enum for different types of objects in the scene."""

    FURNITURE = "furniture"
    MANIPULAND = "manipuland"
    THIN_COVERING = "thin_covering"  # Flat textured surface (no collision geometry)
    WALL_MOUNTED = "wall_mounted"
    CEILING_MOUNTED = "ceiling_mounted"
    WALL = "wall"
    FLOOR = "floor"
    EITHER = "either"  # Ambiguous items for analysis


class AgentType(Enum):
    """Pipeline agent types.

    Order defines pipeline execution order. All agents use the same
    planner/designer/critic trio architecture.
    """

    FLOOR_PLAN = "floor_plan"
    FURNITURE = "furniture"
    WALL_MOUNTED = "wall_mounted"
    CEILING_MOUNTED = "ceiling_mounted"
    MANIPULAND = "manipuland"

    def to_object_type(self) -> ObjectType | None:
        """Convert agent type to corresponding object type.

        Returns None for FLOOR_PLAN which doesn't produce scene objects.
        """
        if self == AgentType.FLOOR_PLAN:
            return None
        return ObjectType(self.value)

    @property
    def is_placement_agent(self) -> bool:
        """Whether this agent places objects via asset router."""
        return self != AgentType.FLOOR_PLAN


@dataclass
class PlacementInfo:
    """Metadata for objects placed on support surfaces.

    Stores the surface-relative placement information (SE(2) on surface) alongside
    the world-frame transform. This enables replay, debugging, and understanding of
    the hierarchical placement structure.
    """

    parent_surface_id: UniqueID
    """ID of the support surface this object is placed on."""

    position_2d: np.ndarray
    """2D position on the support surface in surface coordinates. Shape: (2,).
    Format: [x, y] where x is left-right and y is front-back in surface frame.
    """

    rotation_2d: float
    """Rotation in radians around the surface normal (Z-axis in surface frame)."""

    placement_method: str = "surface_placement"
    """Method used for placement (e.g., 'surface_placement', 'snap_to_edge')."""


@dataclass
class SupportSurface:
    """Represents a support surface where objects can be placed.

    A support surface is a flat surface that can be used to place objects on.
    It is represented by an axis-aligned bounding box and a transform, with optional
    sub-mesh geometry for accurate visualization and future polygon-based placement.
    """

    surface_id: UniqueID
    """Unique identifier for the support surface."""

    bounding_box_min: np.ndarray
    """Minimum corner of the 3D axis-aligned bounding box, in the surface's local
    coordinate frame (origin at surface position, see transform). Shape: (3,).
    """

    bounding_box_max: np.ndarray
    """Maximum corner of the 3D axis-aligned bounding box, in the surface's local
    coordinate frame (origin at surface position, see transform). Shape: (3,).
    """

    transform: RigidTransform
    """Pose of the support surface in the world frame."""

    mesh: "trimesh.Trimesh | None" = None
    """Simplified triangle mesh representing the support surface geometry.
    Mesh is in Z-up coordinate system (Drake/Blender standard), flattened to 2D plane.
    Ready for direct rendering without additional coordinate transformations.
    If None, only bounding box representation is available.

    Note: Future optimization could store mesh_path instead if memory becomes an issue,
    but storing the mesh directly simplifies the API and enables better mesh simplification
    during extraction.
    """

    link_name: str | None = None
    """For articulated objects: name of the link this surface belongs to.
    Used to apply FK transforms when rendering with joints open.
    None for non-articulated objects or if link association failed.
    """

    @property
    def area(self) -> float:
        """Compute surface area from XY bounding box dimensions.

        Returns:
            Surface area in square meters.
        """
        width = self.bounding_box_max[0] - self.bounding_box_min[0]
        depth = self.bounding_box_max[1] - self.bounding_box_min[1]
        return float(width * depth)

    def content_hash(self) -> str:
        """Generate content hash for this support surface."""
        content_dict = {
            "surface_id": str(self.surface_id),
            "bounding_box_min": [
                float(self.bounding_box_min[0]),
                float(self.bounding_box_min[1]),
                float(self.bounding_box_min[2]),
            ],
            "bounding_box_max": [
                float(self.bounding_box_max[0]),
                float(self.bounding_box_max[1]),
                float(self.bounding_box_max[2]),
            ],
            "transform": serialize_rigid_transform(self.transform),
        }

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(content_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_world_pose(
        self, position_2d: np.ndarray, rotation_2d: float, z_offset: float = 0.0
    ) -> RigidTransform:
        """Convert surface-relative SE(2) pose to world SE(3) pose.

        Takes a 2D position and rotation on the support surface and converts it to
        a full 3D pose in world coordinates. This is the key transformation for
        manipuland placement.

        Args:
            position_2d: 2D position on surface [x, y] in surface frame (meters).
            rotation_2d: Rotation around surface normal in radians.
            z_offset: Vertical offset from surface plane (meters). Use negative
                values to place objects below the surface plane (e.g., to
                compensate for gravity settling offset for thin coverings).

        Returns:
            RigidTransform representing the object's pose in world coordinates.
        """
        # Create surface-relative pose.
        # Z=0 is the surface plane, z_offset adjusts from there.
        surface_relative_pose = RigidTransform(
            p=[float(position_2d[0]), float(position_2d[1]), z_offset],
            rpy=RollPitchYaw([0.0, 0.0, rotation_2d]),
        )

        # Compose with surface transform to get world pose.
        world_pose = self.transform @ surface_relative_pose

        return world_pose

    def from_world_pose(
        self, world_transform: RigidTransform
    ) -> tuple[np.ndarray, float]:
        """Convert world SE(3) pose back to surface-relative SE(2).

        Inverse of to_world_pose(). Used after physics resolution to update
        placement_info with new positions.

        Args:
            world_transform: Object's pose in world coordinates.

        Returns:
            Tuple of (position_2d, rotation_2d) where:
            - position_2d: 2D position [x, y] in surface frame (meters)
            - rotation_2d: Rotation around surface normal in radians
        """
        # Compute surface-relative transform.
        surface_relative = self.transform.inverse() @ world_transform

        # Extract 2D position (XY in surface frame).
        position_2d = surface_relative.translation()[:2].copy()

        # Extract yaw rotation (rotation around surface normal / Z-axis).
        rotation_2d = RollPitchYaw(surface_relative.rotation()).yaw_angle()

        return position_2d, rotation_2d

    def contains_point_2d(self, position_2d: np.ndarray) -> bool:
        """Check if a 2D point lies within the surface convex hull.

        Uses the surface mesh's convex hull for accurate placement bounds on
        non-rectangular surfaces. This prevents placing objects outside the actual
        support surface geometry.

        Args:
            position_2d: 2D position to check [x, y] in surface frame (meters).

        Returns:
            True if the point is within the convex hull, False otherwise.

        Raises:
            ValueError: If surface has no mesh (mesh is required for convex hull).
        """
        assert position_2d.shape == (
            2,
        ), f"Expected 2D position, got shape {position_2d.shape}"

        # Fallback to AABB bounds check if no mesh geometry available.
        # This is the case for HSSD pre-validated surfaces.
        if self.mesh is None:
            console_logger.debug(
                f"Surface {self.surface_id} has no mesh geometry, using AABB bounds check"
            )
            in_x = (
                self.bounding_box_min[0] <= position_2d[0] <= self.bounding_box_max[0]
            )
            in_y = (
                self.bounding_box_min[1] <= position_2d[1] <= self.bounding_box_max[1]
            )
            return in_x and in_y

        # Both position_2d and self.mesh.vertices are in surface-local coordinates.
        # The mesh vertices were transformed to surface-local frame during surface creation.
        # No coordinate transformation needed - just use position_2d directly.
        point_xy = position_2d

        # Extract 2D vertices from mesh (XY plane in surface-local frame).
        mesh_xy_vertices = self.mesh.vertices[:, :2]

        # Compute 2D convex hull using safe wrapper.
        hull, processed_vertices = safe_convex_hull_2d(mesh_xy_vertices)
        if hull is None:
            console_logger.warning(
                f"Degenerate convex hull for surface {self.surface_id}. "
                "Falling back to AABB bounds check."
            )
            # Fallback: Check against bounding box instead.
            in_x = (
                self.bounding_box_min[0] <= position_2d[0] <= self.bounding_box_max[0]
            )
            in_y = (
                self.bounding_box_min[1] <= position_2d[1] <= self.bounding_box_max[1]
            )
            return in_x and in_y

        # Point-in-polygon test using convex hull.
        # For a convex polygon, point is inside if it's on the same side of all edges.
        # We use the cross product test for each edge.
        hull_vertices = processed_vertices[hull.vertices]
        n_vertices = len(hull_vertices)

        for i in range(n_vertices):
            # Get edge from vertex i to vertex (i+1) % n.
            v1 = hull_vertices[i]
            v2 = hull_vertices[(i + 1) % n_vertices]

            # Edge vector.
            edge = v2 - v1

            # Vector from v1 to test point.
            to_point = point_xy - v1

            # Cross product (2D): edge × to_point.
            # Positive means point is to the left of edge (inside for CCW hull).
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]

            # If point is to the right of any edge, it's outside the polygon.
            # Use small epsilon to allow points on boundary.
            epsilon = 1e-9
            if cross < -epsilon:
                return False

        return True

    def get_xy_convex_hull(self) -> VPolytope:
        """Get the XY boundary of this surface in world frame as a Drake VPolytope.

        Transforms surface boundary from local (surface) frame to world frame,
        then projects to world XY plane. This is critical for correct IK
        constraints when the surface has non-zero yaw rotation.

        Uses the convex hull of the surface mesh for accurate representation
        of round/irregular surfaces. Falls back to bounding box if no mesh.

        Returns:
            VPolytope representing the 2D XY boundary of the surface in world frame.
        """
        R = self.transform.rotation().matrix()
        t = self.transform.translation()

        if self.mesh is None:
            # No mesh - use axis-aligned bounding box corners in local frame.
            bbox_min = self.bounding_box_min
            bbox_max = self.bounding_box_max
            corners_local = np.array(
                [
                    [bbox_min[0], bbox_min[1], 0.0],
                    [bbox_max[0], bbox_min[1], 0.0],
                    [bbox_max[0], bbox_max[1], 0.0],
                    [bbox_min[0], bbox_max[1], 0.0],
                ]
            )
            # Transform to world frame.
            corners_world = (R @ corners_local.T).T + t
            corners_world_xy = corners_world[:, :2]

            hull, processed_vertices = safe_convex_hull_2d(corners_world_xy)
            if hull is None:
                # Degenerate - use AABB of transformed corners.
                lb = corners_world_xy.min(axis=0)
                ub = corners_world_xy.max(axis=0)
                return VPolytope.MakeBox(lb=lb, ub=ub)

            hull_vertices = processed_vertices[hull.vertices]
            return VPolytope(vertices=hull_vertices.T)

        # Transform mesh vertices to world frame.
        mesh_world = (R @ self.mesh.vertices.T).T + t
        mesh_world_xy = mesh_world[:, :2]

        hull, processed_vertices = safe_convex_hull_2d(mesh_world_xy)
        if hull is None:
            # Degenerate hull - fall back to AABB of transformed mesh.
            lb = mesh_world_xy.min(axis=0)
            ub = mesh_world_xy.max(axis=0)
            return VPolytope.MakeBox(lb=lb, ub=ub)

        hull_vertices = processed_vertices[hull.vertices]
        # VPolytope expects 2xN array (dim x num_vertices).
        return VPolytope(vertices=hull_vertices.T)


@dataclass
class SceneObject:
    """Represents a single object in the scene."""

    object_id: UniqueID
    """Unique identifier for the object."""

    object_type: ObjectType
    """Type of object (furniture or manipuland)."""

    name: str
    """Human-readable name of the object (e.g., 'Dining Table')."""

    description: str
    """Text description used for asset generation (e.g., 'A wooden table')."""

    transform: RigidTransform
    """3D canonical visual pose of the object in world coordinates."""

    internal_model_pose: RigidTransform = field(default_factory=RigidTransform)
    """Additional SDF model-frame pose relative to transform.

    For articulated assets this captures the top-level <model><pose> offset that
    must be applied when mapping raw SDF link/collision geometry into world space.
    """

    geometry_path: Path | None = None
    """Path to the 3D geometry file (e.g., GLB, OBJ)."""

    sdf_path: Path | None = None
    """Path to the Drake SDF file for simulation."""

    image_path: Path | None = None
    """Path to the reference image used for asset generation."""

    support_surfaces: list[SupportSurface] = field(default_factory=list)
    """Support surfaces where other objects can be placed on this object."""

    placement_info: PlacementInfo | None = None
    """Placement metadata for objects placed on support surfaces.

    For manipulands placed on furniture surfaces, this stores the surface-relative
    placement information (parent surface, 2D position, rotation). For furniture
    placed directly on the floor, this is None.
    """

    metadata: dict[str, str | float | bool] = field(default_factory=dict)
    """Additional metadata for the object (e.g., dimensions, material)."""

    bbox_min: np.ndarray | None = None
    """Object-frame AABB minimum corner (x, y, z)."""

    bbox_max: np.ndarray | None = None
    """Object-frame AABB maximum corner (x, y, z)."""

    immutable: bool = False
    """Whether this object is immutable (cannot be moved or removed)."""

    scale_factor: float = 1.0
    """Cumulative scale factor applied to this object's asset (1.0 = original size)."""

    def has_internal_model_pose(self) -> bool:
        """Return whether this object carries a non-identity internal model pose."""
        translation = self.internal_model_pose.translation()
        rotation = self.internal_model_pose.rotation().matrix()
        return (
            not np.allclose(translation, np.zeros(3), atol=1e-9)
            or not np.allclose(rotation, np.eye(3), atol=1e-9)
        )

    def get_effective_transform(self) -> RigidTransform:
        """Return the world pose of raw SDF/link geometry."""
        return self.transform @ self.internal_model_pose

    def get_effective_rotation_matrix(self) -> RotationMatrix:
        """Return the effective world rotation for raw articulated geometry."""
        return self.get_effective_transform().rotation()

    def apply_scale(self, new_scale: float) -> None:
        """Apply scale factor to this object's bounding box and invalidate surfaces.

        This updates the object-frame bounding box and invalidates support surfaces
        (which will need to be recomputed after rescaling).

        Args:
            new_scale: Scale multiplier to apply (e.g., 1.5 = 50% larger).
        """
        if self.bbox_min is not None:
            self.bbox_min = self.bbox_min * new_scale
        if self.bbox_max is not None:
            self.bbox_max = self.bbox_max * new_scale

        self.support_surfaces = []  # Invalidate - must be recomputed.
        self.scale_factor = self.scale_factor * new_scale

    def compute_world_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute world-frame AABB from object-frame bounds and transform.

        Returns:
            Tuple of (world_bbox_min, world_bbox_max) or None if no bounds available.
        """
        if self.bbox_min is None or self.bbox_max is None:
            return None

        # Generate all 8 corners of the object-frame bounding box.
        corners = compute_aabb_corners(self.bbox_min, self.bbox_max)

        # Transform all corners to world coordinates.
        effective_transform = self.get_effective_transform()
        world_corners = []
        for corner in corners:
            world_corner = effective_transform @ corner
            world_corners.append(world_corner)

        world_corners = np.array(world_corners)

        # Find min and max in each dimension.
        world_bbox_min = np.min(world_corners, axis=0)
        world_bbox_max = np.max(world_corners, axis=0)

        return world_bbox_min, world_bbox_max

    def content_hash(self) -> str:
        """Generate content hash for this scene object."""
        obj_dict = {
            "object_id": str(self.object_id),
            "name": self.name,
            "description": self.description,
            "object_type": self.object_type.value if self.object_type else "",
            "transform": serialize_rigid_transform(self.transform),
            "internal_model_pose": serialize_rigid_transform(self.internal_model_pose),
            "geometry_path": str(self.geometry_path) if self.geometry_path else "",
            "sdf_path": str(self.sdf_path) if self.sdf_path else "",
            "image_path": str(self.image_path) if self.image_path else "",
            "support_surfaces": [surf.content_hash() for surf in self.support_surfaces],
            "placement_info": (
                {
                    "parent_surface_id": str(self.placement_info.parent_surface_id),
                    "position_2d": self.placement_info.position_2d.tolist(),
                    "rotation_2d": float(self.placement_info.rotation_2d),
                    "placement_method": self.placement_info.placement_method,
                }
                if self.placement_info
                else None
            ),
            "metadata": dict(sorted(self.metadata.items())),  # Sort for determinism
            "bbox_min": self.bbox_min.tolist() if self.bbox_min is not None else None,
            "bbox_max": self.bbox_max.tolist() if self.bbox_max is not None else None,
            "immutable": self.immutable,
            "scale_factor": self.scale_factor,
        }

        # Hash file contents if they exist.
        for path_key in ["geometry_path", "sdf_path"]:
            path_str = obj_dict[path_key]
            if path_str:
                try:
                    path = Path(path_str)
                    if path.exists():
                        # Determine if file is binary or text based on extension.
                        binary_extensions = {".glb", ".obj", ".ply", ".stl"}
                        is_binary = path.suffix.lower() in binary_extensions

                        if is_binary:
                            # Read binary files in binary mode.
                            with open(path, "rb") as f:
                                content = f.read()
                            obj_dict[f"{path_key}_content_hash"] = hashlib.sha256(
                                content
                            ).hexdigest()
                        else:
                            # Read text files (SDF, XML) in text mode.
                            with open(path, "r", encoding="utf-8") as f:
                                content = f.read()
                            obj_dict[f"{path_key}_content_hash"] = hashlib.sha256(
                                content.encode()
                            ).hexdigest()
                    else:
                        obj_dict[f"{path_key}_content_hash"] = ""
                except Exception as e:
                    console_logger.warning(
                        f"Could not hash file content for {path_str}: {e}"
                    )
                    obj_dict[f"{path_key}_content_hash"] = ""

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(obj_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_dict(self, scene_dir: Path | None = None) -> dict[str, Any]:
        """
        Serialize SceneObject to dictionary.

        Args:
            scene_dir: Optional scene directory for path relativization.
                       If None, paths are stored as absolute paths.

        Returns:
            Dictionary containing complete object state.
        """
        # Serialize support surfaces.
        support_surfaces_data = []
        for surf in self.support_surfaces:
            surf_dict = {
                "surface_id": str(surf.surface_id),
                "bounding_box_min": surf.bounding_box_min.tolist(),
                "bounding_box_max": surf.bounding_box_max.tolist(),
                "transform": serialize_rigid_transform(surf.transform),
                "link_name": surf.link_name,  # For articulated FK transforms.
            }
            # Serialize mesh data if present.
            if surf.mesh is not None:
                surf_dict["mesh"] = {
                    "vertices": surf.mesh.vertices.tolist(),
                    "faces": surf.mesh.faces.tolist(),
                }
            support_surfaces_data.append(surf_dict)

        # Convert paths (relative or absolute).
        geometry_path_str = (
            safe_relative_path(self.geometry_path, scene_dir)
            if self.geometry_path
            else None
        )
        sdf_path_str = (
            safe_relative_path(self.sdf_path, scene_dir) if self.sdf_path else None
        )
        image_path_str = (
            safe_relative_path(self.image_path, scene_dir) if self.image_path else None
        )

        return {
            "object_id": str(self.object_id),
            "object_type": self.object_type.value,
            "name": self.name,
            "description": self.description,
            "transform": serialize_rigid_transform(self.transform),
            "internal_model_pose": serialize_rigid_transform(self.internal_model_pose),
            "geometry_path": geometry_path_str,
            "sdf_path": sdf_path_str,
            "image_path": image_path_str,
            "support_surfaces": support_surfaces_data,
            "placement_info": (
                {
                    "parent_surface_id": str(self.placement_info.parent_surface_id),
                    "position_2d": self.placement_info.position_2d.tolist(),
                    "rotation_2d": float(self.placement_info.rotation_2d),
                    "placement_method": self.placement_info.placement_method,
                }
                if self.placement_info
                else None
            ),
            "metadata": self.metadata,
            "bbox_min": self.bbox_min.tolist() if self.bbox_min is not None else None,
            "bbox_max": self.bbox_max.tolist() if self.bbox_max is not None else None,
            "immutable": self.immutable,
            "scale_factor": self.scale_factor,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], scene_dir: Path | None = None
    ) -> "SceneObject":
        """
        Deserialize SceneObject from dictionary.

        Args:
            data: Dictionary containing object state.
            scene_dir: Optional scene directory for path resolution.
                       If None, paths are treated as absolute.

        Returns:
            Reconstructed SceneObject instance.
        """
        # Reconstruct transform.
        transform_data = data["transform"]
        translation = np.array(transform_data["translation"])
        rotation_wxyz = transform_data["rotation_wxyz"]
        quaternion = Quaternion(wxyz=rotation_wxyz)
        rotation_matrix = RotationMatrix(quaternion)
        transform = RigidTransform(rotation_matrix, translation)

        internal_model_pose = deserialize_rigid_transform(
            data.get('internal_model_pose', serialize_rigid_transform(RigidTransform()))
        )

        # Reconstruct support surfaces.
        support_surfaces = []
        for surf_data in data["support_surfaces"]:
            surf_transform_data = surf_data["transform"]
            surf_translation = np.array(surf_transform_data["translation"])
            surf_rotation_wxyz = surf_transform_data["rotation_wxyz"]
            surf_quaternion = Quaternion(wxyz=surf_rotation_wxyz)
            surf_rotation_matrix = RotationMatrix(surf_quaternion)
            surf_transform = RigidTransform(surf_rotation_matrix, surf_translation)

            # Reconstruct mesh if present.
            mesh = None
            if "mesh" in surf_data and surf_data["mesh"] is not None:
                import trimesh

                mesh = trimesh.Trimesh(
                    vertices=np.array(surf_data["mesh"]["vertices"]),
                    faces=np.array(surf_data["mesh"]["faces"]),
                )

            support_surface = SupportSurface(
                surface_id=UniqueID(surf_data["surface_id"]),
                bounding_box_min=np.array(surf_data["bounding_box_min"]),
                bounding_box_max=np.array(surf_data["bounding_box_max"]),
                transform=surf_transform,
                mesh=mesh,
                link_name=surf_data.get("link_name"),  # For articulated FK transforms.
            )
            support_surfaces.append(support_surface)

        # Resolve paths.
        geometry_path = None
        if data["geometry_path"]:
            geometry_path = (
                scene_dir / data["geometry_path"]
                if scene_dir
                else Path(data["geometry_path"])
            )

        sdf_path = None
        if data["sdf_path"]:
            sdf_path = (
                scene_dir / data["sdf_path"] if scene_dir else Path(data["sdf_path"])
            )

        image_path = None
        if data["image_path"]:
            image_path = (
                scene_dir / data["image_path"]
                if scene_dir
                else Path(data["image_path"])
            )

        # Reconstruct placement_info.
        placement_info = None
        if data.get("placement_info"):
            placement_data = data["placement_info"]
            placement_info = PlacementInfo(
                parent_surface_id=UniqueID(placement_data["parent_surface_id"]),
                position_2d=np.array(placement_data["position_2d"]),
                rotation_2d=float(placement_data["rotation_2d"]),
                placement_method=placement_data["placement_method"],
            )

        return cls(
            object_id=UniqueID(data["object_id"]),
            object_type=ObjectType(data["object_type"]),
            name=data["name"],
            description=data["description"],
            transform=transform,
            internal_model_pose=internal_model_pose,
            geometry_path=geometry_path,
            sdf_path=sdf_path,
            image_path=image_path,
            support_surfaces=support_surfaces,
            placement_info=placement_info,
            metadata=data["metadata"],
            bbox_min=np.array(data["bbox_min"]) if data["bbox_min"] else None,
            bbox_max=np.array(data["bbox_max"]) if data["bbox_max"] else None,
            immutable=data["immutable"],
            scale_factor=data.get("scale_factor", 1.0),
        )


@dataclass
class RoomScene:
    """
    Central state manager and single source of truth for a single room's composition.

    Maintains all objects in the room with transactional-like operations (add, remove,
    move, replace) ensuring consistency. Generates Drake simulation directives for
    rendering and physics simulation.
    """

    room_geometry: "RoomGeometry"
    """The generated 3D geometry for this room (walls, floor, SDF)."""

    scene_dir: Path
    """Base directory for the room (all paths are relative to this)."""

    room_id: str = "main"
    """Unique identifier for this room within a house. Default 'main' for room mode."""

    room_type: str = "room"
    """Type of room (e.g., 'living_room', 'bedroom'). Default 'room' for room mode."""

    objects: dict[UniqueID, SceneObject] = field(default_factory=dict)
    """Dictionary mapping object IDs to SceneObject instances."""

    text_description: str = ""
    """Text description of the overall room."""

    action_log_path: Path | None = None
    """Path to action log file for scene replication/replay."""

    _surface_id_counter: int = field(default=0, init=False, repr=False)
    """Counter for generating sequential surface IDs (S_0, S_1, etc.)."""

    def add_object(self, obj: SceneObject) -> None:
        """Add an object to the scene."""
        self.objects[obj.object_id] = obj

    def remove_object(self, object_id: UniqueID) -> bool:
        """Remove an object from the scene. Returns True if removed."""
        if object_id in self.objects:
            del self.objects[object_id]
            return True
        return False

    def get_object(self, object_id: UniqueID) -> SceneObject | None:
        """Get an object by ID.

        Searches both scene.objects and scene.room_geometry.floor to support
        floor as a placement target for manipulands.

        Args:
            object_id: Unique identifier for the object.

        Returns:
            SceneObject if found, None otherwise.
        """
        # Check regular objects first.
        obj = self.objects.get(object_id)
        if obj:
            return obj

        # Check floor if available.
        if (
            self.room_geometry
            and self.room_geometry.floor
            and self.room_geometry.floor.object_id == object_id
        ):
            return self.room_geometry.floor

        return None

    def generate_unique_id(self, name: str) -> UniqueID:
        """Generate a unique ID that doesn't conflict with existing scene objects.

        Uses base-36 sequential numbering (0-9, a-z) for compact IDs.

        Args:
            name: Human-readable name for the object.

        Returns:
            UniqueID that is guaranteed unique within this scene.
        """
        return UniqueID.generate_unique(name, self.objects)

    def generate_surface_id(self) -> UniqueID:
        """Generate next sequential surface ID using base-36 encoding.

        Returns:
            UniqueID in format S_0, S_1, ..., S_9, S_a, ..., S_z, S_10, etc.
        """
        suffix = _int_to_base36(self._surface_id_counter)
        surface_id = UniqueID(f"S_{suffix}")
        self._surface_id_counter += 1
        return surface_id

    def move_object(self, object_id: UniqueID, new_transform: RigidTransform) -> bool:
        """Move an object to a new position. Returns True if successful."""
        if object_id not in self.objects:
            return False
        self.objects[object_id].transform = new_transform
        return True

    def get_objects_by_type(self, object_type: ObjectType) -> list[SceneObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects.values() if obj.object_type == object_type]

    def get_manipulands(self) -> list[SceneObject]:
        """Get all manipuland objects in the scene.

        Returns:
            List of SceneObject instances with object_type=MANIPULAND.
        """
        return self.get_objects_by_type(ObjectType.MANIPULAND)

    def get_objects_on_surface(self, surface_id: UniqueID) -> list[SceneObject]:
        """Get all objects placed on a specific support surface.

        Args:
            surface_id: The ID of the support surface to query.

        Returns:
            List of SceneObject instances placed on the specified surface.
        """
        return [
            obj
            for obj in self.objects.values()
            if obj.placement_info and obj.placement_info.parent_surface_id == surface_id
        ]

    def to_drake_directive(
        self,
        include_objects: list[UniqueID] | None = None,
        include_object_types: list[ObjectType] | None = None,
        weld_furniture: bool = True,
        free_objects: list[UniqueID] | None = None,
        exclude_room_geometry: bool = False,
        weld_stack_members: bool = True,
        weld_room_geometry: bool = True,
        room_geometry_name: str = "room_geometry",
        model_name_prefix: str = "",
        base_dir: Path | None = None,
        free_mounted_objects_for_collision: bool = False,
        parent_frame: str = "world",
    ) -> str:
        """Generate a Drake directive string from the current scene.

        Args:
            include_objects: If provided, only include these objects in the
                directive. Useful for fast collision checking between 2-3
                objects.
            include_object_types: If provided, only include objects of these
                types (e.g., [ObjectType.FURNITURE, ObjectType.WALL_MOUNTED]).
                Useful for intermediate house snapshots.
            weld_furniture: If True (default), weld furniture to world frame.
                If False, add furniture as free bodies for IK optimization.
            free_objects: If provided, these specific objects will be free bodies
                regardless of weld_furniture setting. Useful for IK when you want
                most furniture welded but one object free to optimize.
            exclude_room_geometry: If True, completely exclude the room geometry
                from the directive. Useful for focused rendering (e.g., manipuland
                agent viewing only furniture + manipulands).
            weld_stack_members: If True (default), weld upper stack members to
                the bottom member, treating stacks as rigid units. If False, all
                stack members are free bodies (legacy behavior).
            weld_room_geometry: If True (default), weld room geometry to world at
                origin. If False, only add the model without weld. HouseScene
                sets this to False and handles welding with transforms.
            room_geometry_name: Model name for room geometry. Use unique names for
                multi-room houses (e.g., "room_geometry_living_room").
            model_name_prefix: Prefix to prepend to all model names. Used by
                HouseScene to ensure globally unique model names across rooms
                (e.g., "living_room_" makes "rug_0" become "living_room_rug_0").
            base_dir: If provided, SDF paths are relative to this directory
                (for portable directives). The directive YAML file should be
                saved in this directory for Drake to resolve paths correctly.
                If None, absolute paths with file:// scheme are used (for temp
                file usage in physics simulation).
            free_mounted_objects_for_collision: If True, wall-mounted and
                ceiling-mounted objects are treated as free bodies instead of
                welded. Used for collision checking where Drake's broadphase
                needs free bodies to detect collisions properly.
            parent_frame: Frame to parent objects to. All object poses and
                welds reference this frame. Free body poses include
                base_frame for Drake frame resolution. Defaults to "world".

        Returns:
            Drake directive in YAML format that can be loaded by Drake's
            ProcessModelDirectives.
        """

        def format_sdf_path(sdf_path: Path | str | None) -> str:
            """Format SDF path as package:// URI or absolute file:// URI."""
            if sdf_path is None:
                return ""
            sdf_path = Path(sdf_path)
            if base_dir is not None:
                # Use package://scene/ for portable scenes.
                # Drake resolves this via PackageMap (set ROS_PACKAGE_PATH or
                # call parser.package_map().Add("scene", scene_dir)).
                rel_path = os.path.relpath(sdf_path, base_dir)
                return f"package://scene/{rel_path}"
            else:
                return f"file://{sdf_path.absolute()}"

        def format_free_body_pose(
            link_name: str,
            tx: float,
            ty: float,
            tz: float,
            angle_deg: float,
            axis: list[float],
        ) -> str:
            """Format default_free_body_pose with base_frame."""
            return f"""
    default_free_body_pose:
      {link_name}:
        base_frame: {parent_frame}
        translation: [{tx}, {ty}, {tz}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""

        if exclude_room_geometry:
            directive = "directives:"
        else:
            room_geom_path = format_sdf_path(self.room_geometry.sdf_path)
            directive = f"""directives:
- add_model:
    name: {room_geometry_name}
    file: {room_geom_path}"""
            if weld_room_geometry:
                directive += f"""
- add_weld:
    parent: {parent_frame}
    child: {room_geometry_name}::room_geometry_body_link"""

        # Filter objects by ID and/or type.
        objects_to_add = list(self.objects.values())
        if include_objects is not None:
            objects_to_add = [
                obj for obj in objects_to_add if obj.object_id in include_objects
            ]
        if include_object_types is not None:
            objects_to_add = [
                obj for obj in objects_to_add if obj.object_type in include_object_types
            ]

        console_logger.info(
            f"to_drake_directive filtering: "
            f"include_objects={len(include_objects) if include_objects else 'None'}, "
            f"total scene objects={len(self.objects)}, "
            f"filtered objects_to_add={len(objects_to_add)}"
        )

        # Add scene objects.
        for obj in objects_to_add:
            # Handle composite objects (e.g., stacks) by expanding member assets.
            if obj.metadata.get("composite_type") == "stack":
                member_assets = obj.metadata.get("member_assets", [])

                # Clear cached model names (regenerated below for collision lookup).
                obj.metadata["member_model_names"] = []

                # Track bottom member info for welding upper members.
                bottom_model_name: str | None = None
                bottom_base_link: str | None = None
                bottom_transform: RigidTransform | None = None

                for i, member in enumerate(member_assets):
                    member_sdf = member.get("sdf_path")
                    if not member_sdf:
                        continue

                    member_transform = deserialize_composite_member_effective_transform(
                        member
                    )
                    translation = [float(x) for x in member_transform.translation()]
                    angle_axis = member_transform.rotation().ToAngleAxis()
                    angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                    axis = [float(x) for x in angle_axis.axis()]

                    # Generate unique model name for each stack member.
                    # The prefix is used by HouseScene to ensure globally unique names.
                    member_name = member.get("name", "stack_member")
                    member_id = member.get("asset_id", "unknown")
                    id_suffix = member_id.split("_")[-1][:8]
                    stack_suffix = str(obj.object_id).split("_")[-1][:4]
                    model_name = (
                        f"{model_name_prefix}{member_name.lower().replace(' ', '_')}_"
                        f"{id_suffix}_s{stack_suffix}_{i}"
                    )

                    # Store model name for direct lookup in collision detection.
                    obj.metadata["member_model_names"].append(model_name)

                    # Extract base link name from member SDF.
                    try:
                        base_link_name = extract_base_link_name_from_sdf(
                            Path(member_sdf)
                        )
                    except ValueError as e:
                        console_logger.warning(
                            f"Warning: {e}. Using 'base_link' as fallback."
                        )
                        base_link_name = "base_link"

                    if weld_stack_members and i == 0:
                        # Bottom member: free body, store info for welding upper members.
                        bottom_model_name = model_name
                        bottom_base_link = base_link_name
                        bottom_transform = member_transform
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )
                    elif weld_stack_members and i > 0:
                        # Upper member: welded to bottom member.
                        t_rel = bottom_transform.inverse() @ member_transform
                        rel_translation = t_rel.translation()
                        rel_angle_axis = t_rel.rotation().ToAngleAxis()
                        rel_angle_deg = float(rel_angle_axis.angle()) * 180.0 / np.pi
                        rel_axis = [float(x) for x in rel_angle_axis.axis()]

                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}
- add_weld:
    parent: {bottom_model_name}::{bottom_base_link}
    child: {model_name}::{base_link_name}
    X_PC:
      translation: [{rel_translation[0]}, {rel_translation[1]}, {rel_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {rel_angle_deg}
        axis: [{rel_axis[0]}, {rel_axis[1]}, {rel_axis[2]}]"""
                    else:
                        # weld_stack_members=False: all members as free bodies.
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                continue

            # Handle filled containers (container + fill objects inside).
            if obj.metadata.get("composite_type") == "filled_container":
                container_asset = obj.metadata.get("container_asset")
                fill_assets = obj.metadata.get("fill_assets", [])

                # Clear cached model names (regenerated below for collision lookup).
                obj.metadata["member_model_names"] = []

                # Track container info for welding fill objects.
                container_model_name: str | None = None
                container_base_link: str | None = None
                container_transform: RigidTransform | None = None

                # Add container as free body (reference member).
                if container_asset:
                    container_sdf = container_asset.get("sdf_path")
                    if container_sdf:
                        container_transform = (
                            deserialize_composite_member_effective_transform(
                                container_asset
                            )
                        )
                        translation = [
                            float(x) for x in container_transform.translation()
                        ]
                        angle_axis = container_transform.rotation().ToAngleAxis()
                        angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                        axis = [float(x) for x in angle_axis.axis()]

                        # Generate unique model name for container.
                        container_name = container_asset.get("name", "container")
                        container_id = container_asset.get("asset_id", "unknown")
                        id_suffix = container_id.split("_")[-1][:8]
                        fill_suffix = str(obj.object_id).split("_")[-1][:4]
                        container_model_name = (
                            f"{model_name_prefix}{container_name.lower().replace(' ', '_')}_"
                            f"{id_suffix}_f{fill_suffix}_c"
                        )

                        obj.metadata["member_model_names"].append(container_model_name)

                        try:
                            container_base_link = extract_base_link_name_from_sdf(
                                Path(container_sdf)
                            )
                        except ValueError as e:
                            console_logger.warning(
                                f"Warning: {e}. Using 'base_link' as fallback."
                            )
                            container_base_link = "base_link"

                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]

                        container_sdf_formatted = format_sdf_path(container_sdf)
                        directive += f"""
- add_model:
    name: {container_model_name}
    file: {container_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=container_base_link,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                # Add fill objects welded to container.
                for i, fill_asset in enumerate(fill_assets):
                    fill_sdf = fill_asset.get("sdf_path")
                    if not fill_sdf:
                        continue

                    fill_transform = deserialize_composite_member_effective_transform(
                        fill_asset
                    )
                    translation = [float(x) for x in fill_transform.translation()]
                    angle_axis = fill_transform.rotation().ToAngleAxis()
                    angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                    axis = [float(x) for x in angle_axis.axis()]

                    fill_name = fill_asset.get("name", "fill_item")
                    fill_id = fill_asset.get("asset_id", "unknown")
                    id_suffix = fill_id.split("_")[-1][:8]
                    fill_suffix = str(obj.object_id).split("_")[-1][:4]
                    fill_model_name = (
                        f"{model_name_prefix}{fill_name.lower().replace(' ', '_')}_"
                        f"{id_suffix}_f{fill_suffix}_{i}"
                    )

                    obj.metadata["member_model_names"].append(fill_model_name)

                    try:
                        fill_base_link = extract_base_link_name_from_sdf(Path(fill_sdf))
                    except ValueError as e:
                        console_logger.warning(
                            f"Warning: {e}. Using 'base_link' as fallback."
                        )
                        fill_base_link = "base_link"

                    if (
                        weld_stack_members
                        and container_model_name
                        and container_transform
                    ):
                        # Fill object welded to container.
                        t_rel = container_transform.inverse() @ fill_transform
                        rel_translation = t_rel.translation()
                        rel_angle_axis = t_rel.rotation().ToAngleAxis()
                        rel_angle_deg = float(rel_angle_axis.angle()) * 180.0 / np.pi
                        rel_axis = [float(x) for x in rel_angle_axis.axis()]

                        fill_sdf_formatted = format_sdf_path(fill_sdf)
                        directive += f"""
- add_model:
    name: {fill_model_name}
    file: {fill_sdf_formatted}
- add_weld:
    parent: {container_model_name}::{container_base_link}
    child: {fill_model_name}::{fill_base_link}
    X_PC:
      translation: [{rel_translation[0]}, {rel_translation[1]}, {rel_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {rel_angle_deg}
        axis: [{rel_axis[0]}, {rel_axis[1]}, {rel_axis[2]}]"""
                    else:
                        # weld_stack_members=False: fill objects as free bodies.
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        fill_sdf_formatted = format_sdf_path(fill_sdf)
                        directive += f"""
- add_model:
    name: {fill_model_name}
    file: {fill_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=fill_base_link,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                continue

            # Handle piles (member assets similar to stack structure).
            if obj.metadata.get("composite_type") == "pile":
                member_assets = obj.metadata.get("member_assets", [])

                # Clear cached model names (regenerated below for collision lookup).
                obj.metadata["member_model_names"] = []

                # Track first member info for welding other members.
                first_model_name: str | None = None
                first_base_link: str | None = None
                first_transform: RigidTransform | None = None

                for i, member in enumerate(member_assets):
                    member_sdf = member.get("sdf_path")
                    if not member_sdf:
                        continue

                    member_transform = (
                        deserialize_composite_member_effective_transform(member)
                    )

                    translation = [float(x) for x in member_transform.translation()]
                    angle_axis = member_transform.rotation().ToAngleAxis()
                    angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                    axis = [float(x) for x in angle_axis.axis()]

                    # Generate unique model name for each pile member.
                    member_name = member.get("name", "pile_member")
                    member_id = member.get("asset_id", "unknown")
                    id_suffix = member_id.split("_")[-1][:8]
                    pile_suffix = str(obj.object_id).split("_")[-1][:4]
                    model_name = (
                        f"{model_name_prefix}{member_name.lower().replace(' ', '_')}_"
                        f"{id_suffix}_p{pile_suffix}_{i}"
                    )

                    # Store model name for direct lookup in collision detection.
                    obj.metadata["member_model_names"].append(model_name)

                    # Extract base link name from member SDF.
                    try:
                        base_link_name = extract_base_link_name_from_sdf(
                            Path(member_sdf)
                        )
                    except ValueError as e:
                        console_logger.warning(
                            f"Warning: {e}. Using 'base_link' as fallback."
                        )
                        base_link_name = "base_link"

                    if weld_stack_members and i == 0:
                        # First member: free body, store info for welding other members.
                        first_model_name = model_name
                        first_base_link = base_link_name
                        first_transform = member_transform
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )
                    elif weld_stack_members and i > 0:
                        # Other members: welded to first member.
                        t_rel = first_transform.inverse() @ member_transform
                        rel_translation = t_rel.translation()
                        rel_angle_axis = t_rel.rotation().ToAngleAxis()
                        rel_angle_deg = float(rel_angle_axis.angle()) * 180.0 / np.pi
                        rel_axis = [float(x) for x in rel_angle_axis.axis()]

                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}
- add_weld:
    parent: {first_model_name}::{first_base_link}
    child: {model_name}::{base_link_name}
    X_PC:
      translation: [{rel_translation[0]}, {rel_translation[1]}, {rel_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {rel_angle_deg}
        axis: [{rel_axis[0]}, {rel_axis[1]}, {rel_axis[2]}]"""
                    else:
                        # weld_stack_members=False: all members as free bodies.
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                continue

            if obj.sdf_path is None:
                continue

            # Extract position and orientation from the effective transform.
            effective_transform = obj.get_effective_transform()
            translation = effective_transform.translation()
            angle_axis = effective_transform.rotation().ToAngleAxis()
            # Create unique model name by combining name with ID suffix.
            # This ensures Drake model instances are unique even for reused assets.
            # The prefix is used by HouseScene to ensure globally unique names.
            id_suffix = str(obj.object_id).split("_")[-1][:8]
            model_name = (
                f"{model_name_prefix}{obj.name.lower().replace(' ', '_')}_{id_suffix}"
            )

            # Extract the base link name from the SDF file.
            try:
                base_link_name = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError as e:
                # Fallback to "base_link" if extraction fails.
                console_logger.warning(f"Warning: {e}. Using 'base_link' as fallback.")
                base_link_name = "base_link"

            # Convert angle to degrees.
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            # Determine if this object should be welded or free.
            # Thin coverings are always welded - they have no collision geometry
            # so would fall through the floor during simulation.
            # Wall-mounted objects are normally welded - they're mounted on walls
            # and shouldn't move during physics simulation.
            # Ceiling-mounted objects are normally welded - they hang from ceiling
            # and shouldn't move during physics simulation.
            # For collision checking, wall/ceiling objects must be free bodies
            # for Drake's broadphase to detect collisions between them.
            is_thin_covering = obj.metadata.get("asset_source") == "thin_covering"
            is_wall_mounted = obj.object_type == ObjectType.WALL_MOUNTED
            is_ceiling_mounted = obj.object_type == ObjectType.CEILING_MOUNTED
            if free_mounted_objects_for_collision:
                # Only thin coverings stay welded for collision checking.
                always_welded = is_thin_covering
            else:
                always_welded = (
                    is_thin_covering or is_wall_mounted or is_ceiling_mounted
                )
            if free_objects is not None:
                # Exclusive mode: ONLY objects in free_objects are free.
                # Used by large scene optimization to reduce DOFs.
                is_free = obj.object_id in free_objects and not always_welded
            else:
                # Original logic when free_objects not specified.
                is_free = (
                    (obj.object_type != ObjectType.FURNITURE) or not weld_furniture
                ) and not always_welded

            if is_free:
                # Free body (in free_objects list or manipuland).
                tx = translation[0]
                ty = translation[1]
                tz = translation[2]
                obj_sdf_formatted = format_sdf_path(obj.sdf_path)
                directive += f"""
- add_model:
    name: {model_name}
    file: {obj_sdf_formatted}"""
                directive += format_free_body_pose(
                    link_name=base_link_name,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                    angle_deg=angle_deg,
                    axis=axis,
                )
            else:
                # Welded (furniture, wall-mounted, or thin covering).
                # Check if model is static (auto-welded by Drake).
                sdf_path = Path(obj.sdf_path).absolute() if obj.sdf_path else None
                is_static = sdf_path and is_static_sdf_model(sdf_path)

                if is_static:
                    # Static models are auto-welded by Drake at their pose.
                    # Use default_free_body_pose to set initial position.
                    tx = translation[0]
                    ty = translation[1]
                    tz = translation[2]
                    obj_sdf_formatted = format_sdf_path(obj.sdf_path)
                    directive += f"""
- add_model:
    name: {model_name}
    file: {obj_sdf_formatted}"""
                    directive += format_free_body_pose(
                        link_name=base_link_name,
                        tx=tx,
                        ty=ty,
                        tz=tz,
                        angle_deg=angle_deg,
                        axis=axis,
                    )
                else:
                    # Non-static models need explicit weld.
                    tx = translation[0]
                    ty = translation[1]
                    tz = translation[2]
                    obj_sdf_formatted = format_sdf_path(obj.sdf_path)
                    directive += f"""
- add_model:
    name: {model_name}
    file: {obj_sdf_formatted}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::{base_link_name}
    X_PC:
      translation: [{tx}, {ty}, {tz}]
      rotation: !AngleAxis
        angle_deg: {angle_deg}
        axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""

        return directive

    def content_hash(self) -> str:
        """
        Generate deterministic content hash of entire Scene state.

        This creates a SHA-256 hash of all scene content including floor plan,
        objects, positions, and metadata. Identical scenes will produce identical
        hashes regardless of object creation order or identity.

        Returns:
            str: SHA-256 hash string of scene content for caching.
        """
        # Collect all content for hashing by delegating to individual class methods.
        content_dict = {
            "room_geometry": self.room_geometry.content_hash(),
            "objects": self._hash_objects(),
            "text_description": self.text_description,
        }

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(content_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_state_dict(self) -> dict[str, Any]:
        """
        Return complete scene state as a dictionary for checkpointing.

        Serializes all scene data including room geometry, objects with full
        metadata needed for restoration via restore_from_state_dict(). All paths
        are saved relative to self.scene_dir for portability.

        Returns:
            Dictionary containing complete scene state including room geometry.
        """
        objects_dict = {}
        for obj in self.objects.values():
            objects_dict[str(obj.object_id)] = obj.to_dict(scene_dir=self.scene_dir)

        # Serialize room geometry if present.
        room_geometry_data = None
        if self.room_geometry:
            room_geometry_data = self.room_geometry.to_dict(scene_dir=self.scene_dir)

        return {
            "room_geometry": room_geometry_data,
            "objects": objects_dict,
            "text_description": self.text_description,
        }

    def restore_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore scene to state from serialized dictionary.

        Resolves all paths relative to self.scene_dir for portability.
        Restores room geometry first, then objects, then populates
        room_geometry.walls from restored wall objects.

        Args:
            state_dict: State dictionary from to_state_dict()
        """
        # Import here to avoid circular import.
        from scenecode.agent_utils.house import RoomGeometry

        # Restore room geometry first if present.
        if state_dict.get("room_geometry"):
            self.room_geometry = RoomGeometry.from_dict(
                state_dict["room_geometry"], scene_dir=self.scene_dir
            )
        else:
            self.room_geometry = None

        # Clear current objects.
        self.objects.clear()

        # Restore text description.
        self.text_description = state_dict["text_description"]

        # Restore objects.
        for obj_data in state_dict["objects"].values():
            scene_object = SceneObject.from_dict(obj_data, scene_dir=self.scene_dir)
            self.objects[scene_object.object_id] = scene_object

        # Populate room_geometry.walls from restored wall objects.
        if self.room_geometry:
            self.room_geometry.walls = [
                obj
                for obj in self.objects.values()
                if obj.object_type == ObjectType.WALL
            ]

    def _hash_objects(self) -> dict:
        """Hash all scene objects using their individual content_hash methods."""
        objects_dict = {}

        # Sort objects by ID for deterministic ordering.
        for object_id in sorted(self.objects.keys(), key=str):
            obj = self.objects[object_id]
            objects_dict[str(object_id)] = obj.content_hash()

        return objects_dict


def extract_and_propagate_support_surfaces(
    scene: "RoomScene",
    furniture_object: SceneObject,
    config: SupportSurfaceExtractionConfig | None = None,
) -> list[SupportSurface]:
    """Extract all support surfaces using HSM algorithm and propagate to identical furniture.

    When furniture is duplicated, all instances share the same geometry but have
    different world poses. This function extracts support surfaces once using the
    HSM face clustering algorithm and propagates them to all furniture with the
    same geometry_path, saving computation.

    Args:
        scene: The scene containing all furniture objects.
        furniture_object: The furniture object to extract support surfaces from.
        config: HSM algorithm configuration (uses defaults if None).

    Returns:
        List of SupportSurface objects for the selected furniture, sorted by area
        (largest first).

    Raises:
        ValueError: If furniture object has no geometry path.
    """
    # Return existing support surfaces if already computed.
    if furniture_object.support_surfaces:
        console_logger.info(
            f"Support surfaces already extracted for {furniture_object.object_id} "
            f"({len(furniture_object.support_surfaces)} surfaces)"
        )
        return furniture_object.support_surfaces

    # Validate furniture has geometry.
    if furniture_object.geometry_path is None:
        raise ValueError(
            f"Furniture object {furniture_object.object_id} has no geometry path"
        )

    # Check if HSSD asset with pre-validated surfaces.
    # Determine surface loading strategy and source.
    if (
        furniture_object.metadata.get("asset_source") == "hssd"
        and "hssd_mesh_id" in furniture_object.metadata
        and not config.recompute_hssd_surfaces
    ):
        from scenecode.agent_utils.hssd_retrieval.support_surface_loader import (
            load_hssd_support_surfaces,
        )

        mesh_id = furniture_object.metadata["hssd_mesh_id"]
        surfaces = load_hssd_support_surfaces(
            mesh_id=mesh_id, config=config, scene=scene
        )
        source = "HSSD"

        if surfaces is None:
            # Fallback to HSM algorithm.
            console_logger.info(
                f"Falling back to HSM algorithm for {furniture_object.object_id}"
            )
            surfaces = extract_support_surfaces_from_mesh(
                mesh_path=furniture_object.geometry_path, config=config
            )
            source = "HSM (HSSD fallback)"
    else:
        # Extract all surfaces using HSM algorithm.
        # For articulated objects with per-link meshes, use per-link extraction
        # for accurate link association.
        is_articulated = furniture_object.metadata.get("is_articulated", False)

        # Use sdf_path.parent for articulated objects (per-link meshes are there).
        if furniture_object.sdf_path:
            sdf_dir = furniture_object.sdf_path.parent
        else:
            sdf_dir = furniture_object.geometry_path.parent

        if is_articulated and furniture_object.sdf_path:
            # Use per-link extraction for articulated objects.
            surfaces = extract_support_surfaces_articulated(
                sdf_dir=sdf_dir, config=config, sdf_path=furniture_object.sdf_path
            )
            source = "HSM (per-link)"
        else:
            surfaces = extract_support_surfaces_from_mesh(
                mesh_path=furniture_object.geometry_path, config=config
            )
            source = "HSM"

        if (
            furniture_object.metadata.get("asset_source") == "hssd"
            and config.recompute_hssd_surfaces
        ):
            source = "HSM (HSSD recomputed)"

    # Log loaded surfaces with areas and clearances.
    if surfaces:
        surfaces_summary = ", ".join(
            [
                f"{surf.surface_id} (area={surf.area:.2f}m², "
                f"clearance={surf.bounding_box_max[2] - surf.bounding_box_min[2]:.2f}m)"
                for surf in surfaces
            ]
        )
        console_logger.info(
            f"Loaded {len(surfaces)} surfaces for {furniture_object.object_id} "
            f"(source: {source}): {surfaces_summary}"
        )

    # Transform surfaces from object-local to world frame.
    # The HSM algorithm extracts surfaces in mesh-local frame (identity transform).
    # We need to transform them to world frame using furniture's transform and scale.
    world_surfaces = []
    scale = furniture_object.scale_factor
    for surface in surfaces:
        # Scale surface translation to match the scaled collision geometry.
        scaled_translation = surface.transform.translation() * scale
        scaled_surface_transform = RigidTransform(
            surface.transform.rotation(), scaled_translation
        )
        world_transform = furniture_object.transform @ scaled_surface_transform

        # Scale bounding box to match scaled geometry.
        scaled_bbox_min = surface.bounding_box_min * scale
        scaled_bbox_max = surface.bounding_box_max * scale

        # Scale mesh vertices to match scaled collision geometry.
        scaled_mesh = None
        if surface.mesh is not None:
            scaled_vertices = surface.mesh.vertices * scale
            scaled_mesh = trimesh.Trimesh(
                vertices=scaled_vertices, faces=surface.mesh.faces
            )

        # Create new surface with world transform and short unique ID.
        # Use scene's generate_surface_id for base-36 sequential IDs (S_0, S_1, ...).
        world_surface = SupportSurface(
            surface_id=scene.generate_surface_id(),
            bounding_box_min=scaled_bbox_min,
            bounding_box_max=scaled_bbox_max,
            transform=world_transform,
            mesh=scaled_mesh,  # Scaled mesh for convex hull computation.
            link_name=surface.link_name,  # Preserve link for FK transforms.
        )
        world_surfaces.append(world_surface)

    furniture_object.support_surfaces = world_surfaces

    console_logger.info(
        f"Extracted {len(world_surfaces)} support surfaces for "
        f"{furniture_object.object_id}"
    )

    # Propagate to all identical furniture (same geometry_path).
    target_geometry_path = furniture_object.geometry_path

    for obj in scene.objects.values():
        # Skip the furniture we just processed.
        if obj.object_id == furniture_object.object_id:
            continue

        # Only propagate to identical furniture (same geometry file).
        if obj.geometry_path != target_geometry_path:
            continue

        # Transform each surface to this object's world frame with scaling.
        obj_surfaces = []
        obj_scale = obj.scale_factor
        for surface in surfaces:
            # Apply this object's scale_factor to surface translation.
            obj_scaled_translation = surface.transform.translation() * obj_scale
            obj_scaled_surface_transform = RigidTransform(
                surface.transform.rotation(), obj_scaled_translation
            )
            obj_world_transform = obj.transform @ obj_scaled_surface_transform

            # Scale bounding box to match this object's scaled geometry.
            obj_scaled_bbox_min = surface.bounding_box_min * obj_scale
            obj_scaled_bbox_max = surface.bounding_box_max * obj_scale

            # Scale mesh vertices to match this object's scale factor.
            obj_scaled_mesh = None
            if surface.mesh is not None:
                obj_scaled_vertices = surface.mesh.vertices * obj_scale
                obj_scaled_mesh = trimesh.Trimesh(
                    vertices=obj_scaled_vertices, faces=surface.mesh.faces
                )

            obj_surface = SupportSurface(
                surface_id=scene.generate_surface_id(),
                bounding_box_min=obj_scaled_bbox_min,
                bounding_box_max=obj_scaled_bbox_max,
                transform=obj_world_transform,
                mesh=obj_scaled_mesh,  # Scaled mesh for convex hull computation.
                link_name=surface.link_name,  # Preserve link for FK transforms.
            )
            obj_surfaces.append(obj_surface)

        obj.support_surfaces = obj_surfaces

        console_logger.info(
            f"Propagated {len(obj_surfaces)} support surfaces from "
            f"{furniture_object.object_id} to {obj.object_id}"
        )

    return world_surfaces


_KEEP_SCENE_OBJECT_FIELD = object()


def clone_scene_object(
    original: SceneObject,
    *,
    object_id: UniqueID | None = None,
    object_type: ObjectType | None = None,
    name: str | None = None,
    description: str | None = None,
    transform: RigidTransform | None = None,
    placement_info: PlacementInfo | None | object = _KEEP_SCENE_OBJECT_FIELD,
    support_surfaces: list[SupportSurface] | object = _KEEP_SCENE_OBJECT_FIELD,
    immutable: bool | None = None,
) -> SceneObject:
    """Clone an asset-like SceneObject while preserving articulated pose metadata."""
    source = vars(original)

    if placement_info is _KEEP_SCENE_OBJECT_FIELD:
        placement_info_value = deepcopy(source.get('placement_info'))
    else:
        placement_info_value = placement_info

    if support_surfaces is _KEEP_SCENE_OBJECT_FIELD:
        support_surfaces_value = deepcopy(source.get('support_surfaces', []))
    else:
        support_surfaces_value = support_surfaces

    bbox_min = source.get('bbox_min')
    bbox_max = source.get('bbox_max')

    return SceneObject(
        object_id=object_id if object_id is not None else source.get('object_id'),
        object_type=(
            object_type if object_type is not None else source.get('object_type')
        ),
        name=name if name is not None else source.get('name', ''),
        description=(
            description if description is not None else source.get('description', '')
        ),
        transform=transform if transform is not None else source.get('transform'),
        internal_model_pose=source.get('internal_model_pose', RigidTransform()),
        geometry_path=source.get('geometry_path'),
        sdf_path=source.get('sdf_path'),
        image_path=source.get('image_path'),
        support_surfaces=support_surfaces_value,
        placement_info=placement_info_value,
        metadata=deepcopy(source.get('metadata', {})),
        bbox_min=None if bbox_min is None else bbox_min.copy(),
        bbox_max=None if bbox_max is None else bbox_max.copy(),
        immutable=source.get('immutable', False) if immutable is None else immutable,
        scale_factor=source.get('scale_factor', 1.0),
    )


def serialize_composite_member_asset(
    asset: SceneObject,
    transform: RigidTransform,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a composite member asset while preserving articulated pose data."""
    source = vars(asset)
    sdf_path = source.get('sdf_path')
    geometry_path = source.get('geometry_path')

    member = {
        'asset_id': str(source.get('object_id', 'unknown')),
        'name': source.get('name', ''),
        'transform': serialize_rigid_transform(transform),
        'internal_model_pose': serialize_rigid_transform(
            source.get('internal_model_pose', RigidTransform())
        ),
        'sdf_path': str(Path(sdf_path).absolute()) if sdf_path else None,
        'geometry_path': (
            str(Path(geometry_path).absolute()) if geometry_path else None
        ),
    }
    if extra_fields:
        member.update(extra_fields)
    return member


def deserialize_composite_member_effective_transform(
    member: dict[str, Any],
) -> RigidTransform:
    """Deserialize a composite member's base-link world pose."""
    identity_data = serialize_rigid_transform(RigidTransform())

    transform_data = member.get('transform', identity_data)
    if not isinstance(transform_data, dict):
        transform_data = identity_data

    internal_model_pose_data = member.get('internal_model_pose', identity_data)
    if not isinstance(internal_model_pose_data, dict):
        internal_model_pose_data = identity_data

    transform = deserialize_rigid_transform(transform_data)
    internal_model_pose = deserialize_rigid_transform(internal_model_pose_data)
    return transform @ internal_model_pose


def copy_scene_object_with_new_pose(
    scene: "RoomScene",
    original: SceneObject,
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> SceneObject:
    """
    Create a copy of a SceneObject with a new pose (position + rotation) and unique ID.

    Generates a guaranteed-unique ID using the scene's sequential numbering system
    (e.g., first "chair", second "chair_2", 11th "chair_a").

    Args:
        scene: Scene instance for generating unique IDs.
        original: Original SceneObject to copy.
        x: New X position.
        y: New Y position.
        z: New Z position.
        roll: Roll rotation in radians (default 0.0).
        pitch: Pitch rotation in radians (default 0.0).
        yaw: Yaw rotation in radians (default 0.0).

    Returns:
        New SceneObject with same asset data but new pose and unique ID.
    """
    # Create new transform with both position and rotation.
    new_transform = RigidTransform(
        rpy=RollPitchYaw(roll=roll, pitch=pitch, yaw=yaw), p=[x, y, z]
    )

    return clone_scene_object(
        original,
        object_id=scene.generate_unique_id(original.name),
        transform=new_transform,
        placement_info=None,
    )
