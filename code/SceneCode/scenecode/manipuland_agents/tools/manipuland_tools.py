import json
import logging
import math
import time

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from agents import function_tool
from omegaconf import DictConfig
from pydrake.all import RollPitchYaw
from scipy.spatial import ConvexHull, QhullError
from typing_extensions import TypedDict

from scenecode.agent_utils.action_logger import log_scene_action
from scenecode.agent_utils.asset_manager import AssetGenerationRequest, AssetManager
from scenecode.agent_utils.loop_detector import LoopDetector
from scenecode.agent_utils.physical_feasibility import apply_surface_projection
from scenecode.agent_utils.placement_noise import (
    PlacementNoiseMode,
    apply_placement_noise,
)
from scenecode.agent_utils.rescale_helpers import rescale_object_common
from scenecode.agent_utils.response_datatypes import (
    AssetGenerationResult,
    AssetInfo,
    BoundingBox3D,
    GeneratedAsset,
)
from scenecode.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
    clone_scene_object,
    deserialize_rigid_transform,
    serialize_rigid_transform,
)
from scenecode.manipuland_agents.tools.arrangement_tools import create_arrangement_impl
from scenecode.manipuland_agents.tools.fill_tools import fill_container_tool_impl
from scenecode.manipuland_agents.tools.pile_tools import create_pile_tool_impl
from scenecode.manipuland_agents.tools.response_dataclasses import (
    AvailableAssetsResult,
    ManipulandErrorType,
    ManipulandInfo,
    ManipulandOperationResult,
    ManipulandPlacementResult,
    PenetrationResolutionResult,
    PileCreationResult,
    Position2D,
    Position3D,
    Rotation3D,
    SimplifiedFurnitureInfo,
    SimplifiedManipulandInfo,
    StackCreationResult,
    SupportSurfaceWithManipulands,
)
from scenecode.manipuland_agents.tools.stack_tools import create_stack_tool_impl

console_logger = logging.getLogger(__name__)


class FillAssetItem(TypedDict):
    """Typed dict for fill asset items in create_arrangement.

    Using TypedDict instead of dict avoids agents SDK schema validation error:
    'additionalProperties should not be set for object types'.
    """

    id: str
    x: float
    y: float
    rotation: float


class ManipulandTools:
    """Agent-callable tools for manipuland asset generation and placement.

    Provides tools for the manipuland designer agent:
    1. Asset Generation: Creates 3D manipulands from text descriptions
    2. Surface Placement: Places manipulands on support surfaces using SE(2) poses
    3. Scene Operations: Removes manipulands, queries scene state

    Tools exposed:
    - generate_manipuland_assets: Generate 3D assets via text-to-3D pipeline
    - place_manipuland_on_surface: Place manipuland on surface with SE(2) pose
    - remove_manipuland: Delete manipuland from scene
    - get_current_scene_state: Get furniture + manipulands for current surface
    - list_available_assets: List all available manipuland assets
    """

    def __init__(
        self,
        scene: RoomScene,
        asset_manager: AssetManager,
        cfg: DictConfig,
        current_furniture_id: UniqueID,
        support_surfaces: dict[str, SupportSurface],
    ):
        """Initialize manipuland tools.

        Args:
            scene: RoomScene instance to manipulate.
            asset_manager: Asset manager for generating 3D assets.
            cfg: Configuration object containing loop detection and validation settings.
            current_furniture_id: ID of furniture currently being populated.
            support_surfaces: Dictionary mapping surface_id (string) to SupportSurface.
                All surfaces for the current furniture item.
        """
        self.scene = scene
        self.asset_manager = asset_manager
        self.cfg = cfg
        self.current_furniture_id = current_furniture_id
        self.support_surfaces = support_surfaces

        # Initialize placement noise configuration.
        # Start with natural profile as default until planner sets it.
        self.placement_noise_config = cfg.placement_noise
        self.active_noise_profile = self.placement_noise_config.natural_profile

        # Initialize placement validation configuration.
        self.top_surface_overlap_tolerance = (
            cfg.placement_validation.top_surface_overlap_tolerance
        )

        # Initialize loop detector from config.
        loop_config = cfg.loop_detection
        loop_detector = LoopDetector(
            max_attempts=loop_config.max_repeated_attempts,
            window_size=loop_config.tracking_window,
            enabled=loop_config.enabled,
            default_error_factory=self._create_loop_error_response,
        )

        # Apply loop detection to implementation methods.
        self._place_manipuland_on_surface_impl = loop_detector(
            self._place_manipuland_on_surface_impl
        )
        self._move_manipuland_impl = loop_detector(self._move_manipuland_impl)
        self._remove_manipuland_impl = loop_detector(self._remove_manipuland_impl)
        self._create_stack_impl = loop_detector(self._create_stack_impl)
        self._create_pile_impl = loop_detector(self._create_pile_impl)
        self._resolve_penetrations_impl = loop_detector(self._resolve_penetrations_impl)

        # Create tool closures.
        self.tools = self._create_tool_closures()

    def set_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Update the active noise profile based on placement style.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        if mode == PlacementNoiseMode.NATURAL:
            self.active_noise_profile = self.placement_noise_config.natural_profile
            console_logger.info("Placement noise set to NATURAL profile")
        elif mode == PlacementNoiseMode.PERFECT:
            self.active_noise_profile = self.placement_noise_config.perfect_profile
            console_logger.info("Placement noise set to PERFECT profile")
        else:
            console_logger.warning(
                f"Unsupported noise mode {mode}, keeping current profile"
            )

    def _create_loop_error_response(
        self, method_name: str, attempt_count: int, _args: tuple, kwargs: dict
    ) -> str:
        """Create manipuland-specific error response for loop detection."""
        # Extract object_id or asset_id from kwargs/args if available.
        identifier = kwargs.get("object_id", kwargs.get("asset_id", ""))

        if method_name == "_remove_manipuland_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to remove '{identifier}' "
                f"{attempt_count} times.\n\n"
                f"Possible causes:\n"
                f"1. Object doesn't exist\n"
                f"2. Object was already removed\n"
                f"3. Wrong object ID\n\n"
                f"Recovery: Call get_current_scene_state() to see actual object IDs."
            )
        elif method_name == "_place_manipuland_on_surface_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to place '{identifier}' "
                f"{attempt_count} times with the same parameters.\n\n"
                f"This suggests placement is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Position is out of surface bounds\n"
                f"2. Asset doesn't exist\n"
                f"3. Invalid placement parameters\n\n"
                f"Recovery: Check surface bounds and available assets."
            )
        elif method_name == "_move_manipuland_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to move '{identifier}' "
                f"{attempt_count} times with the same parameters.\n\n"
                f"This suggests movement is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Position is out of surface bounds\n"
                f"2. Object doesn't exist\n"
                f"3. Already at target position\n\n"
                f"Recovery: Check surface bounds and object status."
            )
        elif method_name == "_create_stack_impl":
            asset_ids = kwargs.get("asset_ids", [])
            diagnostic_message = (
                f"Loop detected: You've tried to create the same stack "
                f"{attempt_count} times with {len(asset_ids)} assets.\n\n"
                f"This suggests stack creation is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Stack is unstable (physics simulation failing)\n"
                f"2. Stack height exceeds surface clearance\n"
                f"3. Invalid asset IDs in stack\n\n"
                f"Recovery: Check simulation feedback and try with fewer items "
                f"or different base objects."
            )
            # Return StackCreationResult for stack operations.
            stack_result = StackCreationResult(
                success=False,
                message=diagnostic_message,
                stack_object_id=None,
                stack_height=None,
                parent_surface_id=(
                    self._current_surface.surface_id if self._current_surface else ""
                ),
                num_items=len(asset_ids),
                error_type=ManipulandErrorType.LOOP_DETECTED,
            )
            return stack_result.to_json()
        elif method_name == "_create_pile_impl":
            asset_ids = kwargs.get("asset_ids", [])
            diagnostic_message = (
                f"Loop detected: You've tried to create the same pile "
                f"{attempt_count} times with {len(asset_ids)} assets.\n\n"
                f"This suggests pile creation is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Objects falling off surface edge\n"
                f"2. Position too close to surface boundary\n"
                f"3. Invalid asset IDs in pile\n\n"
                f"Recovery: Move position toward center of surface or use fewer items."
            )
            # Return PileCreationResult for pile operations.
            pile_result = PileCreationResult(
                success=False,
                message=diagnostic_message,
                pile_object_id=None,
                parent_surface_id=(
                    self._current_surface.surface_id if self._current_surface else ""
                ),
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.LOOP_DETECTED,
            )
            return pile_result.to_json()
        elif method_name == "_resolve_penetrations_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to resolve penetrations on the same "
                f"surface {attempt_count} times.\n\n"
                f"This suggests the solver cannot find a valid configuration.\n\n"
                f"Possible causes:\n"
                f"1. Too many objects for surface area\n"
                f"2. Objects too large to fit\n\n"
                f"Recovery: Remove some objects or use a larger surface."
            )
            return PenetrationResolutionResult(
                success=False,
                message=diagnostic_message,
                num_objects_considered=0,
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.LOOP_DETECTED,
            ).to_json()
        else:
            diagnostic_message = (
                f"Loop detected in {method_name}: {attempt_count} attempts with same "
                f"parameters."
            )

        result = ManipulandOperationResult(
            success=False,
            message=diagnostic_message,
            error_type=ManipulandErrorType.LOOP_DETECTED,
            object_id=identifier if identifier else None,
        )

        return result.to_json()

    def _get_object_convex_hull_2d(
        self, geometry_path: Path, scale_factor: float = 1.0
    ) -> np.ndarray:
        """Extract 2D convex hull vertices from object mesh.

        Loads the object mesh and computes its 2D convex hull by projecting vertices
        to the XY plane. This provides an accurate footprint for placement validation.

        Args:
            geometry_path: Path to object GLB/OBJ file.
            scale_factor: Scale factor to apply to mesh vertices (default 1.0).

        Returns:
            Array of 2D vertices [(x, y), ...] in object-local frame (XY plane).
            Vertices are ordered counter-clockwise around the hull.

        Raises:
            ValueError: If mesh cannot be loaded (fail fast - manipulands must have mesh).
        """
        try:
            mesh = trimesh.load(geometry_path, force="mesh")
        except Exception as e:
            raise ValueError(
                f"Failed to load mesh from {geometry_path} for convex hull "
                f"computation: {e}. Manipulands must have valid geometry."
            )

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(
                f"Loaded geometry from {geometry_path} is not a mesh "
                f"(got {type(mesh)}). Manipulands must have valid mesh geometry."
            )

        # Apply scale factor to mesh vertices.
        if scale_factor != 1.0:
            mesh.vertices *= scale_factor

        # Project mesh vertices to XY plane (drop Z coordinate).
        mesh_xy_vertices = mesh.vertices[:, :2]

        # Compute 2D convex hull.
        try:
            hull = ConvexHull(mesh_xy_vertices)
        except QhullError as e:
            raise ValueError(
                f"Failed to compute convex hull for mesh from {geometry_path}: {e}. "
                f"Mesh may have degenerate geometry."
            )

        # Return hull vertices in counter-clockwise order.
        hull_vertices = mesh_xy_vertices[hull.vertices]
        return hull_vertices

    def _is_top_surface(self, surface_id: str) -> bool:
        """Check if the given surface is the highest (top) surface of its parent object.

        Top surfaces are the highest support surfaces of furniture pieces. They allow
        natural overlap (e.g., books extending over table edges) and skip strict
        boundary validation. Lower surfaces (shelves) require objects to fit entirely
        within their boundaries.

        Args:
            surface_id: ID of the surface to check.

        Returns:
            True if the surface is the highest surface of its parent furniture object.
        """
        # Get the parent object for this surface.
        parent_object = None
        for obj in self.scene.objects.values():
            for surface in obj.support_surfaces:
                if str(surface.surface_id) == surface_id:
                    parent_object = obj
                    break
            if parent_object:
                break

        if parent_object is None:
            console_logger.warning(
                f"Could not find parent object for surface {surface_id}, "
                f"treating as non-top surface (strict validation)"
            )
            return False

        # Find the highest surface by Z-coordinate in world frame.
        max_height = float("-inf")
        highest_surface_id = None
        for surface in parent_object.support_surfaces:
            surface_height = surface.transform.translation()[2]
            if surface_height > max_height:
                max_height = surface_height
                highest_surface_id = str(surface.surface_id)

        return surface_id == highest_surface_id

    def _validate_convex_hull_footprint(
        self,
        target_surface: SupportSurface,
        geometry_path: Path,
        position_2d: np.ndarray,
        rotation_degrees: float,
        allow_overlap_ratio: float = 0.0,
        scale_factor: float = 1.0,
    ) -> tuple[bool, str | None]:
        """Validate that object's convex hull fits within surface with optional overlap.

        Uses the object's actual mesh convex hull for accurate validation (not the
        conservative bounding box). For overlap tolerance, the convex hull is first
        centered at the origin, then shrunk toward the origin before checking
        containment. This ensures correct validation regardless of mesh centering.

        For top surfaces, an overlap tolerance can be specified to allow natural
        overhang (e.g., books extending slightly over table edges). The tolerance
        is relative to the object's size.

        Args:
            target_surface: Surface to validate against.
            geometry_path: Path to object mesh.
            position_2d: Placement position in surface frame [x, y].
            rotation_degrees: Placement rotation in degrees.
            allow_overlap_ratio: Ratio by which to shrink the convex hull
                (0.0 = no shrinking/strict containment, 0.15 = shrink by 15%).
            scale_factor: Scale factor to apply to mesh vertices (default 1.0).

        Returns:
            Tuple of (is_valid, error_message):
            - is_valid: True if shrunk hull vertices are within surface boundary.
            - error_message: Descriptive error if validation fails, None otherwise.
        """
        # Get object convex hull vertices.
        hull_vertices = self._get_object_convex_hull_2d(
            geometry_path=geometry_path, scale_factor=scale_factor
        )

        # Compute hull centroid and center the hull at origin.
        # This ensures shrinking works correctly even if mesh is not perfectly
        # centered. When we place an object at position (x, y), we expect the
        # object's geometric center to be at (x, y), not its mesh origin.
        hull_centroid = hull_vertices.mean(axis=0)
        hull_vertices_centered = hull_vertices - hull_centroid  # Center at (0, 0).

        # Shrink centered hull toward origin by the overlap ratio.
        shrink_factor = 1.0 - allow_overlap_ratio
        shrunk_hull_vertices = shrink_factor * hull_vertices_centered

        # Convert rotation to radians.
        rotation_radians = np.deg2rad(rotation_degrees)

        # Build 2D rotation matrix.
        cos_theta = np.cos(rotation_radians)
        sin_theta = np.sin(rotation_radians)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Check each shrunk hull vertex.
        for i, vertex in enumerate(shrunk_hull_vertices):
            # Apply rotation.
            rotated_vertex = rotation_matrix @ vertex

            # Translate to placement position.
            transformed_vertex = rotated_vertex + position_2d

            # Check if vertex is within surface boundary.
            if not target_surface.contains_point_2d(transformed_vertex):
                allowed_percent = allow_overlap_ratio * 100
                return (
                    False,
                    f"Object convex hull extends beyond surface boundary by more than "
                    f"{allowed_percent:.1f}% of object size. "
                    f"Shrunk hull vertex {i+1} at "
                    f"({transformed_vertex[0]:.3f}, {transformed_vertex[1]:.3f}) "
                    f"is outside surface {target_surface.surface_id}.\n\n"
                    f"Try:\n"
                    f"- Moving object toward the center of the surface\n"
                    f"- Using a smaller object that fits within the surface\n"
                    f"- Placing on a different surface",
                )

        return (True, None)

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create tool closures that capture current furniture/surface context."""

        @function_tool
        def generate_manipuland_assets(
            object_descriptions: list[str],
            short_names: list[str],
            desired_dimensions: list[list[float]],
            style_context: str | None = None,
        ) -> str:
            """Generate 3D manipuland assets from text descriptions.

            Creates small objects like lamps, books, decorations, kitchenware, etc.
            Each object goes through: text → image → 3D geometry → Drake SDF.

            Args:
                object_descriptions: List of manipuland descriptions
                    (e.g., "Ceramic coffee mug", "Hardcover book").
                short_names: List of filesystem-safe names
                    (e.g., "coffee_mug", "book").
                desired_dimensions: List of [width, depth, height] in meters.
                    Manipulands are typically smaller: 0.05-0.3m range.
                style_context: Optional style context for visual consistency
                    (e.g., "modern kitchen", "cozy bedroom").

            Returns:
                IDs and details of created manipuland models.
            """
            console_logger.info("Tool called: generate_manipuland_assets")
            request = AssetGenerationRequest(
                object_descriptions=object_descriptions,
                short_names=short_names,
                object_type=ObjectType.MANIPULAND,
                desired_dimensions=desired_dimensions,
                style_context=style_context,
                scene_id=self.scene.scene_dir.name,
            )
            return self._generate_assets_impl(request)

        @function_tool
        def list_support_surfaces() -> str:
            """List all support surfaces available on the current furniture.

            Returns:
                JSON string with list of surfaces, each containing:
                - surface_id: Unique identifier for the surface
                - area_m2: Surface area in square meters
                - height_m: Approximate height in meters (Z coordinate of surface)
                - clearance_height: Vertical clearance in meters (max object height)
            """
            surfaces_info = []
            for surface_id_str, surface in self.support_surfaces.items():
                # Get approximate height from transform.
                height = surface.transform.translation()[2]
                surfaces_info.append(
                    {
                        "surface_id": surface_id_str,
                        "area_m2": round(surface.area, 4),
                        "height_m": round(height, 3),
                        "clearance_height": round(
                            float(
                                surface.bounding_box_max[2]
                                - surface.bounding_box_min[2]
                            ),
                            3,
                        ),
                    }
                )

            # Sort by height (top to bottom).
            surfaces_info.sort(key=lambda s: s["height_m"], reverse=True)

            result = {
                "furniture_id": str(self.current_furniture_id),
                "num_surfaces": len(surfaces_info),
                "surfaces": surfaces_info,
            }

            return json.dumps(result, indent=2)

        @function_tool
        def place_manipuland_on_surface(
            asset_id: str,
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Place manipuland on a specific support surface.

            The manipuland is placed using 2D coordinates on the surface plane.
            The system automatically converts to 3D world coordinates.

            Each placement gets a unique ID so you can move or remove it later.
            The same manipuland can be placed multiple times.

            Coordinate system:
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center
            - Rotation: degrees around surface normal (Z-axis)

            Args:
                asset_id: ID of the manipuland asset to place.
                surface_id: ID of the support surface to place on.
                    Use list_support_surfaces() to see available surfaces.
                position_x: X position on surface (meters, left-right).
                position_z: Z position on surface (meters, front-back).
                rotation_degrees: Rotation around surface normal (degrees).
                    Positive values rotate counterclockwise when viewed from above.

            Returns:
                Placement result with world pose and surface-relative pose.
            """
            return self._place_manipuland_on_surface_impl(
                asset_id=asset_id,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def move_manipuland(
            object_id: str,
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Move existing manipuland to a new position on a support surface.

            Use this to reposition manipulands or move them between surfaces.
            You need the object ID from when you placed it or from
            'get_current_scene_state'.

            Coordinate system same as placement:
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters, front-back)
            - Origin (0, 0) is at surface center
            - Rotation: degrees around surface normal (Z-axis)

            Args:
                object_id: ID of the manipuland to move.
                surface_id: ID of the target support surface.
                    Can be same or different from current surface.
                position_x: New X position on surface (meters).
                position_z: New Y position on surface (meters, front-back).
                rotation_degrees: New rotation around surface normal (degrees).
                    Positive values rotate counterclockwise when viewed from above.

            Returns:
                Result of the move operation with new world pose.
            """
            return self._move_manipuland_impl(
                object_id=object_id,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def remove_manipuland(object_id: str) -> str:
            """Remove a manipuland from the scene.

            Args:
                object_id: ID of the manipuland to remove.

            Returns:
                Result of the removal operation.
            """
            return self._remove_manipuland_impl(
                object_id=object_id,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                },
            )

        @function_tool
        def get_current_scene_state() -> str:
            """Get current scene state filtered to current furniture + manipulands.

            Shows:
            - Current furniture being populated (with dimensions)
            - Manipulands already placed on this furniture (with dimensions)
            - Current support surface bounds and clearance_height

            Does NOT show:
            - Other furniture in the scene
            - Manipulands on other furniture

            Returns:
                Scene state with furniture, manipulands, and surface info including:
                - surfaces[].clearance_height: Max object height that fits (meters)
                - manipulands[].dimensions: Object size (width, depth, height)
            """
            return self._get_current_scene_state_impl()

        @function_tool
        def list_available_assets() -> str:
            """List all available manipuland assets (from all furniture).

            This includes manipulands generated for previous furniture, enabling
            asset reuse (e.g., same plate on multiple tables).

            Returns:
                List of all available manipuland assets with IDs and descriptions.
            """
            return self._list_available_assets_impl()

        @function_tool
        def create_stack(
            asset_ids: list[str],
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Create a vertical stack of objects on a support surface.

            Stacks objects bottom-to-top. Creates a single composite object that can be
            moved or removed as a unit.

            Use cases:
            - Stack of plates: ["plate_0", "plate_0", "plate_0"]
            - Stack of books: ["book_red", "book_blue", "book_green"]
            - Mixed items: ["plate_0", "bowl_0", "cup_0"]

            Coordinate system (same as place_manipuland_on_surface):
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center

            Args:
                asset_ids: List of asset IDs to stack (bottom to top). Must have
                    at least 2 items. Use same ID multiple times for identical
                    objects. Use list_available_assets() to see available IDs.
                surface_id: ID of the support surface to place stack on.
                    Use list_support_surfaces() to see available surfaces.
                position_x: X position of stack base on surface (meters, left-right).
                position_z: Z position of stack base on surface (meters, front-back).
                rotation_degrees: Rotation around surface normal (degrees).
                    Applied to entire stack.

            Returns:
                StackCreationResult with composite object ID and height.
                On failure, includes actionable feedback in message.
            """
            return self._create_stack_impl(
                asset_ids=asset_ids,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def fill_container(
            container_asset_id: str,
            fill_asset_ids: list[str],
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Fill a container with objects.

            Places a container (bowl, basket, pen holder) at the specified position
            and fills it with objects.

            Use cases:
            - Fruit bowl with apples and oranges
            - Pen holder with pens and pencils
            - Breadbasket with rolls
            - Toy bin with toys

            Coordinate system (same as place_manipuland_on_surface):
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center

            Args:
                container_asset_id: ID of the container asset (bowl, basket, etc.).
                fill_asset_ids: List of asset IDs to put inside container.
                    Must have at least 1 item. Can use same ID multiple times.
                surface_id: ID of the support surface to place container on.
                position_x: X position of container on surface (meters, left-right).
                position_z: Z position of container on surface (meters, front-back).
                rotation_degrees: Container rotation around surface normal (degrees).

            Returns:
                FillContainerResult with composite object ID and fill count.
                On failure, includes actionable feedback in message.
            """
            return self._fill_container_impl(
                container_asset_id=container_asset_id,
                fill_asset_ids=fill_asset_ids,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def create_arrangement(
            container_asset_id: str,
            fill_assets: list[FillAssetItem],
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Place items at specified positions on a flat container (tray, platter, board).

            Unlike fill_container (random positions inside cavity), this places items
            at your exact x,y coordinates on a flat surface. Fails if items collide
            or fall off.

            Two coordinate systems:
            1. Furniture surface (position_x, position_z, rotation_degrees):
               Same as place_manipuland.

            2. Container local (fill_assets x, y, rotation):
               Origin at container center, in meters.
               +X = right, -X = left (when facing container front)
               +Y = front (near edge), -Y = back (far edge)
               Positions rotate with the container.

            Args:
                container_asset_id: Flat container asset ID.
                fill_assets: List of FillAssetItem with id, x, y, rotation.
                    All fields required. x/y in meters from container center.
                    rotation in degrees (use 0 if no rotation needed).
                surface_id: Furniture surface ID.
                position_x: Container X on surface (meters).
                position_z: Container Z on surface (meters).
                rotation_degrees: Container rotation (degrees).

            Returns:
                FillContainerResult. On failure: container bounds + feedback.
            """
            return self._create_arrangement_impl(
                container_asset_id=container_asset_id,
                fill_assets=fill_assets,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def create_pile(
            asset_ids: list[str], surface_id: str, position_x: float, position_z: float
        ) -> str:
            """Create a random pile of objects on a support surface.

            Drops objects in a random cluster and lets physics settle them into a
            natural, messy arrangement. Creates a composite that moves as a unit.

            Use cases:
            - Toys on floor: ["block_0", "block_0", "toy_car_1"] in kid's room
            - Dirty dishes in sink: ["plate_0", "mug_1", "bowl_2"] in built-in sink
            - Firewood by fireplace: ["log_0", "log_0", "log_0"] on hearth
            - Papers on desk: ["paper_0", "paper_0", "folder_1"] messily stacked
            - Laundry on floor: ["shirt_0", "pants_1", "sock_2"] in messy pile

            NOT for (use other tools instead):
            - Neat table settings → place items individually with place_manipuland
            - Stacked plates/bowls → use create_stack (neat, aligned stacking)
            - Glasses/cups/mugs → place individually (must stand upright!)
            - Any arrangement meant to look tidy or organized

            Pile vs Fill distinction:
            - fill_container: Container is a SEPARATE manipuland (bowl, basket, vase)
            - create_pile: Objects dropped on surface OR into BUILT-IN container (sink)

            Coordinate system (same as place_manipuland_on_surface):
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center

            Args:
                asset_ids: List of asset IDs to pile (minimum 2 items).
                    Use same ID multiple times for identical objects.
                    Use list_available_assets() to see available IDs.
                surface_id: ID of the support surface to place pile on.
                    Use list_support_surfaces() to see available surfaces.
                position_x: X position of pile center on surface (meters, left-right).
                position_z: Z position of pile center on surface (meters, front-back).

            Returns:
                PileCreationResult with composite object ID and pile count.
                On failure, includes actionable feedback in message.
            """
            return self._create_pile_impl(
                asset_ids=asset_ids,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def rescale_manipuland(object_id: str, scale_factor: float) -> str:
            """Resize manipuland by a uniform scale factor.

            IMPORTANT: This rescales the underlying ASSET. All instances of the same
            asset will be affected. This is usually what you want - if one instance
            is too small, they all are.

            NOTE: Composite objects CANNOT be rescaled. This includes:
            - Stacks (created by create_stack)
            - Piles (created by create_pile)
            - Filled containers (created by fill_container)
            To resize items in a composite, remove the composite and recreate it with
            rescaled individual assets.

            Use this when proportions are correct but size is wrong.
            For shape/proportion issues, regenerate the asset instead.

            Args:
                object_id: ID of the manipuland to rescale.
                scale_factor: Scale multiplier (e.g., 1.5 = 50% larger, 0.8 = 20% smaller).

            Returns:
                Result with new dimensions and list of affected objects.
            """
            return self._rescale_manipuland_impl(
                object_id=object_id,
                scale_factor=scale_factor,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                },
            )

        @function_tool
        def resolve_penetrations(object_ids: list[str]) -> str:
            """Resolve collisions between specified objects on a surface.

            Spreads overlapping objects apart while keeping them on the surface.
            All orientations are preserved exactly.

            The surface is inferred from the objects - all objects must be on the
            same surface. Fails if objects are on different surfaces.

            IMPORTANT: This is a LAST RESORT tool. Always prefer manual placement
            with calculated positions that avoid overlaps. Only use this when:
            - Space is genuinely too tight for manual collision avoidance
            - You need many objects in a small area (e.g., 10 bottles on narrow shelf)
            - Multiple stacks must be grouped closer than their footprints allow

            WARNING: Objects may end up at positions you didn't intend. The solver
            spreads objects minimally, but you lose precise control over final XY
            positions. Orientations are always preserved.

            Args:
                object_ids: List of object IDs to resolve. These objects will be
                    spread apart if they overlap. All objects must be on the same
                    surface.

            Returns:
                PenetrationResolutionResult with list of moved objects and
                displacement magnitudes.
            """
            return self._resolve_penetrations_impl(
                object_ids=object_ids,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                },
            )

        return {
            "list_support_surfaces": list_support_surfaces,
            "generate_manipuland_assets": generate_manipuland_assets,
            "place_manipuland_on_surface": place_manipuland_on_surface,
            "move_manipuland": move_manipuland,
            "remove_manipuland": remove_manipuland,
            "rescale_manipuland": rescale_manipuland,
            "get_current_scene_state": get_current_scene_state,
            "list_available_assets": list_available_assets,
            "create_stack": create_stack,
            "fill_container": fill_container,
            "create_arrangement": create_arrangement,
            "create_pile": create_pile,
            # "resolve_penetrations": resolve_penetrations,  # Disabled: 0% success rate in experiments
        }

    @log_scene_action
    def _place_manipuland_on_surface_impl(
        self,
        asset_id: str,
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for placing manipuland on support surface."""
        console_logger.info("Tool called: place_manipuland_on_surface")

        try:
            # Validate surface_id exists.
            if surface_id not in self.support_surfaces:
                available_ids = list(self.support_surfaces.keys())
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Invalid surface_id: {surface_id}. "
                        f"Available surfaces: {available_ids}"
                    ),
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                )

            # Get the target surface.
            target_surface = self.support_surfaces[surface_id]

            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(asset_id)
            except Exception:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=f"Invalid asset ID format: {asset_id}",
                    error_type=ManipulandErrorType.ASSET_NOT_FOUND,
                )

            # Get asset from registry.
            original_asset = self.asset_manager.get_asset_by_id(unique_id)
            if not original_asset:
                # Get all assets and filter for manipulands.
                all_assets = self.asset_manager.list_available_assets()
                available_assets = [
                    asset
                    for asset in all_assets
                    if asset.object_type == ObjectType.MANIPULAND
                ]
                available_ids = [str(a.object_id) for a in available_assets]
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Asset {asset_id} not found. Available manipulands: "
                        f"{available_ids}"
                    ),
                    error_type=ManipulandErrorType.ASSET_NOT_FOUND,
                )

            # Validate position is within surface bounds (convex hull).
            position_2d = np.array([position_x, position_z])
            try:
                if not target_surface.contains_point_2d(position_2d):
                    return self._create_placement_failure_result(
                        asset_id=asset_id,
                        message=(
                            f"Position ({position_x:.3f}, {position_z:.3f}) is outside "
                            f"the convex hull of surface {surface_id}. "
                            f"Use list_support_surfaces() to see available surfaces."
                        ),
                        error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                    )
            except ValueError as e:
                # Surface has no mesh - this shouldn't happen with HSM extraction.
                console_logger.error(f"Surface {surface_id} has no mesh: {e}")
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Surface {surface_id} has no mesh geometry for "
                        f"placement validation."
                    ),
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                )

            # Validate object convex hull fits within surface boundary.
            # Top surfaces allow configurable overlap tolerance for natural overhang.
            # Non-top surfaces (shelves) require strict containment (0% overlap).
            overlap_ratio = (
                self.top_surface_overlap_tolerance
                if self._is_top_surface(surface_id)
                else 0.0
            )
            is_valid, error_msg = self._validate_convex_hull_footprint(
                target_surface=target_surface,
                geometry_path=original_asset.geometry_path,
                position_2d=position_2d,
                rotation_degrees=rotation_degrees,
                allow_overlap_ratio=overlap_ratio,
                scale_factor=original_asset.scale_factor,
            )
            if not is_valid:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=error_msg,
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            # Validate object height fits within surface clearance.
            object_height = float(
                original_asset.bbox_max[2] - original_asset.bbox_min[2]
            )
            surface_clearance = float(
                target_surface.bounding_box_max[2] - target_surface.bounding_box_min[2]
            )

            console_logger.info(
                f"Clearance check: object_height={object_height:.3f}m, "
                f"surface_clearance={surface_clearance:.3f}m"
            )

            if object_height > surface_clearance:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Object height {object_height:.3f}m exceeds surface "
                        f"clearance {surface_clearance:.3f}m. Make sure you are "
                        f"placing on the correct surface that you planned to use. "
                        f"If this is the intended surface, choose a shorter object "
                        f"or find a surface with more clearance."
                    ),
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            console_logger.info(
                f"Placing manipuland {asset_id} ({original_asset.name}) at surface "
                f"position ({position_x:.3f}, {position_z:.3f}), "
                f"rotation {rotation_degrees:.1f}°"
            )

            # Thin coverings have no collision geometry and are welded, so they don't
            # fall during physics simulation. Compensate for the surface gravity offset
            # by placing them directly on the physical surface.
            is_thin_covering = (
                original_asset.metadata.get("asset_source") == "thin_covering"
            )
            z_offset = 0.0
            if is_thin_covering:
                z_offset = -self.cfg.support_surface_extraction.height.surface_offset_m
                console_logger.debug(
                    f"Thin covering detected: applying z_offset={z_offset:.3f}m"
                )

            # Convert SE(2) on surface to SE(3) in world.
            rotation_radians = math.radians(rotation_degrees)
            world_transform = target_surface.to_world_pose(
                position_2d=position_2d, rotation_2d=rotation_radians, z_offset=z_offset
            )

            # Apply placement noise for realistic variation.
            world_transform = apply_placement_noise(
                transform=world_transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )

            # Create new scene object with unique ID.
            object_id = self.scene.generate_unique_id(original_asset.name)
            scene_object = clone_scene_object(
                original_asset,
                object_id=object_id,
                object_type=ObjectType.MANIPULAND,
                transform=world_transform,
                placement_info=PlacementInfo(
                    parent_surface_id=target_surface.surface_id,
                    position_2d=position_2d.copy(),
                    rotation_2d=rotation_radians,
                    placement_method="surface_placement",
                ),
            )

            # Add to scene.
            self.scene.add_object(scene_object)

            # Extract world pose for response.
            world_position = world_transform.translation()
            world_rpy = RollPitchYaw(world_transform.rotation())

            console_logger.info(
                f"Successfully placed manipuland '{original_asset.name}' as "
                f"{object_id} on surface {surface_id}"
            )

            # Create success result.
            result = ManipulandPlacementResult(
                success=True,
                message=(
                    f"Successfully placed '{original_asset.name}' on surface at "
                    f"({position_x:.3f}, {position_z:.3f})"
                ),
                asset_id=asset_id,
                object_id=str(object_id),
                world_position=Position3D(
                    x=float(world_position[0]),
                    y=float(world_position[1]),
                    z=float(world_position[2]),
                ),
                world_rotation=Rotation3D(
                    roll=math.degrees(world_rpy.roll_angle()),
                    pitch=math.degrees(world_rpy.pitch_angle()),
                    yaw=math.degrees(world_rpy.yaw_angle()),
                ),
                surface_position=Position2D(x=position_x, y=position_z),
                surface_rotation_deg=rotation_degrees,
                parent_surface_id=surface_id,
                has_geometry=scene_object.geometry_path is not None,
            )

            return result.to_json()

        except Exception as e:
            console_logger.error(f"Error placing manipuland: {e}", exc_info=True)
            return self._create_placement_failure_result(
                asset_id=asset_id,
                message=f"Unexpected error: {str(e)}",
                error_type=None,
            )

    @log_scene_action
    def _move_manipuland_impl(
        self,
        object_id: str,
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for moving manipuland to new surface position."""
        console_logger.info("Tool called: move_manipuland")

        try:
            # Validate surface_id exists.
            if surface_id not in self.support_surfaces:
                available_ids = list(self.support_surfaces.keys())
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"Invalid surface_id: {surface_id}. "
                        f"Available surfaces: {available_ids}"
                    ),
                    object_id=object_id,
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                ).to_json()

            # Get the target surface.
            target_surface = self.support_surfaces[surface_id]

            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(object_id)
            except Exception:
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=f"Invalid object ID format: {object_id}",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                )

            # Check if object exists.
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Object with ID '{object_id}' not found in scene",
                    object_id=object_id,
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Validate position is within surface bounds (convex hull).
            position_2d = np.array([position_x, position_z])
            try:
                if not target_surface.contains_point_2d(position_2d):
                    return self._create_placement_failure_result(
                        asset_id=object_id,
                        message=(
                            f"Position ({position_x:.3f}, {position_z:.3f}) is outside "
                            f"the convex hull of surface {surface_id}."
                        ),
                        error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                    )
            except ValueError as e:
                # Surface has no mesh.
                console_logger.error(f"Surface {surface_id} has no mesh: {e}")
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=(
                        f"Surface {surface_id} has no mesh geometry for "
                        f"placement validation."
                    ),
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                )

            # Validate object convex hull fits within surface boundary.
            # Top surfaces allow configurable overlap tolerance for natural overhang.
            # Non-top surfaces (shelves) require strict containment (0% overlap).
            overlap_ratio = (
                self.top_surface_overlap_tolerance
                if self._is_top_surface(surface_id)
                else 0.0
            )

            # For composite objects, use reference member's geometry for footprint validation.
            geometry_path = scene_obj.geometry_path
            composite_type = scene_obj.metadata.get("composite_type")
            if geometry_path is None and composite_type == "stack":
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    bottom_geometry = member_assets[0].get("geometry_path")
                    if bottom_geometry:
                        geometry_path = Path(bottom_geometry)
            elif geometry_path is None and composite_type == "filled_container":
                container_asset = scene_obj.metadata.get("container_asset")
                if container_asset:
                    container_geometry = container_asset.get("geometry_path")
                    if container_geometry:
                        geometry_path = Path(container_geometry)
            elif geometry_path is None and composite_type == "pile":
                # Use first member's geometry for pile footprint validation.
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    first_geometry = member_assets[0].get("geometry_path")
                    if first_geometry:
                        geometry_path = Path(first_geometry)

            if geometry_path is None:
                return ManipulandOperationResult(
                    success=False,
                    message="Cannot validate placement: no geometry available",
                    object_id=object_id,
                    error_type=ManipulandErrorType.INVALID_OPERATION,
                ).to_json()

            is_valid, error_msg = self._validate_convex_hull_footprint(
                target_surface=target_surface,
                geometry_path=geometry_path,
                position_2d=position_2d,
                rotation_degrees=rotation_degrees,
                allow_overlap_ratio=overlap_ratio,
                scale_factor=scene_obj.scale_factor,
            )
            if not is_valid:
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=error_msg,
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            # Validate object height fits within surface clearance.
            object_height = float(scene_obj.bbox_max[2] - scene_obj.bbox_min[2])
            surface_clearance = float(
                target_surface.bounding_box_max[2] - target_surface.bounding_box_min[2]
            )

            console_logger.info(
                f"Clearance check (move): object_height={object_height:.3f}m, "
                f"surface_clearance={surface_clearance:.3f}m"
            )

            if object_height > surface_clearance:
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=(
                        f"Object height {object_height:.3f}m exceeds surface "
                        f"clearance {surface_clearance:.3f}m. Make sure you are "
                        f"placing on the correct surface that you planned to use. "
                        f"If this is the intended surface, choose a shorter object "
                        f"or find a surface with more clearance."
                    ),
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            # Get current surface-relative pose from placement info.
            if scene_obj.placement_info is None:
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"Object '{object_id}' has no placement info - "
                        "cannot determine current position"
                    ),
                    object_id=object_id,
                    error_type=ManipulandErrorType.INVALID_OPERATION,
                ).to_json()

            # Note: We allow moving objects between surfaces (no validation of current
            # surface).
            current_position_2d = scene_obj.placement_info.position_2d
            current_rotation_2d = scene_obj.placement_info.rotation_2d

            # Check if both position and rotation are unchanged.
            rotation_radians = math.radians(rotation_degrees)
            position_unchanged = np.allclose(
                current_position_2d, position_2d, atol=1e-6
            )
            rotation_unchanged = np.allclose(
                current_rotation_2d, rotation_radians, atol=1e-6
            )

            if position_unchanged and rotation_unchanged:
                console_logger.info(
                    f"Manipuland '{scene_obj.name}'/'{object_id}' is already at "
                    f"position ({position_x:.3f}, {position_z:.3f}) and rotation "
                    f"{rotation_degrees:.1f}° - no movement needed"
                )
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"{scene_obj.name} is already at the target position and "
                        "rotation - no movement needed"
                    ),
                    object_id=object_id,
                    error_type=ManipulandErrorType.NO_MOVEMENT,
                ).to_json()

            console_logger.info(
                f"Moving manipuland {object_id} ({scene_obj.name}) to surface "
                f"position ({position_x:.3f}, {position_z:.3f}), "
                f"rotation {rotation_degrees:.1f}°"
            )

            # Convert SE(2) on surface to SE(3) in world.
            world_transform = target_surface.to_world_pose(
                position_2d=position_2d, rotation_2d=rotation_radians
            )

            # Apply placement noise for realistic variation.
            world_transform = apply_placement_noise(
                transform=world_transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )

            # For stacks, capture old transform before moving to compute delta.
            old_stack_transform = scene_obj.transform

            # Update object to new pose.
            self.scene.move_object(object_id=unique_id, new_transform=world_transform)

            # For composite objects, also update member transforms to match new position.
            composite_type = scene_obj.metadata.get("composite_type")
            if composite_type == "stack":
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    t_delta = world_transform @ old_stack_transform.inverse()

                    for member in member_assets:
                        old_member_transform = deserialize_rigid_transform(
                            member["transform"]
                        )
                        new_member_transform = t_delta @ old_member_transform
                        member["transform"] = serialize_rigid_transform(
                            new_member_transform
                        )
            elif composite_type == "filled_container":
                # Filled container: update container_asset + all fill_assets transforms.
                t_delta = world_transform @ old_stack_transform.inverse()

                # Update container transform.
                container_asset = scene_obj.metadata.get("container_asset")
                if container_asset:
                    old_container_transform = deserialize_rigid_transform(
                        container_asset["transform"]
                    )
                    new_container_transform = t_delta @ old_container_transform
                    container_asset["transform"] = serialize_rigid_transform(
                        new_container_transform
                    )

                # Update all fill asset transforms.
                fill_assets = scene_obj.metadata.get("fill_assets", [])
                for fill_asset in fill_assets:
                    old_fill_transform = deserialize_rigid_transform(
                        fill_asset["transform"]
                    )
                    new_fill_transform = t_delta @ old_fill_transform
                    fill_asset["transform"] = serialize_rigid_transform(
                        new_fill_transform
                    )
            elif composite_type == "pile":
                # Pile: update all member_assets transforms (same structure as stack).
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    t_delta = world_transform @ old_stack_transform.inverse()

                    for member in member_assets:
                        old_member_transform = deserialize_rigid_transform(
                            member["transform"]
                        )
                        new_member_transform = t_delta @ old_member_transform
                        member["transform"] = serialize_rigid_transform(
                            new_member_transform
                        )

            # Update placement info (including new surface_id if moved between surfaces).
            scene_obj.placement_info.parent_surface_id = target_surface.surface_id
            scene_obj.placement_info.position_2d = position_2d.copy()
            scene_obj.placement_info.rotation_2d = rotation_radians

            # Extract world pose for response.
            world_position = world_transform.translation()
            world_rpy = RollPitchYaw(world_transform.rotation())

            console_logger.info(
                f"Successfully moved manipuland '{scene_obj.name}' ({object_id}) "
                f"to surface {surface_id}"
            )

            # Create success result.
            # For move operations, asset_id uses object_id since no new asset is placed.
            result = ManipulandPlacementResult(
                success=True,
                message=(
                    f"Successfully moved '{scene_obj.name}' to surface position "
                    f"({position_x:.3f}, {position_z:.3f})"
                ),
                asset_id=object_id,
                object_id=str(unique_id),
                world_position=Position3D(
                    x=float(world_position[0]),
                    y=float(world_position[1]),
                    z=float(world_position[2]),
                ),
                world_rotation=Rotation3D(
                    roll=float(math.degrees(world_rpy.roll_angle())),
                    pitch=float(math.degrees(world_rpy.pitch_angle())),
                    yaw=float(math.degrees(world_rpy.yaw_angle())),
                ),
                surface_position=Position2D(x=float(position_x), y=float(position_z)),
                surface_rotation_deg=float(rotation_degrees),
                parent_surface_id=surface_id,
                has_geometry=scene_obj.geometry_path is not None,
            )
            return result.to_json()

        except Exception as e:
            console_logger.error(f"Error moving manipuland: {e}", exc_info=True)
            return self._create_placement_failure_result(
                asset_id=object_id,
                message=f"Unexpected error: {str(e)}",
                error_type=None,
            )

    @log_scene_action
    def _remove_manipuland_impl(self, object_id: str, **kwargs) -> str:
        """Implementation for removing manipuland from scene."""
        console_logger.info(f"Tool called: remove_manipuland({object_id})")

        try:
            # Convert string to UniqueID.
            try:
                unique_id = UniqueID(object_id)
            except Exception:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Invalid object ID format: {object_id}",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

            # Get object from scene.
            obj = self.scene.get_object(unique_id)
            if not obj:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

            # Verify it's a manipuland.
            if obj.object_type != ObjectType.MANIPULAND:
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a manipuland "
                        f"(type: {obj.object_type.value})"
                    ),
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

            # Remove from scene.
            success = self.scene.remove_object(unique_id)

            if success:
                console_logger.info(f"Successfully removed manipuland {object_id}")
                return ManipulandOperationResult(
                    success=True,
                    message=f"Successfully removed '{obj.name}' ({object_id})",
                    object_id=object_id,
                ).to_json()
            else:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Failed to remove {object_id}",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing manipuland: {e}", exc_info=True)
            return ManipulandOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
            ).to_json()

    @log_scene_action
    def _rescale_manipuland_impl(
        self, object_id: str, scale_factor: float, **kwargs
    ) -> str:
        """Implementation for rescaling manipuland."""
        console_logger.info(
            f"Tool called: rescale_manipuland (id={object_id}, scale={scale_factor})"
        )
        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="manipuland",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()

    def _generate_assets_impl(self, request: AssetGenerationRequest) -> str:
        """Implementation for generating manipuland assets."""
        console_logger.info(
            f"Generating batch of {len(request.object_descriptions)} manipuland assets"
        )
        start_time = time.time()

        # Generate assets using asset manager.
        result = self.asset_manager.generate_assets(request)

        # Convert successful assets to DTOs.
        generated_assets = [
            GeneratedAsset(
                name=obj.name,
                object_id=str(obj.object_id),
                description=obj.description,
                width=(
                    float(obj.bbox_max[0] - obj.bbox_min[0])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                depth=(
                    float(obj.bbox_max[1] - obj.bbox_min[1])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                height=(
                    float(obj.bbox_max[2] - obj.bbox_min[2])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
            )
            for obj in result.successful_assets
        ]

        elapsed_time = time.time() - start_time

        # Handle partial success (failures exist).
        if result.has_failures:
            failures_detail = "\n".join(
                [f"- {f.description}: {f.error_message}" for f in result.failed_assets]
            )

            message = (
                f"Partially successful: {len(result.successful_assets)} succeeded, "
                f"{len(result.failed_assets)} failed in {elapsed_time:.1f}s"
            )

            console_logger.warning(message)
            console_logger.warning(f"Failures:\n{failures_detail}")

            return AssetGenerationResult(
                success=False,
                assets=generated_assets,
                message=message,
                successful_count=len(result.successful_assets),
                failed_count=len(result.failed_assets),
                failures=failures_detail,
            ).to_json()

        # All succeeded.
        message = (
            f"Successfully generated {len(generated_assets)} manipuland(s) "
            f"in {elapsed_time:.1f}s"
        )
        console_logger.info(message)

        return AssetGenerationResult(
            success=True,
            assets=generated_assets,
            message=message,
        ).to_json()

    def _get_current_scene_state_impl(self) -> str:
        """Implementation for getting current scene state (filtered)."""
        console_logger.info("Tool called: get_current_scene_state")

        # Get current furniture object.
        furniture = self.scene.get_object(self.current_furniture_id)
        if not furniture:
            return ManipulandOperationResult(
                success=False,
                message=f"Furniture {self.current_furniture_id} not found",
                error_type=ManipulandErrorType.SURFACE_NOT_FOUND,
            ).to_json()

        # Convert furniture to simplified DTO.
        furniture_info = self._scene_object_to_simplified_furniture_info(furniture)

        # Build surfaces with their manipulands grouped together.
        total_manipuland_count = 0
        surface_infos = []
        for surface in self.support_surfaces.values():
            manipulands_on_surface = self.scene.get_objects_on_surface(
                surface.surface_id
            )
            total_manipuland_count += len(manipulands_on_surface)
            surface_info = self._support_surface_to_dto_with_manipulands(
                surface=surface, manipulands=manipulands_on_surface
            )
            surface_infos.append(surface_info)

        # Build structured response with manipulands grouped by surface.
        result = {
            "current_furniture": asdict(furniture_info),
            "surfaces": [asdict(s) for s in surface_infos],
            "num_surfaces": len(surface_infos),
            "total_manipuland_count": total_manipuland_count,
        }

        return json.dumps(result, indent=2)

    def _list_available_assets_impl(self) -> str:
        """Implementation for listing all available manipuland assets."""
        console_logger.info("Tool called: list_available_assets")

        # Get all assets from registry and filter for manipulands.
        all_assets = self.asset_manager.list_available_assets()
        available_assets = [
            asset for asset in all_assets if asset.object_type == ObjectType.MANIPULAND
        ]

        # Convert to simplified DTOs.
        asset_dtos = [AssetInfo.from_scene_object(asset) for asset in available_assets]

        result = AvailableAssetsResult(
            assets=asset_dtos,
            total_count=len(asset_dtos),
            message=f"Found {len(asset_dtos)} available manipuland assets",
        )

        return result.to_json()

    @log_scene_action
    def _create_stack_impl(
        self,
        asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for creating a stack of objects on a support surface.

        Delegates to create_stack_tool_impl in stack_tools.py.
        """
        return create_stack_tool_impl(
            asset_ids=asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
        )

    @log_scene_action
    def _fill_container_impl(
        self,
        container_asset_id: str,
        fill_asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for filling a container with objects.

        Delegates to fill_container_tool_impl in fill_tools.py.
        """
        return fill_container_tool_impl(
            container_asset_id=container_asset_id,
            fill_asset_ids=fill_asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
            top_surface_overlap_tolerance=self.top_surface_overlap_tolerance,
            is_top_surface_fn=self._is_top_surface,
            validate_footprint_fn=self._validate_convex_hull_footprint,
        )

    @log_scene_action
    def _create_arrangement_impl(
        self,
        container_asset_id: str,
        fill_assets: list[FillAssetItem],
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for creating a controlled arrangement on a flat container.

        Delegates to create_arrangement_impl in arrangement_tools.py.
        """
        return create_arrangement_impl(
            container_asset_id=container_asset_id,
            fill_assets=fill_assets,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
            validate_footprint_fn=self._validate_convex_hull_footprint,
            top_surface_overlap_tolerance=self.top_surface_overlap_tolerance,
            is_top_surface_fn=self._is_top_surface,
        )

    @log_scene_action
    def _create_pile_impl(
        self,
        asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
        **kwargs,
    ) -> str:
        """Implementation for creating a pile of objects.

        Delegates to create_pile_tool_impl in pile_tools.py.
        """
        return create_pile_tool_impl(
            asset_ids=asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
        )

    def _scene_object_to_manipuland_info(self, obj: SceneObject) -> ManipulandInfo:
        """Convert SceneObject to ManipulandInfo DTO."""
        position = obj.transform.translation()
        rpy = RollPitchYaw(obj.transform.rotation())

        # Extract placement info if available.
        surface_position = None
        surface_rotation_deg = None
        parent_surface_id = None

        if obj.placement_info:
            surface_position = Position2D(
                x=float(obj.placement_info.position_2d[0]),
                y=float(obj.placement_info.position_2d[1]),
            )
            surface_rotation_deg = math.degrees(obj.placement_info.rotation_2d)
            parent_surface_id = str(obj.placement_info.parent_surface_id)

        return ManipulandInfo(
            object_id=str(obj.object_id),
            description=obj.description,
            object_type=obj.object_type.value,
            position=Position3D(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            rotation=Rotation3D(
                roll=math.degrees(rpy.roll_angle()),
                pitch=math.degrees(rpy.pitch_angle()),
                yaw=math.degrees(rpy.yaw_angle()),
            ),
            surface_position=surface_position,
            surface_rotation_deg=surface_rotation_deg,
            parent_surface_id=parent_surface_id,
            has_geometry=obj.geometry_path is not None,
        )

    def _scene_object_to_simplified_manipuland_info(
        self, obj: SceneObject
    ) -> SimplifiedManipulandInfo:
        """Convert SceneObject to SimplifiedManipulandInfo (minimal fields)."""
        surface_position = None
        surface_rotation_deg = None

        if obj.placement_info:
            surface_position = Position2D(
                x=float(obj.placement_info.position_2d[0]),
                y=float(obj.placement_info.position_2d[1]),
            )
            surface_rotation_deg = math.degrees(obj.placement_info.rotation_2d)

        dimensions = None
        if obj.bbox_min is not None and obj.bbox_max is not None:
            bbox_size = obj.bbox_max - obj.bbox_min
            dimensions = BoundingBox3D(
                width=float(bbox_size[0]),
                depth=float(bbox_size[1]),
                height=float(bbox_size[2]),
            )

        # Build composite metadata if this is a composite object.
        composite_metadata = None
        composite_type = obj.metadata.get("composite_type")
        if composite_type == "stack":
            member_assets = obj.metadata.get("member_assets", [])
            composite_metadata = {
                "type": "stack",
                "members": [m.get("asset_id", "unknown") for m in member_assets],
            }
        elif composite_type == "filled_container":
            container_asset = obj.metadata.get("container_asset", {})
            fill_assets = obj.metadata.get("fill_assets", [])
            placement_method = obj.metadata.get("placement_method", "random")

            if placement_method == "controlled":
                # For arrangements: show poses (x, y, rotation) and shape-aware bounds.
                composite_metadata = {
                    "type": "filled_container",
                    "container_id": container_asset.get("asset_id", "unknown"),
                    "fill_count": len(fill_assets),
                    "fill_items": [
                        {
                            "id": f.get("asset_id", "unknown"),
                            "x": f.get("user_pose", {}).get("x", 0),
                            "y": f.get("user_pose", {}).get("y", 0),
                            "rotation": f.get("user_pose", {}).get("rotation", 0),
                        }
                        for f in fill_assets
                    ],
                    "container_bounds": obj.metadata.get("container_bounds"),
                }
            else:
                # For random fills (fill_container): existing behavior.
                composite_metadata = {
                    "type": "filled_container",
                    "container_id": container_asset.get("asset_id", "unknown"),
                    "fill_object_ids": [
                        f.get("asset_id", "unknown") for f in fill_assets
                    ],
                    "fill_count": len(fill_assets),
                }
        elif composite_type == "pile":
            member_assets = obj.metadata.get("member_assets", [])
            composite_metadata = {
                "type": "pile",
                "members": [m.get("asset_id", "unknown") for m in member_assets],
                "pile_count": len(member_assets),
            }

        return SimplifiedManipulandInfo(
            object_id=str(obj.object_id),
            description=obj.description,
            surface_position=surface_position,
            surface_rotation_deg=surface_rotation_deg,
            dimensions=dimensions,
            composite_metadata=composite_metadata,
        )

    def _scene_object_to_simplified_furniture_info(
        self, obj: SceneObject
    ) -> SimplifiedFurnitureInfo:
        """Convert SceneObject (furniture) to SimplifiedFurnitureInfo."""
        dimensions = None
        if obj.bbox_min is not None and obj.bbox_max is not None:
            bbox_size = obj.bbox_max - obj.bbox_min
            dimensions = BoundingBox3D(
                width=float(bbox_size[0]),
                depth=float(bbox_size[1]),
                height=float(bbox_size[2]),
            )

        return SimplifiedFurnitureInfo(
            object_id=str(obj.object_id),
            description=obj.description,
            dimensions=dimensions,
        )

    def _support_surface_to_dto_with_manipulands(
        self, surface: SupportSurface, manipulands: list[SceneObject]
    ) -> SupportSurfaceWithManipulands:
        """Convert SupportSurface to DTO with its manipulands grouped together."""
        bounds_min_2d = Position2D(
            x=float(surface.bounding_box_min[0]),
            y=float(surface.bounding_box_min[1]),
        )
        bounds_max_2d = Position2D(
            x=float(surface.bounding_box_max[0]),
            y=float(surface.bounding_box_max[1]),
        )

        # Extract world-frame position from surface transform.
        world_position = surface.transform.translation()

        # Convert manipulands to simplified DTOs.
        manipuland_infos = [
            self._scene_object_to_simplified_manipuland_info(obj) for obj in manipulands
        ]

        return SupportSurfaceWithManipulands(
            surface_id=str(surface.surface_id),
            bounds_min=bounds_min_2d,
            bounds_max=bounds_max_2d,
            world_x=float(world_position[0]),
            world_y=float(world_position[1]),
            world_z=float(world_position[2]),
            clearance_height=float(surface.bounding_box_max[2]),
            manipulands=manipuland_infos,
        )

    @log_scene_action
    def _resolve_penetrations_impl(self, object_ids: list[str], **kwargs) -> str:
        """Implementation for resolving penetrations.

        All rotations (roll, pitch, yaw) are fixed. Only XY translation is solved.
        Surface is inferred from the objects.
        """
        console_logger.info(
            f"Tool called: resolve_penetrations with object IDs: {object_ids}"
        )

        # Empty list is a no-op.
        if len(object_ids) == 0:
            console_logger.warning("resolve_penetrations: No objects to resolve")
            return PenetrationResolutionResult(
                success=True,
                message="No objects to resolve",
                num_objects_considered=0,
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
            ).to_json()

        # Convert string IDs to UniqueID and validate they exist.
        unique_ids: list[UniqueID] = []
        scene_objects: list[SceneObject] = []
        for obj_id in object_ids:
            unique_id = UniqueID(obj_id)
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                console_logger.warning(
                    f"resolve_penetrations: Object {obj_id} not found in scene"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=f"Object {obj_id} not found in scene",
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                ).to_json()
            unique_ids.append(unique_id)
            scene_objects.append(scene_obj)

        # Infer surface from objects - all must be on same surface.
        # Also validate all objects are on surfaces of the current furniture.
        surface_ids = set()
        valid_surface_ids = set(self.support_surfaces.keys())
        for scene_obj in scene_objects:
            if scene_obj.placement_info is None:
                console_logger.warning(
                    f"resolve_penetrations: Object {scene_obj.object_id} has no placement info"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=f"Object {scene_obj.object_id} has no placement info",
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                ).to_json()

            parent_surface = str(scene_obj.placement_info.parent_surface_id)
            if parent_surface not in valid_surface_ids:
                console_logger.warning(
                    f"resolve_penetrations: Object {scene_obj.object_id} is on surface "
                    f"{parent_surface} which is not on furniture {self.current_furniture_id}"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=(
                        f"Object {scene_obj.object_id} is on surface {parent_surface} "
                        f"which is not on the current furniture {self.current_furniture_id}"
                    ),
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                ).to_json()

            surface_ids.add(parent_surface)

        # Check for unsupported composite types (piles).
        # Piles have scattered members whose footprints cannot be accurately computed.
        for scene_obj in scene_objects:
            if scene_obj.metadata.get("composite_type") == "pile":
                console_logger.warning(
                    f"resolve_penetrations: Cannot include pile '{scene_obj.object_id}' - "
                    "piles have scattered members with complex footprints"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=(
                        f"Cannot resolve penetrations for pile '{scene_obj.object_id}'. "
                        "Piles have scattered members whose combined footprint cannot be "
                        "accurately computed for collision resolution.\n\n"
                        "Instead:\n"
                        "1. Move other objects away from the pile, OR\n"
                        "2. Remove and recreate the pile at a different location"
                    ),
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.UNSUPPORTED_COMPOSITE_TYPE,
                ).to_json()

        if len(surface_ids) > 1:
            console_logger.warning(
                f"resolve_penetrations: Objects are on {len(surface_ids)} different "
                f"surfaces: {sorted(surface_ids)}"
            )
            return PenetrationResolutionResult(
                success=False,
                message=(
                    f"Objects are on {len(surface_ids)} different surfaces: "
                    f"{sorted(surface_ids)}. All objects must be on the same surface."
                ),
                num_objects_considered=len(object_ids),
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.OBJECTS_ON_DIFFERENT_SURFACES,
            ).to_json()

        surface_id = surface_ids.pop()
        surface = self.support_surfaces[surface_id]

        # Call projection (rotations always fixed, only XY translation solved).
        try:
            _, success, moved_ids, max_displacement = apply_surface_projection(
                scene=self.scene,
                surface=surface,
                object_ids=unique_ids,
                influence_distance=self.cfg.penetration_resolution.influence_distance,
                solver_name=self.cfg.penetration_resolution.solver_name,
                iteration_limit=self.cfg.penetration_resolution.iteration_limit,
                time_limit_s=self.cfg.penetration_resolution.time_limit_s,
            )
        except Exception as e:
            console_logger.error(f"Surface projection failed: {e}", exc_info=True)
            return PenetrationResolutionResult(
                success=False,
                message=f"Physics resolution failed with error: {e}",
                num_objects_considered=len(object_ids),
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.PHYSICS_RESOLUTION_FAILED,
            ).to_json()

        # Build result.
        if success:
            moved_id_strs = [str(uid) for uid in moved_ids]
            return PenetrationResolutionResult(
                success=True,
                message=f"Resolved penetrations. Moved {len(moved_ids)} objects.",
                num_objects_considered=len(object_ids),
                num_objects_moved=len(moved_ids),
                moved_object_ids=moved_id_strs,
                max_displacement_m=max_displacement,
            ).to_json()
        else:
            return PenetrationResolutionResult(
                success=False,
                message=(
                    "Failed to resolve penetrations - objects cannot fit on surface.\n\n"
                    "Recovery options:\n"
                    "1. Remove some objects from the surface\n"
                    "2. Use a larger surface\n"
                    "3. Re-orient objects before calling (e.g., rotate knife parallel to edge)"
                ),
                num_objects_considered=len(object_ids),
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.PHYSICS_RESOLUTION_FAILED,
            ).to_json()

    def _create_placement_failure_result(
        self, asset_id: str, message: str, error_type: ManipulandErrorType | None
    ) -> str:
        """Create failure result for placement operations."""
        result = ManipulandPlacementResult(
            success=False,
            message=message,
            asset_id=asset_id,
            object_id="",
            world_position=Position3D(x=0.0, y=0.0, z=0.0),
            world_rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=0.0),
            surface_position=Position2D(x=0.0, y=0.0),
            surface_rotation_deg=0.0,
            parent_surface_id="",  # No surface association for error results.
            has_geometry=False,
            error_type=error_type,
        )
        return result.to_json()
