"""Mixin class for surface rendering functionality in BlenderRenderer.

This mixin encapsulates all surface mesh management methods for multi-surface
furniture rendering, including overlay creation, visibility management, and cleanup.
"""

import logging

import bmesh
import bpy
import numpy as np

from mathutils import Vector

console_logger = logging.getLogger(__name__)


class SurfaceRenderingMixin:
    """Mixin providing surface mesh management methods for BlenderRenderer.

    This mixin contains methods for creating, managing, and cleaning up surface
    meshes and overlay meshes during multi-surface furniture rendering.

    Attributes:
        _client_objects: Blender collection containing imported scene objects.
        _support_surfaces: List of support surface data dicts.
        _surface_mesh_objects: List of temporary surface mesh objects.
        _overlay_mesh_objects: List of overlay mesh objects.
        _hidden_objects: List of objects hidden for per-surface rendering.
    """

    def _create_surface_overlay_mesh(
        self, surface_data: dict, color: tuple[int, int, int]
    ) -> bpy.types.Object | None:
        """Create semi-transparent colored overlay mesh for support surface.

        Args:
            surface_data: Dictionary containing surface geometry data with keys:
                - convex_hull_vertices: List of vertex positions in world space
                - mesh_faces: List of face indices
                - surface_id: Identifier for the surface
            color: RGB color tuple (0-255 range).

        Returns:
            Created Blender mesh object or None if mesh data missing.
        """
        vertices = surface_data.get("convex_hull_vertices")
        faces = surface_data.get("mesh_faces")
        surface_id = surface_data.get("surface_id", "unknown")

        if vertices is None or faces is None:
            console_logger.warning(
                f"Surface {surface_id} missing mesh data, skipping overlay creation"
            )
            return None

        # Create Blender mesh.
        mesh = bpy.data.meshes.new(name=f"Overlay_{surface_id}_mesh")
        mesh.from_pydata(vertices=vertices, edges=[], faces=faces)
        mesh.update()

        # Use BMesh to calculate normals for programmatically created meshes.
        # EEVEE_NEXT requires proper normal calculation, unlike old EEVEE.
        # Reference: https://developer.blender.org/T57366
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()

        # Validate and finalize mesh.
        mesh.validate()
        mesh.update()

        # Create object.
        obj = bpy.data.objects.new(f"Overlay_{surface_id}", mesh)

        # Add to MAIN SCENE COLLECTION for EEVEE_NEXT compatibility.
        # Testing: client_objects collection may have visibility issues in 4.5.
        bpy.context.scene.collection.objects.link(obj)

        # ALSO add to client_objects for consistency (object can be in multiple
        # collections).
        if self._client_objects is not None:
            try:
                self._client_objects.objects.link(obj)
            except RuntimeError:
                # Already linked, that's fine.
                pass

        # Create material matching add_support_surface_debug_volume() exactly.
        # Use Principled BSDF with transparency for EEVEE_NEXT compatibility.
        mat = bpy.data.materials.new(name=f"Overlay_{surface_id}_material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        # Convert RGB (0-255) to RGBA (0-1).
        rgba = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)

        # Add Principled BSDF shader with full opacity for vibrant colors.
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.5

        # Add Material Output.
        output = nodes.new(type="ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        # Opaque material (no transparency) for vibrant, saturated colors.
        mat.blend_method = "OPAQUE"
        mat.use_backface_culling = False

        obj.data.materials.append(mat)

        return obj

    def _setup_per_surface_rendering(self, surface_data: dict) -> None:
        """Setup rendering for a single support surface.

        Creates a mesh object from the surface geometry and hides furniture objects
        so only the surface mesh and its manipulands are visible.

        Args:
            surface_data: Dictionary containing surface geometry data with keys:
                - convex_hull_vertices: List of vertex positions in world space
                - mesh_faces: List of face indices
                - surface_id: Identifier for the surface
        """
        # Restore visibility of objects hidden by previous surface renders.
        self._restore_object_visibility()

        # Extract mesh geometry.
        vertices = surface_data.get("convex_hull_vertices")
        faces = surface_data.get("mesh_faces")

        if vertices is None or faces is None:
            console_logger.warning(
                f"Surface {surface_data.get('surface_id')} missing mesh data, "
                "skipping surface mesh creation"
            )
            return

        # Create new mesh and object.
        mesh = bpy.data.meshes.new(name=f"SurfaceMesh_{surface_data['surface_id']}")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # CRITICAL FIX for EEVEE_NEXT (Blender 4.5+):
        # Use BMesh to calculate normals for programmatically created meshes.
        # EEVEE_NEXT requires proper normal calculation, unlike old EEVEE.
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()

        # Validate and finalize mesh.
        mesh.validate()
        mesh.update()

        surface_obj = bpy.data.objects.new(
            name=f"Surface_{surface_data['surface_id']}", object_data=mesh
        )

        # Add to scene and client objects collection.
        if self._client_objects is not None:
            self._client_objects.objects.link(surface_obj)
        else:
            # Fallback: add to main scene collection.
            bpy.context.scene.collection.objects.link(surface_obj)

        # Apply simple material for visibility.
        mat = bpy.data.materials.new(name=f"SurfaceMat_{surface_data['surface_id']}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
        surface_obj.data.materials.append(mat)

        # Track created surface mesh for cleanup.
        if not hasattr(self, "_surface_mesh_objects"):
            self._surface_mesh_objects = []
        self._surface_mesh_objects.append(surface_obj)

        # Hide furniture and manipulands not on this surface.
        current_surface_id = surface_data.get("surface_id")
        surface_corners = surface_data.get("corners")
        self._hide_objects_not_on_surface(current_surface_id, surface_corners)

    def _is_point_in_surface_volume(self, point: Vector, surface_corners: list) -> bool:
        """Check if a point is within the bounding box volume of a support surface.

        Args:
            point: Point in world space (Blender coordinates).
            surface_corners: List of 8 corner positions defining the bounding box.

        Returns:
            True if point is within the bounding box volume.
        """
        if len(surface_corners) != 8:
            console_logger.warning(
                f"Surface corners expected 8 points, got {len(surface_corners)}"
            )
            return False

        # Convert corners to numpy array for easier computation.
        corners = np.array(surface_corners)

        # Compute min and max bounds in each dimension.
        min_bounds = corners.min(axis=0)
        max_bounds = corners.max(axis=0)

        # Add small epsilon for numerical tolerance.
        epsilon = 0.01

        # Check if point is within axis-aligned bounding box.
        return (
            min_bounds[0] - epsilon <= point.x <= max_bounds[0] + epsilon
            and min_bounds[1] - epsilon <= point.y <= max_bounds[1] + epsilon
            and min_bounds[2] - epsilon <= point.z <= max_bounds[2] + epsilon
        )

    def _build_position_to_metadata_map(
        self,
    ) -> dict[tuple[float, float, float], dict]:
        """Build a mapping from world positions to metadata for spatial matching.

        Since Drake GLTF export doesn't preserve object_id names, we need to match
        Blender objects to metadata by their world positions.

        Returns:
            Dict mapping (x, y, z) position tuples to metadata dicts.
        """
        position_map = {}
        if hasattr(self, "_scene_objects") and self._scene_objects is not None:
            for obj_meta in self._scene_objects:
                position = obj_meta.get("position")
                if position and len(position) == 3:
                    # Round to 3 decimal places for matching tolerance.
                    key = (
                        round(position[0], 3),
                        round(position[1], 3),
                        round(position[2], 3),
                    )
                    position_map[key] = obj_meta
        return position_map

    def _find_metadata_by_position(
        self,
        blender_obj,
        position_map: dict[tuple[float, float, float], dict],
        tolerance: float = 0.1,
    ) -> dict | None:
        """Find metadata for a Blender object by matching its world position.

        Both Drake and Blender use Z-up coordinates. The GLTF import rotation
        (pi/2 around X) just reverts GLTF's Y-up convention back to Z-up, so
        Blender world positions match Drake metadata positions directly.

        Note: We match using the object's origin (from matrix_world.translation),
        NOT the bounding box center. The metadata position is the object's origin
        in world space (from RigidTransform.translation()), so we must compare
        origins, not geometry centers.

        Args:
            blender_obj: Blender object to find metadata for.
            position_map: Mapping from positions to metadata.
            tolerance: Maximum distance for position matching.

        Returns:
            Metadata dict if found, None otherwise.
        """
        # Get object's origin in world space (NOT bounding box center).
        # The metadata position is the object's origin, so we must match origins.
        origin = blender_obj.matrix_world.translation

        # Try exact match first (rounded to 3 decimals).
        rounded_origin = (round(origin.x, 3), round(origin.y, 3), round(origin.z, 3))
        if rounded_origin in position_map:
            return position_map[rounded_origin]

        # Fuzzy match: find closest position within tolerance.
        best_match = None
        best_distance = tolerance
        for pos, meta in position_map.items():
            dx = origin.x - pos[0]
            dy = origin.y - pos[1]
            dz = origin.z - pos[2]
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_match = meta

        return best_match

    def _hide_objects_not_on_surface(
        self, current_surface_id: str, surface_corners: list | None = None
    ) -> None:
        """Hide furniture and manipulands not on the current support surface.

        Uses position-based matching to identify objects since Drake GLTF export
        doesn't preserve object_id names. Falls back to size heuristic for
        unmatched objects.

        Additionally applies z-cutoff filtering to hide objects that are clearly
        on different height levels than the current surface (e.g., objects on a
        nightstand top vs drawer surfaces).

        Args:
            current_surface_id: Surface ID to show manipulands for.
            surface_corners: List of 8 corner positions defining surface bounding box.
                Used for centroid containment fallback and z-cutoff filtering.
        """
        if not hasattr(self, "_hidden_objects"):
            self._hidden_objects = []

        # Check if client objects collection exists.
        if self._client_objects is None:
            console_logger.warning(
                "Client objects collection not initialized, skipping object hiding"
            )
            return

        # Compute surface z-range for z-cutoff filtering.
        # This helps filter out objects from other surfaces at different heights.
        surface_z_min = None
        surface_z_max = None
        if surface_corners is not None and len(surface_corners) == 8:
            corners_np = np.array(surface_corners)
            surface_z_min = corners_np[:, 2].min()
            surface_z_max = corners_np[:, 2].max()
            # Z-tolerance for filtering (handles slight position differences).
            z_tolerance = 0.05

        # Build position-based mapping for spatial matching.
        # This is needed because Drake GLTF export doesn't preserve object_id names.
        position_map = self._build_position_to_metadata_map()

        # Hide objects not on current surface.
        for obj in self._client_objects.objects:
            # Skip already hidden objects and temporary surface meshes.
            if not obj.visible_get() or obj.name.startswith("Surface_"):
                continue

            # Determine if object should be hidden.
            should_hide = False

            # Handle overlay meshes: hide overlays that don't match current surface.
            if obj.name.startswith("Overlay_"):
                # Extract surface ID from overlay name (format: "Overlay_{surface_id}").
                overlay_surface_id = obj.name.replace("Overlay_", "")
                if overlay_surface_id != current_surface_id:
                    should_hide = True
            # Filter other objects by metadata.
            elif obj.type == "MESH" and obj.data:
                # Match object to metadata by world position.
                obj_meta = self._find_metadata_by_position(obj, position_map)

                if obj_meta:
                    # Use metadata-based filtering.
                    obj_type = obj_meta.get("object_type", "")
                    obj_id = obj_meta.get("object_id")
                    parent_surface_id = obj_meta.get("parent_surface_id")

                    # Hide furniture, except for current and context furniture.
                    if obj_type == "furniture":
                        # Don't hide current furniture (the one owning the surfaces).
                        is_current_furniture = (
                            hasattr(self, "_current_furniture_id")
                            and self._current_furniture_id is not None
                            and obj_id == self._current_furniture_id
                        )
                        # Don't hide context furniture (nearby objects for spatial context).
                        is_context_furniture = (
                            hasattr(self, "_context_furniture_ids")
                            and self._context_furniture_ids
                            and obj_id in self._context_furniture_ids
                        )
                        if not is_current_furniture and not is_context_furniture:
                            should_hide = True
                    # Hide manipulands not on current surface.
                    elif obj_type == "manipuland":
                        if parent_surface_id != current_surface_id:
                            should_hide = True
                        # Even if metadata matches, apply z-cutoff filtering.
                        # This handles cases where metadata surface IDs match but the
                        # object is actually on a different z-level (articulated issue).
                        elif surface_z_min is not None and surface_z_max is not None:
                            bbox_corners = [
                                obj.matrix_world @ Vector(corner)
                                for corner in obj.bound_box
                            ]
                            obj_z_min = min(c.z for c in bbox_corners)
                            # Hide if object bottom is clearly above or below surface.
                            if (
                                obj_z_min > surface_z_max + z_tolerance
                                or obj_z_min < surface_z_min - z_tolerance
                            ):
                                should_hide = True
                else:
                    # Fallback: no metadata match (e.g., room geometry).
                    # Use hybrid approach: centroid containment for small objects
                    # (potential manipulands), size heuristic for large objects.
                    bbox_corners = [
                        obj.matrix_world @ Vector(corner) for corner in obj.bound_box
                    ]
                    bbox_size = max(
                        max(c.x for c in bbox_corners) - min(c.x for c in bbox_corners),
                        max(c.y for c in bbox_corners) - min(c.y for c in bbox_corners),
                        max(c.z for c in bbox_corners) - min(c.z for c in bbox_corners),
                    )

                    # Large objects: hide using size heuristic (floor/walls).
                    if bbox_size > 0.5:
                        should_hide = True
                    # Small objects: use centroid containment if available.
                    elif surface_corners is not None:
                        # Compute object centroid.
                        centroid = sum(bbox_corners, Vector()) / len(bbox_corners)

                        # Hide if centroid NOT within surface volume.
                        if not self._is_point_in_surface_volume(
                            centroid, surface_corners
                        ):
                            should_hide = True

            if should_hide:
                obj.hide_render = True
                obj.hide_viewport = True
                self._hidden_objects.append(obj)

    def _filter_objects_by_surface(
        self, scene_objects: list[dict], current_surface_id: str
    ) -> list[dict]:
        """Filter scene objects to only include those on the specified surface.

        Uses metadata-based filtering to determine which objects belong to the surface.

        Args:
            scene_objects: List of scene object metadata dicts.
            current_surface_id: Surface ID to filter by.

        Returns:
            Filtered list of scene object metadata dicts.
        """
        # Filter objects by parent_surface_id metadata.
        filtered_objects = []
        for obj_meta in scene_objects:
            # Skip room geometry, wall, and floor objects.
            obj_type = obj_meta.get("object_type")
            if obj_type in ["room_geometry", "wall", "floor"]:
                continue

            # Skip furniture.
            if obj_type == "furniture":
                continue

            # Include manipulands on the current surface.
            if obj_type == "manipuland":
                parent_surface_id = obj_meta.get("parent_surface_id")
                if parent_surface_id == current_surface_id:
                    filtered_objects.append(obj_meta)

        return filtered_objects

    def _cleanup_surface_meshes(self) -> None:
        """Remove temporary surface mesh objects created for per-surface rendering.

        Note: This method only cleans up temporary surface meshes created for
        per-surface top views. Overlay meshes are cleaned separately via
        _cleanup_overlay_meshes() to allow them to persist across all views.
        """
        if hasattr(self, "_surface_mesh_objects") and self._surface_mesh_objects:
            for obj in self._surface_mesh_objects:
                try:
                    # Store mesh data reference before unlinking.
                    mesh_data = obj.data if obj.data else None

                    # Remove from collection.
                    if (
                        self._client_objects is not None
                        and obj.name in self._client_objects.objects
                    ):
                        self._client_objects.objects.unlink(obj)

                    # Remove mesh data.
                    if mesh_data is not None:
                        bpy.data.meshes.remove(mesh_data)

                    # Remove object.
                    bpy.data.objects.remove(obj)
                except ReferenceError:
                    # Object already removed, skip.
                    console_logger.debug(
                        "Surface mesh object already removed, skipping"
                    )

            self._surface_mesh_objects.clear()

    def _cleanup_overlay_meshes(self) -> None:
        """Remove overlay mesh objects created for multi-surface visualization.

        Overlay meshes should persist across all views in a render and only be
        cleaned up at the end of the render call.
        """
        if hasattr(self, "_overlay_mesh_objects") and self._overlay_mesh_objects:
            for obj in self._overlay_mesh_objects:
                try:
                    # Store mesh data reference before unlinking.
                    mesh_data = obj.data if obj.data else None

                    # Remove from client objects collection.
                    if (
                        self._client_objects is not None
                        and obj.name in self._client_objects.objects
                    ):
                        self._client_objects.objects.unlink(obj)

                    # Remove mesh data.
                    if mesh_data is not None:
                        bpy.data.meshes.remove(mesh_data)

                    # Remove object.
                    bpy.data.objects.remove(obj)
                except ReferenceError:
                    # Object already removed, skip.
                    console_logger.debug(
                        "Overlay mesh object already removed, skipping"
                    )

            self._overlay_mesh_objects.clear()

    def _restore_object_visibility(self) -> None:
        """Restore visibility of objects hidden for per-surface rendering."""
        if not hasattr(self, "_hidden_objects"):
            return

        for obj in self._hidden_objects:
            try:
                obj.hide_render = False
                obj.hide_viewport = False
            except ReferenceError:
                # Object already removed, skip.
                pass

        self._hidden_objects.clear()
