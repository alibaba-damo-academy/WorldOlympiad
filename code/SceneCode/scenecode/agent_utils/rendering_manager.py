import logging
import random
import shutil
import time

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from scenecode.agent_utils.rendering import render_scene_for_agent_observation
from scenecode.agent_utils.room import RoomScene
from scenecode.utils.logging import BaseLogger

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class RenderingManager:
    """
    Generates and caches 2D renders of 3D scenes for visual analysis.

    Converts Drake simulation scenes into multi-view images via a Blender server,
    with automatic caching to avoid re-rendering unchanged scenes. Includes retry
    logic for reliability and saves scene state for before/after comparisons.
    """

    def __init__(
        self, cfg: OmegaConf, logger: BaseLogger, subdirectory: str | None = None
    ) -> None:
        """Initialize rendering manager.

        Args:
            cfg: Rendering configuration.
            logger: Logger instance for saving outputs.
            subdirectory: Optional subdirectory within scene_renders/ for organizing
                renders (e.g., "furniture", "manipulands_desk_0").
        """
        self.cfg = cfg
        self.logger = logger
        self._subdirectory = subdirectory
        self._render_counter = 0
        self._render_cache: dict[str, Path] = {}
        """Cache to avoid redundant renders. Maps scene content hash to directory
        containing rendered images."""
        self._base_output_dir = self.logger.output_dir
        self._last_render_dir: Path | None = None
        """Most recent render directory. Used by agents to save scores alongside renders."""

    def clear_cache(self) -> None:
        """Clear the render cache to force new renders for all scenes.

        This is useful when you want to ensure fresh renders are created,
        such as after resetting a scene to a previous checkpoint state.
        """
        self._render_cache.clear()
        self._last_render_dir = None
        console_logger.info("Render cache cleared")

    @property
    def last_render_dir(self) -> Path | None:
        """Get the directory of the most recent render.

        Returns:
            Path to the most recent render directory, or None if no renders yet.
        """
        return self._last_render_dir

    def render_scene(
        self,
        scene: RoomScene,
        blender_server: "BlenderServer",
        include_objects: list | None = None,
        exclude_room_geometry: bool = False,
        rendering_mode: str = "furniture",
        render_name: str | None = None,
        support_surfaces: list["SupportSurface"] | None = None,
        show_support_surface: bool = False,
        articulated_open: bool = False,
        wall_surfaces: list[dict] | None = None,
        annotate_object_types: list[str] | None = None,
        wall_surfaces_for_labels: list[dict] | None = None,
        wall_furniture_map: dict[str, list] | None = None,
        room_bounds: tuple[float, float, float, float] | None = None,
        ceiling_height: float | None = None,
        context_furniture_ids: list | None = None,
        side_view_elevation_degrees: float | None = None,
        side_view_start_azimuth_degrees: float | None = None,
        include_vertical_views: bool = True,
        override_side_view_count: int | None = None,
    ) -> Path:
        """Render scene with automatic content-based caching.

        Args:
            scene: RoomScene to render.
            include_objects: Optional list of UniqueID objects to include in rendering.
                If provided, only these objects will be rendered. Useful for focused
                rendering (e.g., manipuland agent viewing only current furniture).
            exclude_room_geometry: If True, completely exclude room geometry from rendering.
                Useful for focused rendering of furniture + manipulands only.
            rendering_mode: Rendering mode - "furniture" for room-scale annotations,
                "manipuland" for surface-focused annotations, "wall" for combined
                context top-down + per-wall orthographic views, "ceiling_perspective"
                for elevated ceiling view, "furniture_selection" for minimal annotation
                renders. Defaults to "furniture".
            render_name: Optional descriptive name for the render directory (e.g.,
                "furniture_selection"). If provided, replaces the default
                "renders_XXX" naming. Useful for semantic organization of renders.
            support_surfaces: For manipuland mode, list of SupportSurface objects
                containing transforms and local bounds. Each surface generates a separate
                rendering view with its own coordinate markers and labels.
            show_support_surface: If True, render green wireframe bbox showing support
                surface bounds for debugging. Defaults to False.
            articulated_open: If True, render articulated furniture with doors/drawers
                open (joints at max values). Useful for manipuland placement to show
                internal surfaces. Defaults to False.
            wall_surfaces: List of wall surface dicts for wall rendering modes.
                Each dict contains wall_id, direction, length, height, transform,
                and excluded_regions.
            annotate_object_types: Optional list of object types to annotate. If provided,
                only objects of these types get annotations.
            wall_surfaces_for_labels: Wall surfaces for top-down wall labels.
            wall_furniture_map: For wall mode, mapping from surface_id to list of
                furniture UniqueIDs to include in each wall's orthographic render.
            room_bounds: For ceiling_perspective mode, room XY bounds
                (min_x, min_y, max_x, max_y) in meters.
            ceiling_height: For ceiling_perspective mode, ceiling height in meters.
            context_furniture_ids: For manipuland mode, list of furniture IDs to keep
                visible in per-surface top-down renders. These provide spatial context
                for item placement (e.g., chairs around a table).
            side_view_elevation_degrees: Optional elevation angle in degrees for side
                view cameras. Overrides default (30 degrees). Useful for context image
                rendering where different angles work better for different furniture.
            side_view_start_azimuth_degrees: Optional starting azimuth angle in degrees
                for side views. 90 degrees positions camera at +Y (front). Overrides
                default (0 degrees with 45° offset for corner views).
            include_vertical_views: Whether to include pure vertical views (top/bottom).
                Defaults to True. Set to False for angled-only context image rendering.
            override_side_view_count: Optional override for number of side views. If
                provided, overrides cfg.side_view_count. Set to 1 for single angled view.

        Returns:
            Path to directory containing rendered images.

        Raises:
            RuntimeError: If all rendering attempts fail.
        """
        console_logger.info(
            f"render_scene called with include_objects="
            f"{[str(obj_id) for obj_id in include_objects] if include_objects is not None else 'None'}"
        )

        # Generate cache key from scene content and rendering parameters.
        # Scene content hash includes all objects, transforms, and support surfaces,
        # so we only need to add rendering parameters that affect visual output.
        cache_key_parts = [f"scene_content_{scene.content_hash()}"]

        # Include rendering parameters that affect visual output.
        if exclude_room_geometry:
            cache_key_parts.append("nofloor")

        if rendering_mode != "furniture":
            cache_key_parts.append(f"mode_{rendering_mode}")

        if articulated_open:
            cache_key_parts.append("articulated_open")

        # Include wall surface IDs in cache key for wall rendering modes.
        if wall_surfaces is not None:
            wall_ids = sorted(ws.get("wall_id", "unknown") for ws in wall_surfaces)
            cache_key_parts.append(f"walls_{'_'.join(wall_ids)}")

        # Include include_objects in cache key since it affects which objects are rendered.
        # This prevents returning cached renders with wrong objects when include_objects
        # differs but other parameters are the same.
        if include_objects is not None:
            # Sort object IDs for consistent hashing regardless of list order.
            sorted_ids = sorted(str(obj_id) for obj_id in include_objects)
            objects_hash = hash(tuple(sorted_ids)) & 0xFFFFFFFF  # 32-bit positive hash.
            cache_key_parts.append(f"objs_{objects_hash:08x}")

        # Include context_furniture_ids in cache key since it affects visibility
        # in per-surface top-down views.
        if context_furniture_ids is not None and len(context_furniture_ids) > 0:
            sorted_ctx_ids = sorted(str(ctx_id) for ctx_id in context_furniture_ids)
            ctx_hash = hash(tuple(sorted_ctx_ids)) & 0xFFFFFFFF
            cache_key_parts.append(f"ctx_{ctx_hash:08x}")

        # Include camera angle parameters in cache key.
        if side_view_elevation_degrees is not None:
            cache_key_parts.append(f"elev_{int(side_view_elevation_degrees)}")
        if side_view_start_azimuth_degrees is not None:
            cache_key_parts.append(f"azim_{int(side_view_start_azimuth_degrees)}")
        if not include_vertical_views:
            cache_key_parts.append("no_vert")
        if override_side_view_count is not None:
            cache_key_parts.append(f"sides_{override_side_view_count}")

        cache_key = "_".join(cache_key_parts)

        # Check cache first.
        if cache_key in self._render_cache:
            console_logger.info(f"CACHE HIT - returning cached render")
            self._last_render_dir = self._render_cache[cache_key]
            return self._last_render_dir

        console_logger.info(f"CACHE MISS - creating new render")

        # Try rendering with error handling and retries.
        num_attempts = self.cfg.retry_count
        for attempt in range(num_attempts):
            try:
                console_logger.info(f"Rendering attempt {attempt + 1}/{num_attempts}")

                # Render.
                image_paths = render_scene_for_agent_observation(
                    scene=scene,
                    cfg=self.cfg,
                    blender_server=blender_server,
                    include_objects=include_objects,
                    exclude_room_geometry=exclude_room_geometry,
                    rendering_mode=rendering_mode,
                    support_surfaces=support_surfaces,
                    show_support_surface=show_support_surface,
                    articulated_open=articulated_open,
                    wall_surfaces=wall_surfaces,
                    annotate_object_types=annotate_object_types,
                    wall_surfaces_for_labels=wall_surfaces_for_labels,
                    wall_furniture_map=wall_furniture_map,
                    room_bounds=room_bounds,
                    ceiling_height=ceiling_height,
                    context_furniture_ids=context_furniture_ids,
                    side_view_elevation_degrees=side_view_elevation_degrees,
                    side_view_start_azimuth_degrees=side_view_start_azimuth_degrees,
                    include_vertical_views=include_vertical_views,
                    override_side_view_count=override_side_view_count,
                )

                # Validate rendering output.
                if not image_paths:
                    raise RuntimeError("Rendering failed: No images returned")
                for img_path in image_paths:
                    if not img_path.exists():
                        raise RuntimeError(f"Rendered image missing: {img_path}")

                # Copy images to output directory.
                # Use render_name for descriptive directory naming if provided.
                # Only increment counter when using default naming to avoid gaps.
                if render_name:
                    dir_name = render_name
                else:
                    self._render_counter += 1
                    dir_name = f"renders_{self._render_counter:03d}"
                if self._subdirectory:
                    images_dir = (
                        self._base_output_dir
                        / f"scene_renders/{self._subdirectory}"
                        / dir_name
                    )
                else:
                    images_dir = self._base_output_dir / f"scene_renders/{dir_name}"
                images_dir.mkdir(parents=True, exist_ok=True)

                for img_path in image_paths:
                    shutil.copy(img_path, images_dir / img_path.name)

                # Save scene checkpoint for validation comparisons.
                self.logger.log_scene(scene=scene, output_dir=images_dir)

                # Cache render for reuse and track as most recent.
                self._render_cache[cache_key] = images_dir
                self._last_render_dir = images_dir
                console_logger.info(f"Cached render with key: {cache_key}")

                return images_dir

            except Exception as e:
                console_logger.error(f"Rendering attempt {attempt + 1} failed: {e}")
                if attempt == num_attempts - 1:
                    console_logger.error("All rendering attempts failed, raising error")
                    raise RuntimeError(f"Scene rendering failed after 3 attempts: {e}")
                else:
                    base_delay = self.cfg.retry_delay
                    jitter = random.uniform(0, 2)
                    retry_delay = base_delay + jitter
                    console_logger.warning(
                        f"Retrying rendering in {retry_delay:.1f} seconds..."
                    )
                    time.sleep(retry_delay)

        # This should never be reached due to the exception handling above.
        raise RuntimeError("Unexpected error in rendering loop")
