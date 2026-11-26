import base64
import json
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import ObjectType, RoomScene, UniqueID
from scenecode.agent_utils.vlm_service import VLMService
from scenecode.prompts import prompt_manager
from scenecode.prompts.registry import ManipulandAgentPrompts
from scenecode.utils.omegaconf import OmegaConf
from scenecode.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenecode.agent_utils.blender import BlenderServer

console_logger = logging.getLogger(__name__)


@dataclass
class FurnitureSelection:
    """Selection result for a furniture piece to receive manipulands.

    Contains both the furniture identifier and the structured assignment context
    from the initial VLM analysis. These fields are passed directly to the
    manipuland agent prompts.
    """

    furniture_id: UniqueID
    """ID of the selected furniture piece."""
    suggested_items: str
    """Items to place, with REQUIRED vs Optional distinction."""
    prompt_constraints: str
    """What the prompt explicitly requested for this surface."""
    style_notes: str
    """Style guidance (density, aesthetic) for this surface."""
    context_furniture_ids: list[UniqueID] = field(default_factory=list)
    """IDs of nearby furniture for context (e.g., chairs around a table)."""


def _compute_aabb_edge_distance(
    bounds_a: tuple[np.ndarray, np.ndarray],
    bounds_b: tuple[np.ndarray, np.ndarray],
) -> float:
    """Compute minimum XY-plane distance between two AABBs (edge-to-edge).

    Returns 0.0 if boxes overlap in XY plane.
    """
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b

    # For each axis, compute gap (negative means overlap).
    dx = max(min_a[0] - max_b[0], min_b[0] - max_a[0], 0.0)
    dy = max(min_a[1] - max_b[1], min_b[1] - max_a[1], 0.0)

    return float(np.sqrt(dx**2 + dy**2))


def _compute_direction(from_center: np.ndarray, to_center: np.ndarray) -> str:
    """Compute 8-way direction from XY delta.

    Y+ = NORTH, X+ = EAST (room coordinates).
    """
    dx = to_center[0] - from_center[0]
    dy = to_center[1] - from_center[1]

    angle = np.degrees(np.arctan2(dy, dx))  # 0° = EAST, 90° = NORTH.
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    index = int((angle + 22.5) % 360 / 45)
    return directions[index]


def compute_nearby_furniture_candidates(
    scene: RoomScene,
    target_furniture_id: UniqueID,
    distance_threshold_m: float = 2.0,
    max_candidates: int = 30,
) -> list[dict]:
    """Compute nearby furniture with distance and direction.

    Uses edge-to-edge AABB distance (XY plane only) for proximity.

    Args:
        scene: RoomScene with all furniture.
        target_furniture_id: Furniture to find candidates around.
        distance_threshold_m: Max distance to consider.
        max_candidates: Max candidates to return.

    Returns:
        List of candidate dicts sorted by distance:
        {
            "furniture_id": str,
            "name": str,
            "description": str,
            "distance_m": float,
            "direction": str,  # "N", "NE", "E", etc.
        }
    """
    target = scene.get_object(target_furniture_id)
    if target is None:
        return []

    target_bounds = target.compute_world_bounds()
    if target_bounds is None:
        return []

    target_center = (target_bounds[0] + target_bounds[1]) / 2

    candidates = []
    placeable_types = (ObjectType.FURNITURE, ObjectType.WALL_MOUNTED)

    for obj in scene.objects.values():
        # Skip self and non-furniture.
        if obj.object_id == target_furniture_id:
            continue
        if obj.object_type not in placeable_types:
            continue
        if obj.immutable:
            continue

        obj_bounds = obj.compute_world_bounds()
        if obj_bounds is None:
            continue

        distance = _compute_aabb_edge_distance(target_bounds, obj_bounds)
        if distance > distance_threshold_m:
            continue

        obj_center = (obj_bounds[0] + obj_bounds[1]) / 2
        direction = _compute_direction(target_center, obj_center)

        candidates.append(
            {
                "furniture_id": str(obj.object_id),
                "name": obj.name,
                "description": obj.description or "",
                "distance_m": round(distance, 2),
                "direction": direction,
            }
        )

    # Sort by distance, limit count.
    candidates.sort(key=lambda c: c["distance_m"])
    return candidates[:max_candidates]


class SceneAnalyzer:
    """
    VLM-based scene analysis for spatial and contextual understanding.

    Provides both general-purpose and domain-specific analysis methods:
    - General: analyze_scene() for any VLM-based analysis task
    - Furniture: analyze_furniture_for_manipulands() for object placement selection

    All methods follow the pattern: render → VLM → parse, with appropriate
    error handling, validation, and logging.
    """

    def __init__(
        self,
        vlm_service: VLMService,
        rendering_manager: RenderingManager,
        cfg: OmegaConf,
        blender_server: "BlenderServer",
    ) -> None:
        """Initialize scene analyzer.

        Args:
            vlm_service: VLM service instance.
            rendering_manager: Rendering manager instance.
            cfg: Configuration for OpenAI model settings.
            blender_server: BlenderServer instance for rendering.
        """
        self.vlm_service = vlm_service
        self.rendering_manager = rendering_manager
        self.cfg = cfg
        self.blender_server = blender_server

    def analyze_scene(
        self,
        scene: RoomScene,
        prompt_enum: Any,
        prompt_kwargs: dict[str, Any] | None = None,
        rendering_params: dict[str, Any] | None = None,
        settings_key: str = "scene_critique",
        user_message: str | None = None,
    ) -> str:
        """General-purpose VLM-based scene analysis.

        This method provides a flexible foundation for any VLM-based scene
        analysis task following the pattern: render → VLM → return response.

        Args:
            scene: RoomScene to analyze.
            prompt_enum: Prompt enum from prompt registry.
            prompt_kwargs: Additional kwargs for prompt formatting.
            rendering_params: Parameters for render_scene() (e.g., rendering_mode).
            settings_key: Key in cfg.openai.reasoning_effort and cfg.openai.verbosity.
            user_message: Optional user message (defaults to generic request).

        Returns:
            Raw VLM response string (usually JSON).
        """
        # Render scene with optional parameters.
        rendering_params = rendering_params or {}
        images_dir = self.rendering_manager.render_scene(
            scene=scene, blender_server=self.blender_server, **rendering_params
        )

        # Load and encode images.
        image_paths = sorted(images_dir.glob("*.png"))
        images = [encode_image_to_base64(str(img_path)) for img_path in image_paths]

        # Get system prompt.
        prompt_kwargs = prompt_kwargs or {}
        system_prompt = prompt_manager.get_prompt(
            prompt_name=prompt_enum,
            **prompt_kwargs,
        )

        # Default user message.
        if user_message is None:
            user_message = (
                "Analyze the rendered scene views and provide your assessment "
                "in the requested format."
            )

        # Build messages.
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                        for img in images
                    ],
                ],
            },
        ]

        # Call VLM.
        openai_config = self.cfg.openai
        reasoning_effort = getattr(openai_config.reasoning_effort, settings_key)
        verbosity = getattr(openai_config.verbosity, settings_key)

        response = self.vlm_service.create_completion(
            model=openai_config.model,
            messages=messages,
            usage_label="scene_analyzer.furniture_selection",
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            response_format={"type": "json_object"},
            vision_detail=openai_config.vision_detail,
        )

        return response

    def analyze_furniture_for_manipulands(
        self, scene: RoomScene, prompt_enum: Any
    ) -> list[FurnitureSelection]:
        """Analyze which furniture should have manipulands placed on them.

        This method handles furniture filtering, VLM-based selection, and
        validation of results. It renders the scene in furniture mode (room-scale
        view) to give the VLM appropriate context for furniture selection.

        The VLM returns structured selection data including:
        - suggested_items: REQUIRED vs Optional items for each surface
        - prompt_constraints: What the prompt explicitly requested
        - style_notes: Style guidance (minimalist, cozy, etc.)

        Args:
            scene: Furnished scene to analyze.
            prompt_enum: Prompt enum for furniture analysis.

        Returns:
            List of FurnitureSelection objects with assignment context.
        """
        console_logger.info("Analyzing furniture for manipuland placement (VLM-based)")

        # Filter to furniture and wall-mounted objects (shelves, racks, etc.).
        # Wall-mounted objects can have support surfaces for manipulands.
        placeable_types = (ObjectType.FURNITURE, ObjectType.WALL_MOUNTED)
        furniture_objects = [
            obj
            for obj in scene.objects.values()
            if obj.object_type in placeable_types
            and not obj.immutable
            and obj.bbox_min is not None
            and obj.bbox_max is not None
        ]

        # Add floor if available.
        if scene.room_geometry and scene.room_geometry.floor:
            floor = scene.room_geometry.floor
            if floor.bbox_min is not None and floor.bbox_max is not None:
                furniture_objects.append(floor)

        if not furniture_objects:
            console_logger.info("No furniture found in scene")
            return []

        # Build furniture list for VLM reference.
        furniture_list = "\n".join(
            f"- {obj.object_id}: {obj.name} - {obj.description}"
            for obj in furniture_objects
        )

        # Only render placeable objects (furniture + wall-mounted).
        # Excludes ceiling objects which aren't relevant for furniture selection.
        include_objects = [obj.object_id for obj in furniture_objects]

        # Call VLM with retry logic for JSON parsing failures.
        max_retries = self.cfg.openai.furniture_analysis_max_retries
        response_str = ""
        analysis: dict = {}

        for attempt in range(max_retries):
            try:
                # Use general analysis method.
                response_str = self.analyze_scene(
                    scene=scene,
                    prompt_enum=prompt_enum,
                    prompt_kwargs={
                        "scene_description": scene.text_description,
                        "furniture_list": furniture_list,
                    },
                    rendering_params={
                        "rendering_mode": "furniture_selection",
                        "render_name": "furniture_selection",
                        "include_objects": include_objects,
                    },
                    settings_key="furniture_analysis",
                    user_message=(
                        "Analyze the rendered scene views and identify which furniture "
                        "pieces should have manipulands placed on them. Provide your "
                        "selections in JSON format."
                    ),
                )

                # Parse response.
                analysis = json.loads(response_str)
                break  # Success.
            except json.JSONDecodeError as e:
                preview = repr(response_str[:200]) if response_str else "empty"
                if attempt < max_retries - 1:
                    console_logger.warning(
                        f"VLM returned invalid JSON (attempt {attempt + 1}/{max_retries}). "
                        f"Parse error: {e}. Preview: {preview}. Retrying..."
                    )
                    continue
                # Final attempt failed.
                raise RuntimeError(
                    f"VLM returned invalid JSON after {max_retries} attempts. "
                    f"Parse error: {e}. Preview: {preview}"
                ) from e

        furniture_selections = analysis.get("furniture_selections", [])

        # Build valid IDs set for validation.
        valid_furniture_ids = {obj.object_id for obj in furniture_objects}

        # Extract scene-level style from analysis (fallback if not present).
        scene_style = analysis.get("scene_style", "")

        # Extract and validate selections.
        furniture_data: list[FurnitureSelection] = []
        for selection in furniture_selections:
            furniture_id_str = selection.get("furniture_id")
            suggested_items = selection.get("suggested_items", "")
            prompt_constraints = selection.get(
                "prompt_constraints", "No specific requirements"
            )
            style_notes = selection.get("style_notes", scene_style or "default style")

            if not furniture_id_str:
                console_logger.warning("VLM selection missing furniture_id")
                continue

            fid = UniqueID(furniture_id_str)
            if fid in valid_furniture_ids:
                furniture_data.append(
                    FurnitureSelection(
                        furniture_id=fid,
                        suggested_items=suggested_items,
                        prompt_constraints=prompt_constraints,
                        style_notes=style_notes,
                    )
                )
            else:
                console_logger.warning(
                    f"VLM selected invalid furniture ID: {furniture_id_str}"
                )

        console_logger.info(
            f"Selected {len(furniture_data)} furniture pieces for placement:\n"
            f"{[f.furniture_id for f in furniture_data]}"
        )

        return furniture_data

    def select_context_furniture(
        self,
        scene: RoomScene,
        furniture_selections: list[FurnitureSelection],
        furniture_selection_images_dir: Path | None = None,
    ) -> dict[UniqueID, list[UniqueID]]:
        """Select context furniture for each selected furniture piece.

        Uses text-based context (distances, directions) with optional images
        from the initial furniture selection render.

        Args:
            scene: RoomScene with all furniture.
            furniture_selections: Output from analyze_furniture_for_manipulands().
            furniture_selection_images_dir: Optional path to furniture_selection
                render directory for image context.

        Returns:
            Mapping from furniture_id to list of context_furniture_ids.
        """
        if not furniture_selections:
            return {}

        cfg_context = self.cfg.context_furniture
        if not cfg_context.enabled:
            return {}

        # Build candidates for each selected furniture.
        all_candidates: dict[str, list[dict]] = {}
        for selection in furniture_selections:
            candidates = compute_nearby_furniture_candidates(
                scene=scene,
                target_furniture_id=selection.furniture_id,
                distance_threshold_m=cfg_context.distance_threshold_m,
                max_candidates=cfg_context.max_candidates,
            )
            if candidates:
                all_candidates[str(selection.furniture_id)] = candidates

        # If no furniture has candidates, return empty.
        if not all_candidates:
            console_logger.info("No nearby furniture candidates found")
            return {}

        # Format for VLM prompt.
        furniture_with_candidates = self._format_candidates_for_prompt(
            scene=scene,
            furniture_selections=furniture_selections,
            all_candidates=all_candidates,
        )

        # Load images if configured and available.
        images: list[str] = []
        if cfg_context.include_images and furniture_selection_images_dir:
            images = self._load_images_from_dir(furniture_selection_images_dir)

        # Call VLM with retry logic.
        max_retries = self.cfg.openai.context_selection_max_retries
        response_str = ""
        result: dict = {}

        for attempt in range(max_retries):
            try:
                response_str = self._call_vlm_for_context_selection(
                    furniture_with_candidates=furniture_with_candidates,
                    images=images,
                )
                result = json.loads(response_str)
                break
            except json.JSONDecodeError as e:
                preview = repr(response_str[:200]) if response_str else "empty"
                if attempt < max_retries - 1:
                    console_logger.warning(
                        f"Context selection VLM returned invalid JSON "
                        f"(attempt {attempt + 1}/{max_retries}). "
                        f"Parse error: {e}. Preview: {preview}. Retrying..."
                    )
                    continue
                console_logger.warning(
                    f"Context selection VLM failed after {max_retries} attempts. "
                    f"Returning empty context."
                )
                return {}
            except Exception as e:
                console_logger.warning(f"Context selection VLM call failed: {e}")
                return {}

        # Parse and validate response.
        return self._parse_context_selection_response(result, all_candidates)

    def _format_candidates_for_prompt(
        self,
        scene: RoomScene,
        furniture_selections: list[FurnitureSelection],
        all_candidates: dict[str, list[dict]],
    ) -> str:
        """Format candidates as text for VLM prompt."""
        lines = []
        for selection in furniture_selections:
            fid = str(selection.furniture_id)
            candidates = all_candidates.get(fid, [])
            if not candidates:
                continue

            # Get furniture name from scene.
            furniture = scene.get_object(selection.furniture_id)
            furniture_name = furniture.name if furniture else fid

            lines.append(f"\n## {furniture_name} ({fid})")
            lines.append(f"Items to place: {selection.suggested_items}")
            lines.append("\nNearby furniture:")
            for c in candidates:
                lines.append(
                    f"  - {c['furniture_id']}: {c['name']} "
                    f"({c['distance_m']}m {c['direction']})"
                )

        return "\n".join(lines)

    def _load_images_from_dir(self, images_dir: Path) -> list[str]:
        """Load PNG images from directory as base64 strings."""
        images = []
        for img_path in sorted(images_dir.glob("*.png")):
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                images.append(base64.b64encode(img_bytes).decode())
        return images

    def _call_vlm_for_context_selection(
        self,
        furniture_with_candidates: str,
        images: list[str],
    ) -> str:
        """Make VLM call for context selection."""
        # Get system prompt with template variables.
        system_prompt = prompt_manager.get_prompt(
            prompt_name=ManipulandAgentPrompts.SELECT_CONTEXT_FURNITURE,
            furniture_with_candidates=furniture_with_candidates,
        )

        user_message = (
            "Analyze the rendered scene views and select context furniture "
            "for each piece. Provide your selections in JSON format."
        )

        # Build messages.
        if images:
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": user_message},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                    for img in images
                ],
            ]
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        # Call VLM.
        openai_config = self.cfg.openai
        reasoning_effort = openai_config.reasoning_effort.context_selection
        verbosity = openai_config.verbosity.context_selection

        response = self.vlm_service.create_completion(
            model=openai_config.model,
            messages=messages,
            usage_label="scene_analyzer.context_selection",
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            response_format={"type": "json_object"},
            vision_detail=openai_config.vision_detail,
        )

        return response

    def _parse_context_selection_response(
        self,
        result: dict,
        all_candidates: dict[str, list[dict]],
    ) -> dict[UniqueID, list[UniqueID]]:
        """Parse and validate VLM response."""
        context_map: dict[UniqueID, list[UniqueID]] = {}

        for selection in result.get("context_selections", []):
            furniture_id_str = selection.get("furniture_id")
            context_ids = selection.get("context_furniture_ids", [])

            if not furniture_id_str:
                continue

            # Validate against candidates.
            valid_ids = {
                c["furniture_id"] for c in all_candidates.get(furniture_id_str, [])
            }
            validated_context = []

            for ctx_id in context_ids:
                if ctx_id in valid_ids:
                    validated_context.append(UniqueID(ctx_id))
                else:
                    console_logger.warning(
                        f"VLM returned invalid context ID '{ctx_id}' "
                        f"for {furniture_id_str}"
                    )

            if validated_context:
                context_map[UniqueID(furniture_id_str)] = validated_context
                console_logger.info(
                    f"Context for {furniture_id_str}: "
                    f"{[str(c) for c in validated_context]}"
                )

        return context_map
