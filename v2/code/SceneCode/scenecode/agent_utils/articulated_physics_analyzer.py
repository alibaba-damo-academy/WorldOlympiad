"""Articulated physics analyzer using VLM for PartNet-Mobility assets.

This module analyzes articulated objects (cabinets, dressers, etc.) from
PartNet-Mobility dataset using vision-language models to predict physics
properties for simulation.
"""

import json
import logging
import math
import tempfile
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from omegaconf import DictConfig

from scenecode.agent_utils.blender.render_dataclasses import LinkMeshInfo
from scenecode.agent_utils.urdf_to_sdf import extract_link_meshes
from scenecode.agent_utils.vlm_service import VLMService
from scenecode.prompts import prompt_manager
from scenecode.prompts.registry import MeshPhysicsPrompts
from scenecode.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenecode.agent_utils.blender import BlenderServer

console_logger = logging.getLogger(__name__)


@dataclass
class PlacementOptions:
    """Valid placement locations for an articulated object."""

    on_floor: bool = True
    on_wall: bool = False
    on_ceiling: bool = False
    on_object: bool = False


@dataclass
class LinkAnalysis:
    """Physics analysis for a single articulated link."""

    material: str
    mass_kg: float
    is_static: bool
    description: str


@dataclass
class ArticulatedPhysicsAnalysis:
    """Results from VLM-based articulated object physics analysis."""

    placement_options: PlacementOptions
    """Valid placement locations for this object."""

    front_axis: str
    """Front axis in Blender coordinates (e.g., "+Y", "+X")."""

    scale_correct: bool
    """Whether the object dimensions are realistic."""

    scale_factor: float
    """Scale correction factor (1.0 if scale_correct is True)."""

    link_materials: dict[str, str]
    """Mapping of link names to material types."""

    link_masses: dict[str, float]
    """Mapping of link names to masses in kilograms."""

    link_analysis: dict[str, LinkAnalysis] = field(default_factory=dict)
    """Full analysis for each link."""

    total_mass_kg: float = 0.0
    """Total mass of the entire object in kilograms."""

    link_descriptions: dict[str, str] = field(default_factory=dict)
    """Mapping of link names to descriptions (for JSON output)."""

    object_description: str | None = None
    """Overall description of the object (e.g., 'tall wooden storage cabinet')."""

    vlm_images_dir: Path | None = None
    """Directory containing VLM analysis images (if saved)."""

    front_view_image_index: int | None = None
    """The image index VLM identified as showing the front (for debugging)."""

    is_manipuland: bool = False
    """Whether this object is a manipuland (can be picked up/manipulated)."""


def get_front_axis_from_image_number(
    image_number: int,
    num_side_views: int = 4,
    include_vertical_views: bool = True,
) -> str:
    """Map image index to front axis direction.

    In the articulated pipeline, combined side-view images are interpreted
    directly in Blender coordinates. That means the image index for each side
    view maps to the same axis as the camera direction used by the renderer:
    +X, +Y, -X, -Y for the default 4 side views.

    Args:
        image_number: The image index (0-based).
        num_side_views: Number of side views rendered.
        include_vertical_views: Whether vertical views were included.

    Returns:
        Axis string like "+Y", "-X", etc.
    """
    if include_vertical_views:
        if image_number == 0:
            return "+Z"
        elif image_number == 1:
            return "-Z"
        side_index = image_number - 2
    else:
        side_index = image_number

    # Side views are rendered at equal angles around the object.
    # View 0 (or 2 with vertical views) is from +X direction, and the front
    # axis is interpreted directly from that rendered camera direction.
    angle = 2 * math.pi * side_index / num_side_views
    x = math.cos(angle)
    y = math.sin(angle)

    # Determine primary axis based on largest component.
    if abs(x) > abs(y):
        return "+X" if x > 0 else "-X"
    else:
        return "+Y" if y > 0 else "-Y"


def analyze_articulated_physics(
    urdf_path: Path,
    link_names: list[str],
    bounding_box: dict,
    vlm_service: VLMService,
    cfg: DictConfig,
    blender_server: "BlenderServer",
    category: str | None = None,
    num_side_views: int = 4,
    debug_output_dir: Path | None = None,
) -> ArticulatedPhysicsAnalysis:
    """Analyze articulated object physics using VLM with per-link rendering.

    This function:
    1. Extracts link-to-mesh mappings from the URDF
    2. Renders combined views (all links) and per-link views (each link isolated)
    3. Sends images to VLM with per-link dimensions for accurate analysis
    4. Parses structured response for placement, materials, and masses

    Args:
        urdf_path: Path to the URDF file (meshes are resolved relative to it).
        link_names: List of articulated link names from the URDF.
        bounding_box: Bounding box dict with 'min' and 'max' keys.
        vlm_service: VLM service instance for analysis.
        cfg: Configuration with OpenAI model settings.
        category: Optional object category from dataset metadata (e.g., "Remote",
            "Cabinet"). Used for scale validation - helps VLM detect if dimensions
            are unrealistic for the expected object type.
        num_side_views: Number of equidistant side views for combined render.
        debug_output_dir: Optional directory to save rendered images for debugging.

    Returns:
        ArticulatedPhysicsAnalysis with placement, materials, and mass predictions.

    Raises:
        RuntimeError: If rendering or VLM analysis fails.
        ValueError: If VLM response cannot be parsed.
    """
    # Import constant inside function to keep module import light.
    from scenecode.agent_utils.blender.renderer import ARTICULATED_LIGHT_ENERGY

    start_time = time.time()

    console_logger.info(f"Analyzing articulated physics for {urdf_path}")

    if blender_server is None or not blender_server.is_running():
        raise RuntimeError(
            "Articulated physics analysis requires a running BlenderServer"
        )

    # Determine output directory for rendered images.
    if debug_output_dir is not None:
        output_dir = debug_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="articulated_analysis_")
        output_dir = Path(temp_dir)

    # Extract link-to-mesh mappings from URDF.
    urdf_link_meshes = extract_link_meshes(urdf_path)

    # Convert to renderer's LinkMeshInfo format.
    link_meshes = [
        LinkMeshInfo(
            link_name=lm.link_name,
            mesh_paths=lm.mesh_paths,
            origins=lm.origins,
            world_position=lm.world_position,
            world_rotation=lm.world_rotation,
        )
        for lm in urdf_link_meshes
    ]

    if not link_meshes:
        raise RuntimeError(f"No visual geometry found in URDF: {urdf_path}")

    # Render multi-view images with per-link isolation.
    try:
        render_result = blender_server.render_multiview_articulated(
            link_meshes=link_meshes,
            output_dir=output_dir,
            num_combined_side_views=num_side_views,
            num_link_side_views=4,
            light_energy=ARTICULATED_LIGHT_ENERGY,
            taa_samples=cfg.get("asset_manager", {}).get(
                "validation_taa_samples", None
            ),
        )
        total_images = len(render_result.combined_image_paths) + sum(
            len(paths) for paths in render_result.link_image_paths.values()
        )
        console_logger.info(
            f"Rendered {total_images} articulated views "
            f"({len(render_result.combined_image_paths)} combined, "
            f"{len(render_result.link_image_paths)} links) to {output_dir}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to render articulated object views: {e}") from e

    # Compute dimensions from bounding box.
    bbox_min = bounding_box["min"]
    bbox_max = bounding_box["max"]
    dimensions = [
        bbox_max[0] - bbox_min[0],
        bbox_max[1] - bbox_min[1],
        bbox_max[2] - bbox_min[2],
    ]

    # Load prompt.
    system_prompt = prompt_manager.get_prompt(
        prompt_name=MeshPhysicsPrompts.ARTICULATED
    )

    # Build per-link dimension info for VLM.
    link_dim_lines = []
    for link_name, (w, d, h) in render_result.link_dimensions.items():
        if w > 0 or d > 0 or h > 0:
            link_dim_lines.append(
                f"  - {link_name}: width={w:.3f}m, depth={d:.3f}m, height={h:.3f}m"
            )

    link_dims_text = "\n".join(link_dim_lines) if link_dim_lines else "  (no geometry)"

    # Build category text for VLM.
    category_text = ""
    if category:
        category_text = (
            f"**Dataset category:** {category}\n"
            f"(Use this to validate scale - dimensions should be realistic for this "
            f"object type. If dimensions seem ~10x too large, the data may be in "
            f"wrong units.)\n\n"
        )

    # Build user message text describing the image structure.
    user_text = (
        f"Please analyze this articulated object and provide physics properties.\n\n"
        f"{category_text}"
        f"**Overall bounding box dimensions (meters):**\n"
        f"width={dimensions[0]:.3f}, depth={dimensions[1]:.3f}, "
        f"height={dimensions[2]:.3f}\n\n"
        f"**Articulated links:** {', '.join(link_names)}\n\n"
        f"**Per-link dimensions (meters):**\n{link_dims_text}\n\n"
        f"**Image structure:**\n"
        f"- First {len(render_result.combined_image_paths)} images show the "
        f"complete object from different angles (combined views)\n"
        f"- Following images show each link in isolation to help you identify "
        f"which geometry belongs to which link"
    )

    # Prepare messages for VLM.
    user_content = [
        {"type": "text", "text": system_prompt},
        {"type": "text", "text": user_text},
    ]

    # Add combined images first.
    for img_path in render_result.combined_image_paths:
        img_base64 = encode_image_to_base64(img_path)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
            }
        )

    # Add per-link images with separator text.
    for link_name, img_paths in render_result.link_image_paths.items():
        if img_paths:
            user_content.append({"type": "text", "text": f"\n**Link: {link_name}**"})
            for img_path in img_paths:
                img_base64 = encode_image_to_base64(img_path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    }
                )

    messages = [{"role": "user", "content": user_content}]

    # Call VLM service.
    console_logger.info("Calling VLM service for articulated physics analysis")
    try:
        openai_config = cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.mesh_analysis
        verbosity = openai_config.verbosity.mesh_analysis
        vision_detail = openai_config.vision_detail

        response_text = vlm_service.create_completion(
            model=model,
            messages=messages,
            usage_label="articulated_physics_analysis",
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            response_format={"type": "json_object"},
            vision_detail=vision_detail,
        )

        response_json = json.loads(response_text)
        console_logger.info(
            f"Articulated analysis response ({urdf_path.parent.name}): {response_json}"
        )

    except Exception as e:
        raise RuntimeError(f"VLM analysis failed: {e}") from e

    # Parse VLM response.
    try:
        # Parse placement options.
        placement_opts = response_json.get("placement_options", {})
        placement_options = PlacementOptions(
            on_floor=placement_opts.get("on_floor", True),
            on_wall=placement_opts.get("on_wall", False),
            on_ceiling=placement_opts.get("on_ceiling", False),
            on_object=placement_opts.get("on_object", False),
        )

        # Parse front axis from image index.
        front_view_image_index = response_json.get("front_view_image_index")
        if front_view_image_index is None:
            front_view_idx = 5
            console_logger.info("Use default value: image 5 (front as -Y)")
        else:
            front_view_idx = int(front_view_image_index)
            console_logger.info(
                "Get front_view_image_index as %s from VLM response",
                front_view_idx,
            )
        front_axis = get_front_axis_from_image_number(
            image_number=front_view_idx,
            num_side_views=num_side_views,
            include_vertical_views=True,
        )

        # Parse scale info.
        scale_correct = response_json.get("scale_correct", True)
        scale_factor_raw = response_json.get("scale_factor")
        target_dimensions = response_json.get("target_dimensions")

        # Compute scale_factor: prefer explicit value, fall back to computing from
        # target_dimensions using median (same pattern as mesh_utils.py).
        if scale_factor_raw is not None:
            scale_factor = float(scale_factor_raw)
        elif target_dimensions is not None and not scale_correct:
            # Step 4 fallback: compute uniform scale from target dimensions.
            current_dims = np.array(dimensions)
            target_dims = np.array(target_dimensions)
            # Avoid division by zero.
            valid_mask = current_dims > 0
            if valid_mask.any():
                scale_factors = target_dims[valid_mask] / current_dims[valid_mask]
                scale_factor = float(np.median(scale_factors))
                console_logger.info(
                    f"Computed scale_factor={scale_factor:.4f} from target_dimensions"
                )
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0

        # Parse is_manipuland (default False for articulated furniture).
        is_manipuland = response_json.get("is_manipuland", False)

        # Parse object_description.
        object_description = response_json.get("object_description")

        # Parse link analysis.
        link_analysis_raw = response_json.get("link_analysis", {})
        link_materials = {}
        link_masses = {}
        link_analysis = {}
        link_descriptions = {}
        for link_name, analysis in link_analysis_raw.items():
            material = analysis.get("material", "wood")
            mass_kg = float(analysis.get("mass_kg", 5.0))
            is_static = analysis.get("is_static", True)
            description = analysis.get("description", "")

            link_materials[link_name] = material
            link_masses[link_name] = mass_kg
            link_descriptions[link_name] = description
            link_analysis[link_name] = LinkAnalysis(
                material=material,
                mass_kg=mass_kg,
                is_static=is_static,
                description=description,
            )

        # Fill in any missing links with defaults.
        for link_name in link_names:
            if link_name not in link_materials:
                link_materials[link_name] = "wood"
                link_masses[link_name] = 5.0
                link_descriptions[link_name] = ""

        # Parse total mass (or compute from link masses if not provided).
        total_mass_kg = response_json.get("total_mass_kg")
        if total_mass_kg is not None:
            total_mass_kg = float(total_mass_kg)
        else:
            # Compute from link masses.
            total_mass_kg = sum(link_masses.values())

        console_logger.info(
            f"Articulated analysis complete: front={front_axis}, "
            f"scale_correct={scale_correct}, links={len(link_materials)}, "
            f"total_mass={total_mass_kg:.1f}kg"
        )

        console_logger.info(
            f"Articulated analysis complete in {time.time() - start_time:.2f} seconds"
        )

        return ArticulatedPhysicsAnalysis(
            placement_options=placement_options,
            front_axis=front_axis,
            scale_correct=scale_correct,
            scale_factor=scale_factor,
            link_materials=link_materials,
            link_masses=link_masses,
            link_analysis=link_analysis,
            total_mass_kg=total_mass_kg,
            link_descriptions=link_descriptions,
            object_description=object_description,
            vlm_images_dir=output_dir,
            front_view_image_index=front_view_idx,
            is_manipuland=is_manipuland,
        )

    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse VLM response: {e}") from e
