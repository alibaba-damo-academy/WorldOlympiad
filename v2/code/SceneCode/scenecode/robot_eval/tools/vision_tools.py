"""Vision tools for validation and policy interface agents.

Provides rendering capabilities for visual inspection of scenes.
Requires a BlenderServer to generate renders of the current scene state.
"""

import base64
import logging
import math

from pathlib import Path
from typing import TYPE_CHECKING

import requests

from agents import FunctionTool, ToolOutputImage, function_tool
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ApplyCameraConfig,
    CameraConfig,
    DiagramBuilder,
    Parser,
    Quaternion,
    RenderEngineGltfClientParams,
    Rgba,
    RigidTransform,
    Transform,
)

from scenecode.agent_utils.blender.request_lock import acquire_blender_request_lock
from scenecode.agent_utils.drake_utils import create_plant_from_dmd
from scenecode.robot_eval.dmd_scene import DMDScene

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


def _compute_wall_normals_from_scene_state(scene_state: dict) -> dict[str, list[float]]:
    """Compute wall normals from scene_state wall data.

    Normals point from wall center toward room center (0, 0) in the XY plane.
    Used by Blender to hide camera-facing walls in side views.

    Args:
        scene_state: Scene state dict with "_walls" list (from normalization).

    Returns:
        Dict mapping wall name to [x, y] normal vector.
    """
    wall_normals = {}

    # Get walls from _walls field (populated during normalization).
    walls = scene_state.get("_walls", [])
    for wall_data in walls:
        obj_id = wall_data.get("object_id", "")
        if not obj_id:
            continue

        transform = wall_data.get("transform", {})
        translation = transform.get("translation", [0, 0, 0])

        # Wall center in XY plane.
        wall_x, wall_y = translation[0], translation[1]

        # Normal points from wall toward room center (0, 0).
        normal_x = -wall_x
        normal_y = -wall_y

        # Normalize to unit vector.
        length = math.sqrt(normal_x * normal_x + normal_y * normal_y)
        if length > 1e-6:
            normal_x /= length
            normal_y /= length

        wall_normals[obj_id] = [normal_x, normal_y]

    return wall_normals


def _image_path_to_data_url(image_path: Path) -> str:
    """Convert an image file to a base64 data URL.

    Args:
        image_path: Path to PNG image file.

    Returns:
        Data URL string like "data:image/png;base64,..."
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def _render_validation_scene(
    scene: DMDScene,
    blender_server: "BlenderServer",
    output_dir: Path,
    include_object_ids: list[str] | None = None,
    layout: str = "top_plus_sides",
    side_view_count: int = 4,
) -> list[Path]:
    """Render DMDScene for validation using existing Blender infrastructure.

    For full scene renders (include_object_ids=None), loads the entire DMD file
    directly, which includes room geometry and all objects with correct poses.

    For filtered renders, builds a custom plant with only the specified objects.

    Args:
        scene: DMDScene with scene_state metadata and dmd_path.
        blender_server: Running BlenderServer instance.
        output_dir: Directory to save rendered images.
        include_object_ids: If provided, only render these objects (no room geometry).
            If None, render full scene from DMD (includes room geometry).
        layout: Blender layout mode (e.g., "top_plus_sides").
        side_view_count: Number of side views to render.

    Returns:
        List of paths to rendered PNG images.

    Raises:
        RuntimeError: If Blender config fails or no images are rendered.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if include_object_ids is None:
        # Full scene render: load entire DMD file directly.
        # This includes room geometry and all objects with correct poses.
        if scene.dmd_path is None:
            raise RuntimeError("No dmd_path available for full scene render")

        builder, plant, scene_graph = create_plant_from_dmd(
            directive_path=scene.dmd_path, scene_dir=scene.scene_dir
        )
        console_logger.info(
            f"Loaded full scene from DMD with {plant.num_model_instances()} models"
        )
    else:
        # Filtered render: build custom plant with only specified objects.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(plant)

        # Enable auto-renaming to handle multiple instances of same SDF.
        # Without this, loading 6 green_apple.sdf files would fail because
        # Drake requires unique model instance names.
        parser.SetAutoRenaming(True)

        # Register package path for package:// URI resolution.
        parser.package_map().Add("scene", str(scene.scene_dir))

        objects = scene.scene_state.get("objects", {})
        objects = {k: v for k, v in objects.items() if k in include_object_ids}

        if not objects:
            console_logger.warning("No valid objects to render")
            return []

        for obj_id, obj_data in objects.items():
            sdf_path_str = obj_data.get("sdf_path")
            if not sdf_path_str:
                console_logger.warning(f"Object {obj_id} has no sdf_path, skipping")
                continue

            try:
                # Resolve path to absolute - parser.AddModels expects file paths.
                sdf_path = None

                if sdf_path_str.startswith("package://scene/"):
                    # Resolve package://scene/... to absolute path.
                    relative_part = sdf_path_str[len("package://scene/") :]
                    sdf_path = str(scene.scene_dir / relative_part)
                elif Path(sdf_path_str).is_absolute():
                    sdf_path = sdf_path_str
                else:
                    # Relative path - resolve from scene_dir.
                    # Note: _normalize_scene_state prefixes sdf_path with room_xxx/
                    # for combined house states, so paths should resolve directly.
                    candidate = scene.scene_dir / sdf_path_str
                    if candidate.exists():
                        sdf_path = str(candidate)

                if sdf_path is None:
                    console_logger.warning(
                        f"SDF not found for {obj_id}: {sdf_path_str}"
                    )
                    continue

                model_instances = parser.AddModels(sdf_path)
                if not model_instances:
                    console_logger.warning(f"No models loaded from {sdf_path_str}")
                    continue
                model_instance = model_instances[0]

                # Weld first body to world at the object's transform.
                transform_data = obj_data.get("transform", {})
                trans = transform_data.get("translation", [0, 0, 0])
                rot_wxyz = transform_data.get("rotation_wxyz", [1, 0, 0, 0])
                transform = RigidTransform(
                    Quaternion(rot_wxyz[0], rot_wxyz[1], rot_wxyz[2], rot_wxyz[3]),
                    trans,
                )
                body_indices = plant.GetBodyIndices(model_instance)
                if body_indices:
                    first_body = plant.get_body(body_indices[0])
                    plant.WeldFrames(
                        plant.world_frame(),
                        first_body.body_frame(),
                        transform,
                    )
            except Exception as e:
                console_logger.warning(f"Failed to load object {obj_id}: {e}")
                continue

        plant.Finalize()
        console_logger.info(
            f"Built filtered plant with {plant.num_model_instances()} models"
        )

    # Step 2: Add camera with RenderEngineGltfClientParams.
    # Placeholder pose - Blender handles actual camera positioning.
    placeholder_pose = RigidTransform.Identity()
    camera_config = CameraConfig(
        X_PB=Transform(placeholder_pose),
        width=4,  # Minimal - Blender handles actual resolution.
        height=4,
        background=Rgba(1.0, 1.0, 1.0, 1.0),
        renderer_class=RenderEngineGltfClientParams(
            base_url=blender_server.get_url(),
            render_endpoint="render_overlay",
        ),
    )

    ApplyCameraConfig(
        config=camera_config,
        builder=builder,
        plant=plant,
        scene_graph=scene_graph,
    )

    builder.ExportOutput(
        builder.GetSubsystemByName(
            f"rgbd_sensor_{camera_config.name}"
        ).color_image_output_port(),
        "rgba_image",
    )

    # Step 3: Build diagram.
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Step 4: Configure Blender BEFORE triggering render.
    # Compute wall normals for hiding camera-facing walls in side views.
    # Only for full scene renders of single rooms (not combined houses).
    # Combined houses have rooms offset from origin, which breaks Blender's
    # position-based wall matching (assumes room centered at origin).
    is_combined_house = scene.scene_state.get("_source") == "combined_house"
    if include_object_ids is None and not is_combined_house:
        wall_normals = _compute_wall_normals_from_scene_state(scene.scene_state)
    else:
        wall_normals = {}

    config_payload = {
        "output_dir": str(output_dir.absolute()),
        "layout": layout,
        "top_view_width": 2048,
        "top_view_height": 2048,
        "side_view_count": side_view_count,
        "side_view_width": 1024,
        "side_view_height": 1024,
        "wall_normals": wall_normals,
        "annotations": {
            "enable_set_of_mark_labels": False,
            "enable_bounding_boxes": False,
            "enable_direction_arrows": False,
            "enable_coordinate_grid": False,
            "enable_partial_walls": bool(wall_normals),  # Hide walls facing camera.
            "enable_support_surface_debug": False,
            "enable_convex_hull_debug": False,
            "show_coordinate_frame": False,
            "rendering_mode": "furniture",
        },
        "scene_objects": [],  # No annotations.
        "taa_samples": 16,
    }

    with acquire_blender_request_lock("render_overlay.robot_eval"):
        response = requests.post(
            f"{blender_server.get_url()}/set_overlay_config",
            json=config_payload,
            timeout=10,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to set overlay config: {response.status_code} "
                f"{response.text}"
            )
        console_logger.info("Overlay config set on Blender server")

        # Step 5: Trigger render by evaluating output port.
        # Drake auto-exports GLTF and POSTs to /render_overlay.
        _ = diagram.GetOutputPort("rgba_image").Eval(context)

        # Step 6: Collect rendered images.
        image_paths = sorted(output_dir.glob("*.png"))
    console_logger.info(f"Rendered {len(image_paths)} validation views")
    return image_paths


def create_vision_tools(
    scene: DMDScene, blender_server: "BlenderServer", render_id: str | None = None
) -> list[FunctionTool]:
    """Create vision tools for agents.

    Creates tools that provide visual observations of the scene by
    rendering via Blender. Renders reflect the current scene state
    after robot manipulation.

    Args:
        scene: DMDScene with scene_state metadata and scene_dir.
        blender_server: BlenderServer for generating renders.
        render_id: Optional unique ID for render output directories.
            Use this when running multiple validations in parallel to
            avoid race conditions. If not provided, renders go to
            scene_dir/validation_renders/. If provided, renders go to
            scene_dir/validation_renders_{render_id}/.

    Returns:
        List of FunctionTool objects for the agent.
    """
    # Build render output base directory with optional unique suffix.
    if render_id is not None:
        render_base = scene.scene_dir / f"validation_renders_{render_id}"
    else:
        render_base = scene.scene_dir / "validation_renders"

    @function_tool
    def observe_scene() -> list[ToolOutputImage]:
        """Take visual snapshots of the entire scene from multiple viewpoints.

        Returns images showing the full scene from top-down, front, and
        perspective views. Use this to get an overall impression of the
        scene and verify task completion visually.

        Returns:
            List of scene observation images.
        """
        console_logger.info("Tool called: observe_scene()")

        output_dir = render_base / "scene"
        render_paths = _render_validation_scene(
            scene=scene,
            blender_server=blender_server,
            output_dir=output_dir,
            include_object_ids=None,  # All objects.
            layout="top_plus_sides",
            side_view_count=4,
        )

        return [
            ToolOutputImage(image_url=_image_path_to_data_url(p)) for p in render_paths
        ]

    @function_tool
    def observe_objects(object_ids: list[str]) -> list[ToolOutputImage]:
        """Take focused visual snapshots of specific objects.

        Renders the specified objects from multiple angles for detailed
        inspection. Use this to closely examine relationships between
        specific objects.

        Args:
            object_ids: List of object IDs to observe.

        Returns:
            List of focused observation images.
        """
        console_logger.info(f"Tool called: observe_objects({object_ids})")

        # Validate object IDs exist in scene.
        valid_ids = [
            oid for oid in object_ids if oid in scene.scene_state.get("objects", {})
        ]
        if not valid_ids:
            console_logger.warning(f"No valid object IDs in {object_ids}")
            return []

        output_dir = render_base / "focused"
        render_paths = _render_validation_scene(
            scene=scene,
            blender_server=blender_server,
            output_dir=output_dir,
            include_object_ids=valid_ids,  # Only these objects.
            layout="top_plus_sides",
            side_view_count=4,
        )

        return [
            ToolOutputImage(image_url=_image_path_to_data_url(p)) for p in render_paths
        ]

    return [observe_scene, observe_objects]
