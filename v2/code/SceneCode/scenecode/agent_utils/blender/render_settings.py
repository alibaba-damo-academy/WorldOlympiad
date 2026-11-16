"""Utilities for configuring Blender rendering settings."""

import logging

import bpy

from scenecode.agent_utils.blender.params import RenderParams

logger = logging.getLogger(__name__)

# Rendering constants.
_UINT16_MAX = 2**16 - 1
BACKGROUND_COLOR_LIGHT_PEACH = (1.0, 0.855, 0.725, 1.0)


def apply_depth_render_settings(min_depth: float, max_depth: float) -> None:
    """Configure depth rendering settings.

    Args:
        min_depth: Minimum depth value for depth mapping.
        max_depth: Maximum depth value for depth mapping.
    """
    # Turn anti-aliasing off.
    bpy.context.scene.render.filter_size = 0

    world_nodes = bpy.data.worlds["World"].node_tree.nodes
    # Set the background.
    world_nodes["Background"].inputs[0].default_value = (
        _UINT16_MAX,
        _UINT16_MAX,
        _UINT16_MAX,
        1,
    )

    # Update the render method to use depth image.
    create_depth_node_layer(min_depth=min_depth, max_depth=max_depth)


def apply_label_render_settings(client_objects: bpy.types.Collection) -> None:
    """Configure label rendering settings.

    Sets up material nodes and colors for label rendering mode,
    distinguishing between glTF imported objects and blend file objects.

    Args:
        client_objects: Collection containing glTF imported objects.
    """
    scene = bpy.context.scene

    # Turn anti-aliasing off.
    scene.render.filter_size = 0

    # Set dither to zero because the 8-bit color image tries to create a
    # better perceived transition in color where there is a limited palette.
    scene.render.dither_intensity = 0

    # Meshes from a blend file and the background will be painted to white.
    background_color = (1.0, 1.0, 1.0, 1.0)
    world_nodes = bpy.data.worlds["World"].node_tree.nodes
    world_nodes["Background"].inputs[0].default_value = background_color

    # Every object imported from the glTF file has been placed in a
    # special collection; simply test for its presence.
    assert client_objects is not None

    def is_from_gltf(object):
        return object.name in client_objects.objects

    # Iterate over all meshes and set their label values.
    for bpy_object in bpy.data.objects:
        assert bpy_object is not None
        # Ensure the object is a mesh.
        if bpy_object.type != "MESH":
            continue

        # If a mesh is imported from a glTF, we will set its label value to
        # its diffuse color. If a mesh is loaded from a blend file, its
        # label value will be set to white (same as the background).
        if is_from_gltf(bpy_object):
            mesh_color = bpy_object.data.materials[0].diffuse_color
        else:
            mesh_color = background_color
        bpy_object.data.materials[0].use_nodes = True
        links = bpy_object.data.materials[0].node_tree.links
        nodes = bpy_object.data.materials[0].node_tree.nodes

        # Clear all material nodes before adding necessary nodes.
        nodes.clear()
        rendered_surface = nodes.new("ShaderNodeOutputMaterial")
        # Use 'ShaderNodeBackground' node as it produces a flat color.
        unlit_flat_mesh_color = nodes.new("ShaderNodeBackground")

        links.new(
            unlit_flat_mesh_color.outputs[0],
            rendered_surface.inputs["Surface"],
        )
        unlit_flat_mesh_color.inputs["Color"].default_value = mesh_color


def create_depth_node_layer(min_depth: float = 0.01, max_depth: float = 10.0) -> None:
    """Create a node layer to render depth images.

    Args:
        min_depth: Minimum depth value in meters (default: 0.01).
        max_depth: Maximum depth value in meters (default: 10.0).

    Raises:
        AssertionError: If max_depth would overflow a UINT16 depth image.
    """
    # Get node and node tree.
    bpy.context.scene.use_nodes = True
    nodes = bpy.data.scenes["Scene"].node_tree.nodes
    links = bpy.data.scenes["Scene"].node_tree.links

    # Clear all nodes before adding necessary nodes.
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    map_value = nodes.new("CompositorNodeMapValue")

    # Convert depth measurements via a MapValueNode.
    assert (
        max_depth * 1000 / _UINT16_MAX <= 1.0
    ), f"Provided max_depth '{max_depth}' overflows an UINT16 depth image"
    map_value.use_min = True
    map_value.use_max = True
    map_value.size = [1000 / _UINT16_MAX]
    map_value.min = [min_depth * 1000 / _UINT16_MAX]
    map_value.max = [1.0]

    # Make links to a depth image.
    bpy.data.scenes["Scene"].view_layers["ViewLayer"].use_pass_z = True
    links.new(render_layers.outputs.get("Depth"), map_value.inputs.get("Value"))
    links.new(map_value.outputs.get("Value"), composite.inputs.get("Image"))


def apply_render_settings(params: RenderParams, view_size: int | None = None) -> None:
    """Apply common render settings.

    Args:
        params: Rendering parameters containing image dimensions and focal lengths.
        view_size: Optional override for image resolution. Use the value of
            params.width and params.height if not provided.
    """
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = view_size or params.width
    scene.render.resolution_y = view_size or params.height

    # Set pixel aspect ratios based on focal lengths.
    if params.focal_x > params.focal_y:
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = params.focal_x / params.focal_y
    else:
        scene.render.pixel_aspect_x = params.focal_y / params.focal_x
        scene.render.pixel_aspect_y = 1.0


def apply_image_type_settings(
    params: RenderParams, client_objects: bpy.types.Collection
) -> None:
    """Apply image type specific settings.

    Args:
        params: Rendering parameters containing image type and depth/color settings.
        client_objects: Collection containing glTF imported objects.
    """
    scene = bpy.context.scene

    if params.image_type == "color":
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
    elif params.image_type == "depth":
        scene.render.image_settings.color_mode = "BW"
        scene.render.image_settings.color_depth = "16"
        scene.display_settings.display_device = "None"
        apply_depth_render_settings(
            min_depth=params.min_depth, max_depth=params.max_depth
        )
    else:  # image_type == "label"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.display_settings.display_device = "None"
        apply_label_render_settings(client_objects=client_objects)


def _setup_world(world_name: str) -> None:
    """Helper to set up world with light peach background.

    Args:
        world_name: Name for the world (e.g., "World", "MetricWorld").
    """
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new(world_name)
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        # Light peach background for better contrast with cyan direction
        # arrows and blue annotations.
        bg_node.inputs[0].default_value = BACKGROUND_COLOR_LIGHT_PEACH


def setup_metric_world() -> None:
    """Set up world for metric rendering.

    Creates a world node setup with light peach background for consistency.
    """
    _setup_world(world_name="MetricWorld")


def setup_regular_world() -> None:
    """Set up world for regular scene rendering.

    Creates a world node setup with light peach background.
    """
    _setup_world(world_name="World")


def setup_cycles_gpu_rendering() -> None:
    """Configure Cycles to use GPU rendering with persistent memory.

    Enables CUDA/OPTIX GPU compute and persistent data to keep BVH/textures
    in GPU memory between renders for faster subsequent renders.
    Falls back gracefully if no GPU is available.
    """
    # Get Cycles preferences.
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        logger.warning("Cycles addon not available, skipping GPU setup")
        return

    # Try OPTIX first (fastest on NVIDIA), fall back to CUDA.
    gpu_enabled = False
    for compute_type in ["OPTIX", "CUDA"]:
        try:
            prefs.compute_device_type = compute_type
            prefs.get_devices()

            # Enable all available GPU devices.
            for device in prefs.devices:
                device.use = device.type != "CPU"

            # Check if any GPU device was enabled.
            if any(d.use and d.type != "CPU" for d in prefs.devices):
                gpu_enabled = True
                logger.info(f"Enabled {compute_type} GPU rendering")
                break
        except Exception:
            continue

    if not gpu_enabled:
        logger.info("No GPU available for Cycles, using CPU")
        return

    # Configure scene to use GPU.
    scene = bpy.context.scene
    scene.cycles.device = "GPU"

    # Enable persistent data (keeps BVH/textures in GPU memory).
    scene.render.use_persistent_data = True
