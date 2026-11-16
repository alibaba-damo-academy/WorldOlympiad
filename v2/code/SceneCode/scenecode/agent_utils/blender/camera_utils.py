"""Utilities for camera configuration and coordinate transformations in Blender."""

import logging
import math

import bpy

from mathutils import Vector

from scenecode.agent_utils.blender.params import RenderParams

logger = logging.getLogger(__name__)


def configure_camera_from_params(params: RenderParams) -> bpy.types.Camera:
    """Configure camera with parameters from render params.

    Args:
        params: Rendering parameters containing camera configuration.

    Returns:
        bpy.types.Camera: The configured Blender camera object.

    Raises:
        RuntimeError: If no camera is found in the scene.
    """
    camera = bpy.data.objects.get("Camera Node")
    if camera is None:
        camera = bpy.data.objects.get("Camera")
        if camera is None:
            raise RuntimeError(
                "No camera node found. Check the input glTF file " f"'{params.scene}'."
            )

    bpy.context.scene.camera = camera
    camera.data.show_sensor = True

    # Set clipping planes.
    clip_start = params.min_depth if params.min_depth else params.near
    clip_end = params.max_depth if params.max_depth else params.far
    camera.data.clip_start = clip_start
    camera.data.clip_end = clip_end

    # Set camera shifts and FOV.
    shift_x = -1.0 * (params.center_x / params.width - 0.5)
    shift_y = (params.center_y - 0.5 * params.height) / params.width
    camera.data.shift_x = shift_x
    camera.data.shift_y = shift_y
    camera.data.lens_unit = "FOV"
    camera.data.angle_y = params.fov_y

    return camera


def configure_metric_camera(params: RenderParams) -> bpy.types.Camera:
    """Configure camera for metric rendering.

    Args:
        params: Rendering parameters containing camera configuration.

    Returns:
        bpy.types.Camera: The newly created and configured Blender camera object.
    """
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new(name="MetricCamera")
    camera_obj = bpy.data.objects.new("MetricCamera", camera_data)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj

    # Configure camera parameters.
    camera_data.type = "PERSP"
    camera_data.lens = 50
    camera_data.sensor_width = 36
    camera_data.clip_start = 0.01
    camera_data.clip_end = params.far
    camera_data.lens_unit = "FOV"
    camera_data.angle_y = params.fov_y

    return camera_obj


def calculate_camera_distance(
    camera_obj: bpy.types.Camera, max_dim: float, margin_scale: float = 1.8
) -> float:
    """Calculate camera distance with margin for coordinate marker placement.

    Args:
        camera_obj: The Blender camera object.
        max_dim: Maximum dimension of the scene bounding box.
        margin_scale: The scale factor for the camera distance.

    Returns:
        float: The calculated camera distance including margin scaling in meters.
    """
    camera_data = camera_obj.data
    fov = 2 * math.atan((camera_data.sensor_width / 2) / camera_data.lens)
    base_distance = (max_dim / 2) / math.tan(fov / 2)
    return base_distance * margin_scale


def look_at_target(obj: bpy.types.Object, target: Vector) -> None:
    """Point object at target using track constraint.

    Args:
        obj: The Blender object to orient (typically a camera).
        target: The target position (Vector) to point the object towards.
    """
    direction = (target - obj.location).normalized()
    quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()


def world_to_camera_view(
    scene: bpy.types.Scene, camera: bpy.types.Camera, coord: Vector
) -> Vector:
    """Convert world coordinates to camera view coordinates.

    Args:
        scene: Blender scene object.
        camera: Blender camera object.
        coord: World coordinate position of format (x, y, z).

    Returns:
        Vector: Camera view coordinates (x, y, z) in normalized space.
    """
    co_local = camera.matrix_world.normalized().inverted() @ coord
    z = -co_local.z

    camera_data = camera.data
    frame = [-v for v in camera_data.view_frame(scene=scene)[:3]]
    if camera_data.type != "ORTHO":
        frame = [(v / (v.z / z)) for v in frame]

    min_x, max_x = frame[1].x, frame[2].x
    min_y, max_y = frame[0].y, frame[1].y
    x = (co_local.x - min_x) / (max_x - min_x)
    y = (co_local.y - min_y) / (max_y - min_y)
    return Vector((x, y, z))


def get_pixel_coordinates(
    scene: bpy.types.Scene,
    camera: bpy.types.Camera,
    world_coord: Vector | list | tuple,
) -> tuple[int, int]:
    """Get pixel coordinates for a given world coordinate.

    Args:
        scene: Blender scene object.
        camera: Blender camera object.
        world_coord: World coordinate position (Vector, list, or tuple) of format
            (x, y, z).

    Returns:
        tuple: Pixel coordinates (x, y) in screen space.
    """
    if isinstance(world_coord, (list, tuple)):
        world_coord = Vector(world_coord)
    coord_2d = world_to_camera_view(scene=scene, camera=camera, coord=world_coord)

    # Convert normalized coordinates (0-1) to actual pixel coordinates.
    render = scene.render
    pixel_x = coord_2d.x * render.resolution_x
    pixel_y = (1 - coord_2d.y) * render.resolution_y  # Flip Y for screen coordinates.
    return (pixel_x, pixel_y)
