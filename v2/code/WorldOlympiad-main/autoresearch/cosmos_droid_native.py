"""Shared native DROID cameras and contact helpers. Import after Isaac AppLauncher."""
from pathlib import Path
import cv2
import numpy as np
import torch
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.sim import SimulationContext
from pxr import Usd, UsdPhysics
from robolab.robots.droid import DroidCfg
from .robotics_views import compose_cosmos_droid_view
ACTION_HZ = 15

def _contact_active(sensor: ContactSensor) -> bool:
    forces = sensor.data.force_matrix_w
    return bool(forces is not None and torch.linalg.vector_norm(forces).item() > 1e-4)


def _collision_filter_paths(stage, root_path: str) -> list[str]:
    """Resolve every concrete collision Shape below a WorldIR body.

    Static compiled bodies such as the tray are Xforms whose collision API is
    applied to descendant Shapes.  Binding an Isaac Lab filtered contact
    sensor to the Xform root produces an all-zero force matrix even while the
    rigid object is physically supported by those Shapes.
    """

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"missing contact-filter root: {root_path}")
    paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not paths:
        raise RuntimeError(f"contact-filter root has no collision Shapes: {root_path}")
    return paths


def _quat_multiply_wxyz(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate_wxyz(
    rotation: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    conjugate = (rotation[0], -rotation[1], -rotation[2], -rotation[3])
    rotated = _quat_multiply_wxyz(
        _quat_multiply_wxyz(rotation, (0.0, *vector)),
        conjugate,
    )
    return rotated[1:]


def _exterior_camera(
    cfg,
    name: str,
    robot_position: tuple[float, float, float],
    robot_rotation_wxyz: tuple[float, float, float, float],
    world_override: dict | None = None,
) -> TiledCamera:
    """Attach the released RoboLab DROID rig rigidly to the WorldIR robot root."""

    result = cfg.replace(prim_path=f"/World/{name}")
    if world_override is not None:
        values = [*world_override["position_world"], *world_override["quaternion_opengl_wxyz"]]
        if len(values) != 7 or not np.isfinite(values).all():
            raise ValueError("invalid calibrated exterior camera")
        if abs(np.linalg.norm(world_override["quaternion_opengl_wxyz"])-1) > 1e-6:
            raise ValueError("camera quaternion must be normalized")
        actual_fov = np.degrees(2*np.arctan(float(result.spawn.vertical_aperture)/(2*float(result.spawn.focal_length))))
        if abs(actual_fov-world_override["fov_y_deg"])>1e-7:
            raise ValueError("calibrated rig changes native DROID intrinsics")
        if (world_override["width"],world_override["height"])!=(320,180):
            raise ValueError("calibrated rig changes native exterior dimensions")
        result.offset.pos = tuple(world_override["position_world"])
        result.offset.rot = tuple(world_override["quaternion_opengl_wxyz"])
        return TiledCamera(result)
    local_position = tuple(float(value) for value in result.offset.pos)
    local_rotation = tuple(float(value) for value in result.offset.rot)
    rotated_position = _rotate_wxyz(robot_rotation_wxyz, local_position)
    result.offset.pos = tuple(
        origin + delta for origin, delta in zip(robot_position, rotated_position)
    )
    result.offset.rot = _quat_multiply_wxyz(robot_rotation_wxyz, local_rotation)
    return TiledCamera(result)


def _wrist_camera(wrist_mount=None) -> TiledCamera:
    result = DroidCfg().wrist_cam.replace(
        prim_path="/World/Robot/Gripper/Robotiq_2F_85/base_link/wrist_cam"
    )
    optical_delta_opengl_wxyz = (wrist_mount or {}).get("optical_delta_opengl_wxyz")
    if optical_delta_opengl_wxyz is not None:
        values = np.asarray(optical_delta_opengl_wxyz, dtype=float)
        if (values.shape != (4,) or not np.isfinite(values).all()
                or abs(float(np.linalg.norm(values)) - 1.0) > 1e-5):
            raise ValueError("wrist optical delta must be a unit WXYZ quaternion")
        result.offset.rot = _quat_multiply_wxyz(
            tuple(result.offset.rot), tuple(float(value) for value in values))
    translation = (wrist_mount or {}).get("translation_local_xyz")
    if translation is not None:
        values = np.asarray(translation, dtype=float)
        if (values.shape != (3,) or not np.isfinite(values).all()
                or float(np.linalg.norm(values)) > 0.20):
            raise ValueError("wrist camera translation exceeds its bounded mount")
        result.offset.pos = tuple(np.asarray(result.offset.pos, dtype=float) + values)
    return TiledCamera(result)


def _camera_bgr(camera: TiledCamera) -> np.ndarray:
    rgb = camera.data.output["rgb"][0]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    image = np.asarray(rgb, dtype=np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _render_observation(
    *, frame_index: int, window_root: Path, observation_root: Path,
    sim: SimulationContext, wrist: TiledCamera, left: TiledCamera, right: TiledCamera,
) -> Path:
    # Render the current PhysX/Fabric state without advancing simulation time.
    for _ in range(3):
        sim.render()
    for camera in (wrist, left, right):
        camera.update(1.0 / ACTION_HZ, force_recompute=True)
    observation_root.mkdir(parents=True, exist_ok=True)
    images = {
        "wrist": _camera_bgr(wrist),
        "exterior_left": _camera_bgr(left),
        "exterior_right": _camera_bgr(right),
    }
    for view, short in (("wrist", "wrist"), ("exterior_left", "left"), ("exterior_right", "right")):
        image = images[view]
        if image is None or image.shape[:2] != (720, 1280):
            raise RuntimeError(f"bad native Isaac conditioning frame: {view}")
        destination = observation_root / f"step_{frame_index:06d}_{short}_image.png"
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"failed to save native Isaac conditioning frame: {destination}")
    concat = compose_cosmos_droid_view(images["wrist"], images["exterior_left"], images["exterior_right"])
    output = window_root / "conditioning_droid_concat.png"
    if concat.shape != (540, 640, 3) or not cv2.imwrite(str(output), concat):
        raise RuntimeError("failed to produce DROID concat observation")
    return output
