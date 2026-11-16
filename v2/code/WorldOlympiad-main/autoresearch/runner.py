"""PyBullet event execution, deterministic probes, EGL rendering, and recording."""

from __future__ import annotations

import ctypes
import json
import math
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json
from .models import (
    EpisodeManifest,
    EpisodeSpec,
    ObservationCertificate,
    PhysicsCertificate,
    PhysicsReport,
    ProbeResult,
    RepairRecord,
    WorldIR,
)


@dataclass
class SimulationResult:
    timestamps: np.ndarray
    body_names: list[str]
    positions: np.ndarray
    orientations: np.ndarray
    linear_velocities: np.ndarray
    angular_velocities: np.ndarray
    target_name: str
    target_instance_id: int
    body_instance_map: dict[str, int]
    contacts: list[list[dict[str, Any]]]
    pre_settle_positions: np.ndarray
    pre_settle_orientations: np.ndarray
    max_penetration_m: float
    penetration_pairs: list[dict[str, Any]] = field(default_factory=list)
    min_actor_aabb_z_m: float = math.inf
    rgb_frames: list[np.ndarray] = field(default_factory=list)
    depth_frames: list[np.ndarray] = field(default_factory=list)
    segmentation_frames: list[np.ndarray] = field(default_factory=list)
    camera: dict[str, Any] = field(default_factory=dict)
    renderer: str = "none"
    robot_diagnostics: dict[str, Any] = field(default_factory=dict)
    robot_joint_names: list[str] = field(default_factory=list)
    robot_joint_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    robot_joint_velocities: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    robot_end_effector_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    robot_end_effector_orientations: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float64)
    )
    link_names: list[str] = field(default_factory=list)
    link_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.float64)
    )
    link_orientations: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 4), dtype=np.float64)
    )

    @property
    def target_index(self) -> int:
        return self.body_names.index(self.target_name)


def _import_pybullet():
    try:
        import pybullet as bullet
    except ImportError as exc:
        raise RuntimeError("pybullet==3.2.7 is required for simulation") from exc
    return bullet


def _query_egl_vendor() -> str:
    """Query an initialized EGL device instead of inferring from nvidia-smi."""

    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    query_address = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    display_address = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
    if not query_address or not display_address:
        raise RuntimeError("EGL device-platform extensions are unavailable")
    query_devices = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(query_address)
    get_display = ctypes.CFUNCTYPE(
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )(display_address)
    devices = (ctypes.c_void_p * 32)()
    count = ctypes.c_int()
    if not query_devices(len(devices), devices, ctypes.byref(count)):
        raise RuntimeError("eglQueryDevicesEXT failed")
    egl.eglInitialize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglTerminate.argtypes = [ctypes.c_void_p]
    for index in range(count.value):
        display = get_display(0x313F, devices[index], None)  # EGL_PLATFORM_DEVICE_EXT
        major, minor = ctypes.c_int(), ctypes.c_int()
        if not display or not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            continue
        value = egl.eglQueryString(display, 0x3053)  # EGL_VENDOR
        egl.eglTerminate(display)
        if value:
            return value.decode("utf-8", errors="replace")
    raise RuntimeError("no EGL device could be initialized for vendor verification")


def _load_egl(bullet: Any) -> tuple[int, str]:
    module_dir = Path(bullet.__file__).resolve().parent
    candidates = sorted(module_dir.glob("eglRenderer*.so"))
    if not candidates:
        raise RuntimeError(f"no EGL renderer plugin found beside {bullet.__file__}")
    plugin_id = bullet.loadPlugin(str(candidates[0]), "_eglRendererPlugin")
    if plugin_id < 0:
        raise RuntimeError(f"failed to load EGL plugin: {candidates[0]}")
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        raise RuntimeError("EGL mode requires a working NVIDIA driver and nvidia-smi") from exc
    if "NVIDIA" not in query.stdout and "RTX" not in query.stdout:
        raise RuntimeError(f"unexpected GPU inventory: {query.stdout.strip()}")
    vendor = _query_egl_vendor()
    if "NVIDIA" not in vendor.upper():
        bullet.unloadPlugin(plugin_id)
        raise RuntimeError(f"EGL initialized with a non-NVIDIA vendor: {vendor}")
    return plugin_id, vendor


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def _whole_body_aabb(bullet: Any, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a world AABB that includes the base and every articulated link."""

    bounds = [bullet.getAABB(body_id, link) for link in range(-1, bullet.getNumJoints(body_id))]
    lower = np.min(np.asarray([item[0] for item in bounds], dtype=np.float64), axis=0)
    upper = np.max(np.asarray([item[1] for item in bounds], dtype=np.float64), axis=0)
    return lower, upper


def _camera_for_episode(
    bullet: Any,
    target_id: int,
    support_id: int,
    spec: EpisodeSpec,
    *,
    orbit_index: int = 0,
    robot_id: int | None = None,
) -> dict[str, Any]:
    target_min, target_max = [np.asarray(item, dtype=np.float64) for item in bullet.getAABB(target_id)]
    support_min, support_max = [np.asarray(item, dtype=np.float64) for item in bullet.getAABB(support_id)]
    target_center = (target_min + target_max) / 2
    support_center = (support_min + support_max) / 2
    if spec.event_type in {"block_push", "stack_collapse"}:
        camera_vector = np.asarray(spec.trigger.vector, dtype=np.float64)
        # Counterfactual vectors are serialized explicitly for auditability,
        # but camera is a controlled factor and must remain identical to the
        # reference. Recover the reference direction for the opposite control.
        if spec.variant == "opposite_direction":
            camera_vector *= -1
        direction = _normalize(camera_vector)
        # Watch from beyond the departure edge so the table cannot occlude the
        # falling object. Include both the support surface and expected landing.
        edge = support_center.copy()
        dominant_axis = int(np.argmax(np.abs(direction[:2])))
        edge[dominant_axis] = (
            support_max[dominant_axis]
            if direction[dominant_axis] >= 0
            else support_min[dominant_axis]
        )
        landing_focus = support_center.copy()
        landing_focus[:2] = edge[:2] + direction[:2] * 0.28
        landing_focus[2] = 0.48
        focus = landing_focus.copy()
        perpendicular = np.asarray((-direction[1], direction[0], 0.0))
        # Keep the camera inside the generated room walls while retaining a
        # high, outbound-side view of both the tabletop and floor landing zone.
        departure_offsets = (
            direction * 0.72 + perpendicular * 1.35 + np.asarray((0.0, 0.0, 1.80)),
            direction * 0.72 - perpendicular * 1.35 + np.asarray((0.0, 0.0, 1.80)),
            direction * 1.10 + np.asarray((0.0, 0.0, 1.80)),
            perpendicular * 1.10 + np.asarray((0.0, 0.0, 1.80)),
            -perpendicular * 1.10 + np.asarray((0.0, 0.0, 1.80)),
            -direction * 0.72 + perpendicular * 1.35 + np.asarray((0.0, 0.0, 1.80)),
            -direction * 0.72 - perpendicular * 1.35 + np.asarray((0.0, 0.0, 1.80)),
            -direction * 1.10 + np.asarray((0.0, 0.0, 1.80)),
            -direction * 0.55 + perpendicular * 0.55 + np.asarray((0.0, 0.0, 2.05)),
        )
        if not 0 <= orbit_index < len(departure_offsets):
            raise ValueError(f"invalid deterministic departure camera index: {orbit_index}")
        camera_offset = departure_offsets[orbit_index]
        # The rectangular slider needs more image-space resolution for the
        # reference/imagination Dream2Flow trajectory audit while retaining the departure edge and
        # landing zone. Its certified bbox remains inside the 3%-20% contract.
        if spec.event_type == "block_push":
            camera_offset *= 0.65
        elif spec.event_type == "stack_collapse":
            camera_offset = camera_offset.copy()
            camera_offset[:2] *= 0.45
            camera_offset[2] = 1.05
        eye = focus + camera_offset
    elif spec.event_type == "cup_tip":
        camera_vector = np.asarray(spec.trigger.vector, dtype=np.float64)
        if spec.variant == "opposite_direction":
            camera_vector *= -1
        direction = _normalize(camera_vector)
        perpendicular = np.asarray((-direction[1], direction[0], 0.0))
        # A valid tip may continue into a table departure.  Frame both the
        # tabletop pose and a conservative floor landing envelope instead of
        # centering only on the cup's pre-trigger AABB.
        focus = target_center + direction * 0.18
        focus[2] = max(0.35, float(target_center[2]) - 0.35)
        cup_offsets = (
            direction * 0.55 + perpendicular * 0.75 + np.asarray((0.0, 0.0, 1.80)),
            direction * 0.55 - perpendicular * 0.75 + np.asarray((0.0, 0.0, 1.80)),
            direction * 0.65 + np.asarray((0.0, 0.0, 1.90)),
            perpendicular * 0.70 + np.asarray((0.0, 0.0, 1.80)),
            -perpendicular * 0.70 + np.asarray((0.0, 0.0, 1.80)),
        )
        if not 0 <= orbit_index < len(cup_offsets):
            raise ValueError(f"invalid deterministic cup camera index: {orbit_index}")
        eye = focus + cup_offsets[orbit_index]
    else:
        focus = 0.8 * target_center + 0.2 * support_center
        # Generated rooms have arbitrary wall/table orientations.  A single
        # hard-coded azimuth can consequently place the camera behind a wall.
        # Keep an ordered, deterministic orbit so the renderer can audit the
        # first frame and select the first unobstructed SimReady view.
        orbit_offsets = (
            (1.24, 1.16, 0.91),
            (0.72, 0.72, 0.62),
            (-0.72, 0.72, 0.62),
            (-0.72, -0.72, 0.62),
            (0.72, -0.72, 0.62),
            (0.92, 0.00, 0.72),
            (0.00, 0.92, 0.72),
            (-0.92, 0.00, 0.72),
            (0.00, -0.92, 0.72),
        )
        if not 0 <= orbit_index < len(orbit_offsets):
            raise ValueError(f"invalid deterministic camera orbit index: {orbit_index}")
        camera_offset = np.asarray(orbit_offsets[orbit_index], dtype=np.float64)
        eye = focus + camera_offset
    if robot_id is not None:
        # Formal robotics cases still expose exactly one camera.  Expand that
        # camera deterministically around the robot, worktable, target and the
        # event-specific landing focus so the actor is not cropped while the
        # small CAE target remains measurable.
        robot_min, robot_max = _whole_body_aabb(bullet, robot_id)
        if spec.event_type in {"block_sort", "block_push"}:
            # Follow SceneCode's renderer framing principle: fit the combined
            # geometry bounds, look at their center, and retain an image-space
            # margin.  Omitting the support bounds made the camera technically
            # pass target-mask checks while cropping the robot and letting the
            # tabletop dominate the presentation render.
            direction = _normalize(np.asarray(spec.trigger.vector, dtype=np.float64))
            destination_hint = target_center + direction * (
                0.32 if spec.event_type == "block_sort" else 0.30
            )
            task_margin = np.asarray((0.12, 0.12, 0.10))
            frame_min = np.minimum.reduce(
                [target_min - task_margin, support_min, robot_min, destination_hint]
            )
            frame_max = np.maximum.reduce(
                [target_max + task_margin, support_max, robot_max, destination_hint]
            )
        else:
            frame_min = np.minimum.reduce([target_min, support_min, robot_min, focus])
            frame_max = np.maximum.reduce([target_max, support_max, robot_max, focus])
        previous_focus = focus.copy()
        previous_eye = eye.copy()
        focus = (frame_min + frame_max) * 0.5
        original_view = previous_eye - previous_focus
        horizontal_view = original_view.copy()
        horizontal_view[2] = 0.0
        # A presentation-quality robotics view should read as an oblique
        # workspace view, not a ceiling camera.  Keep the selected orbit's
        # azimuth but cap elevation to roughly 29 degrees.
        elevation = (
            0.90 if spec.event_type == "block_sort"
            else (0.72 if spec.event_type == "block_push" else 0.55)
        )
        view_direction = _normalize(
            _normalize(horizontal_view) + np.asarray((0.0, 0.0, elevation))
        )
        half_extent = (frame_max - frame_min) * 0.5
        vertical_extent = max(float(half_extent[2]), float(half_extent[0]), float(half_extent[1]))
        fit_distance = vertical_extent / math.tan(math.radians(spec.camera.fov_y_deg) / 2)
        if spec.event_type in {"block_sort", "block_push"}:
            # SceneCode's Blender renderer uses 1/0.8 so the fitted geometry
            # occupies roughly 80% of the image. Keep the same audited margin.
            distance = max(
                0.85 if spec.event_type == "block_sort" else 0.95,
                fit_distance * (1.0 / 0.8),
            )
        else:
            distance = max(
                float(np.linalg.norm(previous_eye - previous_focus)),
                fit_distance * 1.35,
            )
        eye = focus + view_direction * distance
    forward = _normalize(focus - eye)
    right = _normalize(np.cross(forward, np.asarray((0.0, 0.0, 1.0))))
    up = _normalize(np.cross(right, forward))
    down = -up
    width, height = spec.camera.width, spec.camera.height
    fy = height / (2 * math.tan(math.radians(spec.camera.fov_y_deg) / 2))
    fx = fy
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return {
        "eye": eye.tolist(),
        "look_at": focus.tolist(),
        "up": up.tolist(),
        "intrinsics": [[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]],
        "camera_to_world_opencv": c2w.tolist(),
        "width": width,
        "height": height,
        "fov_y_deg": spec.camera.fov_y_deg,
        "near_m": spec.camera.near_m,
        "far_m": spec.camera.far_m,
        "deterministic_orbit_index": orbit_index,
    }


def _select_camera_for_episode(
    bullet: Any,
    target_id: int,
    support_id: int,
    spec: EpisodeSpec,
    renderer_id: int,
    robot_id: int | None = None,
) -> dict[str, Any]:
    """Select an unobstructed deterministic view from exact simulator masks."""

    candidate_count = 5 if spec.event_type == "cup_tip" else 9
    candidates: list[
        tuple[tuple[int, int, int, float, int, float], dict[str, Any]]
    ] = []
    reference_direction = np.asarray(spec.trigger.vector, dtype=np.float64)
    if spec.variant == "opposite_direction":
        reference_direction *= -1
    reference_direction = _normalize(reference_direction)
    for orbit_index in range(candidate_count):
        camera = _camera_for_episode(
            bullet,
            target_id,
            support_id,
            spec,
            orbit_index=orbit_index,
            robot_id=robot_id,
        )
        _, _, segmentation = _render(bullet, camera, renderer_id)
        pixels = int(np.count_nonzero(segmentation == target_id))
        robot_pixels = (
            int(np.count_nonzero(segmentation == robot_id)) if robot_id is not None else 0
        )
        fraction = target_bbox_fraction(segmentation, target_id)
        conforms = int(
            pixels >= 200
            and spec.camera.target_fraction_min
            <= fraction
            <= spec.camera.target_fraction_max
        )
        # Prefer a protocol-conforming view, then one near the middle of the
        # allowed image-size band, then greater exact visible area.  The orbit
        # order provides a stable final tie-break.
        midpoint = (spec.camera.target_fraction_min + spec.camera.target_fraction_max) / 2
        eye = np.asarray(camera["eye"], dtype=np.float64)
        look_at = np.asarray(camera["look_at"], dtype=np.float64)
        motion_side_alignment = (
            float(np.dot(_normalize(eye - look_at), reference_direction))
            if spec.event_type == "cup_tip"
            else 0.0
        )
        score = (
            conforms,
            int(robot_id is None or robot_pixels >= 1000),
            int(pixels >= 200),
            int(spec.camera.target_fraction_min <= fraction <= spec.camera.target_fraction_max),
            motion_side_alignment,
            pixels,
            -abs(fraction - midpoint),
        )
        camera["selection_first_frame_pixels"] = pixels
        camera["selection_first_frame_bbox_fraction"] = fraction
        camera["selection_robot_first_frame_pixels"] = robot_pixels
        camera["selection_motion_side_alignment"] = motion_side_alignment
        candidates.append((score, camera))
    _, selected = max(candidates, key=lambda item: item[0])
    selected["selection_candidate_count"] = candidate_count
    selected["selection_policy"] = "exact_instance_mask_deterministic_orbit_v1"
    selected["selection_candidate_audit"] = [
        {
            "orbit_index": camera["deterministic_orbit_index"],
            "first_frame_pixels": camera["selection_first_frame_pixels"],
            "first_frame_bbox_fraction": camera["selection_first_frame_bbox_fraction"],
            "robot_first_frame_pixels": camera["selection_robot_first_frame_pixels"],
            "motion_side_alignment": camera["selection_motion_side_alignment"],
        }
        for _, camera in candidates
    ]
    return selected


def _render(bullet: Any, camera: dict[str, Any], renderer_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = bullet.computeViewMatrix(camera["eye"], camera["look_at"], camera["up"])
    projection = bullet.computeProjectionMatrixFOV(
        fov=camera["fov_y_deg"],
        aspect=camera["width"] / camera["height"],
        nearVal=camera["near_m"],
        farVal=camera["far_m"],
    )
    _, _, rgba, depth_buffer, segmentation = bullet.getCameraImage(
        camera["width"],
        camera["height"],
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=renderer_id,
        flags=bullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        shadow=1,
        lightDirection=[-0.4, -0.5, -1.0],
        lightColor=[1.0, 0.98, 0.94],
        lightDistance=4.0,
    )
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(camera["height"], camera["width"], 4)[..., :3]
    buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(camera["height"], camera["width"])
    near, far = camera["near_m"], camera["far_m"]
    depth = (far * near / (far - (far - near) * buffer)).astype(np.float32)
    segmentation = np.asarray(segmentation, dtype=np.int32).reshape(camera["height"], camera["width"])
    instance = np.where(segmentation < 0, -1, segmentation & ((1 << 24) - 1)).astype(np.int32)
    return rgb, depth, instance


def validate_depth_projection(renderer: str = "egl") -> dict[str, float | bool | str]:
    """Measure metric optical-axis depth on an analytic fronto-parallel box."""

    bullet = _import_pybullet()
    client = bullet.connect(bullet.DIRECT)
    plugin_id = -1
    egl_vendor: str | None = None
    try:
        if renderer == "egl":
            plugin_id, egl_vendor = _load_egl(bullet)
            renderer_id = bullet.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer_id = bullet.ER_TINY_RENDERER
        half_thickness = 0.005
        collision = bullet.createCollisionShape(bullet.GEOM_BOX, halfExtents=[0.25, 0.25, half_thickness])
        visual = bullet.createVisualShape(bullet.GEOM_BOX, halfExtents=[0.25, 0.25, half_thickness], rgbaColor=[0.8, 0.2, 0.2, 1.0])
        body_id = bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=[0, 0, 0])
        camera = {
            "eye": [0.0, 0.0, 1.0],
            "look_at": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "width": 160,
            "height": 120,
            "fov_y_deg": 50.0,
            "near_m": 0.02,
            "far_m": 20.0,
        }
        _, depth, segmentation = _render(bullet, camera, renderer_id)
        measured = float(depth[camera["height"] // 2, camera["width"] // 2])
        expected = 1.0 - half_thickness
        error = abs(measured - expected)
        segmented_id = int(segmentation[camera["height"] // 2, camera["width"] // 2])
        return {
            "renderer": renderer,
            "egl_vendor": egl_vendor,
            "expected_depth_m": expected,
            "measured_depth_m": measured,
            "absolute_error_m": error,
            "segmentation_matches": segmented_id == body_id,
            "passed": error <= 0.001 and segmented_id == body_id,
        }
    finally:
        if plugin_id >= 0:
            bullet.unloadPlugin(plugin_id)
        bullet.disconnect(client)


def _contact_rows(bullet: Any, id_to_name: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in bullet.getContactPoints():
        body_a, body_b = int(point[1]), int(point[2])
        rows.append(
            {
                "body_a": id_to_name.get(body_a, str(body_a)),
                "body_b": id_to_name.get(body_b, str(body_b)),
                "link_a": int(point[3]),
                "link_b": int(point[4]),
                "position_a": list(point[5]),
                "position_b": list(point[6]),
                "normal_on_b": list(point[7]),
                "distance_m": float(point[8]),
                "normal_force_n": float(point[9]),
            }
        )
    return rows


def _base_link_pose(
    bullet: Any, body_id: int
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert PyBullet's base COM/inertial pose to the URDF link pose."""

    inertial_world_position, inertial_world_orientation = (
        bullet.getBasePositionAndOrientation(body_id)
    )
    dynamics = bullet.getDynamicsInfo(body_id, -1)
    local_inertial_position, local_inertial_orientation = dynamics[3], dynamics[4]
    inverse_position, inverse_orientation = bullet.invertTransform(
        local_inertial_position, local_inertial_orientation
    )
    return bullet.multiplyTransforms(
        inertial_world_position,
        inertial_world_orientation,
        inverse_position,
        inverse_orientation,
    )


def _apply_trigger(bullet: Any, body_id: int, spec: EpisodeSpec, multiplier: float) -> None:
    vector = np.asarray(spec.trigger.vector, dtype=np.float64) * multiplier
    if spec.trigger.operation == "set_velocity":
        bullet.resetBaseVelocity(body_id, linearVelocity=vector.tolist())
        return
    if spec.trigger.operation == "apply_force":
        position, orientation = _base_link_pose(bullet, body_id)
        local = spec.trigger.position_local or (0.0, 0.0, 0.0)
        world_point, _ = bullet.multiplyTransforms(position, orientation, local, (0.0, 0.0, 0.0, 1.0))
        bullet.applyExternalForce(body_id, -1, vector.tolist(), world_point, bullet.WORLD_FRAME)
        return
    if spec.trigger.operation == "apply_torque":
        bullet.applyExternalTorque(body_id, -1, vector.tolist(), bullet.WORLD_FRAME)
        return
    raise NotImplementedError("remove_support is reserved by EpisodeSpec but not used in pilot.v1")


def simulate(
    bundle_path: str | Path,
    spec: EpisodeSpec,
    *,
    trigger_multiplier: float,
    render: bool = False,
    renderer: str = "egl",
    camera_orbit_index: int | None = None,
) -> SimulationResult:
    bullet = _import_pybullet()
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    client = bullet.connect(bullet.DIRECT)
    plugin_id = -1
    egl_vendor: str | None = None
    try:
        if render and renderer == "egl":
            plugin_id, egl_vendor = _load_egl(bullet)
            renderer_id = bullet.ER_BULLET_HARDWARE_OPENGL
        elif render and renderer == "tiny":
            renderer_id = bullet.ER_TINY_RENDERER
        else:
            renderer_id = bullet.ER_TINY_RENDERER
        bullet.resetSimulation()
        bullet.setGravity(0, 0, -9.81)
        bullet.setTimeStep(spec.physics_dt_s)
        bullet.setPhysicsEngineParameter(
            fixedTimeStep=spec.physics_dt_s,
            numSolverIterations=100,
            deterministicOverlappingPairs=1,
            enableConeFriction=1,
        )
        body_ids: dict[str, int] = {}
        dynamic_actors = set(spec.actors)
        for name, item in bundle["bodies"].items():
            pose = item["pose"]
            dynamic = name in dynamic_actors and not item.get("metadata", {}).get("fixed_base", False)
            body_id = bullet.loadURDF(
                item["urdf_path"],
                basePosition=pose["xyz"],
                baseOrientation=pose["quat_xyzw"],
                useFixedBase=not dynamic,
                flags=bullet.URDF_USE_INERTIA_FROM_FILE,
            )
            if body_id < 0:
                raise RuntimeError(f"PyBullet failed to load {name}: {item['urdf_path']}")
            body_ids[name] = body_id
            material = item["material"]
            for link_index in range(-1, bullet.getNumJoints(body_id)):
                bullet.changeDynamics(
                    body_id,
                    link_index,
                    lateralFriction=material["lateral_friction"],
                    rollingFriction=material["rolling_friction"],
                    restitution=material["restitution"],
                    linearDamping=material["linear_damping"],
                    angularDamping=material["angular_damping"],
                )
            for joint_index in range(bullet.getNumJoints(body_id)):
                bullet.setJointMotorControl2(body_id, joint_index, bullet.VELOCITY_CONTROL, force=0)
        if spec.target not in body_ids or spec.support not in body_ids:
            raise KeyError(f"episode refers to missing target/support: {spec.target}/{spec.support}")
        if any(name not in body_ids for name in spec.actors):
            raise KeyError(f"episode refers to missing actors: {spec.actors}")

        names = list(body_ids)
        id_to_name = {value: key for key, value in body_ids.items()}
        link_records: list[tuple[str, int, int]] = []
        for body_name in names:
            body_id = body_ids[body_name]
            base_name = bullet.getBodyInfo(body_id)[0].decode("utf-8")
            link_records.append((f"{body_name}::{base_name}", body_id, -1))
            for joint_index in range(bullet.getNumJoints(body_id)):
                link_name = bullet.getJointInfo(body_id, joint_index)[12].decode("utf-8")
                link_records.append((f"{body_name}::{link_name}", body_id, joint_index))
        robot_state = None
        if spec.robot_program is not None:
            program = spec.robot_program
            continuous_push = (
                program.operation == "cartesian_push"
                and program.control_mode == "continuous_kinematic_joint_position"
            )
            if program.robot not in body_ids:
                raise KeyError(f"robot program refers to missing body: {program.robot}")
            robot_id = body_ids[program.robot]
            # The benchmark robot is a prescribed motion actor, not another
            # object under test. Make every articulated link kinematic so
            # contact impulses move the task object without kicking the arm
            # into an uncontrolled joint branch between 240 Hz commands.
            if continuous_push:
                for link_index in range(-1, bullet.getNumJoints(robot_id)):
                    bullet.changeDynamics(robot_id, link_index, mass=0.0)
            joint_by_name = {
                bullet.getJointInfo(robot_id, index)[1].decode("utf-8"): index
                for index in range(bullet.getNumJoints(robot_id))
            }
            link_by_name = {
                bullet.getJointInfo(robot_id, index)[12].decode("utf-8"): index
                for index in range(bullet.getNumJoints(robot_id))
            }
            missing_joints = set(program.arm_joint_names) - set(joint_by_name)
            if missing_joints or program.end_effector_link_name not in link_by_name:
                raise KeyError(
                    f"robot program topology mismatch: joints={sorted(missing_joints)}, "
                    f"end_effector={program.end_effector_link_name!r}"
                )
            contact_object = program.contact_object or spec.target
            if contact_object not in body_ids:
                raise KeyError(f"robot contact object is missing: {contact_object}")
            target_id_for_robot = body_ids[contact_object]
            lower, upper = bullet.getAABB(target_id_for_robot)
            lower = np.asarray(lower, dtype=np.float64)
            upper = np.asarray(upper, dtype=np.float64)
            center = (lower + upper) * 0.5
            vector = np.asarray(spec.trigger.vector, dtype=np.float64)
            direction = vector.copy()
            direction[2] = 0.0
            destination_id = None
            destination_center = None
            destination_top = None
            if program.destination_object:
                if program.destination_object not in body_ids:
                    raise KeyError(
                        f"robot destination object is missing: {program.destination_object}"
                    )
                destination_id = body_ids[program.destination_object]
                destination_lower, destination_upper = bullet.getAABB(destination_id)
                destination_lower = np.asarray(destination_lower, dtype=np.float64)
                destination_upper = np.asarray(destination_upper, dtype=np.float64)
                destination_center = (destination_lower + destination_upper) * 0.5
                destination_top = float(destination_upper[2])
                if program.operation == "cartesian_pick_place":
                    direction = destination_center - center
                    direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                raise ValueError("cartesian robot motion requires a horizontal direction")
            direction /= norm
            half_extent_along_push = float(np.dot(np.abs(direction), (upper - lower) * 0.5))
            contact = center.copy()
            if program.operation == "cartesian_pick_place":
                # The released Panda carries a compact, visible suction-style
                # tool.  Its centre reaches the object's top before the
                # deterministic fixed grasp is created.
                contact[2] = upper[2] + 0.018
            else:
                contact[2] = lower[2] + program.contact_height_fraction * (upper[2] - lower[2])
            support_top = float(bullet.getAABB(body_ids[spec.support])[1][2])
            robot_base = contact - direction * program.base_distance_from_contact_m
            # The Panda URDF base geometry extends about 5 cm below its base
            # frame.  Mount the frame on a deterministic 5.5 cm pedestal so
            # the robot/table interface touches without initial penetration.
            robot_base[2] = support_top + 0.055
            bullet.resetBasePositionAndOrientation(
                robot_id,
                robot_base.tolist(),
                (0.0, 0.0, 0.0, 1.0),
            )
            if program.operation == "cartesian_pick_place":
                if destination_center is None or destination_top is None:
                    raise RuntimeError("pick/place destination was not resolved")
                start_position = contact + np.array(
                    [0.0, 0.0, program.approach_height_m], dtype=np.float64
                )
                lift_position = contact + np.array(
                    [0.0, 0.0, program.lift_height_m], dtype=np.float64
                )
                object_half_height = float((upper[2] - lower[2]) * 0.5)
                desired_object_center = destination_center.copy()
                desired_object_center[2] = destination_top + object_half_height + 0.005
                tool_offset_from_object = contact - center
                place_position = desired_object_center + tool_offset_from_object
                transfer_position = place_position + np.array(
                    [0.0, 0.0, program.lift_height_m], dtype=np.float64
                )
                end_position = place_position + np.array(
                    [0.0, 0.0, program.approach_height_m], dtype=np.float64
                )
            else:
                start_position = contact - direction * (
                    half_extent_along_push + program.approach_clearance_m
                )
                end_position = contact + direction * (
                    program.travel_distance_m * trigger_multiplier
                )
                lift_position = transfer_position = place_position = None
            end_effector = link_by_name[program.end_effector_link_name]
            orientation = bullet.getQuaternionFromEuler((math.pi, 0.0, 0.0))
            movable_joint_indices = [
                index
                for index in range(bullet.getNumJoints(robot_id))
                if bullet.getJointInfo(robot_id, index)[2] != bullet.JOINT_FIXED
            ]
            movable_slot_by_index = {
                joint_index: slot
                for slot, joint_index in enumerate(movable_joint_indices)
            }
            ik_lower_limits = [
                float(bullet.getJointInfo(robot_id, index)[8])
                for index in movable_joint_indices
            ]
            ik_upper_limits = [
                float(bullet.getJointInfo(robot_id, index)[9])
                for index in movable_joint_indices
            ]
            ik_joint_ranges = [
                max(1e-3, upper_limit - lower_limit)
                for lower_limit, upper_limit in zip(ik_lower_limits, ik_upper_limits)
            ]
            # Seed the continuous null-space solver from the unconstrained
            # solution that is already known to reach the requested start
            # pose.  The URDF's zero state is far from this side-mounted
            # workspace and is a poor null-space rest pose.
            start_solution = bullet.calculateInverseKinematics(
                robot_id,
                end_effector,
                start_position.tolist(),
                targetOrientation=orientation,
                maxNumIterations=200,
                residualThreshold=1e-5,
            )
            arm_indices = [joint_by_name[name] for name in program.arm_joint_names]
            arm_solution_slots = [
                movable_slot_by_index[index]
                for index in arm_indices
            ]
            start_targets = [
                float(start_solution[slot])
                for slot in arm_solution_slots
            ]
            for joint_index, target in zip(arm_indices, start_targets):
                bullet.resetJointState(robot_id, joint_index, target)
            for name in program.finger_joint_names:
                if name in joint_by_name:
                    bullet.resetJointState(robot_id, joint_by_name[name], 0.01)
            robot_state = {
                "program": program,
                "robot_id": robot_id,
                "target_id": target_id_for_robot,
                "contact_object": contact_object,
                "base_position": robot_base,
                "end_effector": end_effector,
                "arm_indices": arm_indices,
                "arm_solution_slots": arm_solution_slots,
                "ik_lower_limits": ik_lower_limits,
                "ik_upper_limits": ik_upper_limits,
                "ik_joint_ranges": ik_joint_ranges,
                "ik_rest_poses": list(start_solution),
                "start_position": start_position,
                "end_position": end_position,
                "contact_position": contact,
                "lift_position": lift_position,
                "transfer_position": transfer_position,
                "place_position": place_position,
                "orientation": orientation,
                "joint_names": [
                    *program.arm_joint_names,
                    *[
                        name
                        for name in program.finger_joint_names
                        if name in joint_by_name
                    ],
                ],
                "joint_indices": [
                    *arm_indices,
                    *[
                        joint_by_name[name]
                        for name in program.finger_joint_names
                        if name in joint_by_name
                    ],
                ],
                "previous_targets": start_targets,
                "finger_indices": [
                    joint_by_name[name]
                    for name in program.finger_joint_names
                    if name in joint_by_name
                ],
                "grasp_constraint": None,
                "grasp_created": False,
                "released": False,
            }
            if program.contact_only and continuous_push:
                # Only the certified tool link may touch the selected task
                # object. The visual arm remains fully rendered, while its
                # forearm, fingers and pedestal cannot accidentally strike the
                # table or neighbouring objects and inject hidden energy.
                for other_name, other_id in body_ids.items():
                    if other_name == program.robot:
                        continue
                    for robot_link in range(-1, bullet.getNumJoints(robot_id)):
                        for other_link in range(-1, bullet.getNumJoints(other_id)):
                            bullet.setCollisionFilterPair(
                                robot_id,
                                other_id,
                                robot_link,
                                other_link,
                                enableCollision=0,
                            )
                for target_link in range(-1, bullet.getNumJoints(target_id_for_robot)):
                    bullet.setCollisionFilterPair(
                        robot_id,
                        target_id_for_robot,
                        end_effector,
                        target_link,
                        enableCollision=1,
                    )

        def command_robot(time_s: float, enabled: bool = True) -> None:
            if robot_state is None:
                return
            program = robot_state["program"]
            if (
                continuous_push
                and (not enabled or time_s <= program.motion_start_s)
            ):
                # Hidden settle and the pre-trigger interval must hold the
                # already-solved start configuration exactly. Re-solving the
                # identical Cartesian pose can select another redundant-arm
                # branch and move the robot before the event begins.
                for joint_index, target in zip(
                    robot_state["arm_indices"], robot_state["previous_targets"]
                ):
                    bullet.resetJointState(
                        robot_state["robot_id"],
                        joint_index,
                        targetValue=target,
                        targetVelocity=0.0,
                    )
                    bullet.setJointMotorControl2(
                        robot_state["robot_id"],
                        joint_index,
                        bullet.POSITION_CONTROL,
                        targetPosition=target,
                        targetVelocity=0.0,
                        force=program.max_joint_force_n,
                        positionGain=program.position_gain,
                        velocityGain=1.0,
                        maxVelocity=program.max_joint_speed_rad_s,
                    )
                return
            if not enabled or time_s <= program.motion_start_s:
                fraction = 0.0
            elif time_s >= program.motion_end_s:
                fraction = 1.0
            else:
                fraction = (time_s - program.motion_start_s) / (
                    program.motion_end_s - program.motion_start_s
                )
                fraction = fraction * fraction * (3.0 - 2.0 * fraction)

            if program.operation == "cartesian_pick_place":
                keyframes = [
                    (0.00, robot_state["start_position"]),
                    (0.20, robot_state["contact_position"]),
                    (program.grasp_phase, robot_state["contact_position"]),
                    (0.46, robot_state["lift_position"]),
                    (0.66, robot_state["transfer_position"]),
                    (0.82, robot_state["place_position"]),
                    (program.release_phase, robot_state["place_position"]),
                    (1.00, robot_state["end_position"]),
                ]
                position = keyframes[-1][1]
                for (left_t, left), (right_t, right) in zip(keyframes, keyframes[1:]):
                    if fraction <= right_t:
                        local = 0.0 if right_t == left_t else (fraction - left_t) / (right_t - left_t)
                        local = float(np.clip(local, 0.0, 1.0))
                        local = local * local * (3.0 - 2.0 * local)
                        position = left * (1.0 - local) + right * local
                        break
            else:
                position = (
                    robot_state["start_position"] * (1.0 - fraction)
                    + robot_state["end_position"] * fraction
                )
            ik_kwargs = {}
            if continuous_push:
                ik_kwargs = {
                    "lowerLimits": robot_state["ik_lower_limits"],
                    "upperLimits": robot_state["ik_upper_limits"],
                    "jointRanges": robot_state["ik_joint_ranges"],
                    "restPoses": robot_state["ik_rest_poses"],
                    "jointDamping": [0.05] * len(robot_state["ik_rest_poses"]),
                }
            solution = bullet.calculateInverseKinematics(
                robot_state["robot_id"],
                robot_state["end_effector"],
                position.tolist(),
                targetOrientation=robot_state["orientation"],
                maxNumIterations=100,
                residualThreshold=1e-5,
                **ik_kwargs,
            )
            raw_targets = [
                float(solution[slot])
                for slot in robot_state["arm_solution_slots"]
            ]
            # A prescribed position-controlled reference avoids controller-lag
            # becoming a hidden benchmark factor.  The motion is smooth at
            # 240 Hz; object response and all contacts remain Bullet dynamics.
            previous = robot_state["previous_targets"]
            if continuous_push:
                max_delta = program.max_joint_speed_rad_s * spec.physics_dt_s
                targets = [
                    old_target
                    + float(np.clip(raw_target - old_target, -max_delta, max_delta))
                    for raw_target, old_target in zip(raw_targets, previous)
                ]
            else:
                targets = raw_targets
            velocities = [
                (target - old_target) / spec.physics_dt_s
                for target, old_target in zip(targets, previous)
            ]
            for joint_index, target, velocity in zip(
                robot_state["arm_indices"], targets, velocities
            ):
                bullet.resetJointState(
                    robot_state["robot_id"],
                    joint_index,
                    targetValue=target,
                    targetVelocity=velocity,
                )
                if continuous_push:
                    bullet.setJointMotorControl2(
                        robot_state["robot_id"],
                        joint_index,
                        bullet.POSITION_CONTROL,
                        targetPosition=target,
                        targetVelocity=velocity,
                        force=program.max_joint_force_n,
                        positionGain=program.position_gain,
                        velocityGain=1.0,
                        maxVelocity=program.max_joint_speed_rad_s,
                    )
            robot_state["previous_targets"] = targets
            if continuous_push:
                for slot, target in zip(robot_state["arm_solution_slots"], targets):
                    robot_state["ik_rest_poses"][slot] = target

            if program.operation != "cartesian_pick_place":
                return

            finger_value = 0.035
            if program.grasp_phase <= fraction < program.release_phase:
                finger_value = 0.0
            for finger_index in robot_state["finger_indices"]:
                bullet.resetJointState(robot_state["robot_id"], finger_index, finger_value)

            if (
                enabled
                and fraction >= program.grasp_phase
                and fraction < program.release_phase
                and robot_state["grasp_constraint"] is None
                and not robot_state["grasp_created"]
            ):
                closest = bullet.getClosestPoints(
                    robot_state["robot_id"],
                    robot_state["target_id"],
                    distance=0.035,
                    linkIndexA=robot_state["end_effector"],
                )
                if closest:
                    ee_state = bullet.getLinkState(
                        robot_state["robot_id"],
                        robot_state["end_effector"],
                        computeForwardKinematics=True,
                    )
                    target_pose = bullet.getBasePositionAndOrientation(robot_state["target_id"])
                    inverse_target = bullet.invertTransform(*target_pose)
                    child_frame = bullet.multiplyTransforms(
                        inverse_target[0], inverse_target[1], ee_state[4], ee_state[5]
                    )
                    constraint = bullet.createConstraint(
                        robot_state["robot_id"],
                        robot_state["end_effector"],
                        robot_state["target_id"],
                        -1,
                        bullet.JOINT_FIXED,
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        child_frame[0],
                        parentFrameOrientation=(0.0, 0.0, 0.0, 1.0),
                        childFrameOrientation=child_frame[1],
                    )
                    bullet.changeConstraint(
                        constraint,
                        maxForce=program.max_joint_force_n,
                    )
                    robot_state["grasp_constraint"] = constraint
                    robot_state["grasp_created"] = True
            if (
                fraction >= program.release_phase
                and robot_state["grasp_constraint"] is not None
            ):
                bullet.removeConstraint(robot_state["grasp_constraint"])
                robot_state["grasp_constraint"] = None
                robot_state["released"] = True
        settle_steps = round(spec.hidden_settle_s / spec.physics_dt_s)
        pre_positions: list[tuple[float, float, float]] = []
        pre_orientations: list[tuple[float, float, float, float]] = []
        target_id = body_ids[spec.target]
        for step in range(settle_steps):
            command_robot(0.0, enabled=False)
            bullet.stepSimulation()
            if step >= settle_steps - round(0.25 / spec.physics_dt_s):
                position, orientation = _base_link_pose(bullet, target_id)
                pre_positions.append(position)
                pre_orientations.append(orientation)

        max_penetration = 0.0
        penetration_pairs: list[dict[str, Any]] = []
        ids = list(body_ids.values())
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                pair_depth = 0.0
                for point in bullet.getClosestPoints(left, right, distance=0.0):
                    pair_depth = max(pair_depth, max(0.0, -float(point[8])))
                if pair_depth > 0.0:
                    max_penetration = max(max_penetration, pair_depth)
                    penetration_pairs.append(
                        {
                            "body_a": id_to_name[left],
                            "body_b": id_to_name[right],
                            "penetration_m": pair_depth,
                        }
                    )

        camera: dict[str, Any] = {}
        if render:
            if camera_orbit_index is None:
                camera = _select_camera_for_episode(
                    bullet,
                    target_id,
                    body_ids[spec.support],
                    spec,
                    renderer_id,
                    robot_id=(robot_state["robot_id"] if robot_state is not None else None),
                )
            else:
                camera = _camera_for_episode(
                    bullet,
                    target_id,
                    body_ids[spec.support],
                    spec,
                    orbit_index=camera_orbit_index,
                    robot_id=(robot_state["robot_id"] if robot_state is not None else None),
                )
                _, _, initial_segmentation = _render(bullet, camera, renderer_id)
                camera["selection_first_frame_pixels"] = int(
                    np.count_nonzero(initial_segmentation == target_id)
                )
                camera["selection_first_frame_bbox_fraction"] = target_bbox_fraction(
                    initial_segmentation, target_id
                )
                camera["selection_candidate_count"] = 1
                camera["selection_policy"] = "forced_fixed_orbit_v1"
        if egl_vendor is not None:
            camera["egl_vendor"] = egl_vendor
        total_steps = round(spec.duration_s / spec.physics_dt_s)
        render_stride = round(1.0 / (spec.fps * spec.physics_dt_s))
        trigger_step = round(spec.trigger_time_s / spec.physics_dt_s)
        trigger_duration_steps = max(1, round(spec.trigger.duration_s / spec.physics_dt_s))
        state_times: list[float] = []
        positions: list[list[tuple[float, float, float]]] = []
        orientations: list[list[tuple[float, float, float, float]]] = []
        linear_velocities: list[list[tuple[float, float, float]]] = []
        angular_velocities: list[list[tuple[float, float, float]]] = []
        contacts: list[list[dict[str, Any]]] = []
        rgb_frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        segmentation_frames: list[np.ndarray] = []
        min_actor_aabb_z = math.inf
        robot_min_distance_m = math.inf
        robot_contact_steps = 0
        robot_joint_positions: list[list[float]] = []
        robot_joint_velocities: list[list[float]] = []
        robot_end_effector_positions: list[tuple[float, float, float]] = []
        robot_end_effector_orientations: list[tuple[float, float, float, float]] = []
        link_positions: list[list[tuple[float, float, float]]] = []
        link_orientations: list[list[tuple[float, float, float, float]]] = []

        trigger_body = body_ids[spec.actors[0]]
        for step in range(total_steps + 1):
            state_times.append(step * spec.physics_dt_s)
            step_positions, step_orientations, step_linear, step_angular = [], [], [], []
            for name in names:
                position, orientation = _base_link_pose(bullet, body_ids[name])
                linear, angular = bullet.getBaseVelocity(body_ids[name])
                step_positions.append(position)
                step_orientations.append(orientation)
                step_linear.append(linear)
                step_angular.append(angular)
                if name in dynamic_actors:
                    min_actor_aabb_z = min(
                        min_actor_aabb_z,
                        float(bullet.getAABB(body_ids[name])[0][2]),
                    )
            positions.append(step_positions)
            orientations.append(step_orientations)
            linear_velocities.append(step_linear)
            angular_velocities.append(step_angular)
            step_link_positions: list[tuple[float, float, float]] = []
            step_link_orientations: list[tuple[float, float, float, float]] = []
            for _, body_id, link_index in link_records:
                if link_index == -1:
                    link_position, link_orientation = _base_link_pose(bullet, body_id)
                else:
                    link_state = bullet.getLinkState(
                        body_id, link_index, computeForwardKinematics=True
                    )
                    link_position, link_orientation = link_state[4], link_state[5]
                step_link_positions.append(tuple(float(v) for v in link_position))
                step_link_orientations.append(tuple(float(v) for v in link_orientation))
            link_positions.append(step_link_positions)
            link_orientations.append(step_link_orientations)
            contacts.append(_contact_rows(bullet, id_to_name))
            if robot_state is not None:
                joint_states = bullet.getJointStates(
                    robot_state["robot_id"], robot_state["joint_indices"]
                )
                robot_joint_positions.append([float(state[0]) for state in joint_states])
                robot_joint_velocities.append([float(state[1]) for state in joint_states])
                ee_state = bullet.getLinkState(
                    robot_state["robot_id"],
                    robot_state["end_effector"],
                    computeForwardKinematics=True,
                )
                robot_end_effector_positions.append(tuple(float(v) for v in ee_state[4]))
                robot_end_effector_orientations.append(tuple(float(v) for v in ee_state[5]))
                closest = bullet.getClosestPoints(
                    robot_state["robot_id"], robot_state["target_id"], distance=1.0
                )
                if closest:
                    robot_min_distance_m = min(
                        robot_min_distance_m, min(float(point[8]) for point in closest)
                    )
                if any(
                    {row["body_a"], row["body_b"]}
                    == {spec.robot_program.robot, robot_state["contact_object"]}
                    for row in contacts[-1]
                ):
                    robot_contact_steps += 1
            if render and step % render_stride == 0:
                rgb, depth, segmentation = _render(bullet, camera, renderer_id)
                rgb_frames.append(rgb)
                depth_frames.append(depth)
                segmentation_frames.append(segmentation)
            if step == total_steps:
                break
            active = trigger_step <= step < trigger_step + trigger_duration_steps
            one_shot = spec.trigger.operation == "set_velocity" and step == trigger_step
            command_robot(
                step * spec.physics_dt_s,
                enabled=spec.variant != "no_trigger",
            )
            use_external_trigger = (
                spec.robot_program is None or not spec.robot_program.contact_only
            )
            if use_external_trigger and spec.variant != "no_trigger" and (active or one_shot):
                _apply_trigger(bullet, trigger_body, spec, trigger_multiplier)
            bullet.stepSimulation()
        expected_render_frames = round(spec.duration_s * spec.fps) + 1
        if render and len(rgb_frames) != expected_render_frames:
            raise RuntimeError(
                f"expected {expected_render_frames} inclusive frames, got {len(rgb_frames)}"
            )
        return SimulationResult(
            timestamps=np.asarray(state_times, dtype=np.float64),
            body_names=names,
            positions=np.asarray(positions, dtype=np.float64),
            orientations=np.asarray(orientations, dtype=np.float64),
            linear_velocities=np.asarray(linear_velocities, dtype=np.float64),
            angular_velocities=np.asarray(angular_velocities, dtype=np.float64),
            target_name=spec.target,
            target_instance_id=target_id,
            body_instance_map=body_ids,
            contacts=contacts,
            pre_settle_positions=np.asarray(pre_positions, dtype=np.float64),
            pre_settle_orientations=np.asarray(pre_orientations, dtype=np.float64),
            max_penetration_m=max_penetration,
            penetration_pairs=penetration_pairs,
            min_actor_aabb_z_m=min_actor_aabb_z,
            rgb_frames=rgb_frames,
            depth_frames=depth_frames,
            segmentation_frames=segmentation_frames,
            camera=camera,
            renderer=renderer if render else "none",
            robot_diagnostics=(
                {
                    "program": spec.robot_program.model_dump(mode="json"),
                    "contact_object": robot_state["contact_object"],
                    "base_position": robot_state["base_position"].tolist(),
                    "start_position": robot_state["start_position"].tolist(),
                    "end_position": robot_state["end_position"].tolist(),
                    "path_multiplier": trigger_multiplier,
                    "minimum_robot_object_distance_m": robot_min_distance_m,
                    "robot_object_contact_steps": robot_contact_steps,
                    "grasp_created": bool(robot_state["grasp_created"]),
                    "released": bool(robot_state["released"]),
                }
                if robot_state is not None
                else {}
            ),
            robot_joint_names=(robot_state["joint_names"] if robot_state else []),
            robot_joint_positions=np.asarray(robot_joint_positions, dtype=np.float64),
            robot_joint_velocities=np.asarray(robot_joint_velocities, dtype=np.float64),
            robot_end_effector_positions=np.asarray(
                robot_end_effector_positions, dtype=np.float64
            ).reshape((-1, 3)),
            robot_end_effector_orientations=np.asarray(
                robot_end_effector_orientations, dtype=np.float64
            ).reshape((-1, 4)),
            link_names=[item[0] for item in link_records],
            link_positions=np.asarray(link_positions, dtype=np.float64),
            link_orientations=np.asarray(link_orientations, dtype=np.float64),
        )
    finally:
        if plugin_id >= 0:
            bullet.unloadPlugin(plugin_id)
        bullet.disconnect(client)


def _tilt_degrees(quaternions: np.ndarray) -> np.ndarray:
    x, y, z, w = [quaternions[:, index] for index in range(4)]
    # World Z component of the rotated local Z axis.
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))


def _target_support_contact(result: SimulationResult, support: str, step: int) -> bool:
    target = result.target_name
    return any({row["body_a"], row["body_b"]} == {target, support} for row in result.contacts[step])


def outcome_metrics(result: SimulationResult, spec: EpisodeSpec) -> dict[str, Any]:
    index = result.target_index
    position = result.positions[:, index]
    orientation = result.orientations[:, index]
    displacement_xy = np.linalg.norm(position[:, :2] - position[0, :2], axis=1)
    displacement_xyz = np.linalg.norm(position - position[0], axis=1)
    drop = position[0, 2] - position[:, 2]
    trigger_step = round(spec.trigger_time_s / spec.physics_dt_s)
    initial_contacts = {
        tuple(sorted((row["body_a"], row["body_b"]))) for row in result.contacts[max(0, trigger_step - 1)]
    }
    final_contacts = {
        tuple(sorted((row["body_a"], row["body_b"]))) for row in result.contacts[-1]
    }
    metrics = {
        "max_drop_m": float(drop.max()),
        "max_horizontal_displacement_m": float(displacement_xy.max()),
        "max_displacement_m": float(displacement_xyz.max()),
        "max_tilt_deg": float(_tilt_degrees(orientation).max()),
        "support_lost": not _target_support_contact(result, spec.support, -1),
        "contact_changed": initial_contacts != final_contacts,
        "robot_contact_occurred": bool(
            result.robot_diagnostics.get("robot_object_contact_steps", 0) > 0
        ),
        "robot_object_contact_steps": int(
            result.robot_diagnostics.get("robot_object_contact_steps", 0)
        ),
        "robot_grasp_created": bool(result.robot_diagnostics.get("grasp_created", False)),
        "robot_grasp_released": bool(result.robot_diagnostics.get("released", False)),
    }
    if spec.robot_program is not None and spec.robot_program.destination_object:
        destination_index = result.body_names.index(spec.robot_program.destination_object)
        metrics["final_destination_distance_m"] = float(
            np.linalg.norm(position[-1, :2] - result.positions[-1, destination_index, :2])
        )
        destination_contact = any(
            {row["body_a"], row["body_b"]}
            == {result.target_name, spec.robot_program.destination_object}
            for row in result.contacts[-1]
        )
    else:
        destination_contact = False
    metrics["support_retained"] = bool(
        not metrics["support_lost"]
        or destination_contact
        or metrics["max_drop_m"] < 0.12
    )
    expected = spec.expected
    passed = True
    if expected.min_drop_m:
        passed &= metrics["max_drop_m"] >= expected.min_drop_m
    if expected.min_displacement_m:
        displacement_metric = (
            "max_displacement_m"
            if expected.kind == "stack_collapse"
            else "max_horizontal_displacement_m"
        )
        passed &= metrics[displacement_metric] >= expected.min_displacement_m
    if expected.min_tilt_deg:
        passed &= metrics["max_tilt_deg"] >= expected.min_tilt_deg
    if expected.max_final_distance_m is not None:
        passed &= metrics.get("final_destination_distance_m", math.inf) <= expected.max_final_distance_m
    if expected.require_support_loss:
        passed &= metrics["support_lost"]
    if expected.require_support_retained:
        passed &= metrics["support_retained"]
    if expected.require_contact_change:
        passed &= metrics["contact_changed"]
    if spec.robot_program is not None and spec.variant != "no_trigger":
        passed &= metrics["robot_contact_occurred"]
        if spec.robot_program.operation == "cartesian_pick_place":
            passed &= metrics["robot_grasp_created"] and metrics["robot_grasp_released"]
    metrics["passed"] = bool(passed)
    return metrics


def calibrate_trigger(bundle_path: str | Path, spec: EpisodeSpec) -> tuple[float, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for multiplier in spec.trigger_grid:
        result = simulate(bundle_path, spec, trigger_multiplier=multiplier, render=False)
        metrics = outcome_metrics(result, spec)
        attempts.append({"multiplier": multiplier, "metrics": metrics})
        if metrics["passed"]:
            return multiplier, {"attempts": attempts, "selection": "smallest_passing_grid_value"}
    raise RuntimeError(f"no trigger grid value satisfied {spec.episode_id}: {attempts}")


def _orientation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    dots = np.abs(np.sum(left * right, axis=-1))
    return float(np.degrees(2 * np.arccos(np.clip(dots, -1, 1))).max())


def target_bbox_fraction(segmentation: np.ndarray, instance_id: int) -> float:
    """Return the larger target bbox dimension as a fraction of the image."""

    rows, columns = np.nonzero(segmentation == instance_id)
    if not len(rows):
        return 0.0
    height, width = segmentation.shape
    bbox_height = int(rows.max() - rows.min() + 1)
    bbox_width = int(columns.max() - columns.min() + 1)
    return float(max(bbox_height / height, bbox_width / width))


def _pair_aware_penetration_probe(
    result: SimulationResult,
    spec: EpisodeSpec,
    bundle: dict[str, Any],
) -> ProbeResult:
    """Apply the three official SceneCode tolerances by semantic contact pair."""

    from .thresholds import load_threshold_profile

    thresholds = load_threshold_profile().scenecode
    rows: list[dict[str, Any]] = []
    for raw in result.penetration_pairs:
        row = dict(raw)
        left, right = row["body_a"], row["body_b"]
        left_item, right_item = bundle["bodies"][left], bundle["bodies"][right]
        intended_support = (
            left_item.get("support") == right or right_item.get("support") == left
        )
        room_furniture = (
            left_item.get("semantic_category") == "room" and right_item.get("static")
        ) or (
            right_item.get("semantic_category") == "room" and left_item.get("static")
        )
        if intended_support:
            pair_class = "manipuland_support"
            tolerance = thresholds.manipuland_support_tolerance_m
        elif room_furniture:
            pair_class = "furniture_floor"
            tolerance = thresholds.furniture_floor_tolerance_m
        else:
            pair_class = "object_object"
            tolerance = thresholds.object_penetration_m
        row.update(
            pair_class=pair_class,
            tolerance_m=tolerance,
            passed=float(row["penetration_m"]) <= tolerance,
        )
        rows.append(row)
    passed = all(bool(row["passed"]) for row in rows)
    return ProbeResult(
        name="initial_penetration_pair_aware",
        passed=passed,
        measured=max((float(row["penetration_m"]) for row in rows), default=0.0),
        threshold="SceneCode official pair-aware tolerances",
        detail=json.dumps(rows, sort_keys=True),
    )


def certify(
    result: SimulationResult,
    repeats: list[SimulationResult],
    spec: EpisodeSpec,
    world: WorldIR,
    bundle_path: str | Path,
    calibration: dict[str, Any],
) -> PhysicsReport:
    target_index = result.target_index
    pre_drift = float(np.linalg.norm(result.pre_settle_positions[-1] - result.pre_settle_positions[0]))
    pre_angle = _orientation_distance_deg(result.pre_settle_orientations[-1:], result.pre_settle_orientations[:1])
    all_state = np.concatenate([result.positions.reshape(-1), result.linear_velocities.reshape(-1), result.angular_velocities.reshape(-1)])
    max_speed = float(np.linalg.norm(result.linear_velocities, axis=-1).max())
    min_z = result.min_actor_aabb_z_m
    outcome = outcome_metrics(result, spec)
    position_repeat_error = 0.0
    angle_repeat_error = 0.0
    reference_target = result.positions[:, target_index]
    reference_orientation = result.orientations[:, target_index]
    for replay in repeats:
        replay_index = replay.target_index
        position_repeat_error = max(position_repeat_error, float(np.linalg.norm(replay.positions[:, replay_index] - reference_target, axis=-1).max()))
        angle_repeat_error = max(angle_repeat_error, _orientation_distance_deg(replay.orientations[:, replay_index], reference_orientation))

    visibility_counts: list[int] = []
    if result.segmentation_frames:
        visibility_counts = [int(np.count_nonzero(frame == result.target_instance_id)) for frame in result.segmentation_frames]
    visible_fraction = float(np.mean(np.asarray(visibility_counts) >= 1)) if visibility_counts else 0.0
    first_area = visibility_counts[0] if visibility_counts else 0
    first_bbox_fraction = (
        target_bbox_fraction(result.segmentation_frames[0], result.target_instance_id)
        if result.segmentation_frames
        else 0.0
    )
    depth_fixture = result.camera.get("depth_projection_fixture", {})
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    from .thresholds import load_threshold_profile

    profile = load_threshold_profile()
    robot_probes: list[ProbeResult] = []
    if spec.robot_program is not None:
        robot_steps = int(result.robot_diagnostics.get("robot_object_contact_steps", 0))
        robot_instance_id = result.body_instance_map[spec.robot_program.robot]
        robot_first_frame_pixels = (
            int(np.count_nonzero(result.segmentation_frames[0] == robot_instance_id))
            if result.segmentation_frames
            else 0
        )
        robot_state_finite = bool(
            np.isfinite(result.robot_joint_positions).all()
            and np.isfinite(result.robot_joint_velocities).all()
            and np.isfinite(result.robot_end_effector_positions).all()
        )
        expected_robot_samples = len(result.timestamps)
        robot_probes = [
            ProbeResult(
                name="robot_object_contact",
                passed=robot_steps > 0,
                measured=robot_steps,
                threshold="> 0 simulation steps",
                detail=f"certified contact pair: {spec.robot_program.robot}/{spec.robot_program.contact_object or spec.target}",
            ),
            ProbeResult(
                name="robot_joint_trajectory_complete",
                passed=(
                    result.robot_joint_positions.shape[0] == expected_robot_samples
                    and len(result.robot_joint_names) > 0
                    and robot_state_finite
                ),
                measured={
                    "samples": int(result.robot_joint_positions.shape[0]),
                    "joints": len(result.robot_joint_names),
                    "finite": robot_state_finite,
                },
                threshold={"samples": expected_robot_samples, "finite": True},
            ),
            ProbeResult(
                name="robot_first_frame_visible",
                passed=robot_first_frame_pixels >= 1000,
                measured=robot_first_frame_pixels,
                threshold=1000,
                detail="robot pixels decoded from the recorded first-frame instance mask",
            ),
        ]
        if spec.robot_program.operation == "cartesian_pick_place":
            robot_probes.append(
                ProbeResult(
                    name="robot_grasp_and_release",
                    passed=bool(
                        result.robot_diagnostics.get("grasp_created", False)
                        and result.robot_diagnostics.get("released", False)
                    ),
                    measured={
                        "grasp_created": bool(
                            result.robot_diagnostics.get("grasp_created", False)
                        ),
                        "released": bool(result.robot_diagnostics.get("released", False)),
                    },
                    threshold={"grasp_created": True, "released": True},
                    detail="deterministic contact-gated grasp followed by release over the destination",
                )
            )
    probes = [
        ProbeResult(name="compile_geometry_complete", passed=True, measured=True, threshold=True),
        ProbeResult(
            name="egl_vendor_nvidia",
            passed=result.renderer != "egl" or "NVIDIA" in str(result.camera.get("egl_vendor", "")).upper(),
            measured=result.camera.get("egl_vendor"),
            threshold="NVIDIA",
        ),
        _pair_aware_penetration_probe(result, spec, bundle),
        ProbeResult(name="pretrigger_drift_m", passed=pre_drift < 0.005, measured=pre_drift, threshold=0.005),
        ProbeResult(name="pretrigger_rotation_deg", passed=pre_angle < 1.0, measured=pre_angle, threshold=1.0),
        ProbeResult(name="finite_state", passed=bool(np.isfinite(all_state).all()), measured=bool(np.isfinite(all_state).all()), threshold=True),
        ProbeResult(name="max_speed_m_s", passed=max_speed < 50.0, measured=max_speed, threshold=50.0),
        ProbeResult(
            name="min_dynamic_actor_aabb_z_m",
            passed=min_z > -0.05,
            measured=min_z,
            threshold=-0.05,
            detail="minimum world AABB Z over episode actors; excludes static room/floor origins",
        ),
        ProbeResult(name="event_outcome", passed=bool(outcome["passed"]), measured=bool(outcome["passed"]), threshold=True, detail=json.dumps(outcome, sort_keys=True)),
        *robot_probes,
        ProbeResult(name="replay_position_error_m", passed=position_repeat_error <= 0.001, measured=position_repeat_error, threshold=0.001),
        ProbeResult(name="replay_angle_error_deg", passed=angle_repeat_error <= 0.5, measured=angle_repeat_error, threshold=0.5),
        ProbeResult(name="target_visible_fraction", passed=visible_fraction >= profile.observation.visible_ratio_min, measured=visible_fraction, threshold=profile.observation.visible_ratio_min),
        ProbeResult(name="target_first_frame_pixels", passed=first_area >= profile.observation.first_frame_pixels_min, measured=first_area, threshold=profile.observation.first_frame_pixels_min),
        ProbeResult(
            name="target_first_frame_bbox_fraction",
            passed=(
                spec.camera.target_fraction_min
                <= first_bbox_fraction
                <= spec.camera.target_fraction_max
            ),
            measured=first_bbox_fraction,
            threshold=f"[{spec.camera.target_fraction_min}, {spec.camera.target_fraction_max}]",
            detail="max(target_bbox_width/image_width, target_bbox_height/image_height)",
        ),
        ProbeResult(
            name="analytic_depth_projection_error_m",
            passed=bool(depth_fixture.get("passed", False)),
            measured=depth_fixture.get("absolute_error_m"),
            threshold=0.001,
            detail=json.dumps(depth_fixture, sort_keys=True),
        ),
    ]
    repairs = [RepairRecord.model_validate(item) for item in bundle.get("repairs", [])]
    passed = all(probe.passed for probe in probes)
    return PhysicsReport(
        episode_id=spec.episode_id,
        status=("repaired_pass" if repairs else "pass") if passed else "quarantined",
        world_hash=world.content_hash(),
        episode_hash=spec.content_hash(),
        pybullet_version=str(getattr(_import_pybullet(), "__version__", "3.2.7")),
        renderer=result.renderer,
        probes=probes,
        repairs=repairs,
        diagnostics={
            "trigger_calibration": calibration,
            "outcome": outcome,
            "platform": platform.platform(),
            "egl_vendor": result.camera.get("egl_vendor"),
            "robot": result.robot_diagnostics,
            "bundle_sha256": sha256_file(bundle_path),
            "source_file_sha256": {
                body.name: body.source_sha256
                for body in world.bodies
                if body.source_sha256 is not None
            },
        },
    )


def _write_independent_certificates(
    destination: Path,
    result: SimulationResult,
    spec: EpisodeSpec,
    report: PhysicsReport,
) -> tuple[Path, Path]:
    """Split legacy aggregate probes into independently usable M2 certificates."""

    from .thresholds import load_threshold_profile, threshold_profile_hash

    profile = load_threshold_profile()
    profile_hash = threshold_profile_hash(profile)
    observation_names = {
        "egl_vendor_nvidia",
        "robot_first_frame_visible",
        "target_visible_fraction",
        "target_first_frame_pixels",
        "target_first_frame_bbox_fraction",
        "analytic_depth_projection_error_m",
    }
    physics_probes = [probe for probe in report.probes if probe.name not in observation_names]
    physics_passed = all(probe.passed for probe in physics_probes)
    physics = PhysicsCertificate(
        episode_id=spec.episode_id,
        status=("repaired_pass" if report.repairs else "pass") if physics_passed else "quarantined",
        episode_hash=spec.content_hash(),
        threshold_profile_id=profile.profile_id,
        threshold_profile_hash=profile_hash,
        probes=physics_probes,
        failure_codes=[] if physics_passed else [
            f"PHYSICS_{probe.name.upper()}" for probe in physics_probes if not probe.passed
        ],
        diagnostics={"legacy_physics_report": "physics_report.json"},
        world_hash=report.world_hash,
        simulator=f"pybullet-{report.pybullet_version}",
        repairs=report.repairs,
    )

    expected_frames = round(spec.duration_s * spec.fps) + 1
    stream_lengths = {
        "rgb": len(result.rgb_frames),
        "depth": len(result.depth_frames),
        "segmentation": len(result.segmentation_frames),
    }
    resolution_ok = all(
        frame.shape[:2] == (spec.camera.height, spec.camera.width)
        for stream in (result.rgb_frames, result.depth_frames, result.segmentation_frames)
        for frame in stream
    )
    depth_values = (
        np.concatenate([frame.reshape(-1) for frame in result.depth_frames])
        if result.depth_frames
        else np.asarray([], dtype=np.float32)
    )
    depth_finite_ratio = (
        float(np.mean(np.isfinite(depth_values))) if len(depth_values) else 0.0
    )
    target_depths = np.concatenate(
        [
            depth[segmentation == result.target_instance_id]
            for depth, segmentation in zip(result.depth_frames, result.segmentation_frames)
            if np.any(segmentation == result.target_instance_id)
        ]
    ) if any(
        np.any(segmentation == result.target_instance_id)
        for segmentation in result.segmentation_frames
    ) else np.asarray([], dtype=np.float32)
    target_depth_ok = bool(
        len(target_depths)
        and np.isfinite(target_depths).all()
        and (target_depths > spec.camera.near_m).all()
        and (target_depths < spec.camera.far_m).all()
    )
    obs_probes = [
        *(probe for probe in report.probes if probe.name in observation_names),
        ProbeResult(
            name="observation_stream_frame_count",
            passed=all(value == expected_frames for value in stream_lengths.values()),
            measured=json.dumps(stream_lengths, sort_keys=True),
            threshold=expected_frames,
        ),
        ProbeResult(
            name="observation_stream_resolution",
            passed=resolution_ok,
            measured=f"{spec.camera.width}x{spec.camera.height}" if resolution_ok else "mismatch",
            threshold=f"{spec.camera.width}x{spec.camera.height}",
        ),
        ProbeResult(
            name="depth_finite_ratio",
            passed=depth_finite_ratio >= profile.observation.depth_valid_ratio_min,
            measured=depth_finite_ratio,
            threshold=profile.observation.depth_valid_ratio_min,
        ),
        ProbeResult(
            name="target_depth_not_clipped",
            passed=target_depth_ok,
            measured=(
                json.dumps([float(target_depths.min()), float(target_depths.max())])
                if len(target_depths)
                else None
            ),
            threshold=f"({spec.camera.near_m}, {spec.camera.far_m})",
        ),
        ProbeResult(
            name="camera_calibration_complete",
            passed=(
                np.asarray(result.camera.get("intrinsics", [])).shape == (3, 3)
                and np.asarray(result.camera.get("camera_to_world_opencv", [])).shape == (4, 4)
            ),
            measured=True,
            threshold=True,
        ),
    ]
    observation_passed = all(probe.passed for probe in obs_probes)
    observation = ObservationCertificate(
        episode_id=spec.episode_id,
        status="pass" if observation_passed else "quarantined",
        episode_hash=spec.content_hash(),
        threshold_profile_id=profile.profile_id,
        threshold_profile_hash=profile_hash,
        probes=obs_probes,
        failure_codes=[] if observation_passed else [
            f"OBSERVATION_{probe.name.upper()}" for probe in obs_probes if not probe.passed
        ],
        renderer=result.renderer,
        frame_count=len(result.rgb_frames),
        target_instance_id=result.target_instance_id,
    )
    return (
        physics.write_json(destination / "physics_certificate.json"),
        observation.write_json(destination / "observation_certificate.json"),
    )


def record_result(
    result: SimulationResult,
    spec: EpisodeSpec,
    world: WorldIR,
    report: PhysicsReport,
    output_dir: str | Path,
    *,
    presentation_renderer: str = "pybullet_debug",
    presentation_bundle_path: str | Path | None = None,
    blender_executable: str | Path | None = None,
    presentation_width: int = 1280,
    presentation_height: int = 720,
) -> EpisodeManifest:
    destination = Path(output_dir).resolve()
    frames_dir = destination / "rgb_frames"
    depth_dir = destination / "depth"
    segmentation_dir = destination / "segmentation"
    for directory in (frames_dir, depth_dir, segmentation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for episode recording") from exc
    video_path = destination / "rgb.mp4"
    for index, (rgb, depth, segmentation) in enumerate(zip(result.rgb_frames, result.depth_frames, result.segmentation_frames)):
        cv2.imwrite(str(frames_dir / f"{index:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{index:04d}.npy", depth.astype(np.float32))
        np.save(segmentation_dir / f"{index:04d}.npy", segmentation.astype(np.uint32))
    from .media import write_h264_rgb

    write_h264_rgb(video_path, result.rgb_frames, spec.fps, crf=16)
    camera_rgb_path = destination / "camera_rgb.png"
    cv2.imwrite(str(camera_rgb_path), cv2.cvtColor(result.rgb_frames[0], cv2.COLOR_RGB2BGR))
    segmentation_encoding_path = write_json(
        destination / "segmentation_encoding.json",
        {
            "dtype": "uint32",
            "encoding": "pybullet_body_unique_id",
            "background_value": 2**32 - 1,
            "target_instance_id": result.target_instance_id,
            "body_instance_map": result.body_instance_map,
        },
    )

    camera_calibration_path = write_json(
        destination / "camera_calibration_info.json",
        {
            "head_camera": {
                "intrinsics": result.camera["intrinsics"],
                "extrinsics": result.camera["camera_to_world_opencv"],
            }
        },
    )
    object_names = {
        "block_sort": "small red sorting block",
        "cup_tip": "small blue object",
        "block_push": "small brown block",
        "stack_collapse": "yellow top block",
    }
    instructions = {
        "block_sort": "Track the small red block while the robotic arm picks it up and places it inside the blue sorting tray.",
        "cup_tip": "Track the blue ceramic cup while the robotic arm pushes its upper side and tips it over.",
        "block_push": "Track the wooden slider block while the robotic arm pushes it into the green target zone.",
        "stack_collapse": "Track the yellow top block while the robotic arm pushes the middle block and collapses the stack.",
    }
    scene_data_path = destination / "scene_data.yaml"
    scene_data_path.write_text(
        f"instruction: {instructions[spec.event_type]}\n"
        f"object_name: {object_names[spec.event_type]}\n"
        f"canonical_object_name: {spec.target}\n",
        encoding="utf-8",
    )
    dream2flow_input_path = write_json(
        destination / "dream2flow_input.json",
        {
            "schema_version": "simready_dream2flow_input.v1",
            "video": video_path.name,
            "camera_rgb": camera_rgb_path.name,
            "camera_calibration": camera_calibration_path.name,
            "scene_data": scene_data_path.name,
            "depth_npy_dir": depth_dir.name,
            "initial_depth_output": "initial_depth.pt",
            "depth_frames_output": "depth_frames.pt",
            "object_flow_output": "dream2flow/object_flow_result.pt",
            "camera_name": "head_camera",
            "video_role": "reference" if spec.variant == "reference" else "imagination",
            "source_model": None if spec.variant == "reference" else f"pybullet_control_{spec.variant}",
        },
    )

    state_path = destination / "state_trajectory.npz"
    np.savez_compressed(
        state_path,
        timestamps=result.timestamps,
        body_names=np.asarray(result.body_names),
        positions=result.positions,
        orientations_xyzw=result.orientations,
        linear_velocities=result.linear_velocities,
        angular_velocities=result.angular_velocities,
        target_name=np.asarray(result.target_name),
        robot_joint_names=np.asarray(result.robot_joint_names),
        robot_joint_positions=result.robot_joint_positions,
        robot_joint_velocities=result.robot_joint_velocities,
        robot_end_effector_positions=result.robot_end_effector_positions,
        robot_end_effector_orientations_xyzw=result.robot_end_effector_orientations,
        link_names=np.asarray(result.link_names),
        link_positions=result.link_positions,
        link_orientations_xyzw=result.link_orientations,
    )
    target_index = result.target_index
    canonical_path = destination / "target_action_trajectory.npz"
    cae_stride = round(1.0 / (spec.fps * spec.physics_dt_s))
    cae_indices = np.arange(0, len(result.timestamps), cae_stride, dtype=np.int64)
    if cae_indices[-1] != len(result.timestamps) - 1:
        cae_indices = np.append(cae_indices, len(result.timestamps) - 1)
    np.savez_compressed(
        canonical_path,
        positions=result.positions[cae_indices, target_index] - result.positions[0, target_index],
        timestamps=result.timestamps[cae_indices],
        coordinate_frame=np.asarray("pybullet_world_z_up"),
        units=np.asarray("m"),
        metadata_json=np.asarray(json.dumps({"benchmark_status": "pipeline_validation", "episode_id": spec.episode_id, "variant": spec.variant, "source_state_rate_hz": 1.0 / spec.physics_dt_s, "cae_sample_rate_hz": spec.fps}, sort_keys=True)),
    )
    from .surface_gt import write_simulator_object_flow

    simulator_object_flow_path = write_simulator_object_flow(
        destination / "simulator_object_flow.npz", result, spec, world
    )
    contact_path = destination / "contacts.jsonl"
    with contact_path.open("w", encoding="utf-8") as stream:
        for timestamp, contacts in zip(result.timestamps, result.contacts):
            stream.write(json.dumps({"timestamp": float(timestamp), "contacts": contacts}, sort_keys=True) + "\n")
    calibration_path = write_json(destination / "calibration.json", result.camera)
    presentation_report_path: Path | None = None
    if presentation_renderer == "blender_replay":
        if presentation_bundle_path is None:
            raise ValueError("blender_replay requires presentation_bundle_path")
        from .blender_replay import render_blender_replay

        replay = render_blender_replay(
            bundle_path=presentation_bundle_path,
            trajectory_path=state_path,
            calibration_path=calibration_path,
            output_dir=destination / "blender_replay",
            fps=spec.fps,
            target_name=spec.target,
            blender_executable=blender_executable,
            width=presentation_width,
            height=presentation_height,
            segmentation_frames=result.segmentation_frames,
            target_instance_id=result.target_instance_id,
            robot_instance_id=(
                result.body_instance_map.get(spec.robot_program.robot)
                if spec.robot_program is not None
                else None
            ),
        )
        for path in frames_dir.glob("*.png"):
            path.unlink()
        result.rgb_frames = replay.rgb_frames
        for index, rgb in enumerate(result.rgb_frames):
            cv2.imwrite(
                str(frames_dir / f"{index:04d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
        write_h264_rgb(video_path, result.rgb_frames, spec.fps, crf=16)
        cv2.imwrite(
            str(camera_rgb_path),
            cv2.cvtColor(result.rgb_frames[0], cv2.COLOR_RGB2BGR),
        )
        presentation_report_path = replay.report_path
    elif presentation_renderer != "pybullet_debug":
        raise ValueError(f"unknown presentation renderer: {presentation_renderer}")
    world_path = world.write_json(destination / "world_ir.json")
    spec_path = spec.write_json(destination / "episode_spec.json")
    report_path = report.write_json(destination / "physics_report.json")
    physics_certificate_path, observation_certificate_path = _write_independent_certificates(
        destination, result, spec, report
    )
    artifact_paths = [
        video_path,
        camera_rgb_path,
        camera_calibration_path,
        scene_data_path,
        dream2flow_input_path,
        state_path,
        canonical_path,
        simulator_object_flow_path,
        contact_path,
        calibration_path,
        world_path,
        spec_path,
        report_path,
        physics_certificate_path,
        observation_certificate_path,
        segmentation_encoding_path,
    ]
    if presentation_report_path is not None:
        artifact_paths.append(presentation_report_path)
    artifact_paths.extend(sorted(frames_dir.glob("*.png")))
    artifact_paths.extend(sorted(depth_dir.glob("*.npy")))
    artifact_paths.extend(sorted(segmentation_dir.glob("*.npy")))
    manifest = EpisodeManifest(
        episode_id=spec.episode_id,
        prompt=world.prompt,
        world_ir_path=world_path.name,
        episode_spec_path=spec_path.name,
        physics_report_path=report_path.name,
        physics_certificate_path=physics_certificate_path.name,
        observation_certificate_path=observation_certificate_path.name,
        rgb_video_path=video_path.name,
        rgb_frames_dir=frames_dir.name,
        depth_dir=depth_dir.name,
        segmentation_dir=segmentation_dir.name,
        segmentation_encoding_path=segmentation_encoding_path.name,
        calibration_path=calibration_path.name,
        state_trajectory_path=state_path.name,
        simulator_object_flow_path=simulator_object_flow_path.name,
        contact_trajectory_path=contact_path.name,
        target_instance_id=result.target_instance_id,
        body_instance_map=result.body_instance_map,
        presentation_renderer=presentation_renderer,
        presentation_render_report_path=(
            presentation_report_path.relative_to(destination).as_posix()
            if presentation_report_path is not None
            else None
        ),
        artifact_sha256={
            path.relative_to(destination).as_posix(): sha256_file(path)
            for path in artifact_paths
        },
    )
    manifest.write_json(destination / "episode_manifest.json")
    return manifest
