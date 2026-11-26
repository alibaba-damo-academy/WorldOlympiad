"""Freeze native fixed cameras around a certified articulated reference motion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .robotics_native_cameras import (
    DROID_NATIVE_FIXED_FOV_Y_DEG, DROID_NATIVE_FRAME_MARGIN_PX,
    ROBOLAB_DROID_FIXED_CAMERAS, framing_measurement, pinhole_intrinsics,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.asarray([[x, y, z] for x in (low[0], high[0])
                       for y in (low[1], high[1])
                       for z in (low[2], high[2])], dtype=float)


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def fit_articulated_camera_rig(*, trajectory: Path, host_sdf: Path,
                               binding: Path, required_views: dict,
                               output: Path) -> Path:
    """Back fixed lenses away until the full physical motion is in native view.

    Inputs are only the certified reference trajectory, host geometry, joint
    binding and frozen model camera requirements. No WAM output or score is read.
    """
    from scripts.render_scenecode_asset_identity import sdf_collision_bounds

    trajectory, host_sdf, binding, output = (
        Path(value).resolve() for value in (trajectory, host_sdf, binding, output))
    payload = json.loads(trajectory.read_text())
    frames = payload.get("frames") or []
    if not frames or not required_views:
        raise ValueError("articulated camera fit needs a reference motion and native views")
    joint = json.loads(binding.read_text())
    if (joint.get("passed") is not True
            or joint.get("host_sdf_sha256") != _sha(host_sdf)):
        raise ValueError("articulated camera fit has an unbound host asset")
    host_id = joint["host_object_id"]
    child_id = joint.get("child_link")
    if (not isinstance(child_id, str)
            or not all(joint.get(key) is not None for key in (
                "contact_target_local_xyz", "contact_normal_local_xyz",
                "handle_axis_local_xyz"))):
        raise ValueError("articulated camera needs certified contact-side geometry")
    low, high = sdf_collision_bounds(host_sdf)
    # Each recorded link centre already spans the joint sweep. Expanding every
    # link by half of the *whole host* double-counts that span and makes the
    # end effector sub-threshold in model-native views. The rendered masks still
    # decide final admission for every selected view.
    half_extent = 0.30 * (np.asarray(high) - np.asarray(low)) + 0.05
    centers = []
    for frame in frames:
        host = (frame.get("objects") or {}).get(host_id)
        if not isinstance(host, dict) or not host.get("link_states"):
            raise ValueError("articulated trajectory lacks host link states")
        centers.extend(item["position"] for item in host["link_states"].values())
        centers.append(frame["robot"]["end_effector_position"])
    points = np.asarray(centers, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("articulated camera fit received invalid motion positions")
    envelope = _box(points.min(axis=0) - half_extent,
                    points.max(axis=0) + half_extent)
    initial = np.asarray([
        item["position"]
        for item in frames[0]["objects"][host_id]["link_states"].values()
    ], dtype=float)
    host_box = _box(initial.min(axis=0) - half_extent,
                    initial.max(axis=0) + half_extent)
    robot = frames[0]["robot"]
    child = frames[0]["objects"][host_id]["link_states"].get(child_id)
    if not isinstance(child, dict):
        raise ValueError("articulated camera cannot locate the certified moving component")
    child_rotation = Rotation.from_quat(child["orientation_xyzw"])
    contact = np.asarray(child["position"], dtype=float) + child_rotation.apply(
        joint["contact_target_local_xyz"])
    normal = child_rotation.apply(joint["contact_normal_local_xyz"])
    normal[2] = 0.0
    normal /= np.linalg.norm(normal)
    axis = child_rotation.apply(joint["handle_axis_local_xyz"])
    axis[2] = 0.0
    if np.linalg.norm(axis) < 0.2:
        # A vertical pull bar has no reliable horizontal direction. Use the
        # contact-side tangent rather than amplify tiny mesh/PCA tilt noise.
        axis = np.cross([0., 0., 1.], normal)
    axis /= np.linalg.norm(axis)
    first_ee = np.asarray(robot["end_effector_position"], dtype=float)
    aim = 0.55 * contact + 0.45 * first_ee
    if not all(np.isfinite(value) for value in (*contact, *normal, *axis, *aim)):
        raise ValueError("articulated camera has invalid contact geometry")
    cameras = {}
    measurements = {}
    for name, requirement in required_views.items():
        if name not in ROBOLAB_DROID_FIXED_CAMERAS:
            raise ValueError(f"no fixed native camera fit for {name}")
        size = [int(value) for value in requirement["size_wh"]]
        floor = float(requirement["minimum_linear_fraction"])
        side = -1.0 if name == "exterior_left" else 1.0
        native_position = contact + 0.9 * normal + side * 0.28 * axis + np.asarray([0., 0., 0.35])
        # A frontal view of a closed door can reduce the cabinet to a flat
        # panel even when the host and door have distinct colours.  Expose the
        # host's side in the second exterior view while keeping the same
        # lens, task trajectory and full-motion framing constraint.
        if name == "exterior_right":
            # The handle axis may point either way in the authored SDF.  Use
            # its handedness with the outward normal so the side exposure is
            # invariant to the cabinet's global yaw and hinge convention.
            handedness = float(np.cross(normal, axis)[2])
            yaw = math.radians(25.0 if handedness >= 0.0 else -25.0)
            turn = np.asarray([
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ])
            native_position = contact + turn @ (native_position - contact)
        drake_rotation = _look_at(native_position.copy(), aim.copy())
        opengl_rotation = drake_rotation @ np.diag([1.0, -1.0, -1.0])
        intrinsics = pinhole_intrinsics(size, DROID_NATIVE_FIXED_FOV_Y_DEG)
        selected = None
        for index in range(401):
            distance = index / 100.0
            position = native_position - distance * drake_rotation[:, 2]
            pose = np.eye(4)
            pose[:3, :3] = drake_rotation
            pose[:3, 3] = position
            full = framing_measurement(
                camera_from_table=pose, intrinsics=intrinsics,
                points_table=envelope, size_wh=size,
                margin_px=DROID_NATIVE_FRAME_MARGIN_PX)
            host = framing_measurement(
                camera_from_table=pose, intrinsics=intrinsics,
                points_table=host_box, size_wh=size,
                margin_px=DROID_NATIVE_FRAME_MARGIN_PX)
            if full["fully_in_frame"] and host["linear_fraction"] >= floor:
                selected = (distance, position, full, host)
                break
        if selected is None:
            raise ValueError(f"articulated motion cannot fit registered native view {name}")
        distance, position, full, host = selected
        cameras[name] = {
            "width": size[0], "height": size[1],
            "fov_y_deg": DROID_NATIVE_FIXED_FOV_Y_DEG,
            "position_world": position.tolist(),
            "quaternion_opengl_wxyz": [float(value) for value in np.roll(
                Rotation.from_matrix(opengl_rotation).as_quat(), 1)],
        }
        measurements[name] = {
            "backward_translation_m": distance,
            "full_motion": full,
            "initial_host": host,
            "required_minimum_linear_fraction": floor,
        }
    # The top plane is an auxiliary physical reference view. It is not sent
    # to Cosmos DROID, but it must show the door/handle and the full cabinet
    # over the joint sweep. Link-centre padding alone underestimates a tall
    # cabinet: add the complete SDF collision bounds at the measured base pose.
    parent_id = joint.get("parent_link") or next(
        (name for name in frames[0]["objects"][host_id]["link_states"]
         if name != child_id), None)
    parent_state = frames[0]["objects"][host_id]["link_states"].get(parent_id)
    if not isinstance(parent_state, dict):
        raise ValueError("articulated top camera lacks its measured host base")
    parent_rotation = Rotation.from_quat(
        parent_state.get("orientation_xyzw", [0., 0., 0., 1.]))
    host_corners = parent_rotation.apply(_box(np.asarray(low), np.asarray(high))) + np.asarray(
        parent_state["position"], dtype=float)
    top_envelope = _box(
        np.minimum(envelope.min(axis=0), host_corners.min(axis=0)),
        np.maximum(envelope.max(axis=0), host_corners.max(axis=0)))
    top_aim = 0.80 * contact + 0.20 * first_ee
    top_aim[2] += 0.20
    top_eye = contact + 0.90 * normal + np.asarray([0., 0., 1.55])
    top_size = [640, 360]
    top_intrinsics = pinhole_intrinsics(top_size, DROID_NATIVE_FIXED_FOV_Y_DEG)
    top_selected = None
    for index in range(401):
        top_rotation = _look_at(top_eye.copy(), top_aim.copy())
        top_position = top_eye - 0.01 * index * top_rotation[:, 2]
        top_pose = np.eye(4)
        top_pose[:3, :3] = top_rotation
        top_pose[:3, 3] = top_position
        top_full = framing_measurement(
            camera_from_table=top_pose, intrinsics=top_intrinsics,
            points_table=top_envelope, size_wh=top_size,
            margin_px=max(DROID_NATIVE_FRAME_MARGIN_PX, 18))
        if top_full["fully_in_frame"]:
            top_selected = (top_position, top_rotation, top_full, index)
            break
    if top_selected is None:
        raise ValueError("articulated cabinet and full motion cannot fit top view")
    top_position, top_rotation, top_full, top_index = top_selected
    top_opengl = top_rotation @ np.diag([1.0, -1.0, -1.0])
    cameras["top"] = {
        "width": top_size[0], "height": top_size[1],
        "fov_y_deg": DROID_NATIVE_FIXED_FOV_Y_DEG,
        "position_world": top_position.tolist(),
        "quaternion_opengl_wxyz": [float(value) for value in np.roll(
            Rotation.from_matrix(top_opengl).as_quat(), 1)],
    }
    measurements["top"] = {
        "backward_translation_m": top_index / 100.0,
        "full_motion": top_full,
        "contact_target_world": contact.tolist(),
        "aim_world": top_aim.tolist(),
        "auxiliary_reference_view_not_model_conditioning": True,
    }
    rig = {
        "schema_version": "worldolympiad_camera_rig.v1",
        "selection": "native_contact_side_and_full_geometry_top_v5",
        "source_trajectory_sha256": _sha(trajectory),
        "host_sdf_sha256": _sha(host_sdf),
        "articulation_binding_sha256": _sha(binding),
        "camera_measurements": measurements,
        "link_center_envelope_fraction_of_host_extent": 0.30,
        "contact_side_world": normal.tolist(),
        "contact_target_world": contact.tolist(),
        "cameras": cameras,
    }
    if output.exists():
        if json.loads(output.read_text()) != rig:
            raise ValueError("frozen articulated camera rig changed")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rig, indent=2, sort_keys=True) + "\n")
    return output
