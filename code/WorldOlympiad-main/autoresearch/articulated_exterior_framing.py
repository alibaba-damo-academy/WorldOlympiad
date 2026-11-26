"""Propose fixed exterior views from frozen physical motion and measured bounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .robotics_articulated_camera import _box, _look_at
from .robotics_native_cameras import framing_measurement, pinhole_intrinsics


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured_motion_bounds(
    *, trajectory: Path, physics_sdf: Path, host_object_id: str,
    root_link: str, joint_name: str, handle_link: str, robot_urdf: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return runtime AABB corners over every frozen reference frame."""
    import pybullet as bullet

    data = json.loads(Path(trajectory).read_text())
    frames = data["frames"]
    if not frames:
        raise ValueError("reference motion has no frames")
    initial = frames[0]["objects"][host_object_id]["link_states"][root_link]
    client = bullet.connect(bullet.DIRECT)
    try:
        body = bullet.loadSDF(str(Path(physics_sdf).resolve()), physicsClientId=client)[0]
        if initial.get("pose_frame") == "link":
            from .pybullet_frame_contract import reset_base_link_pose
            reset_base_link_pose(bullet, body, initial["position"],
                                 initial["orientation_xyzw"], physicsClientId=client)
        else:
            # Historical recordings explicitly predate corrected root-link
            # telemetry; retain their COM placement for historical diagnostics.
            bullet.resetBasePositionAndOrientation(
                body, initial["position"], initial["orientation_xyzw"],
                physicsClientId=client)
        names = {
            bullet.getJointInfo(body, index, physicsClientId=client)[12].decode(): index
            for index in range(bullet.getNumJoints(body, physicsClientId=client))
        }
        joints = {
            bullet.getJointInfo(body, index, physicsClientId=client)[1].decode(): index
            for index in range(bullet.getNumJoints(body, physicsClientId=client))
        }
        names[root_link] = -1
        if joint_name not in joints or handle_link not in names:
            raise ValueError("runtime physics lacks articulated joint or handle link")
        robot = None
        if robot_urdf is not None:
            if _sha(robot_urdf) != data["authority"]["gripper_urdf_sha256"]:
                raise ValueError("camera motion robot differs from recorded reference")
            first_robot = frames[0]["robot"]
            if first_robot.get("base_pose_frame") != "urdf_root_link":
                raise ValueError("camera motion robot lacks root-link placement")
            robot = bullet.loadURDF(str(Path(robot_urdf).resolve()),
                first_robot["base_position"], first_robot["base_orientation_xyzw"],
                useFixedBase=True, physicsClientId=client)
            robot_joints = {
                bullet.getJointInfo(robot, i, physicsClientId=client)[1].decode(): i
                for i in range(bullet.getNumJoints(robot, physicsClientId=client))}
        motion = []
        initial_host = []
        initial_handle = None
        for frame_index, frame in enumerate(frames):
            bullet.resetJointState(
                body, joints[joint_name], float(frame["articulated"]["position"]),
                physicsClientId=client)
            for name, index in names.items():
                low, high = bullet.getAABB(body, index, physicsClientId=client)
                corners = _box(np.asarray(low), np.asarray(high))
                motion.extend(corners)
                if frame_index == 0:
                    initial_host.extend(corners)
                    if name == handle_link:
                        initial_handle = corners
            motion.append(frame["robot"]["end_effector_position"])
            if robot is not None:
                for name, value in zip(frame["robot"]["joint_names"],
                                       frame["robot"]["joint_positions"], strict=True):
                    bullet.resetJointState(robot, robot_joints[name], value, physicsClientId=client)
                for i in range(-1, bullet.getNumJoints(robot, physicsClientId=client)):
                    if bullet.getCollisionShapeData(robot, i, physicsClientId=client):
                        low, high = bullet.getAABB(robot, i, physicsClientId=client)
                        motion.extend(_box(np.asarray(low), np.asarray(high)))
        if initial_handle is None:
            raise ValueError("reference has no measured handle")
        return {
            "full_motion": np.asarray(motion, dtype=float),
            "initial_host": np.asarray(initial_host, dtype=float),
            "initial_handle": np.asarray(initial_handle, dtype=float),
        }
    finally:
        bullet.disconnect(client)


def fit_measured_exterior_rig(*, parent_rig: Path, trajectory: Path,
        physics_sdf: Path, binding: Path, robot_urdf: Path, output: Path,
        minimum_handle_span_px: float = 5., margin_px: int = 8) -> Path:
    """Choose readable fixed views from actual furniture and robot motion bounds."""
    rig = json.loads(Path(parent_rig).read_text())
    joint = json.loads(Path(binding).read_text())
    data = json.loads(Path(trajectory).read_text())
    if (_sha(physics_sdf) != data["prepared_assets"]["host"]["physics_sha256"]
            or joint["host_sdf_sha256"] != data["binding"]["host_sdf_sha256"]):
        raise ValueError("camera search host differs from recorded reference")
    root_state = data["frames"][0]["objects"][joint["host_object_id"]]["link_states"][joint["parent_link"]]
    if root_state.get("pose_frame") != "link":
        raise ValueError("new camera admission requires measured root-link telemetry")
    bounds = measured_motion_bounds(trajectory=trajectory, physics_sdf=physics_sdf,
        host_object_id=joint["host_object_id"], root_link=joint["parent_link"],
        joint_name=joint["joint_name"], handle_link=joint["contact_frame_evidence"]["handle_link"],
        robot_urdf=robot_urdf)
    child = data["frames"][0]["objects"][joint["host_object_id"]]["link_states"][joint["child_link"]]
    normal = Rotation.from_quat(child["orientation_xyzw"]).apply(joint["contact_normal_local_xyz"])
    normal[2] = 0; normal /= np.linalg.norm(normal)
    tangent = np.cross([0., 0., 1.], normal)
    aim = (bounds["full_motion"].min(0) + bounds["full_motion"].max(0)) / 2
    candidate = json.loads(json.dumps(rig))
    for view, sign in (("exterior_left", -1.), ("exterior_right", 1.)):
        camera = candidate["cameras"][view]
        size = [camera["width"], camera["height"]]
        intrinsics = pinhole_intrinsics(size, camera["fov_y_deg"])
        best = None
        for yaw_degrees in (15., 30., 45.):
            yaw = np.radians(sign * yaw_degrees)
            direction = np.cos(yaw)*normal + np.sin(yaw)*tangent
            for distance in np.arange(.8, 2.51, .05):
                for elevation in (.25, .4, .55):
                    eye = aim + distance*direction + [0, 0, elevation]
                    rotation = _look_at(eye.copy(), aim.copy())
                    pose = np.eye(4); pose[:3,:3] = rotation; pose[:3,3] = eye
                    measurements = {name: framing_measurement(camera_from_table=pose,
                        intrinsics=intrinsics, points_table=points, size_wh=size,
                        margin_px=margin_px) for name, points in bounds.items()}
                    box = measurements["initial_handle"]["projected_bbox_xyxy"]
                    span = min(box[2]-box[0], box[3]-box[1])
                    if (all(row["fully_in_frame"] for row in measurements.values())
                            and span >= minimum_handle_span_px
                            and measurements["initial_host"]["linear_fraction"] >= .35
                            and (best is None or span > best[0])):
                        best = (span, eye, rotation, measurements)
        if best is None:
            raise ValueError(f"no readable full robot/host motion view for {view}")
        span, eye, rotation, measurements = best
        camera.update(position_world=eye.tolist(),
            quaternion_opengl_wxyz=np.roll(Rotation.from_matrix(rotation @ np.diag([1.,-1.,-1.])).as_quat(),1).tolist())
        candidate["camera_measurements"][view] = {
            "policy": "measured_robot_and_host_motion_max_handle_span.v1",
            **measurements, "measured_handle_span_px": span,
            "minimum_handle_span_px": minimum_handle_span_px, "margin_px": margin_px}
    candidate["selection"] = "measured_robot_and_host_motion_max_handle_span.v1"
    candidate["exterior_framing_source"] = {
        "trajectory_sha256": _sha(trajectory), "physics_sdf_sha256": _sha(physics_sdf),
        "robot_urdf_sha256": _sha(robot_urdf), "binding_sha256": _sha(binding),
        "parent_rig_sha256": _sha(parent_rig), "model_output_consulted": False}
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, indent=2, sort_keys=True)+"\n")
    return output


def propose_exterior_view(
    *, parent_rig: Path, trajectory: Path, physics_sdf: Path,
    host_object_id: str, root_link: str, joint_name: str,
    handle_link: str, view_id: str, eye_world: list[float],
    target_world: list[float], output: Path, margin_px: int = 8,
    minimum_host_fraction: float = 0.35,
    minimum_handle_span_px: float = 5.0,
) -> Path:
    """Write one immutable candidate only if measured motion fits native FOV."""
    if view_id not in ("exterior_left", "exterior_right"):
        raise ValueError("candidate must name one native exterior view")
    rig = json.loads(Path(parent_rig).read_text())
    camera = rig["cameras"][view_id]
    size = [camera["width"], camera["height"]]
    eye = np.asarray(eye_world, dtype=float)
    target = np.asarray(target_world, dtype=float)
    if eye.shape != (3,) or target.shape != (3,) or not np.isfinite([*eye, *target]).all():
        raise ValueError("invalid exterior eye or target")
    rotation = _look_at(eye.copy(), target.copy())
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = eye
    intrinsics = pinhole_intrinsics(size, camera["fov_y_deg"])
    bounds = measured_motion_bounds(
        trajectory=trajectory, physics_sdf=physics_sdf,
        host_object_id=host_object_id, root_link=root_link,
        joint_name=joint_name, handle_link=handle_link)
    measurements = {
        name: framing_measurement(
            camera_from_table=camera_pose, intrinsics=intrinsics,
            points_table=points, size_wh=size, margin_px=margin_px)
        for name, points in bounds.items()
    }
    handle = measurements["initial_handle"]["projected_bbox_xyxy"]
    handle_span = min(handle[2] - handle[0], handle[3] - handle[1])
    if (not measurements["full_motion"]["fully_in_frame"]
            or not measurements["initial_host"]["fully_in_frame"]
            or measurements["initial_host"]["linear_fraction"] < minimum_host_fraction
            or not measurements["initial_handle"]["fully_in_frame"]
            or handle_span < minimum_handle_span_px):
        raise ValueError("candidate crops motion, cabinet or handle in native FOV")
    gl_rotation = rotation @ np.diag([1.0, -1.0, -1.0])
    candidate = json.loads(json.dumps(rig))
    candidate["cameras"][view_id].update({
        "position_world": eye.tolist(),
        "quaternion_opengl_wxyz": np.roll(
            Rotation.from_matrix(gl_rotation).as_quat(), 1).tolist(),
    })
    candidate.setdefault("camera_measurements", {})[view_id] = {
        "policy": "measured_full_motion_exterior_candidate_v1",
        **measurements,
        "minimum_handle_span_px": minimum_handle_span_px,
        "measured_handle_span_px": handle_span,
        "margin_px": margin_px,
    }
    candidate["exterior_framing_source"] = {
        "trajectory_sha256": _sha(trajectory),
        "physics_sdf_sha256": _sha(physics_sdf),
        "parent_rig_sha256": _sha(parent_rig),
        "candidate_view": view_id,
        "model_output_consulted": False,
    }
    candidate["selection"] = "measured_full_motion_exterior_candidate_v1"
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return output
