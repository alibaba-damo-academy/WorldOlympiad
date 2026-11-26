"""Derive the DROID CAE frame-zero registration from frozen reference physics."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import cv2

from .robotics_views import sha256_file


def registration_for_cosmos_pair(pair, wam_output_manifest: Path) -> tuple[dict, Path]:
    """Bind the CAE robot state to this exact model/reference pair."""
    from .robotics_contracts import WAMInputManifest, WAMOutputManifest
    from .robotics_reference_materializer import _robot_asset
    from .robotics_views import load_reference_manifest, resolve_artifact

    wam_path = Path(wam_output_manifest).resolve()
    wam = WAMOutputManifest.model_validate_json(wam_path.read_text(encoding="utf-8"))
    if wam.wam != "cosmos3_nano_droid" or wam.case_id != pair.case_id:
        raise ValueError("Cosmos CAE registration belongs to another case")
    input_path = resolve_artifact(wam_path, wam.input_manifest)
    spec = WAMInputManifest.model_validate_json(input_path.read_text(encoding="utf-8"))
    ref_path = Path(spec.reference_manifest).resolve()
    if spec.reference_manifest_sha256 != sha256_file(ref_path):
        raise ValueError("Cosmos CAE reference manifest digest changed")
    ref_path, reference = load_reference_manifest(ref_path)
    if reference.case_id != pair.case_id:
        raise ValueError("Cosmos CAE reference belongs to another task")
    view = next(
        (item for item in reference.views if item.view_id == pair.reference_view),
        None,
    )
    if view is None or spec.formal_reference_transform != "identity_exact":
        raise ValueError("Cosmos CAE needs its frozen identity-camera reference")
    source_video = resolve_artifact(ref_path, view.video)
    paired_video = Path(pair.reference_video).resolve()
    if (sha256_file(source_video) != view.sha256
            or sha256_file(paired_video) != pair.reference_sha256):
        raise ValueError("Cosmos CAE reference video digest changed")
    source_capture = cv2.VideoCapture(str(source_video))
    paired_capture = cv2.VideoCapture(str(paired_video))
    frame_mae = []
    try:
        while True:
            source_ok, source_frame = source_capture.read()
            paired_ok, paired_frame = paired_capture.read()
            if source_ok != paired_ok:
                raise ValueError("Cosmos CAE paired reference timeline changed")
            if not source_ok:
                break
            if source_frame.shape != paired_frame.shape:
                raise ValueError("Cosmos CAE paired reference pixels changed size")
            frame_mae.append(float(np.abs(
                source_frame.astype(np.float32)
                - paired_frame.astype(np.float32)
            ).mean()))
    finally:
        source_capture.release()
        paired_capture.release()
    if (len(frame_mae) != view.frame_count
            or max(frame_mae, default=float("inf")) > 3.0):
        raise ValueError("Cosmos CAE paired reference differs beyond H.264 re-encode")
    trajectory_path = resolve_artifact(ref_path, reference.trajectory_json)
    if sha256_file(trajectory_path) != reference.trajectory_json_sha256:
        raise ValueError("Cosmos CAE reference trajectory digest changed")
    _, urdf = _robot_asset(ref_path.parents[2])
    import json
    registration = register_droid_frame_zero(
        json.loads(trajectory_path.read_text(encoding="utf-8")), urdf
    )
    registration.update({
        "reference_manifest_sha256": sha256_file(ref_path),
        "reference_trajectory_sha256": sha256_file(trajectory_path),
        "paired_reference_video_sha256": pair.reference_sha256,
        "paired_reference_maximum_frame_mae_8bit": max(frame_mae),
    })
    return registration, urdf


DROID_CAE_TARGET_LINK = "base_link"


def register_droid_frame_zero(trajectory: dict, robot_urdf: Path) -> dict:
    """Verify the published robot pose by independent PyBullet FK."""
    robot_urdf = Path(robot_urdf).resolve()
    frames = trajectory.get("frames") or []
    if not frames or int(frames[0].get("frame", -1)) != 0:
        raise ValueError("DROID registration requires immutable frame zero")
    controller_droid = (
        (trajectory.get("controller") or {}).get("robot_embodiment")
        == "robolab_droid")
    articulated_droid = (
        str((trajectory.get("authority") or {}).get("robot", "")).startswith(
            "RoboLab DROID Franka+Robotiq")
        and (trajectory.get("authority") or {}).get("gripper_urdf_sha256")
        == sha256_file(robot_urdf))
    if not (controller_droid or articulated_droid):
        raise ValueError("DROID registration belongs to another embodiment")
    robot = frames[0].get("robot") or {}
    names = robot.get("joint_names") or []
    values = robot.get("joint_positions") or []
    if len(names) != len(values) or len(names) != len(set(names)):
        raise ValueError("reference robot joint state is incomplete")
    positions = dict(zip(names, values, strict=True))
    base_position = np.asarray(robot.get("base_position"), dtype=np.float64)
    base_quaternion = np.asarray(robot.get("base_orientation_xyzw"), dtype=np.float64)
    measured_ee = np.asarray(robot.get("end_effector_position"), dtype=np.float64)
    if (base_position.shape != (3,) or base_quaternion.shape != (4,)
            or measured_ee.shape != (3,) or not all(
                np.isfinite(value).all()
                for value in (base_position, base_quaternion, measured_ee)
            )):
        raise ValueError("reference robot frame-zero pose is invalid")
    root = ET.parse(robot_urdf).getroot()
    by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or not child.get("link"):
            raise ValueError("robot URDF joint has no child link")
        by_child[child.get("link")] = joint
    chain = []
    current = DROID_CAE_TARGET_LINK
    seen = set()
    while current in by_child:
        if current in seen:
            raise ValueError("robot URDF target chain is cyclic")
        seen.add(current)
        joint = by_child[current]
        if joint.get("type") != "fixed":
            chain.append(joint)
        current = joint.find("parent").get("link")
    chain.reverse()
    joint_names = [joint.get("name") for joint in chain]
    if not joint_names or any(name not in positions for name in joint_names):
        raise ValueError("reference joint state does not cover target link chain")
    limits = {}
    for joint in chain:
        limit = joint.find("limit")
        if limit is None:
            raise ValueError("DROID target chain lacks native joint limits")
        limits[joint.get("name")] = [
            float(limit.get("lower")), float(limit.get("upper"))
        ]
    import pybullet as p
    connection = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(
            str(robot_urdf),
            basePosition=base_position.tolist(),
            baseOrientation=base_quaternion.tolist(),
            useFixedBase=True,
        )
        target_index = None
        for index in range(p.getNumJoints(body)):
            info = p.getJointInfo(body, index)
            joint_name = info[1].decode()
            link_name = info[12].decode()
            if joint_name in positions:
                p.resetJointState(body, index, float(positions[joint_name]))
            if link_name == DROID_CAE_TARGET_LINK:
                target_index = index
        if target_index is None:
            raise ValueError("DROID URDF has no registered end-effector link")
        predicted_ee = np.asarray(
            p.getLinkState(body, target_index, computeForwardKinematics=True)[4],
            dtype=np.float64,
        )
    finally:
        p.disconnect(connection)
    residual = float(np.linalg.norm(predicted_ee - measured_ee))
    if not math.isfinite(residual) or residual > 1.0e-4:
        raise ValueError(f"DROID frame-zero FK differs from reference: {residual:.6g} m")
    return {
        "schema": "dream2flow_robot_registration_v1",
        "initialization_kind": "native_frame_zero",
        "source_urdf": str(robot_urdf),
        "source_urdf_sha256": sha256_file(robot_urdf),
        "target_link_name": DROID_CAE_TARGET_LINK,
        "root_pose_xyzw": [*base_position.tolist(), *base_quaternion.tolist()],
        "initial_joint_positions": {
            name: float(positions[name]) for name in joint_names
        },
        "native_joint_limits": limits,
        "native_target_position": measured_ee.tolist(),
        "initial_pose_validation_position": predicted_ee.tolist(),
        "fk_residual_m": residual,
        "motion_evidence_used": "frozen physical reference frame zero only",
    }
