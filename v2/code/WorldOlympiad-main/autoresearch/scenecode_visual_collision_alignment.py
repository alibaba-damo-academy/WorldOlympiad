"""Measure SceneCode visuals against runtime physics, then propose a visual repair.

The SDF collision XML is insufficient: the PyBullet importer can rebase its
link frames, while the reference renderer retains the authored visual frames.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.render_scenecode_asset_identity import (
    _corners, _geometry_bounds, _pose_matrix, _sdf_frame_transforms,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure_visual_collision_alignment(
    sdf: Path, *, link_states: dict, physics_aabbs: dict
) -> dict:
    """Compare per-link visible AABBs to independently measured PyBullet AABBs."""
    sdf = Path(sdf).resolve()
    model, authored_frames = _sdf_frame_transforms(sdf)
    root_link = model.find("link")
    if root_link is None or root_link.get("name") not in link_states:
        raise ValueError("SDF base link has no measured runtime pose")
    root_name = root_link.attrib["name"]
    root_state = link_states[root_name]
    base_world = np.eye(4)
    base_world[:3, :3] = Rotation.from_quat(
        root_state["orientation_xyzw"]).as_matrix()
    base_world[:3, 3] = root_state["position"]
    base_authored_inverse = np.linalg.inv(authored_frames[root_name])
    rows = []
    for link in model.findall("link"):
        name = link.get("name")
        if name not in link_states or name not in physics_aabbs:
            raise ValueError(f"missing runtime physics for link {name}")
        # The renderer follows SDF link/joint frames from the measured base.
        # PyBullet's per-link states may have been rebased by its importer;
        # using those states for visual geometry would hide exactly the drift
        # this admission check is intended to detect.
        world = base_world @ base_authored_inverse @ authored_frames[name]
        rotation = Rotation.from_matrix(world[:3, :3])
        points = []
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            bounds = _geometry_bounds(geometry, sdf) if geometry is not None else None
            if bounds is None:
                continue
            pose = visual.find("pose")
            if pose is not None and pose.get("relative_to") not in (None, name):
                raise ValueError("visual pose must be link-local")
            local = _pose_matrix(pose.text if pose is not None else None)
            points.append((world @ local @ _corners(bounds).T).T[:, :3])
        if not points:
            raise ValueError(f"link {name} has no measurable visual")
        visible = np.concatenate(points)
        visual_low, visual_high = visible.min(axis=0), visible.max(axis=0)
        visible_center = (visual_low + visual_high) / 2
        low, high = [np.asarray(value, dtype=float) for value in physics_aabbs[name]]
        physical_center = (low + high) / 2
        world_delta = physical_center - visible_center
        rows.append({
            "link": name,
            "visual_center_world_xyz": visible_center.tolist(),
            "visual_aabb_low_world_xyz": visual_low.tolist(),
            "visual_aabb_high_world_xyz": visual_high.tolist(),
            "physics_collision_center_world_xyz": physical_center.tolist(),
            "physics_collision_aabb_low_world_xyz": low.tolist(),
            "physics_collision_aabb_high_world_xyz": high.tolist(),
            "collision_minus_visual_link_xyz": rotation.inv().apply(world_delta).tolist(),
            "center_error_m": float(np.linalg.norm(world_delta)),
            "max_aabb_bound_error_m": float(max(
                np.max(np.abs(visual_low - low)),
                np.max(np.abs(visual_high - high)),
            )),
        })
    if len(rows) < 2:
        raise ValueError("alignment requires at least two measured links")
    deltas = np.asarray([row["collision_minus_visual_link_xyz"] for row in rows])
    common = np.median(deltas, axis=0)
    return {
        "schema_version": "scenecode_runtime_visual_collision_alignment.v1",
        "sdf_sha256": _sha(sdf),
        "links": rows,
        "common_link_local_shift_xyz": common.tolist(),
        "max_common_shift_residual_m": float(np.linalg.norm(deltas - common, axis=1).max()),
        "max_center_error_m": max(row["center_error_m"] for row in rows),
        "max_aabb_bound_error_m": max(
            row["max_aabb_bound_error_m"] for row in rows),
    }


def measured_pybullet_aabbs(physics_sdf: Path, *, root_state: dict,
                            joint_positions: dict | None = None) -> dict:
    """Get link bounds from the same SDF loader as the reference executor."""
    import pybullet as bullet

    client = bullet.connect(bullet.DIRECT)
    try:
        body = bullet.loadSDF(str(Path(physics_sdf).resolve()), physicsClientId=client)[0]
        if root_state.get("pose_frame") == "link":
            from .pybullet_frame_contract import reset_base_link_pose
            reset_base_link_pose(bullet, body, root_state["position"],
                                 root_state["orientation_xyzw"], physicsClientId=client)
        else:
            # Older frozen records stored COM here. Preserve their diagnostic
            # interpretation while all new references label link poses.
            bullet.resetBasePositionAndOrientation(
                body, root_state["position"], root_state["orientation_xyzw"],
                physicsClientId=client)
        joints = {}
        links = {bullet.getBodyInfo(body, physicsClientId=client)[0].decode(): -1}
        for index in range(bullet.getNumJoints(body, physicsClientId=client)):
            info = bullet.getJointInfo(body, index, physicsClientId=client)
            joints[info[1].decode()] = index
            links[info[12].decode()] = index
        for name, position in (joint_positions or {}).items():
            if name not in joints:
                raise ValueError(f"unknown runtime joint {name}")
            bullet.resetJointState(body, joints[name], float(position), physicsClientId=client)
        return {
            name: tuple([list(values) for values in bullet.getAABB(
                body, index, physicsClientId=client)])
            for name, index in links.items()
        }
    finally:
        bullet.disconnect(client)


def _without_visuals(sdf: Path) -> bytes:
    tree = ET.parse(sdf)
    for link in tree.findall(".//link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    return ET.tostring(tree.getroot())


def derive_common_visual_shift(
    source_sdf: Path, output_dir: Path, *, link_states: dict, physics_aabbs: dict,
    max_residual_m: float = 1e-3, aligned_tolerance_m: float = 1e-3,
    bound_tolerance_m: float = 0.015,
) -> Path:
    """Copy an asset and shift only its visual poses when all links agree."""
    source_sdf = Path(source_sdf).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir in source_sdf.parent.parents or source_sdf.parent in output_dir.parents:
        raise ValueError("candidate must be separate from source")
    before = measure_visual_collision_alignment(
        source_sdf, link_states=link_states, physics_aabbs=physics_aabbs)
    shift = np.asarray(before["common_link_local_shift_xyz"])
    if (before["max_common_shift_residual_m"] > max_residual_m
            or np.linalg.norm(shift) < aligned_tolerance_m
            or np.linalg.norm(shift) > 1.0 or not np.isfinite(shift).all()):
        raise ValueError("asset has no safe shared runtime visual displacement")
    tree = ET.parse(source_sdf, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    count = 0
    for link in tree.findall(".//link"):
        for visual in link.findall("visual"):
            pose = visual.find("pose")
            if pose is not None and pose.get("relative_to") not in (None, link.get("name")):
                raise ValueError("visual pose uses another frame")
            if pose is None:
                pose = ET.Element("pose")
                visual.insert(0, pose)
            fields = [float(value) for value in (pose.text or "0 0 0 0 0 0").split()]
            if len(fields) != 6 or not all(math.isfinite(value) for value in fields):
                raise ValueError("invalid visual pose")
            fields[:3] = (np.asarray(fields[:3]) + shift).tolist()
            pose.text = " ".join(f"{value:.12g}" for value in fields)
            count += 1
    if not count:
        raise ValueError("asset has no visuals")
    original_files = {
        str(path.relative_to(source_sdf.parent)): _sha(path)
        for path in source_sdf.parent.rglob("*") if path.is_file()
    }
    shutil.copytree(source_sdf.parent, output_dir)
    candidate = output_dir / source_sdf.name
    tree.write(candidate, encoding="utf-8", xml_declaration=True)
    if _without_visuals(source_sdf) != _without_visuals(candidate):
        shutil.rmtree(output_dir)
        raise RuntimeError("candidate changed physical SDF declarations")
    after = measure_visual_collision_alignment(
        candidate, link_states=link_states, physics_aabbs=physics_aabbs)
    if (after["max_center_error_m"] > aligned_tolerance_m
            or after["max_aabb_bound_error_m"] > bound_tolerance_m):
        shutil.rmtree(output_dir)
        raise ValueError("candidate failed runtime alignment")
    for relative, digest in original_files.items():
        if relative != source_sdf.name and _sha(output_dir / relative) != digest:
            shutil.rmtree(output_dir)
            raise RuntimeError("candidate changed source mesh or other asset")
    receipt = {
        "schema_version": "scenecode_runtime_visual_collision_repair.v1",
        "source_sdf_sha256": _sha(source_sdf),
        "candidate_sdf_sha256": _sha(candidate),
        "before": before,
        "after": after,
        "applied_link_local_shift_xyz": shift.tolist(),
        "visual_count": count,
        "physical_sdf_and_other_asset_files_unchanged": True,
        "passed": True,
    }
    (output_dir / "visual_collision_alignment_certificate.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return candidate


def certify_reference_runtime_alignment(
    *, visual_sdf: Path, physics_sdf: Path, trajectory: dict,
    host_object_id: str, base_link: str, output: Path,
    tolerance_m: float = 1e-3, bound_tolerance_m: float = 0.015,
) -> Path:
    """Fail reference admission if rendered geometry misses runtime collision."""
    visual_sdf = Path(visual_sdf).resolve()
    physics_sdf = Path(physics_sdf).resolve()
    frame = trajectory["frames"][0]
    states = frame["objects"][host_object_id]["link_states"]
    aabbs = measured_pybullet_aabbs(
        physics_sdf, root_state=states[base_link])
    measurement = measure_visual_collision_alignment(
        visual_sdf, link_states=states, physics_aabbs=aabbs)
    visual_floor_z = min(
        row["visual_aabb_low_world_xyz"][2] for row in measurement["links"])
    physics_floor_z = min(
        row["physics_collision_aabb_low_world_xyz"][2]
        for row in measurement["links"])
    receipt = {
        "schema_version": "scenecode_reference_runtime_alignment.v1",
        "visual_sdf_sha256": _sha(visual_sdf),
        "physics_sdf_sha256": _sha(physics_sdf),
        "host_object_id": host_object_id,
        "base_link": base_link,
        "measurement": measurement,
        "visual_floor_z_m": visual_floor_z,
        "physics_floor_z_m": physics_floor_z,
        "visual_physics_floor_gap_m": visual_floor_z - physics_floor_z,
        "tolerance_m": tolerance_m,
        "bound_tolerance_m": bound_tolerance_m,
        "passed": (measurement["max_center_error_m"] <= tolerance_m
                   and measurement["max_aabb_bound_error_m"] <= bound_tolerance_m
                   and abs(visual_floor_z - physics_floor_z) <= bound_tolerance_m),
    }
    output = Path(output)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not receipt["passed"]:
        raise ValueError(f"reference visual/physics alignment failed: {output}")
    return output
