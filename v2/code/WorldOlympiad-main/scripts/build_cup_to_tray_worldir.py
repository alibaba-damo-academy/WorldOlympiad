#!/usr/bin/env python3
"""Build the dedicated cup-to-tray WorldIR and Isaac compilation plan."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from autoresearch.isaac_compiler import prepare_isaac_bundle
from autoresearch.io import sha256_file, write_json
from autoresearch.models import Geometry, Pose, WorldIR
from autoresearch.robotics_embodiment_compatibility import (
    validate_parallel_gripper_cylinder,
)
from autoresearch.scenecode_importer import import_sdf_body
from autoresearch.urdf_importer import import_urdf_body


def with_visual_rgba(body, rgba: tuple[float, float, float, float]):
    """Carry the certified SceneCode look into simulator-neutral WorldIR.

    SceneCode's generated SDF currently serializes white diffuse values even
    when the GLTF/Blender material is colored. WorldIR must make the intended
    cross-renderer appearance explicit instead of relying on backend metadata.
    """

    links = []
    for link in body.links:
        visuals = [visual.model_copy(update={"rgba": rgba}) for visual in link.visuals]
        links.append(link.model_copy(update={"visuals": visuals}))
    return body.model_copy(update={"links": links})


def _copy_content_bound(source: Path, destination: Path) -> str:
    """Copy an immutable provenance artifact without retaining its source path."""

    source = source.resolve()
    digest = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise ValueError(f"provenance destination already has different content: {destination}")
        return digest
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != digest:
            raise RuntimeError(f"provenance copy changed bytes: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _portable_body(body, output: Path, mesh_sources: dict[str, Path]):
    """Replace import-time locators with owner-relative snapshots and hashes."""

    source_sha256 = body.source_sha256
    metadata = dict(body.metadata)
    if body.source_path is not None:
        body_source = Path(body.source_path).resolve()
        actual = sha256_file(body_source)
        if source_sha256 is not None and source_sha256 != actual:
            raise ValueError(f"body source SHA256 mismatch: {body_source}")
        source_sha256 = actual
    urdf_source = metadata.pop("precompiled_urdf_source", None)
    if urdf_source is not None:
        urdf_sha256 = sha256_file(Path(str(urdf_source)).resolve())
        if source_sha256 is not None and urdf_sha256 != source_sha256:
            raise ValueError("native URDF metadata and body source identities differ")
        metadata.pop("precompiled_asset_policy", None)
        metadata["native_source_contract"] = {
            "format": "urdf",
            "sha256": urdf_sha256,
            "policy": "content identity only; RoboLab supplies the native runtime embodiment",
        }

    def portable_geometry(geometry: Geometry) -> Geometry:
        if geometry.kind != "mesh":
            return geometry
        source = Path(str(geometry.path)).resolve()
        digest = sha256_file(source)
        suffix = source.suffix.lower() or ".mesh"
        relative = (Path("source_assets") / f"{digest}{suffix}").as_posix()
        if suffix == ".gltf":
            from urllib.parse import unquote
            import hashlib
            payload = json.loads(source.read_text())
            dependency_hashes = {}
            for collection in ("buffers", "images"):
                for resource in payload.get(collection, []):
                    uri = resource.get("uri")
                    if uri and not uri.startswith("data:"):
                        if "://" in uri or Path(uri).is_absolute():
                            raise ValueError("portable glTF requires local relative resources")
                        dependency_hashes[uri] = sha256_file((source.parent / unquote(uri)).resolve())
            bundle_digest = hashlib.sha256(json.dumps(
                {"document": digest, "resources": dependency_hashes}, sort_keys=True).encode()).hexdigest()
            relative = (Path("source_assets") / bundle_digest / source.name).as_posix()
            destination = output / relative
            for collection in ("buffers", "images"):
                for resource in payload.get(collection, []):
                    uri = resource.get("uri")
                    if not uri or uri.startswith("data:"):
                        continue
                    if "://" in uri or Path(uri).is_absolute():
                        raise ValueError("portable glTF requires local relative resources")
                    dependency = (source.parent / unquote(uri)).resolve()
                    name = sha256_file(dependency) + dependency.suffix
                    _copy_content_bound(dependency, destination.parent / "resources" / name)
                    resource["uri"] = "resources/" + name
            destination.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            if destination.exists() and destination.read_text() != serialized:
                raise ValueError("portable glTF resource bindings changed")
            destination.write_text(serialized)
        else:
            _copy_content_bound(source, output / Path(relative))
        previous = mesh_sources.setdefault(relative, source)
        if sha256_file(previous) != digest:
            raise ValueError(f"portable mesh identity collision: {relative}")
        return geometry.model_copy(update={"path": relative})

    links = []
    for link in body.links:
        links.append(
            link.model_copy(
                update={
                    "visuals": [portable_geometry(item) for item in link.visuals],
                    "collisions": [portable_geometry(item) for item in link.collisions],
                }
            )
        )
    return body.model_copy(
        update={
            "links": links,
            "source_path": None,
            "source_sha256": source_sha256,
            "metadata": metadata,
        }
    )


def _compiled_mesh_source_keys(world: WorldIR, collision_proxy_bodies: set[str]) -> set[str]:
    """Select only source meshes that the Isaac plan actually compiles."""

    keys: set[str] = set()
    for body in world.bodies:
        if body.name == "robot":
            continue
        for link in body.links:
            keys.update(
                str(item.path) for item in link.visuals if item.kind == "mesh"
            )
            if body.name not in collision_proxy_bodies:
                keys.update(
                    str(item.path) for item in link.collisions if item.kind == "mesh"
                )
    return keys


def _obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            if raw.startswith("v "):
                values = raw.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"prepared OBJ has no vertices: {path}")
    return points.min(axis=0), points.max(axis=0)


def _geometry_bounds(plan_root: Path, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    pose = geometry.get("pose", {})
    quat = np.asarray(pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    # SDF pose composition emits roundoff around 1e-8 even for an exact
    # identity rotation; keep rejecting real axis changes.
    if not np.allclose(quat, [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("reviewed cup-to-tray task assets require axis-aligned geometry poses")
    center = np.asarray(pose.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(geometry.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    kind = geometry["kind"]
    if kind == "mesh":
        lower, upper = _obj_bounds(plan_root / geometry["asset_path"])
        scaled = np.vstack([lower * scale, upper * scale])
        return scaled.min(axis=0) + center, scaled.max(axis=0) + center
    if kind == "cylinder":
        radius = float(geometry["radius"])
        half = float(geometry["length"]) / 2.0
        extent = np.asarray([radius, radius, half]) * scale
        return center - extent, center + extent
    if kind == "box":
        extent = np.asarray(geometry["size"], dtype=np.float64) * scale / 2.0
        return center - extent, center + extent
    if kind == "sphere":
        extent = np.full(3, float(geometry["radius"])) * scale
        return center - extent, center + extent
    raise ValueError(f"unsupported task asset geometry for axis audit: {kind}")


def _union_bounds(
    plan_root: Path,
    geometries: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    bounds = [_geometry_bounds(plan_root, geometry) for geometry in geometries]
    if not bounds:
        raise ValueError("task asset has no geometry for axis audit")
    return (
        np.vstack([item[0] for item in bounds]).min(axis=0),
        np.vstack([item[1] for item in bounds]).max(axis=0),
    )


def _bounded_inner_proxy_evidence(
    visual_bounds: tuple[np.ndarray, np.ndarray],
    collision_bounds: tuple[np.ndarray, np.ndarray],
    declaration: dict | None,
) -> dict:
    """Verify a gripper-compatible collision proxy stays inside the visual."""
    if not isinstance(declaration, dict) or declaration.get("policy") != (
        "explicit_backend_collision_proxy"
    ):
        return {"passed": False, "reason": "no explicit backend proxy"}
    proxies = declaration.get("compiled_proxies") or []
    if (declaration.get("compiled_proxy_count") != 1 or len(proxies) != 1
            or proxies[0].get("kind") != "cylinder"
            or int(declaration.get("source_worldir_collision_count") or 0) < 1):
        return {"passed": False, "reason": "proxy inventory is not one declared cylinder"}
    visual_low, visual_high = visual_bounds
    collision_low, collision_high = collision_bounds
    visual_extent = visual_high - visual_low
    collision_extent = collision_high - collision_low
    if np.any(collision_extent <= 0) or np.any(visual_extent <= 0):
        return {"passed": False, "reason": "proxy bounds are degenerate"}
    ratios = visual_extent / collision_extent
    proxy = proxies[0]
    radial = 2.0 * float(proxy.get("radius") or 0.0)
    axial = float(proxy.get("length") or 0.0)
    checks = {
        "declared_proxy_matches_collision_bounds": bool(np.allclose(
            collision_extent, [radial, radial, axial], atol=1.0e-6, rtol=0
        )),
        "proxy_enclosed_by_visible_asset": bool(
            np.all(collision_low >= visual_low - 0.005)
            and np.all(collision_high <= visual_high + 0.005)
        ),
        "bounded_radial_clearance": bool(
            np.all(ratios[:2] >= 1.0)
            and np.all(ratios[:2] <= 1.5)
            and abs(float(ratios[0] - ratios[1])) <= 0.05
        ),
        "same_vertical_axis": bool(1.0 <= ratios[2] <= 1.1),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "visual_to_collision_extent_ratio_xyz": ratios.tolist(),
        "maximum_radial_ratio": 1.5,
        "maximum_axial_ratio": 1.1,
        "containment_tolerance_m": 0.005,
    }


def certify_visual_collision_axis_alignment(plan_path: Path, output: Path) -> Path:
    """Fail closed when visible task assets are rotated relative to physics."""

    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    reports: dict[str, dict] = {}
    for body in plan["bodies"]:
        if body["name"] not in {"table", "cup", "tray"}:
            continue
        visuals: list[dict] = []
        collisions: list[dict] = []
        for link in body.get("links", []):
            link_pose = link.get("pose", {})
            if not np.allclose(
                link_pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                [0.0, 0.0, 0.0, 1.0],
                atol=1e-8,
            ) or not np.allclose(link_pose.get("xyz", [0.0, 0.0, 0.0]), 0.0, atol=1e-8):
                raise ValueError("reviewed cup-to-tray task assets require identity link poses")
            visuals.extend(link.get("visuals", []))
            collisions.extend(link.get("collisions", []))
        visual_bounds = _union_bounds(plan_path.parent, visuals)
        collision_bounds = _union_bounds(plan_path.parent, collisions)
        visual_extent = visual_bounds[1] - visual_bounds[0]
        collision_extent = collision_bounds[1] - collision_bounds[0]
        relative_error = np.abs(visual_extent - collision_extent) / np.maximum(
            collision_extent, 1e-9
        )
        direct_match = bool(np.max(relative_error) <= 0.35)
        proxy_evidence = _bounded_inner_proxy_evidence(
            visual_bounds, collision_bounds,
            (plan.get("collision_proxy_overrides") or {}).get(body["name"]),
        )
        passed = direct_match or bool(proxy_evidence["passed"])
        reports[body["name"]] = {
            "passed": passed,
            "policy": (
                "direct_extent_match" if direct_match
                else "bounded_inner_grasp_proxy" if proxy_evidence["passed"]
                else "failed"
            ),
            "proxy_enclosure": proxy_evidence,
            "visual_extent_xyz_m": visual_extent.tolist(),
            "collision_extent_xyz_m": collision_extent.tolist(),
            "relative_extent_error_xyz": relative_error.tolist(),
        }
    if set(reports) != {"table", "cup", "tray"}:
        raise ValueError("axis audit did not cover the exact task assets")
    passed = all(record["passed"] for record in reports.values())
    certificate = write_json(
        output,
        {
            "schema_version": "scenecode_isaac_visual_collision_axis.v2",
            "passed": passed,
            "isaac_compile_plan": plan_path.name,
            "isaac_compile_plan_sha256": sha256_file(plan_path),
            "maximum_relative_extent_error": 0.35,
            "bodies": reports,
        },
    )
    if not passed:
        raise ValueError(f"SceneCode visual/collision axis mismatch: {certificate}")
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--table-sdf", type=Path, required=True)
    parser.add_argument("--cup-sdf", type=Path, required=True)
    parser.add_argument("--tray-sdf", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default=None,
                        help="Frozen typed model prompt for WorldIR provenance.")
    parser.add_argument(
        "--cup-collision-policy",
        choices=("bounded_grasp_proxy", "source_collision_meshes"),
        default="bounded_grasp_proxy",
        help="Preserve a feature-grasp asset's authored collisions when a single cylinder would erase its handle or spout.",
    )
    parser.add_argument(
        "--cup-proxy-radius",
        type=float,
        default=0.0365,
        help="Isaac-only cylindrical cup collision proxy radius in metres.",
    )
    parser.add_argument(
        "--cup-proxy-length",
        type=float,
        default=0.10119048,
        help="Isaac-only cylindrical cup collision proxy height in metres.",
    )
    parser.add_argument(
        "--cup-proxy-center-z",
        type=float,
        default=None,
        help="Local proxy center height; defaults to half of --cup-proxy-length.",
    )
    parser.add_argument(
        "--gripper-max-aperture",
        type=float,
        default=0.085,
        help="Robotiq 2F-85 maximum opening in metres.",
    )
    parser.add_argument(
        "--minimum-closing-travel",
        type=float,
        default=0.008,
        help="Required total inward travel after enclosing the proxy.",
    )
    parser.add_argument(
        "--scene-id",
        default="cup_to_tray_scenecode_isaac_native_v1",
        help="Stable identity for the generated WorldIR scene.",
    )
    args = parser.parse_args()

    if not 0.01 <= args.cup_proxy_length <= 0.30:
        raise ValueError("cup proxy length must lie in [0.01, 0.30] metres")
    cup_proxy_center_z = (
        args.cup_proxy_length / 2.0
        if args.cup_proxy_center_z is None
        else args.cup_proxy_center_z
    )
    if not 0.0 <= cup_proxy_center_z <= args.cup_proxy_length:
        raise ValueError("cup proxy center z must lie within the proxy height")

    compatibility = validate_parallel_gripper_cylinder(
        radius_m=args.cup_proxy_radius,
        max_aperture_m=args.gripper_max_aperture,
        minimum_closing_travel_m=args.minimum_closing_travel,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory_source = args.trajectory.resolve()
    trajectory_path = output / "provenance" / "reference_trajectory.json"
    trajectory_sha256 = _copy_content_bound(trajectory_source, trajectory_path)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory.get("frames"), list) or not trajectory["frames"]:
        raise ValueError("reference action trajectory must contain at least one frame")
    frame0 = trajectory["frames"][0]

    def pose(record: dict) -> Pose:
        return Pose(
            xyz=tuple(float(value) for value in record["link_position"]),
            quat_xyzw=tuple(float(value) for value in record["link_orientation_xyzw"]),
        )

    table = import_sdf_body(
        args.table_sdf,
        body_name="table",
        semantic_category="furniture",
        pose=Pose(),
        static=True,
        target=False,
        metadata={"role": "support_surface", "source": "SceneCode"},
    )
    cup = import_sdf_body(
        args.cup_sdf,
        body_name="cup",
        semantic_category="ceramic",
        pose=pose(frame0["cup"]),
        static=False,
        target=True,
        support="table",
        metadata={"role": "manipulated_object", "color": "red", "source": "SceneCode"},
    )
    tray = import_sdf_body(
        args.tray_sdf,
        body_name="tray",
        semantic_category="plastic",
        pose=pose(frame0["tray"]),
        # The tray is the fixed task receptacle, not a second manipuland.
        static=True,
        target=True,
        support="table",
        metadata={
            "role": "receptacle",
            "color": "blue",
            "source": "SceneCode",
            "physics_role": "fixed_task_destination",
        },
    )
    table = with_visual_rgba(table, (0.70, 0.56, 0.39, 1.0))
    cup = with_visual_rgba(cup, (0.82, 0.06, 0.035, 1.0))
    tray = with_visual_rgba(tray, (0.42, 0.65, 0.82, 1.0))

    robot_record = frame0["robot"]
    source_robot = import_urdf_body(
        args.robot_urdf,
        body_name="robot",
        pose=Pose(),
    )
    robot = source_robot.model_copy(
        update={
            "pose": Pose(
                xyz=tuple(float(value) for value in robot_record["base_position"]),
                quat_xyzw=tuple(
                    float(value) for value in robot_record["base_orientation_xyzw"]
                ),
            ),
            "metadata": {
                **source_robot.metadata,
                "native_backend": "RoboLab DroidCfg Franka+Robotiq",
            },
        }
    )

    mesh_sources: dict[str, Path] = {}
    table = _portable_body(table, output, mesh_sources)
    cup = _portable_body(cup, output, mesh_sources)
    tray = _portable_body(tray, output, mesh_sources)
    robot = _portable_body(robot, output, mesh_sources)

    world = WorldIR(
        scene_id=args.scene_id,
        prompt=args.instruction or "Pick up the red cup and place it in the blue tray.",
        bodies=[table, cup, tray, robot],
        provenance={
            "action_trajectory_authority": {
                "path": "provenance/reference_trajectory.json",
                "sha256": trajectory_sha256,
                "frame_count": len(trajectory["frames"]),
                "initial_state_frame_index": 0,
                "role": "SceneCode action trajectory and initial object state",
            },
            "visual_assets": "SceneCode SDF visual meshes",
            "native_isaac_embodiment": "RoboLab DroidCfg",
            "cup_collision_policy": args.cup_collision_policy,
        },
    )
    world_path = world.write_json(output / "world_ir.json")
    # SceneCode's cup carries 64 PyBullet/CoACD collision pieces.  Keeping all
    # of them in PhysX creates overlapping multi-contact impulses when the
    # RobotiQ fingers close.  The Isaac compiler therefore receives one
    # explicit, backend-only grasp proxy while WorldIR identity, visual mesh,
    # mass, inertia, material, and initial pose remain unchanged.
    cup_isaac_grasp_proxy = Geometry(
        name="cup_isaac_grasp_proxy",
        kind="cylinder",
        pose=Pose(xyz=(0.0, 0.0, cup_proxy_center_z)),
        radius=args.cup_proxy_radius,
        length=args.cup_proxy_length,
    )
    proxy_bodies = ({"cup"} if args.cup_collision_policy == "bounded_grasp_proxy"
                    else set())
    plan_path = prepare_isaac_bundle(
        world,
        output / "isaac",
        native_body_names=["robot"],
        collision_proxy_overrides=(
            {"cup": [cup_isaac_grasp_proxy]} if proxy_bodies else {}
        ),
        mesh_source_overrides={
            key: mesh_sources[key]
            for key in _compiled_mesh_source_keys(world, proxy_bodies)
        },
        authority_bindings={
            "action_trajectory": {
                "path": "../provenance/reference_trajectory.json",
                "sha256": trajectory_sha256,
                "role": "SceneCode action trajectory and initial object state",
            }
        },
    )
    axis_certificate = certify_visual_collision_axis_alignment(
        plan_path,
        output / "isaac" / "visual_collision_axis_certificate.json",
    )
    compatibility_path = output / "isaac" / "embodiment_compatibility.json"
    write_json(
        compatibility_path,
        {
            **compatibility,
            "embodiment": "RoboLab DroidCfg / Robotiq 2F-85",
            "body": "cup",
            "collision_policy": args.cup_collision_policy,
            "grasp_proxy_used_only_for_feature_fit": (
                args.cup_collision_policy == "source_collision_meshes"
            ),
            "proxy_length_m": args.cup_proxy_length,
            "proxy_center_z_m": cup_proxy_center_z,
            "world_hash": world.content_hash(),
            "isaac_plan": "isaac_compile_plan.json",
            "action_trajectory": {
                "path": "../provenance/reference_trajectory.json",
                "sha256": trajectory_sha256,
            },
        },
    )
    print(
        json.dumps(
            {
                "world_ir": str(world_path),
                "isaac_plan": str(plan_path),
                "visual_collision_axis_certificate": str(axis_certificate),
                "embodiment_compatibility": str(compatibility_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
