"""WorldIR -> self-contained Isaac USD compilation.

The preparation phase is ordinary Python and produces a content-addressed
``isaac_compile_plan.v1`` bundle.  Materialization requires an Isaac/Kit
runtime because OpenUSD and PhysX schemas are supplied by Isaac Sim.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

import numpy as np

from .compiler import _default_inertial, _prepare_mesh, _principalize
from .io import sha256_file, write_json
from .math3d import compose
from .models import Geometry, Pose, RepairRecord, WorldIR


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _geometry_payload(
    geometry: Geometry,
    *,
    destination: Path,
    body_name: str,
    link_name: str,
    role: str,
    repairs: list[RepairRecord],
    mesh_source_overrides: Mapping[str, Path],
    consumed_mesh_source_overrides: set[str],
    preserve_visual_materials: bool = False,
) -> dict[str, Any]:
    payload = geometry.model_dump(mode="json", exclude_none=True)
    if geometry.kind != "mesh":
        return payload
    source_key = str(geometry.path)
    source = mesh_source_overrides.get(source_key)
    if source is not None:
        consumed_mesh_source_overrides.add(source_key)
        source = source.resolve()
    else:
        source = Path(source_key).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"WorldIR mesh source is missing: {source}")
    asset_dir = destination / "assets" / _safe_name(body_name) / _safe_name(link_name) / role
    asset_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(source)
    repair_start = len(repairs)
    prepared_name = _prepare_mesh(
        geometry.model_copy(update={"path": str(source)}),
        asset_dir,
        visual=role == "visual",
        repairs=repairs,
        attempt=1,
    )
    prepared = asset_dir / prepared_name
    for index in range(repair_start, len(repairs)):
        repair = repairs[index]
        repairs[index] = repair.model_copy(
            update={
                "target": f"{body_name}/{link_name}/{role}/{geometry.name}",
                "detail": (
                    f"{repair.operation} completed from content-bound mesh "
                    f"sha256:{source_sha256}"
                ),
            }
        )
    # ``path`` is an import-time locator and may point into a disposable
    # SceneCode scratch tree.  The durable plan carries only its content
    # identity plus the owner-relative prepared asset used by materialization.
    payload.pop("path", None)
    payload["source_sha256"] = source_sha256
    payload["source_format"] = source.suffix.lower().removeprefix(".") or "unknown"
    payload["asset_path"] = prepared.relative_to(destination).as_posix()
    payload["asset_sha256"] = sha256_file(prepared)
    if (preserve_visual_materials and role == "visual"
            and source.suffix.lower() in {".gltf", ".glb"}):
        blender = os.environ.get("BLENDER_BIN") or shutil.which("blender")
        if not blender:
            raise RuntimeError("material-preserving Isaac export requires BLENDER_BIN")
        native_dir = asset_dir / f"{_safe_name(geometry.name)}_native"
        native_dir.mkdir(exist_ok=True)
        native_usd = native_dir / "asset.usdc"
        converter = Path(__file__).parents[1] / "scripts/blender_gltf_to_usd.py"
        subprocess.run(
            [blender, "--background", "--factory-startup", "--python", str(converter),
             "--", "--input", str(source), "--output", str(native_usd)],
            check=True, capture_output=True, text=True, timeout=180,
        )
        payload["native_visual_asset"] = {
            "path": native_usd.relative_to(destination).as_posix(),
            "sha256": sha256_file(native_usd),
            "source_sha256": source_sha256,
            "policy": "declared_gltf_scene_materials_uv_textures",
            "dependencies": [
                {"path": file.relative_to(destination).as_posix(),
                 "sha256": sha256_file(file)}
                for file in sorted(native_dir.rglob("*")) if file.is_file()
            ],
        }
    # OBJ companions preserve material/texture references when Blender emitted
    # them. They are part of the bundle even though the initial Isaac compiler
    # uses deterministic WorldIR RGBA as its cross-renderer material contract.
    companions = sorted(
        path
        for path in asset_dir.iterdir()
        if path.is_file() and path != prepared
    )
    payload["companion_assets"] = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in companions
    ]
    return payload


def _body_source_identity(source_path: str | None, declared_sha256: str | None) -> str | None:
    """Bind a source body without serializing its import-time filesystem path."""

    if source_path is None:
        return declared_sha256
    source = Path(source_path).resolve()
    if not source.is_file():
        if declared_sha256 is None:
            raise FileNotFoundError(f"WorldIR body source is missing: {source}")
        return declared_sha256
    actual = sha256_file(source)
    if declared_sha256 is not None and actual != declared_sha256:
        raise ValueError(f"WorldIR body source SHA256 mismatch: {source}")
    return actual


def _validated_authority_bindings(
    destination: Path,
    bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Validate owner-relative, content-bound external authorities.

    ``destination`` owns the compile plan.  The enclosing backend directory is
    the portability boundary, so a binding may use one ``..`` component to
    reach a sibling provenance artifact but may never escape that boundary.
    """

    result: dict[str, dict[str, str]] = {}
    bundle_root = destination.parent.resolve()
    for name, raw in sorted(bindings.items()):
        if not name or set(raw) != {"path", "sha256", "role"}:
            raise ValueError(
                "authority bindings require exactly path, sha256, and role"
            )
        relative = str(raw["path"])
        posix = PurePosixPath(relative)
        windows = PureWindowsPath(relative)
        if (
            not relative
            or "\\" in relative
            or posix.is_absolute()
            or windows.is_absolute()
            or posix.as_posix() != relative
        ):
            raise ValueError(f"authority binding path must be normalized and relative: {relative!r}")
        resolved = (destination / Path(*posix.parts)).resolve()
        try:
            resolved.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(f"authority binding escapes backend bundle: {relative!r}") from exc
        expected = str(raw["sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"authority binding has invalid SHA256: {name!r}")
        if not resolved.is_file():
            raise FileNotFoundError(f"authority artifact is missing: {resolved}")
        if sha256_file(resolved) != expected:
            raise ValueError(f"authority artifact SHA256 mismatch: {resolved}")
        role = str(raw["role"]).strip()
        if not role:
            raise ValueError(f"authority binding role is empty: {name!r}")
        result[name] = {"path": relative, "sha256": expected, "role": role}
    return result


def prepare_isaac_bundle(
    world: WorldIR,
    output_dir: str | Path,
    *,
    native_body_names: Iterable[str] = (),
    collision_proxy_overrides: dict[str, list[Geometry]] | None = None,
    mesh_source_overrides: Mapping[str, str | Path] | None = None,
    authority_bindings: Mapping[str, Mapping[str, str]] | None = None,
    preserve_visual_materials: bool = False,
) -> Path:
    """Prepare a deterministic Isaac compilation plan from one WorldIR.

    ``native_body_names`` reserves embodiments such as RoboLab's DROID robot
    for the model adapter. Every other body is compiled from WorldIR. Reserved
    bodies remain in provenance and receive explicit ``native_runtime`` status;
    they are never silently dropped.
    """

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    native = set(native_body_names)
    collision_proxies = collision_proxy_overrides or {}
    mesh_sources = {
        str(key): Path(value) for key, value in (mesh_source_overrides or {}).items()
    }
    consumed_mesh_sources: set[str] = set()
    validated_authorities = _validated_authority_bindings(
        destination, authority_bindings or {}
    )
    unknown = native - {body.name for body in world.bodies}
    if unknown:
        raise ValueError(f"native Isaac body overrides are not in WorldIR: {sorted(unknown)}")
    unknown_proxies = set(collision_proxies) - {body.name for body in world.bodies}
    if unknown_proxies:
        raise ValueError(
            f"Isaac collision proxy overrides are not in WorldIR: {sorted(unknown_proxies)}"
        )
    native_proxies = native & set(collision_proxies)
    if native_proxies:
        raise ValueError(
            "native Isaac bodies cannot also declare compiler collision proxies: "
            f"{sorted(native_proxies)}"
        )
    repairs: list[RepairRecord] = []
    bodies: list[dict[str, Any]] = []
    expected_counts = {"visual": 0, "collision": 0}
    authored_counts = {"visual": 0, "collision": 0}
    capability_notes: list[dict[str, Any]] = []
    collision_proxy_contract: dict[str, dict[str, Any]] = {}

    for body in world.bodies:
        effective_pose = compose(body.pose, body.internal_model_pose)
        body_source_sha256 = _body_source_identity(
            body.source_path, body.source_sha256
        )
        if body.name in native:
            bodies.append(
                {
                    "name": body.name,
                    "semantic_category": body.semantic_category,
                    "status": "native_runtime",
                    "pose": effective_pose.model_dump(mode="json"),
                    "source_sha256": body_source_sha256,
                    "metadata": body.metadata,
                }
            )
            continue

        if not body.static and len(body.links) != 1:
            from .isaac_articulation import floating_tree_root
            # Fixed subtrees must already have been losslessly merged. Validate
            # before emitting assets, never freeze a movable mechanism.
            floating_tree_root(body.model_dump(mode="json"))
            if any(link.inertial is None or link.inertial.mass <= 0 for link in body.links):
                raise ValueError("floating articulation requires measured positive link masses")
        body_collision_proxy = collision_proxies.get(body.name)
        if body_collision_proxy is not None:
            if len(body.links) != 1:
                raise ValueError(
                    f"Isaac collision proxy for {body.name!r} requires exactly one link"
                )
            if not body_collision_proxy:
                raise ValueError(f"Isaac collision proxy for {body.name!r} is empty")
            source_count = sum(len(link.collisions) for link in body.links)
            collision_proxy_contract[body.name] = {
                "source_worldir_collision_count": source_count,
                "compiled_proxy_count": len(body_collision_proxy),
                "policy": "explicit_backend_collision_proxy",
                "compiled_proxies": [
                    proxy.model_dump(mode="json", exclude_none=True)
                    for proxy in body_collision_proxy
                ],
            }
            capability_notes.append(
                {
                    "body": body.name,
                    "feature": "collision_geometry",
                    "status": "explicit_backend_proxy",
                    "detail": (
                        f"compiled {len(body_collision_proxy)} declared Isaac grasp proxy geometry "
                        f"instead of {source_count} source WorldIR collision parts; visuals, pose, "
                        "mass, inertia, material, and WorldIR identity are unchanged"
                    ),
                }
            )
        body_payload: dict[str, Any] = {
            "name": body.name,
            "prim_name": _safe_name(body.name),
            "semantic_category": body.semantic_category,
            "status": "worldir_compiled",
            "pose": effective_pose.model_dump(mode="json"),
            "static": body.static,
            "target": body.target,
            "support": body.support,
            "affordances": body.affordances,
            "material": body.material.model_dump(mode="json"),
            "source_sha256": body_source_sha256,
            "metadata": body.metadata,
            "links": [],
            "joints": [joint.model_dump(mode="json", exclude_none=True) for joint in body.joints],
        }
        if body.joints:
            capability_notes.append(
                {
                    "body": body.name,
                    "feature": "articulation",
                    "status": "dynamic_usd_joints",
                    "detail": (
                        "joint topology is preserved and materialized as executable USD "
                        "Physics joints; fixed joints remain structural"
                    ),
                }
            )
        for link in body.links:
            inertial = link.inertial or _default_inertial(link, body.semantic_category)
            if link.inertial is None:
                repairs.append(
                    RepairRecord(
                        attempt=2,
                        operation="inject_inertial",
                        target=f"{body.name}/{link.name}",
                        detail="used the same deterministic category prior as the PyBullet compiler",
                    )
                )
            inertial, inertial_pose, changed = _principalize(inertial)
            if changed:
                repairs.append(
                    RepairRecord(
                        attempt=2,
                        operation="principalize_inertia",
                        target=f"{body.name}/{link.name}",
                        detail="diagonalized the symmetric inertia tensor for USD PhysicsMassAPI",
                    )
                )
            visuals = [
                _geometry_payload(
                    geometry,
                    destination=destination,
                    body_name=body.name,
                    link_name=link.name,
                    role="visual",
                    repairs=repairs,
                    mesh_source_overrides=mesh_sources,
                    consumed_mesh_source_overrides=consumed_mesh_sources,
                    preserve_visual_materials=preserve_visual_materials,
                )
                for geometry in link.visuals
            ]
            collision_geometries = (
                body_collision_proxy
                if body_collision_proxy is not None
                else link.collisions
            )
            collisions = [
                _geometry_payload(
                    geometry,
                    destination=destination,
                    body_name=body.name,
                    link_name=link.name,
                    role="collision",
                    repairs=repairs,
                    mesh_source_overrides=mesh_sources,
                    consumed_mesh_source_overrides=consumed_mesh_sources,
                )
                for geometry in collision_geometries
            ]
            expected_counts["visual"] += len(link.visuals)
            expected_counts["collision"] += len(collision_geometries)
            authored_counts["visual"] += len(visuals)
            authored_counts["collision"] += len(collisions)
            body_payload["links"].append(
                {
                    "name": link.name,
                    "prim_name": _safe_name(link.name),
                    "pose": link.pose.model_dump(mode="json"),
                    "inertial": inertial.model_dump(mode="json"),
                    "inertial_pose": inertial_pose.model_dump(mode="json"),
                    "visuals": visuals,
                    "collisions": collisions,
                }
            )
        bodies.append(body_payload)

    unused_mesh_sources = set(mesh_sources) - consumed_mesh_sources
    if unused_mesh_sources:
        raise ValueError(
            "unused WorldIR mesh source overrides: " f"{sorted(unused_mesh_sources)}"
        )

    if authored_counts != expected_counts:
        raise RuntimeError(
            f"Isaac preparation dropped geometry: expected={expected_counts}, "
            f"authored={authored_counts}"
        )
    plan = {
        "schema_version": "isaac_compile_plan.v1",
        "backend": "isaac_lab_usd",
        "world_schema_version": world.schema_version,
        "world_hash": world.content_hash(),
        "units": world.units,
        "up_axis": world.up_axis,
        "scene_id": world.scene_id,
        "authority_bindings": validated_authorities,
        "bodies": bodies,
        "native_body_names": sorted(native),
        "collision_proxy_overrides": collision_proxy_contract,
        "geometry_counts": authored_counts,
        "material_contract": "WorldIR RGBA + contact material; source companions retained",
        "mesh_collision_approximation": "convexHull",
        "capability_notes": capability_notes,
        "repairs": [repair.model_dump(mode="json") for repair in repairs],
    }
    plan_path = write_json(destination / "isaac_compile_plan.json", plan)
    write_json(
        destination / "compile_report.json",
        {
            "schema_version": "simready_compile_report.v2",
            "backend": "isaac_lab_usd",
            "world_hash": world.content_hash(),
            "status": "repaired_pass" if repairs else "pass",
            "geometry_counts": authored_counts,
            "native_body_names": sorted(native),
            "authority_bindings": validated_authorities,
            "repairs": plan["repairs"],
            "capability_notes": capability_notes,
            "semantic_status": (
                "partial"
                if any(
                    note.get("status") in {"frozen_geometry", "unsupported"}
                    for note in capability_notes
                )
                else "complete"
            ),
            "materialization_required": True,
            "materialization_runtime": "Isaac Sim / OpenUSD",
        },
    )
    return plan_path


def _obj_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("v "):
                values = line.split()
                vertices.append(tuple(float(values[index]) for index in range(1, 4)))
            elif line.startswith("f "):
                refs = line.split()[1:]
                indices: list[int] = []
                for ref in refs:
                    raw_index = int(ref.split("/", maxsplit=1)[0])
                    indices.append(raw_index - 1 if raw_index > 0 else len(vertices) + raw_index)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(triangles):
        raise ValueError(f"OBJ has no usable triangles: {path}")
    return points, triangles


def materialize_isaac_usd(
    plan_path: str | Path,
    output_path: str | Path,
    *,
    visual_asset_manifest_path: str | Path | None = None,
    fixed_articulation_bodies: tuple[str, ...] = (),
) -> Path:
    """Materialize one prepared plan inside an initialized Isaac runtime."""

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "Isaac USD materialization requires an initialized Isaac Sim runtime"
        ) from exc

    plan_file = Path(plan_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "isaac_compile_plan.v1":
        raise ValueError(f"unsupported Isaac compile plan: {plan_file}")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    native_visual_assets: dict[str, dict[str, Any]] = {}
    visual_manifest: dict[str, Any] | None = None
    if visual_asset_manifest_path is not None:
        manifest_file = Path(visual_asset_manifest_path).resolve()
        visual_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if visual_manifest.get("schema_version") != "scenecode_isaac_visual_assets.v1":
            raise ValueError(f"unsupported SceneCode visual asset manifest: {manifest_file}")
        for asset in visual_manifest.get("assets", []):
            name = str(asset["name"])
            usd_path = Path(str(asset["usd_path"]))
            if not usd_path.is_absolute():
                usd_path = (manifest_file.parent / usd_path).resolve()
            if not usd_path.is_file():
                raise FileNotFoundError(f"native visual USD is missing for {name}: {usd_path}")
            if sha256_file(usd_path) != asset["usd_sha256"]:
                raise ValueError(f"native visual USD hash mismatch for {name}: {usd_path}")
            native_visual_assets[name] = {**asset, "resolved_usd_path": str(usd_path)}
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world_prim)
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
    scene_root = UsdGeom.Xform.Define(stage, "/World/Scene")
    scene_root.GetPrim().SetCustomData(
        {"worldir": {"sha256": str(plan["world_hash"])}}
    )
    material_root = UsdGeom.Scope.Define(stage, "/World/Materials")
    _ = material_root

    authored_counts = {"visual": 0, "collision": 0}
    body_prim_paths: dict[str, str] = {}
    articulation_joint_paths: list[str] = []

    def set_pose(xformable: Any, pose: dict[str, Any], scale: list[float] | None = None) -> None:
        xyz = pose.get("xyz", [0.0, 0.0, 0.0])
        quat = pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        xformable.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        x, y, z, w = quat
        xformable.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        if scale is not None:
            xformable.AddScaleOp().Set(Gf.Vec3f(*scale))

    def apply_rgba(gprim: Any, rgba: list[float]) -> None:
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgba[:3])])
        gprim.CreateDisplayOpacityAttr().Set([float(rgba[3])])

    def define_geometry(
        parent_path: str,
        geometry: dict[str, Any],
        *,
        role: str,
        native_visual_usd: str | None,
        fallback_rgba: list[float],
        physics_material: Any,
    ) -> None:
        nonlocal authored_counts
        geometry_path = f"{parent_path}/{_safe_name(geometry['name'])}"
        holder = UsdGeom.Xform.Define(stage, geometry_path)
        set_pose(holder, geometry.get("pose", {}), geometry.get("scale"))
        shape_path = f"{geometry_path}/Shape"
        kind = geometry["kind"]
        asset = geometry.get("native_visual_asset") if role == "visual" else None
        if asset is not None:
            if asset.get("source_sha256") != geometry["source_sha256"]:
                raise ValueError("native visual material asset has a different mesh source")
            for dependency in asset["dependencies"]:
                resource = (plan_file.parent / dependency["path"]).resolve()
                resource.relative_to(plan_file.parent)
                if sha256_file(resource) != dependency["sha256"]:
                    raise ValueError(f"native visual dependency changed: {resource}")
            native_visual_usd = str((plan_file.parent / asset["path"]).resolve())
            if sha256_file(Path(native_visual_usd)) != asset["sha256"]:
                raise ValueError("native visual USD changed")
        if role == "visual" and kind == "mesh" and native_visual_usd is not None:
            # Compose the complete Isaac-converted SceneCode asset.  Its mesh
            # hierarchy, UVs, material bindings and texture references remain
            # intact under this transformed WorldIR geometry holder.
            shape = UsdGeom.Xform.Define(stage, shape_path)
            shape.GetPrim().GetReferences().AddReference(
                os.path.relpath(native_visual_usd, output.parent)
            )
            shape.GetPrim().SetCustomData(
                {"worldir": {"visualSource": "scenecode_native_usd"}}
            )
        elif kind == "mesh":
            points, triangles = _obj_triangles(plan_file.parent / geometry["asset_path"])
            shape = UsdGeom.Mesh.Define(stage, shape_path)
            shape.CreatePointsAttr(points.tolist())
            shape.CreateFaceVertexCountsAttr([3] * len(triangles))
            shape.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
            shape.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        elif kind == "box":
            shape = UsdGeom.Cube.Define(stage, shape_path)
            shape.CreateSizeAttr(1.0)
            UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*geometry["size"]))
        elif kind == "sphere":
            shape = UsdGeom.Sphere.Define(stage, shape_path)
            shape.CreateRadiusAttr(float(geometry["radius"]))
        elif kind == "cylinder":
            shape = UsdGeom.Cylinder.Define(stage, shape_path)
            shape.CreateRadiusAttr(float(geometry["radius"]))
            shape.CreateHeightAttr(float(geometry["length"]))
            shape.CreateAxisAttr().Set(UsdGeom.Tokens.z)
        else:
            raise ValueError(f"unsupported WorldIR geometry kind: {kind}")
        if role == "visual":
            if kind == "mesh" and native_visual_usd is None:
                # SceneCode/GLTF assets may contain thin surfaces whose
                # triangle winding is not consistent after axis conversion.
                # Blender renders these assets two-sided; preserve that visual
                # contract in USD instead of dropping tabletops/tray panels.
                shape.CreateDoubleSidedAttr().Set(True)
            if native_visual_usd is None:
                rgba = geometry.get("rgba") or fallback_rgba
                apply_rgba(shape, rgba)
        else:
            # Collision geometry is a physics-only representation. Leaving it
            # visible causes convex-hull/default-gray meshes to cover the
            # authored SceneCode visual mesh in native Isaac RTX observations.
            UsdGeom.Imageable(shape.GetPrim()).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.invisible
            )
            collision = UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            collision.CreateCollisionEnabledAttr(True)
            if kind == "mesh":
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim())
                mesh_collision.GetApproximationAttr().Set("convexHull")
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                physics_material, materialPurpose="physics"
            )
        authored_counts[role] += 1

    for body in plan["bodies"]:
        if body["status"] == "native_runtime":
            continue
        body_path = f"/World/Scene/{body['prim_name']}"
        body_prim_paths[body["name"]] = body_path
        body_xform = UsdGeom.Xform.Define(stage, body_path)
        set_pose(body_xform, body["pose"])
        body_prim = body_xform.GetPrim()
        body_prim.SetCustomData(
            {
                "worldir": {
                    "semanticCategory": body["semantic_category"],
                    "target": bool(body["target"]),
                }
            }
        )

        material_path = f"/World/Materials/{body['prim_name']}"
        material = UsdShade.Material.Define(stage, material_path)
        physics_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        contact = body["material"]
        physics_api.CreateStaticFrictionAttr().Set(float(contact["lateral_friction"]))
        physics_api.CreateDynamicFrictionAttr().Set(float(contact["lateral_friction"]))
        physics_api.CreateRestitutionAttr().Set(float(contact["restitution"]))

        fallback_rgba = [0.55, 0.55, 0.55, 1.0]
        native_visual = native_visual_assets.get(body["name"])
        native_visual_usd = (
            str(native_visual["resolved_usd_path"]) if native_visual is not None else None
        )
        if native_visual_usd is not None:
            visual_count = sum(len(link["visuals"]) for link in body["links"])
            if visual_count != 1:
                raise ValueError(
                    f"native SceneCode visual asset for {body['name']!r} requires exactly "
                    f"one WorldIR visual geometry, found {visual_count}"
                )
        link_prim_paths: dict[str, str] = {}
        movable_children = {
            str(joint["child"])
            for joint in body.get("joints", [])
            if str(joint.get("type")) != "fixed"
        }
        for link in body["links"]:
            link_path = f"{body_path}/{link['prim_name']}"
            link_prim_paths[link["name"]] = link_path
            link_xform = UsdGeom.Xform.Define(stage, link_path)
            set_pose(link_xform, link["pose"])
            for visual in link["visuals"]:
                define_geometry(
                    f"{link_path}/Visuals",
                    visual,
                    role="visual",
                    native_visual_usd=native_visual_usd,
                    fallback_rgba=fallback_rgba,
                    physics_material=material,
                )
            for collision in link["collisions"]:
                define_geometry(
                    f"{link_path}/Collisions",
                    collision,
                    role="collision",
                    native_visual_usd=None,
                    fallback_rgba=fallback_rgba,
                    physics_material=material,
                )
            if link["name"] in movable_children or (not body["static"] and body.get("joints")):
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(link_xform.GetPrim())
                rigid_api.CreateRigidBodyEnabledAttr(True)
                rigid_api.CreateKinematicEnabledAttr(False)
                inertial = link["inertial"]
                inertial_pose = link["inertial_pose"]
                mass_api = UsdPhysics.MassAPI.Apply(link_xform.GetPrim())
                mass_api.CreateMassAttr().Set(float(inertial["mass"]))
                mass_api.CreateCenterOfMassAttr().Set(
                    Gf.Vec3f(*inertial_pose["xyz"])
                )
                ixx, iyy, izz, _ixy, _ixz, _iyz = inertial["inertia"]
                mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, iyy, izz))
                quat = inertial_pose["quat_xyzw"]
                mass_api.CreatePrincipalAxesAttr().Set(
                    Gf.Quatf(quat[3], Gf.Vec3f(quat[0], quat[1], quat[2]))
                )
        for joint in body.get("joints", []):
            if str(joint.get("type")) == "fixed":
                continue
            parent_name = str(joint["parent"])
            child_name = str(joint["child"])
            if parent_name not in link_prim_paths or child_name not in link_prim_paths:
                raise ValueError(
                    f"Isaac articulation joint references unknown links: {joint['name']}"
                )
            joint_path = (
                f"{body_path}/Joints/{_safe_name(str(joint['name']))}"
            )
            joint_type = str(joint["type"])
            if joint_type == "prismatic":
                joint_schema = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            elif joint_type == "revolute":
                joint_schema = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            elif joint_type == "continuous":
                joint_schema = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            else:
                raise ValueError(f"unsupported Isaac articulation joint: {joint_type}")
            joint_schema.CreateBody0Rel().SetTargets(
                [Sdf.Path(link_prim_paths[parent_name])]
            )
            joint_schema.CreateBody1Rel().SetTargets(
                [Sdf.Path(link_prim_paths[child_name])]
            )
            # Local frame rotations carry the full signed, non-cardinal axis.
            joint_schema.CreateAxisAttr().Set("X")
            if joint.get("lower") is not None and joint_type != "continuous":
                joint_schema.CreateLowerLimitAttr().Set(
                    float(joint["lower"]) * 180.0 / np.pi
                    if joint_type == "revolute"
                    else float(joint["lower"])
                )
            if joint.get("upper") is not None and joint_type != "continuous":
                joint_schema.CreateUpperLimitAttr().Set(
                    float(joint["upper"]) * 180.0 / np.pi
                    if joint_type == "revolute"
                    else float(joint["upper"])
                )
            parent_pose = body["links"][
                next(index for index, link in enumerate(body["links"]) if link["name"] == parent_name)
            ]["pose"]
            child_pose = body["links"][
                next(index for index, link in enumerate(body["links"]) if link["name"] == child_name)
            ]["pose"]
            from .isaac_joint_frames import joint_local_frames
            frames = joint_local_frames(parent_pose, child_pose, joint.get("pose"), joint["axis"])
            for frame, position_attr, rotation_attr in (
                (frames[0], joint_schema.CreateLocalPos0Attr(), joint_schema.CreateLocalRot0Attr()),
                (frames[1], joint_schema.CreateLocalPos1Attr(), joint_schema.CreateLocalRot1Attr()),
            ):
                position_attr.Set(Gf.Vec3f(*frame['xyz']))
                x, y, z, w = frame['quat_xyzw']
                rotation_attr.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            articulation_joint_paths.append(joint_path)
        if not body["static"] and not body.get("joints"):
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(body_prim)
            rigid_api.CreateRigidBodyEnabledAttr(True)
            rigid_api.CreateKinematicEnabledAttr(False)
            inertial = body["links"][0]["inertial"]
            inertial_pose = body["links"][0]["inertial_pose"]
            mass_api = UsdPhysics.MassAPI.Apply(body_prim)
            mass_api.CreateMassAttr().Set(float(inertial["mass"]))
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*inertial_pose["xyz"]))
            ixx, iyy, izz, _ixy, _ixz, _iyz = inertial["inertia"]
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, iyy, izz))
            quat = inertial_pose["quat_xyzw"]
            mass_api.CreatePrincipalAxesAttr().Set(
                Gf.Quatf(quat[3], Gf.Vec3f(quat[0], quat[1], quat[2]))
            )

    floating_articulations = []
    from .isaac_articulation import author_floating_articulation
    for body in plan['bodies']:
        if body['status'] != 'native_runtime' and not body['static'] and body.get('joints'):
            floating_articulations.append(author_floating_articulation(stage, body))

    fixed_articulations = []
    if fixed_articulation_bodies:
        from .isaac_articulation import author_fixed_articulation
        for name in fixed_articulation_bodies:
            matches = [body for body in plan['bodies'] if body['name'] == name]
            if len(matches) != 1:
                raise ValueError(f'unknown or ambiguous fixed articulation body: {name}')
            fixed_articulations.append(author_fixed_articulation(stage, matches[0]))
    if authored_counts != plan["geometry_counts"]:
        raise RuntimeError(
            f"USD authoring dropped geometry: plan={plan['geometry_counts']}, "
            f"stage={authored_counts}"
        )
    if len(articulation_joint_paths) != sum(
        str(joint.get("type")) != "fixed"
        for body in plan["bodies"]
        for joint in body.get("joints", [])
        if body["status"] != "native_runtime"
    ):
        raise RuntimeError("USD authoring dropped one or more articulation joints")
    stage.GetRootLayer().customLayerData = {
        "worldir_sha256": str(plan["world_hash"]),
        "worldir_schema_version": str(plan["world_schema_version"]),
        "compiler_schema_version": "isaac_usd_bundle.v1",
    }
    stage.Save()
    reopened = Usd.Stage.Open(str(output))
    if reopened is None or reopened.GetDefaultPrim().GetPath() != Sdf.Path("/World"):
        raise RuntimeError(f"Isaac USD validation could not reopen {output}")
    reopened_counts = {"visual": 0, "collision": 0}
    collision_api_count = 0
    rigid_body_api_count = 0
    reopened_joint_paths: list[str] = []
    for prim in reopened.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.endswith("/Shape") and "/Visuals/" in prim_path:
            reopened_counts["visual"] += 1
        if prim_path.endswith("/Shape") and "/Collisions/" in prim_path:
            reopened_counts["collision"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api_count += 1
        if prim.GetTypeName() in {"PhysicsPrismaticJoint", "PhysicsRevoluteJoint"}:
            reopened_joint_paths.append(prim_path)
    missing_bodies = [
        name
        for name, prim_path in body_prim_paths.items()
        if not reopened.GetPrimAtPath(prim_path).IsValid()
    ]
    layer_data = dict(reopened.GetRootLayer().customLayerData)
    validation_errors: list[str] = []
    if reopened_counts != plan["geometry_counts"]:
        validation_errors.append(
            f"reopened geometry mismatch: expected={plan['geometry_counts']} "
            f"actual={reopened_counts}"
        )
    if collision_api_count != plan["geometry_counts"]["collision"]:
        validation_errors.append(
            f"collision API mismatch: expected={plan['geometry_counts']['collision']} "
            f"actual={collision_api_count}"
        )
    if missing_bodies:
        validation_errors.append(f"missing body prims: {missing_bodies}")
    if layer_data.get("worldir_sha256") != plan["world_hash"]:
        validation_errors.append("reopened layer WorldIR hash does not match the compile plan")
    if not reopened.GetPrimAtPath("/World/PhysicsScene").IsValid():
        validation_errors.append("physics scene is missing")
    if sorted(reopened_joint_paths) != sorted(articulation_joint_paths):
        validation_errors.append(
            f"reopened articulation joints differ: expected={sorted(articulation_joint_paths)} "
            f"actual={sorted(reopened_joint_paths)}"
        )
    for mechanism in fixed_articulations:
        anchor = reopened.GetPrimAtPath(mechanism['root_joint'])
        base = reopened.GetPrimAtPath(mechanism['root_link'])
        if (not anchor.HasAPI(UsdPhysics.ArticulationRootAPI)
                or not base.HasAPI(UsdPhysics.RigidBodyAPI)
                or [str(path) for path in UsdPhysics.FixedJoint(anchor).GetBody0Rel().GetTargets()] != [mechanism['root_link']]):
            validation_errors.append(f"reopened fixed articulation is incomplete: {mechanism['body']}")
    from .isaac_articulation import validate_floating_articulation
    for mechanism in floating_articulations:
        body = next(body for body in plan['bodies'] if body['name'] == mechanism['body'])
        try:
            validate_floating_articulation(reopened, body)
        except ValueError as exc:
            validation_errors.append(f"reopened floating articulation is incomplete: {exc}")
    if validation_errors:
        raise RuntimeError("; ".join(validation_errors))
    bundle = {
        "schema_version": "isaac_usd_bundle.v1",
        "backend": "isaac_lab_usd",
        "world_hash": plan["world_hash"],
        "usd_path": output.name,
        "usd_sha256": sha256_file(output),
        "geometry_counts": authored_counts,
        "native_body_names": plan["native_body_names"],
        "body_prim_paths": body_prim_paths,
        "articulation_joint_paths": articulation_joint_paths,
        "fixed_articulations": fixed_articulations,
        "floating_articulations": floating_articulations,
        "capability_notes": plan.get("capability_notes", []),
        "semantic_status": (
            "partial"
            if any(
                note.get("status") in {"frozen_geometry", "unsupported"}
                for note in plan.get("capability_notes", [])
            )
            else "complete"
        ),
        "visual_asset_contract": (
            "SceneCode GLTF/GLB -> Isaac native USD references; WorldIR collisions separate"
            if visual_manifest is not None
            else "legacy WorldIR flattened visual geometry"
        ),
        "native_visual_assets": {
            name: {
                "source_sha256": asset["source_sha256"],
                "usd_sha256": asset["usd_sha256"],
                "usd_path": asset["resolved_usd_path"],
            }
            for name, asset in sorted(native_visual_assets.items())
        },
        "validation": {
            "stage_reopened": True,
            "default_prim": "/World",
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(reopened)),
            "up_axis": str(UsdGeom.GetStageUpAxis(reopened)),
            "world_hash_matches": True,
            "physics_scene_present": True,
            "reopened_geometry_counts": reopened_counts,
            "collision_api_count": collision_api_count,
            "rigid_body_api_count": rigid_body_api_count,
            "articulation_joint_count": len(reopened_joint_paths),
            "missing_body_prims": missing_bodies,
        },
    }
    return write_json(output.parent / "isaac_bundle.json", bundle)
