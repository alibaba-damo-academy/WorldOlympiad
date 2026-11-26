"""Adapter-owned seams for binding WOV2 references into native RoboTwin tasks.

This module intentionally does not pretend to import WOV2 SDF assets into
SAPIEN. It provides strict config, command, identity/pose binding, and receipt
boundaries for a native bridge. A bridge must supply native handles keyed by
canonical WOV2 object IDs; it must never fall back to RoboTwin randomized
``load_actors`` objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class RoboTwinRuntime:
    workspace: Path
    repo: Path
    python: Path


def _configured_path(value: str, workspace: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def runtime_from_profile(profile_path: str | Path) -> RoboTwinRuntime:
    """Resolve the pinned RoboTwin checkout/interpreter from the eval profile."""
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    workspace = Path(profile["workspace_root"]).expanduser().resolve()
    tested = profile["tested_model"]
    repo = _configured_path(tested["robotwin_repo"], workspace)
    python = _configured_path(tested["robotwin_python"], workspace)
    if not repo.is_dir():
        raise FileNotFoundError(f"configured RoboTwin repository is missing: {repo}")
    if not python.is_file():
        raise FileNotFoundError(f"configured RoboTwin Python is missing: {python}")
    return RoboTwinRuntime(workspace=workspace, repo=repo, python=python)


def build_bridge_command(runtime: RoboTwinRuntime, bridge_script: str | Path, *,
                         stock_task: str, reference: str | Path, output: str | Path,
                         port: int, seed: int = 0, window_s: float = 60,
                         prompt: str | None = None) -> list[str]:
    """Construct the source-wrapper invocation with absolute paths and a ref."""
    if stock_task != "stack_blocks_two":
        raise ValueError("this native bridge seam currently admits only stack_blocks_two")
    if int(port) <= 0 or float(window_s) <= 0:
        raise ValueError("port and window_s must be positive")
    command = [str(runtime.python.resolve()), str(Path(bridge_script).resolve()),
               "--robotwin-repo", str(runtime.repo.resolve()), "--stock-task", stock_task,
               "--ref", str(Path(reference).resolve()), "--out", str(Path(output).resolve()),
               "--port", str(int(port)), "--seed", str(int(seed)),
               "--window-s", str(float(window_s))]
    if prompt:
        command.extend(("--prompt", prompt))
    return command


def _pose_factory():
    import sapien.core as sapien
    return sapien.Pose


def native_table_surface_translation(task: Any) -> tuple[float, float, float]:
    """Return the configured RoboTwin table-top origin in native world metres.

    Read RoboTwin's
    actual table entity pose rather than assuming a nominal height; this keeps
    authored relative poses aligned when the native episode has table XY/Z bias.
    """
    table = getattr(task, "table", None)
    getter = getattr(table, "get_pose", None)
    if not callable(getter):
        raise ValueError("native RoboTwin environment exposes no table pose for frame alignment")
    position = np.asarray(getter().p, dtype=float).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("native RoboTwin table pose must have a finite 3D position")
    return tuple(float(value) for value in position)


def bind_reference_objects(reference_state: Mapping[str, Any],
                           native_objects: Mapping[str, Any], *,
                           pose_factory: Callable[[list[float], list[float]], Any] | None = None,
                           position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
                           ) -> dict[str, dict[str, Any]]:
    """Bind frame-zero WOV2 poses after translating into the native table frame.

    WOV2 quaternions are xyzw while SAPIEN Pose.q is wxyz. Positions remain
    relative to the WOV2 tabletop origin; RoboTwin's tabletop is elevated above
    its ground plane, so the caller supplies that frame translation. The handle
    identity set must match exactly; asset creation is a separate bridge step.
    """
    initial = reference_state.get("initial_objects")
    if reference_state.get("schema_version") != "wo_reference_state.v1" or not isinstance(initial, dict):
        raise ValueError("expected canonical wo_reference_state.v1 initial_objects")
    expected_ids, actual_ids = set(initial), set(native_objects)
    if actual_ids != expected_ids:
        missing, extra = sorted(expected_ids - actual_ids), sorted(actual_ids - expected_ids)
        raise ValueError(f"native/reference object identity mismatch; missing={missing}, extra={extra}")
    pose_factory = pose_factory or _pose_factory()
    evidence = {}
    import math
    offset = [float(v) for v in position_offset]
    if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
        raise ValueError("native table position_offset must contain three finite values")
    for object_id, state in initial.items():
        source_position = [float(v) for v in state["link_position"]]
        q_xyzw = [float(v) for v in state["link_orientation_xyzw"]]
        if len(source_position) != 3 or len(q_xyzw) != 4:
            raise ValueError(f"invalid canonical pose for {object_id}")
        if not all(math.isfinite(v) for v in source_position + q_xyzw):
            raise ValueError(f"nonfinite canonical pose for {object_id}")
        position = [value + delta for value, delta in zip(source_position, offset)]
        norm = math.sqrt(sum(v * v for v in q_xyzw))
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"nonunit canonical quaternion for {object_id}")
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        handle = native_objects[object_id]
        pose = pose_factory(position, q_wxyz)
        if hasattr(handle, "set_root_pose"):
            handle.set_root_pose(pose)
        elif hasattr(handle, "set_pose"):
            handle.set_pose(pose)
        else:
            raise TypeError(f"native object {object_id} cannot accept a SAPIEN pose")
        if hasattr(handle, "set_root_velocity"):
            handle.set_root_velocity([0.0] * 6)
        else:
            if hasattr(handle, "set_velocity"):
                handle.set_velocity([0.0] * 3)
            if hasattr(handle, "set_angular_velocity"):
                handle.set_angular_velocity([0.0] * 3)
        evidence[object_id] = {"source_position": source_position,
                               "position": position, "position_offset": offset,
                               "orientation_xyzw": q_xyzw,
                               "pose_source": "WOV2 frame0 link pose translated into RoboTwin tabletop frame"}
    return evidence


def bind_stack_blocks_two(task: Any, reference_objects: Mapping[str, Any],
                           task_object_ids: Mapping[str, str]) -> dict[str, str]:
    """Bind existing native predicate fields to explicitly named reference IDs."""
    if set(task_object_ids) != {"block1", "block2"}:
        raise ValueError("stack_blocks_two requires explicit block1 and block2 reference IDs")
    ids = list(task_object_ids.values())
    if len(set(ids)) != 2 or any(object_id not in reference_objects for object_id in ids):
        raise ValueError("stack_blocks_two binding must name two distinct loaded reference objects")
    for field, object_id in task_object_ids.items():
        setattr(task, field, reference_objects[object_id])
    return dict(task_object_ids)


def stack_blocks_two_object_ids(task_spec: Mapping[str, Any]) -> dict[str, str]:
    """Bind the stock two-actor native environment to a supported WOV2 goal.

    The RoboTwin task supplies a dual-arm action/camera environment and its
    tabletop as native geometry, not goal semantics for arbitrary authored
    assets. Only one direct ``on`` or ``stacked`` relation between two authored
    objects is supported. An optional explicitly fixed ``table`` support may
    also be declared in TaskSpec; it remains represented by the native table
    and is not loaded as a duplicate reference actor.
    """
    if task_spec.get("schema_version") != "wo_task_spec.v1":
        raise ValueError("stack_blocks_two requires a bound wo_task_spec.v1 TaskSpec")
    if len(task_spec.get("steps", [])) != 1:
        raise ValueError("stack_blocks_two reference must contain one native on/stacked goal")
    goal = task_spec["steps"][0].get("goal", {})
    if goal.get("kind") not in {"on", "stacked"} or not goal.get("target"):
        raise ValueError("stack_blocks_two reference must specify on(subject, target) or stacked(subject, target)")
    subject, target = goal["subject"], goal["target"]
    objects = {item["object_id"]: item for item in task_spec.get("objects", [])}
    if subject == target or subject not in objects or target not in objects:
        raise ValueError("on/stacked goal must name two distinct TaskSpec objects")
    if objects[subject].get("role") == "support":
        raise ValueError("stack_blocks_two subject must be a movable non-support object")
    extra_objects = set(objects) - {subject, target}
    # The RoboTwin manipulation task already supplies its table as native
    # environment geometry. A TaskSpec may therefore include that explicit
    # fixed support alongside the two reference-bound object actors without
    # treating it as a distractor or loading a duplicate table mesh.
    unsupported = [object_id for object_id in extra_objects
                   if objects[object_id].get("role") != "support"
                   or objects[object_id].get("category") != "table"
                   or objects[object_id].get("geometry_constraints", {}).get("fixed") is not True]
    if unsupported:
        raise ValueError(
            "stack_blocks_two admits only the two related objects plus a fixed native table support"
        )
    return {"block1": target, "block2": subject}


def native_on_predicate(subject_aabb, target_aabb, touching: bool) -> tuple[bool, dict[str, Any]]:
    """Evaluate WOV2's direct on/stacked geometric/contact condition on native AABBs."""
    s_lo, s_hi = (np.asarray(v, dtype=float) for v in subject_aabb)
    t_lo, t_hi = (np.asarray(v, dtype=float) for v in target_aabb)
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (s_lo, s_hi, t_lo, t_hi)):
        raise ValueError("native object AABBs must be finite 3-vectors")
    gap = float(s_lo[2] - t_hi[2])
    width = max(1e-9, float(s_hi[0] - s_lo[0]))
    depth = max(1e-9, float(s_hi[1] - s_lo[1]))
    overlap = (max(0., min(s_hi[0], t_hi[0]) - max(s_lo[0], t_lo[0]))
               * max(0., min(s_hi[1], t_hi[1]) - max(s_lo[1], t_lo[1])) / (width * depth))
    from ..predicates import THRESHOLDS
    evidence = {"gap_to_target_top_m": gap, "xy_overlap": overlap, "touching": bool(touching)}
    # A positive footprint intersection plus native contact is the available
    # SAPIEN evidence that the target, rather than a lateral face, supports the
    # subject; unlike inside/covering, WOV2 on does not demand a fixed overlap
    # percentage.
    passed = bool(touching and overlap > 0.0
                  and gap > -0.5 * float(s_hi[2] - s_lo[2])
                  and abs(gap) < THRESHOLDS["stack_gap_m"] + 0.02)
    return passed, evidence


def sdf_collision_vertices_model(sdf_path: str | Path) -> np.ndarray:
    """Read collision mesh vertices in the prepared SDF model frame."""
    from ..sim.sdf_urdf import _Frames, _T
    import trimesh

    sdf_path = Path(sdf_path).resolve()
    root = ET.parse(sdf_path).getroot()
    model = _sdf_child(root, "model")
    if model is None:
        raise ValueError(f"prepared SDF has no model: {sdf_path}")
    vertices = []
    frames = _Frames(model)
    for link in _sdf_children(model, "link"):
        link_T_model = np.linalg.inv(frames.model(link.get("name")))
        for collision in _sdf_children(link, "collision"):
            geometry = _sdf_child(collision, "geometry")
            mesh = _sdf_child(geometry, "mesh") if geometry is not None else None
            uri = _sdf_child(mesh, "uri") if mesh is not None else None
            if uri is None or not (uri.text or "").strip():
                raise ValueError("native WOV2 on predicate requires collision mesh geometry")
            path = Path(uri.text.strip())
            if not path.is_absolute():
                path = (sdf_path.parent / path).resolve()
            scale_el = _sdf_child(mesh, "scale")
            scale = np.asarray([float(v) for v in scale_el.text.split()]
                               if scale_el is not None and scale_el.text else [1., 1., 1.])
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("invalid collision mesh scale in prepared SDF")
            pose_el = _sdf_child(collision, "pose")
            pose_transform, relative = _T(pose_el)
            local = (link_T_model @ frames.model(relative) @ pose_transform
                     if relative else pose_transform)
            loaded = trimesh.load(str(path), force="scene", process=False)
            mesh_data = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
            if not hasattr(mesh_data, "vertices") or not len(mesh_data.vertices):
                raise ValueError(f"collision mesh has no vertices: {path}")
            points = np.asarray(mesh_data.vertices, dtype=float) * scale
            vertices.append(points @ local[:3, :3].T + local[:3, 3])
    if not vertices:
        raise ValueError(f"prepared SDF has no collision meshes: {sdf_path}")
    return np.concatenate(vertices, axis=0)


def native_actor_aabb(actor: Any, local_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform prepared model-frame collision vertices into a native AABB."""
    pose = actor.get_pose()
    p = np.asarray(pose.p, dtype=float)
    w, x, y, z = np.asarray(pose.q, dtype=float)
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    world = np.asarray(local_vertices, dtype=float) @ rotation.T + p
    return world.min(axis=0), world.max(axis=0)


def _sdf_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sdf_child(element, name):
    return next((child for child in element if _sdf_local_name(child.tag) == name), None)


def _sdf_children(element, name):
    return [child for child in element if _sdf_local_name(child.tag) == name]


def _sapien_pose_from_transform(transform, sapien_module):
    """Convert a rigid transform to SAPIEN's [w, x, y, z] pose quaternion."""
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s,
             (rotation[1, 0] - rotation[0, 1]) / s]
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            q = [(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                 (rotation[0, 1] + rotation[1, 0]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s]
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            q = [(rotation[0, 2] - rotation[2, 0]) / s,
                 (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                 (rotation[1, 2] + rotation[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            q = [(rotation[1, 0] - rotation[0, 1]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s,
                 (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s]
    return sapien_module.Pose(p=transform[:3, 3].tolist(), q=q)


def add_sdf_mesh_actor(scene, sdf_path: str | Path, object_id: str, *, fixed: bool,
                       sapien_module=None):
    """Build a one-link canonical prepared SDF as a real native SAPIEN actor.

    RoboTwin's configured SAPIEN ActorBuilder already supports the mesh assets
    used by prepared SceneCode SDFs.  This bridge intentionally rejects
    articulated/multi-link or non-mesh representations instead of dropping
    visuals, flattening links, or inventing a stock box.
    """
    if sapien_module is None:
        import sapien.core as sapien_module
    from ..sim.sdf_urdf import _Frames, _T

    sdf_path = Path(sdf_path).resolve()
    root = ET.parse(sdf_path).getroot()
    model = _sdf_child(root, "model")
    if model is None:
        raise ValueError(f"prepared SDF has no model: {sdf_path}")
    links = _sdf_children(model, "link")
    if len(links) != 1:
        raise ValueError(f"{object_id}: native stack bridge requires a one-link SDF asset")
    link = links[0]
    frames = _Frames(model)
    link_name = link.get("name")
    link_T_model = np.linalg.inv(frames.model(link_name))
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if fixed else "dynamic")
    visual_count = collision_count = 0

    def mesh_spec(geometry, owner):
        mesh = _sdf_child(geometry, "mesh")
        if mesh is None:
            raise ValueError(f"{object_id}: only canonical mesh {owner} geometry is supported")
        uri_el = _sdf_child(mesh, "uri")
        if uri_el is None or not (uri_el.text or "").strip():
            raise ValueError(f"{object_id}: mesh {owner} has no URI")
        mesh_path = Path(uri_el.text.strip())
        if not mesh_path.is_absolute():
            mesh_path = (sdf_path.parent / mesh_path).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"{object_id}: prepared SDF mesh is missing: {mesh_path}")
        scale_el = _sdf_child(mesh, "scale")
        scale = ([float(x) for x in scale_el.text.split()]
                 if scale_el is not None and scale_el.text else [1.0, 1.0, 1.0])
        if len(scale) != 3 or not np.isfinite(scale).all() or any(x <= 0 for x in scale):
            raise ValueError(f"{object_id}: invalid SDF mesh scale")
        pose_el = _sdf_child(owner, "pose")
        pose_transform, relative = _T(pose_el)
        local = (link_T_model @ frames.model(relative) @ pose_transform
                 if relative else pose_transform)
        return str(mesh_path), scale, _sapien_pose_from_transform(local, sapien_module)

    visuals = _sdf_children(link, "visual")
    if visuals:
        for visual in visuals:
            geometry = _sdf_child(visual, "geometry")
            if geometry is None:
                raise ValueError(f"{object_id}: visual geometry is absent")
            mesh, scale, local_pose = mesh_spec(geometry, visual)
            builder.add_visual_from_file(filename=mesh, scale=scale, pose=local_pose)
            visual_count += 1
    else:
        # Some SceneCode exports provide only a collision decomposition in
        # their canonical SDF. Preserve native-camera visibility by rendering
        # those same integrity-bound meshes rather than rejecting the actor or
        # substituting an unrelated stock shape. Assets with explicit visuals
        # continue to use those (typically higher quality) meshes.
        for collision in _sdf_children(link, "collision"):
            geometry = _sdf_child(collision, "geometry")
            if geometry is None:
                raise ValueError(f"{object_id}: collision geometry is absent")
            mesh, scale, local_pose = mesh_spec(geometry, collision)
            builder.add_visual_from_file(filename=mesh, scale=scale, pose=local_pose)
            visual_count += 1
    for collision in _sdf_children(link, "collision"):
        geometry = _sdf_child(collision, "geometry")
        if geometry is None:
            raise ValueError(f"{object_id}: collision geometry is absent")
        mesh, scale, local_pose = mesh_spec(geometry, collision)
        # Convex decomposition is RoboTwin's supported dynamic-mesh import
        # path.  Measured reference mass/inertia/material is rebound below.
        builder.add_multiple_convex_collisions_from_file(
            filename=mesh, scale=scale, pose=local_pose)
        collision_count += 1
    if not visual_count or not collision_count:
        raise ValueError(f"{object_id}: SDF must provide visual and collision meshes")
    return builder.build(name=object_id)


def prepared_sdf_for_object(reference: str | Path, object_id: str) -> Path:
    reference = Path(reference).resolve()
    markers = sorted((reference / "physics_sdf").glob(f"{object_id}_s*/.path"))
    if len(markers) != 1:
        raise ValueError(f"{object_id}: expected one canonical prepared-SDF marker, found {len(markers)}")
    marker = markers[0]
    sdf = Path(marker.read_text(encoding="utf-8").strip()).resolve()
    if reference not in sdf.parents or not sdf.is_file():
        raise ValueError(f"{object_id}: prepared SDF is not a file in the bound reference")
    return sdf


def _verify_bound_reference(reference: str | Path, reference_state: Mapping[str, Any]) -> Path:
    """Verify the archived frame-zero record and every bound asset dependency.

    Prepared SceneCode SDFs live inside the reference, while their mesh URIs
    may intentionally resolve to immutable generated assets outside it. Such
    external files are accepted only when they are in the transitive asset
    closure of a reference-contained prepared SDF and their recorded digest
    still matches.
    """
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    reference = Path(reference).resolve()
    if Path(str(reference_state.get("reference", ""))).resolve() != reference:
        raise ValueError("reference-state archive is not bound to the requested WOV2 reference")
    sources = reference_state.get("source_sha256")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("reference-state archive has no source-integrity manifest")

    # Recompute the permitted external dependency set from canonical SDFs,
    # rather than trusting arbitrary out-of-reference paths in the manifest.
    permitted_external: set[Path] = set()
    seen_assets: set[Path] = set()
    asset_hashes: dict[str, str] = {}

    def visit_asset(path: Path) -> None:
        path = path.resolve()
        if path in seen_assets:
            return
        if not path.is_file():
            raise ValueError(f"bound reference asset is missing: {path}")
        seen_assets.add(path)
        asset_hashes[str(path)] = sha(path)
        if path.suffix.lower() not in {".urdf", ".sdf", ".xml"}:
            return
        root = ET.parse(path).getroot()
        for node in root.iter():
            value = node.get("filename") or (node.text.strip()
                     if node.tag == "uri" and node.text else None)
            if value:
                if "://" in value:
                    raise ValueError(f"unresolved bound asset URI: {value}")
                dependency = (path.parent / value).resolve()
                if reference not in dependency.parents:
                    permitted_external.add(dependency)
                visit_asset(dependency)

    physics = reference / "physics_sdf"
    # Each prepared object directory may retain multiple converted model
    # variants, but its .path marker identifies the one SDF actually loaded by
    # the native adapter. Bind only those canonical inputs; scanning every
    # cached *.sdf would reject unused conversion variants omitted from the
    # reference's source manifest.
    markers = sorted(physics.rglob(".path")) if physics.is_dir() else ()
    for marker in markers:
        try:
            resolved_sdf = Path(marker.read_text(encoding="utf-8").strip()).resolve()
        except OSError as exc:
            raise ValueError(f"cannot read prepared SDF marker: {marker}") from exc
        if reference not in resolved_sdf.parents or not resolved_sdf.is_file():
            raise ValueError(f"prepared SDF escaped the bound reference: {marker}")
        visit_asset(resolved_sdf)

    for raw_path, expected_sha in sources.items():
        path = Path(raw_path).resolve()
        internal = reference in path.parents
        if (not internal and path not in permitted_external) or not path.is_file() or sha(path) != expected_sha:
            raise ValueError(f"WOV2 reference source changed or is not bound to its assets: {path}")
    for raw_path, expected_sha in asset_hashes.items():
        if sources.get(raw_path) != expected_sha:
            raise ValueError(f"prepared reference asset is absent from its integrity manifest: {raw_path}")
    return reference


def initialize_reference_stack_blocks_two(task: Any, *, reference: str | Path,
                                           task_spec: Mapping[str, Any],
                                           reference_state: Mapping[str, Any],
                                           setup_kwargs: Mapping[str, Any],
                                           actor_loader: Callable[..., Any] | None = None,
                                           sapien_module=None,
                                           dynamics_binder: Callable[..., Any] | None = None
                                           ) -> dict[str, Any]:
    """Run stock RoboTwin setup with only its native random-actor hook replaced.

    The native table, robot, cameras and task class remain upstream-owned.
    ``load_actors`` is intercepted before setup_demo reaches it, so randomized
    stock actors are never used. Stock stacked goals retain the upstream
    ``check_success``; a WOV2 direct ``on`` goal uses its declared collision
    geometry/contact predicate because RoboTwin's cube-center tolerance is not
    valid for arbitrary authored object sizes. SAPIEN settling still runs;
    canonical frame-zero poses are reapplied and zero-velocity bound afterward.
    """
    reference = _verify_bound_reference(reference, reference_state)
    ids = stack_blocks_two_object_ids(task_spec)
    initial = reference_state.get("initial_objects", {})
    expected = set(ids.values())
    if not expected.issubset(initial):
        raise ValueError("reference_state lacks one or more stacked object initial poses")
    if reference_state.get("schema_version") != "wo_reference_state.v1":
        raise ValueError("expected canonical wo_reference_state.v1")
    if actor_loader is None:
        actor_loader = add_sdf_mesh_actor
    if sapien_module is None:
        import sapien.core as sapien_module
    if dynamics_binder is None:
        from .robotwin_dynamics import apply_loaded as dynamics_binder

    original_load_actors = task.load_actors
    predicate_function = getattr(task.check_success, "__func__", task.check_success)
    handles: dict[str, Any] = {}

    def load_bound_actors():
        for object_id in sorted(expected):
            sdf = prepared_sdf_for_object(reference, object_id)
            obj_spec = next(item for item in task_spec["objects"] if item["object_id"] == object_id)
            handle = actor_loader(task.scene, sdf, object_id,
                                  fixed=bool(obj_spec.get("geometry_constraints", {}).get("fixed", False)),
                                  sapien_module=sapien_module)
            dynamics = reference_state.get("measured_dynamics", {}).get(object_id)
            if not dynamics:
                raise ValueError(f"{object_id}: reference has no measured native dynamics")
            dynamics_binder(task.scene, handle, dynamics, sapien_module.Pose)
            handles[object_id] = handle
        task.wo_handles = handles
        task.wo_task_object_ids = bind_stack_blocks_two(task, handles, ids)
        task.block1_target_pose = [0, -0.13, 0.75 + task.table_z_bias, 0, 1, 0, 0]

    task.load_actors = load_bound_actors
    try:
        task.setup_demo(**dict(setup_kwargs))
    finally:
        task.load_actors = original_load_actors

    _verify_bound_reference(reference, reference_state)
    bound_state = {"schema_version": "wo_reference_state.v1",
                   "initial_objects": {oid: initial[oid] for oid in expected}}
    position_offset = np.asarray(native_table_surface_translation(task))
    for obj_spec in task_spec["objects"]:
        if obj_spec.get("role") != "support":
            continue
        from scipy.spatial.transform import Rotation

        support_id = obj_spec["object_id"]
        support_pose = initial[support_id]
        origin = np.asarray(support_pose["link_position"], dtype=float)
        vertices = sdf_collision_vertices_model(prepared_sdf_for_object(reference, support_id))
        rotation = Rotation.from_quat(support_pose["link_orientation_xyzw"])
        surface = origin.copy()
        surface[2] += rotation.apply(vertices)[:, 2].max()
        position_offset -= surface

    pose_evidence = bind_reference_objects(
        bound_state, handles,
        pose_factory=lambda p, q: sapien_module.Pose(p=p, q=q),
        position_offset=position_offset)
    goal = task_spec["steps"][0]["goal"]
    if goal["kind"] == "on":
        # RoboTwin's block-center tolerance is specific to equal 5 cm cubes and
        # is not a valid success check for arbitrary authored object geometry.
        # Evaluate the declared WOV2 on predicate using the actual SAPIEN actor
        # poses, prepared collision meshes, and RoboTwin's current contact set.
        local_vertices = {oid: sdf_collision_vertices_model(prepared_sdf_for_object(reference, oid))
                          for oid in ids.values()}
        subject, target = goal["subject"], goal["target"]

        def check_bound_on_goal():
            subject_aabb = native_actor_aabb(handles[subject], local_vertices[subject])
            target_aabb = native_actor_aabb(handles[target], local_vertices[target])
            touching = bool(task.check_actors_contact(subject, target))
            passed, evidence = native_on_predicate(subject_aabb, target_aabb, touching)
            task.wo_goal_evidence = evidence
            return passed

        task.check_success = check_bound_on_goal
        task.wo_goal_predicate_source = "WOV2 on predicate on native SAPIEN collision geometry/contact"
        task.wo_goal_evidence = {}
        task.eval_success = False
    elif getattr(task.check_success, "__func__", task.check_success) is not predicate_function:
        raise RuntimeError("native RoboTwin check_success predicate was replaced")
    else:
        task.wo_goal_predicate_source = "upstream RoboTwin check_success"
        task.wo_goal_evidence = {}
    return {"task_object_ids": dict(ids), "initial_pose_evidence": pose_evidence,
            "stock_random_actor_reset_bypassed": True,
            "native_predicate": task.wo_goal_predicate_source}


def validate_capture_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate native result evidence required by policy_check without rewriting it."""
    if not isinstance(metadata, Mapping) or type(metadata.get("task_success")) is not bool:
        raise ValueError("control produced no measured task result")
    try:
        predicate = metadata["completion_evidence"]["native_predicate_evidence"]
    except (KeyError, TypeError) as exc:
        raise ValueError("control produced no native predicate evidence") from exc
    if predicate.get("predicate_source") != "upstream RoboTwin check_success":
        raise ValueError("stock control must use the native task predicate")
    return dict(metadata)


def policy_check_receipt(task: str, model: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the established policy_check receipt shape after evidence validation."""
    return {"control_only": True, "task": task, "model": model,
            "native_result": validate_capture_metadata(metadata)}
