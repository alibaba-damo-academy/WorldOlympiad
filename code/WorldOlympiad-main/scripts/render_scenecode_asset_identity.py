#!/usr/bin/env python3
"""Render published SceneCode task assets for the final visual-identity gate.

The script deliberately does not claim semantic success.  It proves that each
matched task asset can be loaded and framed from several independent views,
then writes contact sheets that a vision-capable reviewer must compare with the
full task ``visual_description``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import shutil
import socket
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw

from autoresearch.scenecode_identity_render_contract import (
    asset_material_profile, source_digests, validate_dependencies,
)
from autoresearch.scenecode_identity_lighting import (
    identity_lighting_plan,
    validate_identity_lighting,
)
from autoresearch.scenecode_structure import (
    failed_structure_reason,
    structure_contract,
    validate_mesh_structure,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_matrix(text: str | None) -> np.ndarray:
    values = [float(value) for value in (text or "0 0 0 0 0 0").split()]
    if len(values) != 6:
        raise ValueError(f"SDF pose must contain six values, got {values!r}")
    x, y, z, roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = (x, y, z)
    return transform


def _corners(bounds: np.ndarray) -> np.ndarray:
    low, high = bounds
    return np.array(
        [
            [x, y, z, 1.0]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )


def _mesh_path(uri: str, sdf_path: Path) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"unsupported mesh URI scheme in {uri!r}")
    raw = unquote(parsed.path if parsed.scheme == "file" else uri)
    path = Path(raw)
    if not path.is_absolute():
        path = sdf_path.parent / path
    return path.resolve()


def _mesh_points_in_sdf_frame(bounds: np.ndarray, mesh_path: Path) -> np.ndarray:
    """Return mesh-bound corners in Drake/SDF's Z-up coordinates.

    glTF stores geometry in a Y-up coordinate system.  Drake rotates glTF
    geometry into its Z-up world while loading it, so framing the raw trimesh
    bounds would aim the camera at the wrong center and can crop tall assets.
    """

    points = _corners(bounds)[:, :3]
    if mesh_path.suffix.lower() in {".glb", ".gltf"}:
        points = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
    return points


def _geometry_bounds(geometry: ET.Element, sdf_path: Path) -> np.ndarray | None:
    box = geometry.find("box/size")
    if box is not None and box.text:
        size = np.asarray([float(value) for value in box.text.split()], dtype=np.float64)
        return np.stack((-size / 2.0, size / 2.0))
    sphere = geometry.find("sphere/radius")
    if sphere is not None and sphere.text:
        radius = float(sphere.text)
        return np.array([[-radius] * 3, [radius] * 3], dtype=np.float64)
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.findtext("radius", "0"))
        length = float(cylinder.findtext("length", "0"))
        return np.array(
            [[-radius, -radius, -length / 2.0], [radius, radius, length / 2.0]],
            dtype=np.float64,
        )
    capsule = geometry.find("capsule")
    if capsule is not None:
        radius = float(capsule.findtext("radius", "0"))
        length = float(capsule.findtext("length", "0"))
        half_z = length / 2.0 + radius
        return np.array(
            [[-radius, -radius, -half_z], [radius, radius, half_z]],
            dtype=np.float64,
        )
    mesh = geometry.find("mesh")
    if mesh is not None:
        uri = mesh.findtext("uri")
        if not uri:
            return None
        import trimesh

        mesh_path = _mesh_path(uri, sdf_path)
        if not mesh_path.is_file():
            raise FileNotFoundError(f"SDF mesh does not exist: {mesh_path}")
        loaded = trimesh.load(mesh_path, process=False)
        bounds = np.asarray(loaded.bounds, dtype=np.float64)
        points = _mesh_points_in_sdf_frame(bounds, mesh_path)
        scale_text = mesh.findtext("scale")
        if scale_text:
            scale = np.asarray([float(value) for value in scale_text.split()], dtype=np.float64)
            if scale.shape != (3,):
                raise ValueError(f"invalid mesh scale in {sdf_path}: {scale_text!r}")
            points = points * scale
        return np.stack((points.min(axis=0), points.max(axis=0)))
    return None


def _sdf_frame_transforms(sdf_path: Path) -> tuple[ET.Element, dict[str, np.ndarray]]:
    """Resolve model, link, and named SDF frame poses into model coordinates.

    SceneCode articulations commonly express a moving link with
    ``pose relative_to=parent_link``.  Identity framing must understand that
    graph rather than flattening links at the origin or rejecting an otherwise
    valid official asset.  The resolver is deliberately small and fail-closed:
    unknown frames and cycles are reported as invalid SDF structure.
    """

    root = ET.parse(sdf_path).getroot()
    model = root.find("model")
    if model is None:
        raise ValueError(f"SDF has no model element: {sdf_path}")

    model_pose = _pose_matrix(model.findtext("pose"))
    frame_defs: dict[str, tuple[str, np.ndarray]] = {"__model__": ("", model_pose)}
    for link in model.findall("link"):
        pose = link.find("pose")
        frame_defs[link.get("name", "")] = (
            pose.get("relative_to") if pose is not None else "__model__",
            _pose_matrix(pose.text if pose is not None else None),
        )
    for frame in model.findall("frame"):
        name = frame.get("name")
        if not name:
            raise ValueError(f"unnamed SDF frame in {sdf_path}")
        pose = frame.find("pose")
        frame_defs[name] = (
            (pose.get("relative_to") if pose is not None else None)
            or frame.get("attached_to") or "__model__",
            _pose_matrix(pose.text if pose is not None else None),
        )

    resolved: dict[str, np.ndarray] = {"__model__": model_pose}
    visiting: set[str] = set()

    def resolve(name: str) -> np.ndarray:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError(f"cyclic SDF frame graph at {name!r}: {sdf_path}")
        if name not in frame_defs:
            raise ValueError(f"unknown SDF relative_to frame {name!r}: {sdf_path}")
        visiting.add(name)
        parent, local = frame_defs[name]
        # Explicit pose coordinates may differ from the frame's attachment.
        parent_name = parent or "__model__"
        resolved[name] = resolve(parent_name) @ local
        visiting.remove(name)
        return resolved[name]

    for name in frame_defs:
        resolve(name)
    return model, resolved


def _sdf_element_bounds(
    sdf_path: Path, *, primary_tag: str, fallback_tag: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-aligned bounds from one SDF geometry element class."""

    model, frames = _sdf_frame_transforms(sdf_path)
    points: list[np.ndarray] = []
    candidates = model.findall(f".//{primary_tag}")
    if not candidates:
        candidates = model.findall(f".//{fallback_tag}")
    for node in candidates:
        link = next((parent for parent in model.iter("link") if node in list(parent)), None)
        if link is None:
            continue
        geometry = node.find("geometry")
        if geometry is None:
            continue
        local_bounds = _geometry_bounds(geometry, sdf_path)
        if local_bounds is None:
            continue
        link_name = link.get("name")
        if not link_name or link_name not in frames:
            raise ValueError(f"geometry belongs to unresolved SDF link: {sdf_path}")
        node_pose = node.find("pose")
        if node_pose is not None and node_pose.get("relative_to"):
            node_frame = node_pose.get("relative_to")
            if node_frame not in frames:
                raise ValueError(
                    f"unknown SDF visual relative_to frame {node_frame!r}: {sdf_path}"
                )
            transform = frames[node_frame] @ _pose_matrix(node_pose.text)
        else:
            transform = frames[link_name] @ _pose_matrix(
                node_pose.text if node_pose is not None else None
            )
        transformed = (transform @ _corners(local_bounds).T).T[:, :3]
        points.append(transformed)
    if not points:
        raise ValueError(
            f"could not derive {primary_tag}/{fallback_tag} bounds from {sdf_path}"
        )
    stacked = np.concatenate(points, axis=0)
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError(f"non-finite visual bounds in {sdf_path}")
    if float(np.linalg.norm(high - low)) <= 1e-6:
        raise ValueError(f"degenerate visual bounds in {sdf_path}")
    return low, high


def sdf_visual_bounds(sdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return SDF visual bounds, falling back to collision geometry."""

    return _sdf_element_bounds(
        sdf_path, primary_tag="visual", fallback_tag="collision"
    )


def sdf_collision_bounds(sdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return SDF collision bounds, falling back to visual geometry."""

    return _sdf_element_bounds(
        sdf_path, primary_tag="collision", fallback_tag="visual"
    )


def _combined_visual_mesh_path(sdf_path: Path) -> Path:
    """Locate SceneCode's official combined visual mesh for Blender identity.

    Blender's ordinary Drake client endpoint expects an imported camera node,
    while SceneCode's generated per-link GLTF parts intentionally contain only
    geometry.  The official Blender multiview endpoint creates its own camera
    and accepts the combined GLB emitted alongside articulated SDF assets.
    """

    model = ET.parse(sdf_path).getroot().find("model")
    model_name = model.get("name") if model is not None else None
    candidates = [
        sdf_path.parent / "mesh" / f"{model_name}.glb" if model_name else None,
        sdf_path.parent / "combined_scene.gltf",
        sdf_path.parent / "mesh" / "combined_scene.gltf",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"official combined visual mesh is missing for articulated SDF: {sdf_path}"
    )


def sdf_visual_mesh_paths(sdf_path: Path) -> list[Path]:
    """Return distinct visual mesh files referenced by one SDF asset."""

    root = ET.parse(sdf_path).getroot()
    paths: list[Path] = []
    for uri_node in root.findall(".//visual/geometry/mesh/uri"):
        if not uri_node.text:
            continue
        path = _mesh_path(uri_node.text, sdf_path)
        if path not in paths:
            paths.append(path)
    return paths


def native_visual_plan(sdf_path: Path) -> list[dict]:
    """Resolve the exact declared visual assembly, keeping mesh bytes untouched."""
    model, frames = _sdf_frame_transforms(sdf_path)
    visuals = []
    for link in model.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.findtext("uri"):
                raise ValueError("native material identity currently requires declared visual meshes")
            pose = visual.find("pose")
            frame = pose.get("relative_to") if pose is not None else None
            transform = frames[frame or link.get("name")] @ _pose_matrix(
                pose.text if pose is not None else None
            )
            scale = [float(value) for value in mesh.findtext("scale", "1 1 1").split()]
            if len(scale) != 3 or not np.all(np.isfinite(scale)) or min(scale) <= 0:
                raise ValueError("native identity mesh scale must be three positive finite values")
            if visual.find("material") is not None:
                raise ValueError("native material identity requires resolving the SDF material override")
            visuals.append({"mesh_path": str(_mesh_path(mesh.findtext("uri"), sdf_path)),
                            "transform": transform.tolist(), "scale": scale})
    if not visuals:
        raise ValueError("native identity has no declared visual meshes")
    return visuals


def render_native_material_asset(
    *, sdf_path: Path, output_dir: Path, object_id: str, width: int, height: int,
    views: list[tuple[float, float]], visual_description: str, material_profile: dict,
) -> dict:
    """Use the native importer and CPU path tracing for unsupported VTK materials."""
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_gate = validate_sdf_structure(sdf_path, visual_description)
    lighting = identity_lighting_plan()
    validate_identity_lighting(lighting)
    plan = {"schema_version": "scenecode_native_material_render.v1",
            "visuals": native_visual_plan(sdf_path), "output_dir": str(output_dir.resolve()),
            "object_id": object_id, "width": width, "height": height, "views": views,
            "lighting": lighting}
    plan_path = output_dir / "native_material_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    blender = os.environ.get("BLENDER_EXECUTABLE") or shutil.which("blender")
    if not blender:
        raise RuntimeError("native material identity requires the installed Blender executable")
    command = [blender, "--background", "--factory-startup", "--threads", "4",
               "--python", str(Path(__file__).with_name("render_scenecode_identity_blender.py")),
               "--", "--plan", str(plan_path.resolve())]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    log_path = output_dir / "native_material_render.log"
    with log_path.open("w") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if completed.returncode != 0:
        raise RuntimeError(f"native material render failed: {log_path}")
    validate_dependencies(material_profile["dependencies"])
    receipt = json.loads((output_dir / "native_material_render.json").read_text())
    paths, labels = [], []
    for view in receipt["views"]:
        path = Path(view["image"])
        rgb = np.asarray(Image.open(path).convert("RGB"))
        foreground, border = _render_coverage(rgb)
        view.update(foreground_fraction=foreground, border_foreground_fraction=border,
                    render_passed=0.005 <= foreground <= 0.75 and border <= 0.01)
        paths.append(path)
        labels.append(f"yaw={view['yaw_deg']:g}, elevation={view['elevation_deg']:g}")
    sheet = output_dir / f"{object_id}_contact_sheet.png"
    _contact_sheet(paths, labels, sheet)
    observed = bool(paths) and all(view["render_passed"] for view in receipt["views"])
    return {"model_instance": None, "bounds_low": receipt["bounds_low"],
            "bounds_high": receipt["bounds_high"], "views": receipt["views"],
            "contact_sheet": str(sheet), "material_profile": material_profile,
            "effective_renderer": "blender_native_cpu", "native_material_receipt": receipt,
            "lighting": receipt.get("lighting", lighting),
            "render_observability_passed": observed, "structure_gate": structure_gate,
            "structural_gate_passed": bool(structure_gate["passed"]),
            "render_gate_passed": observed and bool(structure_gate["passed"])}


def validate_sdf_structure(sdf_path: Path, visual_description: str) -> dict[str, object]:
    """Apply a description-derived structure contract to the actual visual mesh."""

    contract = structure_contract(visual_description)
    if contract is None:
        return validate_mesh_structure(sdf_path, visual_description)
    mesh_paths = sdf_visual_mesh_paths(sdf_path)
    if len(mesh_paths) != 1:
        return {
            "schema_version": "scenecode_structure_gate.v1",
            "contract": contract,
            "applicable": True,
            "passed": False,
            "visual_description": visual_description,
            "error": (
                "structure-bearing SceneCode assets must expose exactly one combined "
                f"visual mesh, found {len(mesh_paths)}"
            ),
            "visual_meshes": [str(path) for path in mesh_paths],
        }
    result = validate_mesh_structure(mesh_paths[0], visual_description)
    result["visual_meshes"] = [str(mesh_paths[0])]
    return result


def _look_at(eye: np.ndarray, target: np.ndarray):
    from pydrake.all import RigidTransform, RotationMatrix

    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return RigidTransform(RotationMatrix(np.column_stack((right, down, forward))), eye)


def _parse_views(value: str) -> list[tuple[float, float]]:
    views: list[tuple[float, float]] = []
    for item in value.split(","):
        yaw_text, elevation_text = item.strip().split(":", maxsplit=1)
        views.append((float(yaw_text), float(elevation_text)))
    if len(views) < 3:
        raise argparse.ArgumentTypeError("identity critique requires at least three views")
    return views


def _contact_sheet(paths: list[Path], labels: list[str], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    columns = min(3, len(images))
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + 32)), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = (index % columns) * width
        y = (index // columns) * (height + 32)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + height + 7), label, fill=(235, 235, 235))
    sheet.save(output)


def _free_loopback_port() -> int:
    """Reserve and return a concrete loopback port for BlenderServer.

    ``BlenderServer(port=0)`` forwards zero to its child, whose Flask server
    chooses a real port that the parent manager does not learn.  Choosing the
    port here keeps the manager and child on the same endpoint.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _render_coverage(rgb: np.ndarray) -> tuple[float, float]:
    """Measure visible foreground and reject framing that touches an edge."""

    height, width = rgb.shape[:2]
    patch = max(4, min(height, width) // 32)
    corners = np.concatenate(
        (
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, -patch:].reshape(-1, 3),
            rgb[-patch:, :patch].reshape(-1, 3),
            rgb[-patch:, -patch:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(corners.astype(np.float32), axis=0)
    delta = np.max(np.abs(rgb.astype(np.float32) - background), axis=2)
    foreground = delta > 12.0
    border = np.concatenate(
        (foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1])
    )
    return float(np.mean(foreground)), float(np.mean(border))


def blender_executable() -> str | None:
    """Return the installed Blender executable, if the native path is available."""

    return os.environ.get("BLENDER_EXECUTABLE") or shutil.which("blender")


def identity_render_path(
    *, material_profile: dict, renderer: str, native_available: bool
) -> str:
    """Choose the identity renderer for one asset.

    The identity gate compares a rendered asset against a declared material,
    which is only meaningful when the render displays that material faithfully.
    The native importer plus the calibrated neutral lighting is the
    authoritative path for every asset; the Drake/VTK path uses an uncalibrated
    default rig that was measured to display a declared albedo of 0.86 near
    0.46, so it is retained only as the fallback when the native renderer is
    unavailable.
    """

    if (
        material_profile["requires_native_material_renderer"]
        or renderer == "blender"
        or native_available
    ):
        return "blender_native_cpu"
    return "vtk"


def render_asset(
    *,
    sdf_path: Path,
    output_dir: Path,
    object_id: str,
    width: int,
    height: int,
    views: list[tuple[float, float]],
    renderer: str,
    visual_description: str,
) -> dict[str, object]:
    material_profile = asset_material_profile(sdf_path)
    if identity_render_path(
        material_profile=material_profile,
        renderer=renderer,
        native_available=blender_executable() is not None,
    ) == "blender_native_cpu":
        return render_native_material_asset(
            sdf_path=sdf_path, output_dir=output_dir, object_id=object_id,
            width=width, height=height, views=views, visual_description=visual_description,
            material_profile=material_profile,
        )
    from pydrake.all import (
        AddMultibodyPlantSceneGraph,
        ApplyCameraConfig,
        CameraConfig,
        DiagramBuilder,
        Parser,
        RenderEngineGltfClientParams,
        RenderEngineVtkParams,
        Rgba,
        Transform,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    low, high = sdf_visual_bounds(sdf_path)
    structure_gate = validate_sdf_structure(sdf_path, visual_description)
    center = (low + high) / 2.0
    radius = float(np.linalg.norm(high - low) / 2.0)
    half_fov = math.radians(42.0 / 2.0)
    distance = max(radius / math.sin(half_fov) * 1.30, 0.25)

    server = None
    if renderer == "blender":
        from scenecode.agent_utils.blender.server_manager import BlenderServer

        server = BlenderServer(
            port=_free_loopback_port(), log_file=output_dir / "blender.log"
        )
        server.start()
        server.wait_until_ready(timeout=90.0)

    try:
        visual_mesh_paths = sdf_visual_mesh_paths(sdf_path)
        # Articulated SceneCode assets expose one visual mesh per link.  Route
        # these through SceneCode's own multiview helper, which creates the
        # camera and renders the official combined GLB without flattening links
        # or asking the Drake client to invent a camera node.
        if renderer == "blender" and len(visual_mesh_paths) > 1:
            combined_mesh = _combined_visual_mesh_path(sdf_path)
            image_paths = server.render_multiview_for_analysis(
                mesh_path=combined_mesh,
                output_dir=output_dir,
                elevation_degrees=25.0,
                num_side_views=max(4, len(views)),
                include_vertical_views=True,
                width=width,
                height=height,
                show_coordinate_frame=False,
            )
            view_records: list[dict[str, object]] = []
            paths: list[Path] = []
            labels: list[str] = []
            for index, image_path in enumerate(image_paths):
                image = Image.open(image_path).convert("RGB")
                rgb = np.asarray(image)
                foreground_fraction, border_foreground_fraction = _render_coverage(rgb)
                name = image_path.stem
                record = {
                    "index": index,
                    "name": name,
                    "image": str(image_path),
                    "foreground_fraction": foreground_fraction,
                    "border_foreground_fraction": border_foreground_fraction,
                    "render_passed": (
                        0.005 <= foreground_fraction <= 0.75
                        and border_foreground_fraction <= 0.01
                    ),
                }
                view_records.append(record)
                paths.append(image_path)
                labels.append(name)
            contact_sheet = output_dir / f"{object_id}_contact_sheet.png"
            _contact_sheet(paths, labels, contact_sheet)
            return {
                "model_instance": None,
                "material_profile": material_profile,
                "effective_renderer": renderer,
                "bounds_low": low.tolist(),
                "bounds_high": high.tolist(),
                "contact_sheet": str(contact_sheet),
                "views": view_records,
                "render_observability_passed": all(
                    bool(record["render_passed"]) for record in view_records
                ),
                "structure_gate": structure_gate,
                "structural_gate_passed": bool(structure_gate["passed"]),
                "render_gate_passed": all(
                    bool(record["render_passed"]) for record in view_records
                )
                and bool(structure_gate["passed"]),
            }

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        model_instance = Parser(plant).AddModels(str(sdf_path.resolve()))[0]
        plant.Finalize()

        view_records: list[dict[str, object]] = []
        for index, (yaw_deg, elevation_deg) in enumerate(views):
            yaw = math.radians(yaw_deg)
            elevation = math.radians(elevation_deg)
            direction = np.array(
                [
                    math.cos(elevation) * math.cos(yaw),
                    math.cos(elevation) * math.sin(yaw),
                    math.sin(elevation),
                ],
                dtype=np.float64,
            )
            eye = center + distance * direction
            name = f"identity_{index}"
            renderer_class = (
                RenderEngineGltfClientParams(base_url=server.get_url())
                if server is not None
                else RenderEngineVtkParams()
            )
            camera = CameraConfig(
                name=name,
                X_PB=Transform(_look_at(eye, center)),
                width=width,
                height=height,
                focal=CameraConfig.FovDegrees(y=42.0),
                background=Rgba(0.10, 0.10, 0.10, 1.0),
                renderer_name=f"{name}_{renderer}",
                renderer_class=renderer_class,
            )
            ApplyCameraConfig(
                config=camera,
                builder=builder,
                plant=plant,
                scene_graph=scene_graph,
            )
            builder.ExportOutput(
                builder.GetSubsystemByName(f"rgbd_sensor_{name}").color_image_output_port(),
                f"rgba_{index}",
            )
            view_records.append(
                {
                    "index": index,
                    "yaw_deg": yaw_deg,
                    "elevation_deg": elevation_deg,
                    "eye": eye.tolist(),
                }
            )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        paths: list[Path] = []
        labels: list[str] = []
        for record in view_records:
            index = int(record["index"])
            rgba = np.asarray(diagram.GetOutputPort(f"rgba_{index}").Eval(context).data)
            rgb = rgba[:, :, :3].astype(np.uint8)
            image_path = output_dir / f"{object_id}_view{index}.png"
            Image.fromarray(rgb).save(image_path)
            foreground_fraction, border_foreground_fraction = _render_coverage(rgb)
            record["image"] = str(image_path)
            record["foreground_fraction"] = foreground_fraction
            record["border_foreground_fraction"] = border_foreground_fraction
            record["render_passed"] = (
                0.005 <= foreground_fraction <= 0.75
                and border_foreground_fraction <= 0.01
            )
            paths.append(image_path)
            labels.append(
                f"yaw={record['yaw_deg']:g}, elevation={record['elevation_deg']:g}"
            )
        contact_sheet = output_dir / f"{object_id}_contact_sheet.png"
        _contact_sheet(paths, labels, contact_sheet)
        return {
            "model_instance": int(model_instance),
            "material_profile": material_profile,
            "effective_renderer": renderer,
            "bounds_low": low.tolist(),
            "bounds_high": high.tolist(),
            "contact_sheet": str(contact_sheet),
            "views": view_records,
            "render_observability_passed": all(
                bool(record["render_passed"]) for record in view_records
            ),
            "structure_gate": structure_gate,
            "structural_gate_passed": bool(structure_gate["passed"]),
            # Compatibility field consumed by the identity tooling. A candidate
            # is render-admissible only when both framing and deterministic
            # structure pass.
            "render_gate_passed": all(
                bool(record["render_passed"]) for record in view_records
            )
            and bool(structure_gate["passed"]),
        }
    finally:
        if server is not None:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-certificate", type=Path, required=True)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--views",
        type=_parse_views,
        default=_parse_views("0:25,90:25,180:25,270:25,45:70"),
        help="comma-separated yaw:elevation pairs",
    )
    parser.add_argument("--renderer", choices=("blender", "vtk"), default="blender")
    args = parser.parse_args()

    render_sources = source_digests(Path(__file__).resolve().parents[1])
    inventory = json.loads(args.inventory_certificate.resolve().read_text(encoding="utf-8"))
    task = json.loads(args.task_json.resolve().read_text(encoding="utf-8"))
    descriptions = {
        item["object_id"]: item["visual_description"] for item in task.get("objects", [])
    }
    results: dict[str, object] = {}
    errors: list[str] = []
    for match in inventory.get("matches", []):
        if match.get("status") != "matched":
            continue
        object_id = str(match["task_object_id"])
        sdf_value = match.get("sdf_path")
        if not sdf_value:
            errors.append(f"{object_id}: matched inventory entry has no sdf_path")
            continue
        try:
            rendered = render_asset(
                sdf_path=Path(sdf_value),
                output_dir=args.output_dir / object_id,
                object_id=object_id,
                width=args.width,
                height=args.height,
                views=args.views,
                renderer=args.renderer,
                visual_description=descriptions.get(object_id, ""),
            )
            validate_dependencies(rendered["material_profile"]["dependencies"])
            results[object_id] = {
                "sdf_path": str(sdf_value),
                "visual_description": descriptions.get(object_id, ""),
                **rendered,
            }
        except Exception as exc:
            errors.append(f"{object_id}: {type(exc).__name__}: {exc}")

    for object_id, result in results.items():
        structure_gate = result.get("structure_gate") or {}
        if structure_gate.get("passed") is not True:
            errors.append(
                f"{object_id}: {failed_structure_reason(structure_gate)}"
            )

    expected_ids = set(descriptions)
    rendered_ids = set(results)
    missing = sorted(expected_ids - rendered_ids)
    if missing:
        errors.append(f"task objects without identity renders: {missing}")
    render_gate_passed = not errors and all(
        bool(result["render_gate_passed"]) for result in results.values()
    )
    validate_dependencies(render_sources)
    report = {
        "schema_version": "scenecode_asset_identity_render.v3",
        "renderer": "+".join(sorted({str(result["effective_renderer"]) for result in results.values()})),
        "requested_renderer": args.renderer,
        "render_source_digests": render_sources,
        "inventory_certificate": str(args.inventory_certificate.resolve()),
        "inventory_certificate_sha256": _sha256(args.inventory_certificate.resolve()),
        "task_json": str(args.task_json.resolve()),
        "task_sha256": _sha256(args.task_json.resolve()),
        "render_gate_passed": render_gate_passed,
        "identity_critique": {
            "status": (
                "pending_vision_review" if render_gate_passed else "blocked_by_render_or_structure_gate"
            ),
            "note": (
                "Rendering and deterministic structure success are not complete "
                "semantic identity success."
            ),
        },
        "errors": errors,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "asset_identity_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    if not render_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
