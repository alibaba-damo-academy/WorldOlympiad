#!/usr/bin/env python3
"""Export an existing scene to self-contained MuJoCo MJCF format.

Takes a scene directory (e.g., outputs/2025-12-05/13-39-27/scene_039) and exports
it to a self-contained MuJoCo directory with the scene.xml and all referenced
mesh assets.

Can also export a single Drake SDF file to MuJoCo MJCF format.

Usage:
    python scripts/export_scene_to_mujoco.py <scene_path> [--output <output_path>]

Example:
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039 \
        --output /tmp/mujoco_scene
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REPO_ROOT_STR = str(REPO_ROOT)
while REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

import mujoco
import numpy as np
import trimesh
import yaml

from PIL import Image
from pydrake.all import Quaternion, RigidTransform, RotationMatrix

from scenecode.agent_utils.drake_utils import create_plant_from_dmd
from scenecode.agent_utils.house import HouseScene
from scenecode.agent_utils.room import (
    ObjectType,
    SceneObject,
    UniqueID,
    deserialize_composite_member_effective_transform,
)
from scenecode.utils.sdf_utils import parse_scale

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def apply_scale_to_trimesh(mesh: trimesh.Trimesh, scale: list[float]) -> None:
    """Apply scale transformation to mesh vertices in-place.

    Args:
        mesh: Trimesh object to scale.
        scale: [sx, sy, sz] scale factors.
    """
    if scale == [1.0, 1.0, 1.0]:
        return
    scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    mesh.apply_transform(scale_matrix)


def build_mesh_asset_filename(
    mesh_path: Path,
    sdf_dir: Path,
    room_id: str,
    scale: list[float],
) -> str:
    """Build a collision/visual mesh filename that is stable and collision-free.

    Drake-generated articulated assets commonly store per-link collision pieces in
    sibling directories like:

      E_body_1_combined_coacd/convex_piece_000.obj
      E_door_1_16_combined_coacd/convex_piece_000.obj

    Flattening those into a shared MuJoCo meshes/ directory by basename alone
    aliases unrelated meshes together. Include the asset directory and relative
    subpath so per-link meshes stay distinct.
    """

    room_prefix = f"{room_id}_" if room_id else ""
    scale_suffix = ""
    if scale != [1.0, 1.0, 1.0]:
        scale_suffix = f"_s{'_'.join(f'{s:.3g}' for s in scale)}"

    try:
        relative_parts = mesh_path.relative_to(sdf_dir).parts[:-1]
        relative_prefix = "_".join(relative_parts)
    except ValueError:
        relative_prefix = mesh_path.parent.name

    parts = [room_prefix.rstrip("_"), sdf_dir.name, relative_prefix, mesh_path.stem]
    stem = "_".join(part for part in parts if part)
    return f"{stem}{scale_suffix}{mesh_path.suffix}"


def get_degenerate_trimesh_reason(mesh: trimesh.Trimesh) -> str | None:
    """Return a human-readable reason when a mesh is not a valid 3D collider."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return f"unexpected vertex shape {vertices.shape}"
    if vertices.shape[0] < 4:
        return f"only {vertices.shape[0]} vertices"
    if not np.isfinite(vertices).all():
        return "non-finite vertices"

    centered = vertices - vertices.mean(axis=0, keepdims=True)
    extents = np.ptp(vertices, axis=0)
    scale = max(float(np.max(extents)), 1.0)
    tol = max(scale * 1e-6, 1e-9)
    rank = np.linalg.matrix_rank(centered, tol=tol)
    if rank < 3:
        extents_str = ", ".join(f"{extent:.6g}" for extent in extents)
        return f"rank {rank} geometry with extents [{extents_str}]"

    return None


def maybe_drop_degenerate_collision_geom(
    spec: mujoco.MjSpec,
    geom: mujoco._specs.MjsGeom,
    geom_name: str,
    mesh: trimesh.Trimesh,
    mesh_path: Path,
) -> bool:
    """Delete one collision geom when its mesh is too degenerate for MuJoCo."""
    reason = get_degenerate_trimesh_reason(mesh)
    if reason is None:
        return False

    console_logger.warning(
        "Dropping degenerate collision geom '%s' from '%s': %s",
        geom_name,
        mesh_path,
        reason,
    )
    spec.delete(geom)
    return True


def drop_bad_collision_mesh_from_spec(
    spec: mujoco.MjSpec,
    bad_mesh: str,
    mesh_assets: dict[str, str],
) -> bool:
    """Remove a single collision geom+mesh pair when MuJoCo rejects it."""
    if not bad_mesh.endswith("_mesh"):
        return False

    geom_name = bad_mesh.removesuffix("_mesh")
    if not geom_name.endswith("_collision"):
        return False

    geom = next((geom for geom in spec.geoms if geom.name == geom_name), None)
    if geom is None:
        return False

    console_logger.warning(
        "Dropping collision geom '%s' after MuJoCo rejected mesh '%s'",
        geom_name,
        bad_mesh,
    )
    spec.delete(geom)
    mesh_assets.pop(bad_mesh, None)

    mesh = next((mesh for mesh in spec.meshes if mesh.name == bad_mesh), None)
    if mesh is not None:
        spec.delete(mesh)
    return True


def get_bad_mesh_name_from_compile_error(err_str: str) -> str | None:
    """Extract the offending mesh name from common MuJoCo compile errors."""
    volume_match = re.search(r"mesh volume is too small: (\S+)", err_str)
    if volume_match:
        return volume_match.group(1)

    if "qhull error" not in err_str:
        return None

    qhull_match = re.search(r"Element name '([^']+)'", err_str)
    if qhull_match:
        return qhull_match.group(1)

    return None


def parse_transform_dict(transform_data: dict) -> RigidTransform:
    """Parse a serialized transform dict into a RigidTransform.

    Args:
        transform_data: Dict with 'translation' and 'rotation_wxyz' keys.

    Returns:
        RigidTransform constructed from the data.
    """
    translation = np.array(transform_data.get("translation", [0, 0, 0]))
    rotation_wxyz = transform_data.get("rotation_wxyz", [1, 0, 0, 0])
    quaternion = Quaternion(wxyz=rotation_wxyz)
    rotation_matrix = RotationMatrix(quaternion)
    return RigidTransform(rotation_matrix, translation)


def expand_composite_to_members(
    obj: SceneObject, room_offset: np.ndarray
) -> list[tuple[Path, RigidTransform, str]]:
    """Expand composite object into (sdf_path, transform, name) tuples.

    Composites (stack, pile, filled_container) contain multiple physical objects
    that are grouped together. This function extracts each member so they can be
    individually exported.

    Adapted from scenecode/robot_eval/dmd_scene.py::_expand_composite_members().

    Args:
        obj: SceneObject with composite_type in metadata.
        room_offset: [x, y, z] offset to apply to member transforms.

    Returns:
        List of (sdf_path, transform, name) for each member object.
    """
    metadata = obj.metadata
    composite_type = metadata.get("composite_type")
    members = []

    if composite_type in ("stack", "pile"):
        for asset in metadata.get("member_assets", []):
            sdf_path_str = asset.get("sdf_path")
            if not sdf_path_str:
                continue
            sdf_path = Path(sdf_path_str)
            transform = deserialize_composite_member_effective_transform(asset)
            # Apply room offset.
            new_pos = transform.translation() + room_offset
            new_transform = RigidTransform(R=transform.rotation(), p=new_pos)
            members.append((sdf_path, new_transform, asset.get("name", "member")))

    elif composite_type == "filled_container":
        # Container.
        container = metadata.get("container_asset", {})
        if container.get("sdf_path"):
            sdf_path = Path(container["sdf_path"])
            transform = deserialize_composite_member_effective_transform(container)
            new_pos = transform.translation() + room_offset
            new_transform = RigidTransform(R=transform.rotation(), p=new_pos)
            members.append(
                (sdf_path, new_transform, container.get("name", "container"))
            )
        # Fill items.
        for asset in metadata.get("fill_assets", []):
            sdf_path_str = asset.get("sdf_path")
            if not sdf_path_str:
                continue
            sdf_path = Path(sdf_path_str)
            transform = deserialize_composite_member_effective_transform(asset)
            new_pos = transform.translation() + room_offset
            new_transform = RigidTransform(R=transform.rotation(), p=new_pos)
            members.append((sdf_path, new_transform, asset.get("name", "fill")))

    return members


def parse_dmd_yaml(dmd_path: Path) -> list[dict]:
    """Parse Drake Model Directive YAML file.

    Args:
        dmd_path: Path to the DMD YAML file.

    Returns:
        List of directives from the DMD file.
    """

    def angle_axis_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> dict:
        return {"!AngleAxis": loader.construct_mapping(node)}

    yaml.SafeLoader.add_constructor("!AngleAxis", angle_axis_constructor)

    with open(dmd_path) as f:
        data = yaml.safe_load(f)

    return data.get("directives", []) if data else []


def get_model_directives(directives: list[dict]) -> list[dict]:
    """Extract add_model directives in order."""
    return [d["add_model"] for d in directives if "add_model" in d]


def get_weld_directives_by_model(directives: list[dict]) -> dict[str, dict]:
    """Index add_weld directives by child model name."""
    welds: dict[str, dict] = {}
    for directive in directives:
        if "add_weld" not in directive:
            continue
        child = directive["add_weld"].get("child", "")
        if "::" not in child:
            continue
        model_name = child.split("::", 1)[0]
        welds[model_name] = directive["add_weld"]
    return welds


def get_welded_models(directives: list[dict]) -> set[str]:
    """Extract model names that are welded (to any parent) from DMD directives.

    Any model in an add_weld directive is static, regardless of the parent
    frame (world, room_*_frame, etc.).

    Args:
        directives: List of DMD directives.

    Returns:
        Set of model names that are welded (should be static).
    """
    welded = set()
    for d in directives:
        if "add_weld" not in d:
            continue
        child = d["add_weld"].get("child", "")
        # child format is "model_name::link_name", extract model_name.
        model_name = child.split("::")[0]
        welded.add(model_name)
    return welded


def resolve_package_uri(uri: str, sdf_dir: Path) -> Path | None:
    """Resolve package:// URI to filesystem path.

    Tries common resolution strategies for Drake and ROS package URIs.

    Args:
        uri: URI string, potentially starting with 'package://'.
        sdf_dir: Directory containing the SDF file.

    Returns:
        Resolved filesystem path, or None if resolution fails.
    """
    if not uri.startswith("package://"):
        return sdf_dir / uri

    # Strip package:// prefix.
    package_path = uri[len("package://") :]

    # Strategy 1: Look relative to SDF directory (common for packaged models).
    # package://pkg_name/path/... -> try sdf_dir/../path/...
    parts = package_path.split("/", 1)
    if len(parts) == 2:
        pkg_name, rel_path = parts

        # Try common parent directories.
        for parent in [
            sdf_dir.parent,
            sdf_dir.parent.parent,
            sdf_dir.parent.parent.parent,
        ]:
            candidate = parent / rel_path
            if candidate.exists():
                return candidate

            # Also try with package name prefix (pkg_name/rel_path).
            candidate = parent / pkg_name / rel_path
            if candidate.exists():
                return candidate

    # Strategy 2: Try relative to SDF dir directly.
    candidate = sdf_dir / package_path
    if candidate.exists():
        return candidate

    return None


def resolve_scene_file_uri(uri: str, scene_dir: Path, dmd_dir: Path) -> Path | None:
    """Resolve a DMD add_model file URI to a filesystem path."""
    if uri.startswith("package://scene/"):
        rel_path = uri[len("package://scene/") :]
        candidate = scene_dir / rel_path
        if candidate.exists():
            return candidate
        return None

    if uri.startswith("file://"):
        candidate = Path(uri[len("file://") :])
        return candidate if candidate.exists() else None

    candidate = Path(uri)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    candidate = dmd_dir / uri
    return candidate if candidate.exists() else None


def get_sdf_link_poses_and_roots(
    sdf_path: Path,
) -> tuple[dict[str, tuple[list[float], list[float]]], list[str]]:
    """Return model-frame link poses and root links for an SDF model."""
    tree = parse_sdf_with_drake_namespace(sdf_path)
    root = tree.getroot()
    model_elem = root.find(".//model")
    if model_elem is None:
        return {}, []

    link_absolute_poses: dict[str, tuple[list[float], list[float]]] = {}
    links = model_elem.findall("./link")
    for link_elem in links:
        link_name = link_elem.get("name")
        if not link_name:
            continue
        link_absolute_poses[link_name] = parse_pose(link_elem.find("pose"))

    child_links = set()
    for joint_elem in model_elem.findall("./joint"):
        child_elem = joint_elem.find("child")
        if child_elem is not None and child_elem.text:
            child_links.add(child_elem.text)

    root_links = [
        link_elem.get("name")
        for link_elem in links
        if link_elem.get("name") and link_elem.get("name") not in child_links
    ]
    return link_absolute_poses, root_links


def get_dmd_reference_link_name(
    add_model_data: dict,
    weld_data: dict | None,
    link_absolute_poses: dict[str, tuple[list[float], list[float]]],
    root_links: list[str],
) -> str | None:
    """Choose the link whose DMD-authored world pose should anchor the model."""
    free_body_pose = add_model_data.get("default_free_body_pose", {})
    if free_body_pose:
        link_name = next(iter(free_body_pose))
        if link_name in link_absolute_poses:
            return link_name

    if weld_data:
        child = weld_data.get("child", "")
        if "::" in child:
            link_name = child.split("::", 1)[1]
            if link_name in link_absolute_poses:
                return link_name

    if len(root_links) == 1:
        return root_links[0]

    if root_links:
        return root_links[0]

    if link_absolute_poses:
        return next(iter(link_absolute_poses))

    return None


def infer_is_furniture_from_sdf_path(sdf_path: Path) -> bool:
    """Infer furniture-ness from the packaged scene asset path."""
    path_str = sdf_path.as_posix()
    return "/generated_assets/furniture/" in path_str


def infer_room_id_from_scene_asset_path(scene_dir: Path, sdf_path: Path) -> str:
    """Infer room_id from a packaged scene asset path.

    Examples:
    - scene_dir/room_bedroom/generated_assets/... -> bedroom
    - scene_dir/room_geometry/room_geometry_bathroom.sdf -> bathroom
    """
    try:
        rel_path = sdf_path.relative_to(scene_dir)
    except ValueError:
        rel_path = sdf_path

    if rel_path.parts:
        first = rel_path.parts[0]
        if first.startswith("room_") and first != "room_geometry":
            return first[len("room_") :]

    stem = sdf_path.stem
    if stem.startswith("room_geometry_"):
        return stem[len("room_geometry_") :]

    return ""


def parse_sdf_with_drake_namespace(sdf_path: Path) -> ET.ElementTree:
    """Parse SDF file that may contain Drake-specific namespace extensions.

    Drake SDFs often use the 'drake:' prefix for custom elements without
    declaring the namespace. This causes ET.parse() to fail with
    'unbound prefix' error. We handle this by adding namespace declarations.

    Args:
        sdf_path: Path to SDF file.

    Returns:
        Parsed ElementTree.
    """
    # Read file content.
    with open(sdf_path, "r") as f:
        content = f.read()

    # Check if file uses drake: prefix without namespace declaration.
    if "drake:" in content and "xmlns:drake" not in content:
        # Add namespace declaration to sdf root element.
        content = content.replace(
            '<sdf version="1.7">',
            '<sdf version="1.7" xmlns:drake="http://drake.mit.edu">',
        )
        content = content.replace(
            '<sdf version="1.8">',
            '<sdf version="1.8" xmlns:drake="http://drake.mit.edu">',
        )
        content = content.replace(
            '<sdf version="1.9">',
            '<sdf version="1.9" xmlns:drake="http://drake.mit.edu">',
        )

    return ET.ElementTree(ET.fromstring(content))


def create_mujoco_spec_with_environment(
    model_name: str, ground_collides: bool = False
) -> mujoco.MjSpec:
    """Create MuJoCo spec with skybox and ground plane.

    Args:
        model_name: Name for the model.
        ground_collides: Whether ground plane should have collision.

    Returns:
        Configured MuJoCo spec.
    """
    spec = mujoco.MjSpec()
    spec.modelname = model_name
    spec.compiler.degree = False
    spec.compiler.balanceinertia = True
    spec.compiler.boundmass = 0.001
    spec.compiler.boundinertia = 0.001

    # Visual settings.
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]

    # Skybox.
    skybox = spec.add_texture(name="skybox")
    skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox.rgb1 = [0.3, 0.5, 0.7]
    skybox.rgb2 = [0.0, 0.0, 0.0]
    skybox.width = 512
    skybox.height = 512

    # Ground plane texture and material.
    grid_texture = spec.add_texture(name="grid")
    grid_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid_texture.rgb1 = [0.2, 0.3, 0.4]
    grid_texture.rgb2 = [0.1, 0.2, 0.3]
    grid_texture.width = 512
    grid_texture.height = 512
    grid_texture.mark = mujoco.mjtMark.mjMARK_EDGE
    grid_texture.markrgb = [0.8, 0.8, 0.8]

    grid_material = spec.add_material(name="grid")
    grid_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    grid_material.texrepeat = [10, 10]
    grid_material.reflectance = 0.0

    # Ground plane geom.
    ground = spec.worldbody.add_geom(name="ground_plane")
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [0, 0, 0.05]
    ground.pos = [0, 0, -0.001]
    ground.material = "grid"
    ground.contype = 1 if ground_collides else 0
    ground.conaffinity = 1 if ground_collides else 0

    return spec


def find_gltf_color_texture(gltf_path: Path) -> Path | None:
    """Find color texture referenced by a GLTF file.

    Parses the GLTF JSON to find external texture references.

    Args:
        gltf_path: Path to GLTF file.

    Returns:
        Path to color texture file, or None if not found.
    """
    try:
        with open(gltf_path, "r") as f:
            gltf_data = json.load(f)

        # Find images referenced in the GLTF.
        images = gltf_data.get("images", [])
        if not images:
            return None

        # Look for color texture (usually index 0 in PBR materials).
        # Check materials to find baseColorTexture index.
        materials = gltf_data.get("materials", [])
        color_texture_idx = None
        for mat in materials:
            pbr = mat.get("pbrMetallicRoughness", {})
            base_color = pbr.get("baseColorTexture", {})
            if "index" in base_color:
                # Get texture, then get its source (image index).
                tex_idx = base_color["index"]
                textures = gltf_data.get("textures", [])
                if tex_idx < len(textures):
                    color_texture_idx = textures[tex_idx].get("source", 0)
                    break

        # Default to first image if no specific color texture found.
        if color_texture_idx is None:
            color_texture_idx = 0

        if color_texture_idx < len(images):
            image_uri = images[color_texture_idx].get("uri")
            if image_uri:
                texture_path = gltf_path.parent / image_uri
                if texture_path.exists():
                    return texture_path
                # Try resolving relative path.
                texture_path = (gltf_path.parent / image_uri).resolve()
                if texture_path.exists():
                    return texture_path

        return None
    except Exception as e:
        console_logger.debug(f"Could not parse GLTF for textures: {e}")
        return None


def convert_gltf_to_obj(
    gltf_path: Path,
    obj_path: Path,
    texture_dir: Path | None = None,
    scale: list[float] | None = None,
) -> tuple[bool, Path | None, list[float] | None]:
    """Convert GLTF file to OBJ format using trimesh.

    GLTF uses Y-up coordinate system, OBJ uses Z-up. This function applies
    the necessary rotation during conversion. Optionally applies scale.

    Args:
        gltf_path: Path to input GLTF file.
        obj_path: Path for output OBJ file.
        texture_dir: Directory to save extracted textures.
        scale: Optional [sx, sy, sz] scale factors from SDF.

    Returns:
        Tuple of (success, texture_path or None, base_color_rgba or None).
    """
    try:
        # Use force='mesh' to get a single mesh with UV coordinates preserved.
        # Using Scene + concatenate loses UVs.
        mesh = trimesh.load(gltf_path, force="mesh")

        # Validate mesh has minimum vertices required by MuJoCo.
        if mesh.vertices.shape[0] < 4:
            console_logger.warning(
                f"Mesh {gltf_path.name} has only {mesh.vertices.shape[0]} vertices "
                f"(MuJoCo requires at least 4). Skipping conversion."
            )
            return False, None, None

        # GLTF is Y-up, MuJoCo uses Z-up. Apply the same Y-up to Z-up transform
        # that scenecode uses for consistency with collision geometry.
        # This is a +90° rotation around X axis: x'=x, y'=-z, z'=y.
        yup_to_zup = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        mesh.apply_transform(yup_to_zup)

        # Apply scale if provided.
        if scale is not None and scale != [1.0, 1.0, 1.0]:
            apply_scale_to_trimesh(mesh, scale)

        mesh.export(obj_path)
        console_logger.info(
            f"Converted {gltf_path.name} -> {obj_path.name} (Y-up -> Z-up)"
        )

        # Extract texture if available.
        texture_path = None
        base_color = None

        if texture_dir:
            # Method 1: Try to find external texture referenced in GLTF.
            external_texture = find_gltf_color_texture(gltf_path)
            if external_texture:
                # MuJoCo only supports PNG textures, so convert if necessary.
                dest_texture = texture_dir / f"{obj_path.stem}_texture.png"
                if not dest_texture.exists():
                    if external_texture.suffix.lower() in (".jpg", ".jpeg"):
                        # Convert JPG to PNG.
                        img = Image.open(external_texture)
                        img.save(dest_texture, "PNG")
                        console_logger.info(
                            f"Converted texture: {external_texture.name} -> "
                            f"{dest_texture.name}"
                        )
                    else:
                        shutil.copy(external_texture, dest_texture)
                        console_logger.info(
                            f"Copied external texture: {dest_texture.name}"
                        )
                texture_path = dest_texture

            # Method 2: Try to extract embedded texture from mesh.
            if texture_path is None:
                try:
                    if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
                        material = mesh.visual.material
                        image = None

                        # Try different ways trimesh stores textures.
                        if hasattr(material, "image") and material.image is not None:
                            image = material.image
                        elif (
                            hasattr(material, "baseColorTexture")
                            and material.baseColorTexture is not None
                        ):
                            image = material.baseColorTexture

                        if image is not None:
                            texture_path = texture_dir / f"{obj_path.stem}_texture.png"
                            image.save(texture_path)
                            console_logger.info(
                                f"Extracted embedded texture: {texture_path.name}"
                            )
                except Exception as tex_err:
                    console_logger.debug(
                        f"Could not extract embedded texture: {tex_err}"
                    )

        # Method 3: Extract base color from PBR material if no texture found.
        if texture_path is None:
            base_color = get_gltf_base_color(gltf_path)

        return True, texture_path, base_color
    except Exception as e:
        console_logger.warning(f"Failed to convert {gltf_path}: {e}")
        return False, None, None


def get_gltf_base_color(gltf_path: Path) -> list[float] | None:
    """Extract base color from GLTF PBR material.

    Args:
        gltf_path: Path to GLTF file.

    Returns:
        RGBA color as [r, g, b, a] or None if not found.
    """
    try:
        with open(gltf_path, "r") as f:
            gltf_data = json.load(f)

        materials = gltf_data.get("materials", [])
        if materials:
            # Get first material's base color.
            mat = materials[0]
            pbr = mat.get("pbrMetallicRoughness", {})
            base_color = pbr.get("baseColorFactor")
            if base_color and len(base_color) >= 3:
                # Ensure we have RGBA.
                if len(base_color) == 3:
                    base_color = base_color + [1.0]
                return base_color
        return None
    except Exception:
        return None


def _remap_room_geometry_paths(state_dict: dict, scene_dir: Path) -> None:
    """Remap stale absolute sdf_path in room_geometry to valid paths.

    Scenes moved after generation have stale absolute paths pointing to the
    original outputs/ directory. The SDF files still exist at the same path
    relative to the scene root (e.g. room_geometry/room_geometry_bedroom.sdf).
    Since RoomGeometry.from_dict resolves paths relative to the room subdir
    (not the scene root), we rewrite the path as an absolute path pointing
    to the actual file location.
    """
    for room_data in state_dict.get("rooms", {}).values():
        rg = room_data.get("room_geometry", {})
        sdf_path_str = rg.get("sdf_path")
        if not sdf_path_str:
            continue
        sdf_path = Path(sdf_path_str)
        if sdf_path.is_absolute() and not sdf_path.exists():
            # Extract the portion after scene_NNN/.
            parts = sdf_path.parts
            for i, part in enumerate(parts):
                if part.startswith("scene_"):
                    relative = str(Path(*parts[i + 1 :]))
                    candidate = scene_dir / relative
                    if candidate.exists():
                        rg["sdf_path"] = str(candidate.resolve())
                        console_logger.debug(
                            f"Remapped sdf_path: {sdf_path_str} -> " f"{candidate}"
                        )
                    break


def load_house_from_directory(scene_dir: Path) -> HouseScene:
    """Load a house scene from a scene directory.

    Args:
        scene_dir: Path to scene directory (e.g., outputs/.../scene_039).

    Returns:
        Reconstructed HouseScene object with all rooms.

    Raises:
        FileNotFoundError: If house_state.json not found.
    """
    house_state_path = scene_dir / "combined_house" / "house_state.json"
    if not house_state_path.exists():
        raise FileNotFoundError(f"House state not found: {house_state_path}")

    with open(house_state_path, "r") as f:
        state_dict = json.load(f)

    # Remap absolute sdf_path values in room_geometry to be relative to the
    # scene directory. Scenes that were moved after generation have stale
    # absolute paths but the files still exist at the same relative location.
    _remap_room_geometry_paths(state_dict, scene_dir)

    # Use HouseScene.from_state_dict to restore the full house.
    house = HouseScene.from_state_dict(state_dict, house_dir=scene_dir.resolve())

    # Log summary.
    total_furniture = 0
    total_manipulands = 0
    for room_id, room in house.rooms.items():
        furniture_count = len(room.get_objects_by_type(ObjectType.FURNITURE))
        manipuland_count = len(room.get_manipulands())
        total_furniture += furniture_count
        total_manipulands += manipuland_count
        console_logger.info(
            f"Room '{room_id}': {furniture_count} furniture, {manipuland_count} manipulands"
        )

    console_logger.info(
        f"Loaded house with {len(house.rooms)} rooms, "
        f"{total_furniture} furniture, {total_manipulands} manipulands"
    )

    return house


def process_sdf_model(
    sdf_path: Path,
    sdf_dir: Path,
    model_name: str,
    transform_pos: list[float],
    transform_quat: list[float],
    is_static: bool,
    spec: mujoco.MjSpec,
    meshes_dir: Path,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
    color_assets: dict[str, list[float]],
    room_id: str = "",
) -> list[str]:
    """Process an SDF model and add bodies/joints to spec.

    Supports both single-link (rigid) and multi-link (articulated) models.

    Args:
        sdf_path: Path to SDF file.
        sdf_dir: Directory containing SDF (for mesh resolution).
        model_name: Unique name for the model in MuJoCo.
        transform_pos: [x, y, z] position for model root.
        transform_quat: [w, x, y, z] quaternion for model root.
        is_static: Whether model should be static (no freejoint).
        spec: MuJoCo spec to add bodies to.
        meshes_dir: Directory for mesh assets.
        mesh_assets: Dict to track mesh assets (name -> filename).
        texture_assets: Dict to track texture assets (name -> filename).
        color_assets: Dict to track base color assets (name -> [r,g,b,a]).
        room_id: Optional room identifier for unique mesh naming across rooms.

    Returns:
        List of body names created for this model. Used for self-collision
        filtering on articulated models.
    """
    # Parse SDF (handles Drake namespace extensions).
    tree = parse_sdf_with_drake_namespace(sdf_path)
    root = tree.getroot()
    model_elem = root.find(".//model")
    if model_elem is None:
        console_logger.warning(f"No model element in SDF: {sdf_path}")
        return []

    # Check if SDF model is marked static.
    static_elem = model_elem.find("static")
    sdf_is_static = static_elem is not None and static_elem.text.lower() == "true"

    # Build link→parent mapping from joints.
    link_parents: dict[str, tuple[str, ET.Element | None]] = {}
    for joint_elem in model_elem.findall(".//joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is not None and child_elem is not None:
            link_parents[child_elem.text] = (parent_elem.text, joint_elem)

    # Collect links and their absolute poses in model frame.
    links = model_elem.findall(".//link")
    link_absolute_poses: dict[str, tuple[list[float], list[float]]] = {}
    for link_elem in links:
        link_name = link_elem.get("name")
        pos, quat = parse_pose(link_elem.find("pose"))
        link_absolute_poses[link_name] = (pos, quat)

    # Track created bodies and their absolute poses.
    link_bodies: dict[str, mujoco._specs.MjsBody] = {}

    # Create model root body with transform.
    model_root = spec.worldbody.add_body(name=model_name)
    model_root.pos = transform_pos
    model_root.quat = transform_quat

    # Add freejoint for dynamic objects.
    if not is_static and not sdf_is_static:
        model_root.add_freejoint(name=f"{model_name}_freejoint")

    # Process links in topological order (parents before children).
    processed: set[str] = set()
    while len(processed) < len(links):
        progress_made = False
        for link_elem in links:
            link_name = link_elem.get("name")
            if link_name in processed:
                continue

            parent_link_name, joint_elem = link_parents.get(link_name, (None, None))

            # Get parent body and its absolute pose.
            if parent_link_name is None:
                # Root link - parent is model root at origin in model frame.
                parent_body = model_root
                parent_abs_pos = [0.0, 0.0, 0.0]
                parent_abs_quat = [1.0, 0.0, 0.0, 0.0]
            elif parent_link_name in link_bodies:
                parent_body = link_bodies[parent_link_name]
                parent_abs_pos, parent_abs_quat = link_absolute_poses[parent_link_name]
            else:
                # Parent not yet processed - skip for now.
                continue

            body_name = f"{model_name}_{link_name}"
            child_abs_pos, child_abs_quat = link_absolute_poses[link_name]

            # Compute child pose relative to parent.
            rel_pos, rel_quat = compute_relative_pose(
                parent_abs_pos, parent_abs_quat, child_abs_pos, child_abs_quat
            )

            body = parent_body.add_body(name=body_name)
            body.pos = rel_pos
            body.quat = rel_quat

            inertial_elem = link_elem.find("inertial")
            if inertial_elem is not None:
                apply_inertial(body, inertial_elem)

            if joint_elem is not None:
                joint_type = joint_elem.get("type", "fixed")
                if joint_type != "fixed":
                    # Parse joint pose (anchor position relative to child body).
                    joint_pose_elem = joint_elem.find("pose")
                    joint_pos, _ = parse_pose(joint_pose_elem)
                    # Pass the child's absolute quaternion for axis transformation.
                    add_joint_from_sdf(
                        body=body,
                        joint_elem=joint_elem,
                        child_abs_quat=child_abs_quat,
                        joint_pos=joint_pos,
                        name_prefix=body_name,
                    )

            for collision_elem in link_elem.findall("collision"):
                add_geom_from_sdf(
                    spec=spec,
                    body=body,
                    geom_elem=collision_elem,
                    sdf_dir=sdf_dir,
                    meshes_dir=meshes_dir,
                    mesh_assets=mesh_assets,
                    texture_assets=texture_assets,
                    color_assets=color_assets,
                    is_collision=True,
                    name_prefix=body_name,
                    room_id=room_id,
                )

            for visual_elem in link_elem.findall("visual"):
                add_geom_from_sdf(
                    spec=spec,
                    body=body,
                    geom_elem=visual_elem,
                    sdf_dir=sdf_dir,
                    meshes_dir=meshes_dir,
                    mesh_assets=mesh_assets,
                    texture_assets=texture_assets,
                    color_assets=color_assets,
                    is_collision=False,
                    name_prefix=body_name,
                    room_id=room_id,
                )

            link_bodies[link_name] = body
            processed.add(link_name)
            progress_made = True

        if not progress_made:
            unprocessed = [
                l.get("name") for l in links if l.get("name") not in processed
            ]
            console_logger.warning(
                f"Could not process all links for {model_name}. "
                f"Unprocessed: {unprocessed}"
            )
            break

    return [f"{model_name}_{l.get('name')}" for l in links]


def export_scene_to_mujoco(
    house: HouseScene,
    output_dir: Path,
    include_floor_plan: bool = True,
    weld_furniture: bool = False,
) -> Path:
    """Export house scene to self-contained MuJoCo directory.

    Creates a directory with:
    - scene.xml: Main MJCF file
    - meshes/: All referenced mesh files (converted to OBJ if necessary)

    Args:
        house: HouseScene object to export (contains all rooms).
        output_dir: Output directory path.
        include_floor_plan: Whether to include floor plan objects.
        weld_furniture: Whether to weld furniture (make static). Default False
            means furniture has freejoints and can move/fall.

    Returns:
        Path to the exported scene.xml file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    output_path = output_dir / "scene.xml"

    # Create MuJoCo spec with environment (skybox, ground).
    spec = create_mujoco_spec_with_environment(
        model_name=f"scene_{house.house_dir.name}",
        ground_collides=False,  # Scene has its own floor.
    )

    # Track mesh, texture, and color assets.
    mesh_assets: dict[str, str] = {}
    texture_assets: dict[str, str] = {}
    color_assets: dict[str, list[float]] = {}

    # Parse DMD to get welded models (should be static in MuJoCo).
    # Use house.dmd.yaml which only welds room geometry (floors, walls).
    # Furniture and manipulands keep their freejoints.
    dmd_path = house.house_dir / "combined_house" / "house.dmd.yaml"
    if dmd_path.exists():
        directives = parse_dmd_yaml(dmd_path)
        welded_models = get_welded_models(directives)
        console_logger.info(
            f"Found {len(welded_models)} welded models in " f"{dmd_path.name}"
        )
    else:
        welded_models = set()
        console_logger.debug("No DMD file found, using heuristic welding")

    # Track body names for articulated models (for self-collision filtering).
    articulated_model_bodies: list[list[str]] = []

    def process_scene_object(
        obj: SceneObject, is_static: bool, room_id: str = ""
    ) -> list[str]:
        """Process a SceneObject and return list of body names created."""
        if not obj.sdf_path or not obj.sdf_path.exists():
            console_logger.warning(f"SDF not found for {obj.name}: {obj.sdf_path}")
            return []

        # Parse SDF to get model name for unique naming.
        tree = parse_sdf_with_drake_namespace(obj.sdf_path)
        model_elem = tree.getroot().find(".//model")
        if model_elem is None:
            console_logger.warning(f"No model element in SDF: {obj.sdf_path}")
            return []

        # Use room_id and full object_id to ensure unique model names. Different
        # objects can share the same SDF model name, so the object_id is needed.
        room_prefix = f"{room_id}_" if room_id else ""
        model_name = f"{room_prefix}{obj.object_id}"

        # Extract transform.
        translation = obj.transform.translation()
        rotation = obj.transform.rotation().ToQuaternion()

        return process_sdf_model(
            sdf_path=obj.sdf_path,
            sdf_dir=obj.sdf_path.parent,
            model_name=model_name,
            transform_pos=[
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
            ],
            transform_quat=[
                float(rotation.w()),
                float(rotation.x()),
                float(rotation.y()),
                float(rotation.z()),
            ],
            is_static=is_static,
            spec=spec,
            meshes_dir=meshes_dir,
            mesh_assets=mesh_assets,
            texture_assets=texture_assets,
            color_assets=color_assets,
            room_id=room_id,
        )

    # Process all rooms in the house.
    for room_id, room in house.rooms.items():
        # Get room position offset for multi-room scenes.
        room_offset_x, room_offset_y = house._get_room_position(room_id)
        room_offset = np.array([room_offset_x, room_offset_y, 0.0])

        # Add floor plan (static).
        # The room_geometry.sdf contains both floor and walls as a single model.
        if include_floor_plan and room.room_geometry and room.room_geometry.sdf_path:
            sdf_path = room.room_geometry.sdf_path
            if sdf_path.exists():
                # Create a pseudo SceneObject for the room geometry.
                room_geometry_obj = SceneObject(
                    object_id=UniqueID(f"room_geometry_{room_id}"),
                    object_type=ObjectType.FLOOR,
                    name=f"room_geometry_{room_id}",
                    description=f"Room geometry for {room_id}",
                    transform=RigidTransform(p=room_offset),
                    sdf_path=sdf_path,
                )
                process_scene_object(room_geometry_obj, is_static=True, room_id=room_id)

        # Add furniture and manipulands with room offset applied.
        for obj in room.objects.values():
            if obj.object_type in (ObjectType.WALL, ObjectType.FLOOR):
                continue

            # Handle composite objects by expanding into member components.
            composite_type = obj.metadata.get("composite_type")
            if composite_type in ("stack", "pile", "filled_container"):
                members = expand_composite_to_members(obj, room_offset)
                # Use parent object_id directly for uniqueness.
                parent_id = str(obj.object_id)
                for idx, (sdf_path, transform, name) in enumerate(members):
                    if not sdf_path.exists():
                        console_logger.warning(
                            f"SDF not found for composite member: {sdf_path}"
                        )
                        continue
                    # Create pseudo SceneObject for member with unique ID.
                    # Use format that puts unique suffix at end for id_suffix extraction.
                    # Include both parent_id and member index for uniqueness.
                    unique_suffix = f"{parent_id}m{idx}"
                    member_obj = SceneObject(
                        object_id=UniqueID(unique_suffix),
                        object_type=ObjectType.MANIPULAND,
                        name=f"{name}_m{idx}",
                        description=f"Member of {obj.name}",
                        transform=transform,
                        sdf_path=sdf_path,
                    )
                    process_scene_object(member_obj, is_static=False, room_id=room_id)
                continue  # Don't process composite container itself.

            # Determine if object should be static (no freejoint).
            # DMD welding is the source of truth for which objects are welded.
            # The weld_furniture flag is an override for all furniture.
            # Model name format must match room.py:1413-1416.
            id_suffix = str(obj.object_id).split("_")[-1][:8]
            dmd_model_name = (
                f"{room_id}_{obj.name.lower().replace(' ', '_')}_{id_suffix}"
            )
            is_welded_in_dmd = dmd_model_name in welded_models

            should_be_static = is_welded_in_dmd or (
                weld_furniture and obj.object_type == ObjectType.FURNITURE
            )

            # Apply room offset to object transform.
            obj_translation = obj.transform.translation() + room_offset
            obj_with_offset = SceneObject(
                object_id=obj.object_id,
                object_type=obj.object_type,
                name=obj.name,
                description=obj.description,
                transform=RigidTransform(R=obj.transform.rotation(), p=obj_translation),
                sdf_path=obj.sdf_path,
                geometry_path=obj.geometry_path,
                image_path=obj.image_path,
                support_surfaces=obj.support_surfaces,
                placement_info=obj.placement_info,
                metadata=obj.metadata,
            )
            body_names = process_scene_object(
                obj_with_offset, is_static=should_be_static, room_id=room_id
            )
            # Track articulated models (multi-link with joints) for
            # self-collision filtering.
            if len(body_names) > 1:
                articulated_model_bodies.append(body_names)

    add_articulated_self_collision_exclusions(spec, articulated_model_bodies)
    add_mujoco_assets_to_spec(spec, mesh_assets, texture_assets)
    return compile_and_write_mjcf(
        spec=spec,
        output_path=output_path,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
    )


def export_dmd_scene_to_mujoco(
    scene_dir: Path,
    dmd_path: Path,
    output_dir: Path,
    include_floor_plan: bool = True,
    weld_furniture: bool = False,
) -> Path:
    """Export a scene directly from house.dmd.yaml plus referenced SDF assets.

    This is the clean fallback for archived scenes that still contain the
    authoritative Drake directives and packaged SDF assets but no longer keep
    house_state.json metadata around.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    output_path = output_dir / "scene.xml"

    spec = create_mujoco_spec_with_environment(
        model_name=f"scene_{scene_dir.name}",
        ground_collides=False,
    )

    mesh_assets: dict[str, str] = {}
    texture_assets: dict[str, str] = {}
    color_assets: dict[str, list[float]] = {}
    articulated_model_bodies: list[list[str]] = []

    directives = parse_dmd_yaml(dmd_path)
    model_directives = get_model_directives(directives)
    welded_models = get_welded_models(directives)
    weld_directives = get_weld_directives_by_model(directives)

    console_logger.info(
        f"Loaded {len(model_directives)} model directive(s) from {dmd_path.name}"
    )

    builder, plant, scene_graph = create_plant_from_dmd(dmd_path, scene_dir)
    del builder, scene_graph
    context = plant.CreateDefaultContext()

    for add_model_data in model_directives:
        model_name = add_model_data.get("name")
        file_uri = add_model_data.get("file")
        if not model_name or not file_uri:
            continue

        if not include_floor_plan and model_name.startswith("room_geometry_"):
            continue

        sdf_path = resolve_scene_file_uri(
            file_uri, scene_dir=scene_dir, dmd_dir=dmd_path.parent
        )
        if sdf_path is None or not sdf_path.exists():
            console_logger.warning(f"SDF not found for {model_name}: {file_uri}")
            continue

        room_id = infer_room_id_from_scene_asset_path(scene_dir, sdf_path)

        link_absolute_poses, root_links = get_sdf_link_poses_and_roots(sdf_path)
        weld_data = weld_directives.get(model_name)
        reference_link_name = get_dmd_reference_link_name(
            add_model_data=add_model_data,
            weld_data=weld_data,
            link_absolute_poses=link_absolute_poses,
            root_links=root_links,
        )
        if reference_link_name is None:
            console_logger.warning(
                f"Could not determine reference link for {model_name}: {sdf_path}"
            )
            continue

        try:
            model_instance = plant.GetModelInstanceByName(model_name)
            reference_body = plant.GetBodyByName(reference_link_name, model_instance)
        except RuntimeError as exc:
            console_logger.warning(
                f"Skipping {model_name}; Drake model lookup failed: {exc}"
            )
            continue

        x_wl = plant.EvalBodyPoseInWorld(context, reference_body)
        ref_pos, ref_quat = link_absolute_poses.get(
            reference_link_name, ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        )
        x_ml = RigidTransform(
            RotationMatrix(Quaternion(wxyz=ref_quat)),
            ref_pos,
        )
        x_wm = x_wl @ x_ml.inverse()
        q_wm = x_wm.rotation().ToQuaternion()

        is_static = model_name in welded_models or (
            weld_furniture and infer_is_furniture_from_sdf_path(sdf_path)
        )

        body_names = process_sdf_model(
            sdf_path=sdf_path,
            sdf_dir=sdf_path.parent,
            model_name=model_name,
            transform_pos=[
                float(x_wm.translation()[0]),
                float(x_wm.translation()[1]),
                float(x_wm.translation()[2]),
            ],
            transform_quat=[
                float(q_wm.w()),
                float(q_wm.x()),
                float(q_wm.y()),
                float(q_wm.z()),
            ],
            is_static=is_static,
            spec=spec,
            meshes_dir=meshes_dir,
            mesh_assets=mesh_assets,
            texture_assets=texture_assets,
            color_assets=color_assets,
            room_id=room_id,
        )
        if len(body_names) > 1:
            articulated_model_bodies.append(body_names)

    add_articulated_self_collision_exclusions(spec, articulated_model_bodies)
    add_mujoco_assets_to_spec(spec, mesh_assets, texture_assets)
    return compile_and_write_mjcf(
        spec=spec,
        output_path=output_path,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
    )


def add_articulated_self_collision_exclusions(
    spec: mujoco.MjSpec,
    articulated_model_bodies: list[list[str]],
) -> None:
    """Disable self-collisions within each articulated model."""
    for body_names in articulated_model_bodies:
        for i in range(len(body_names)):
            for j in range(i + 1, len(body_names)):
                exclude = spec.add_exclude()
                exclude.name = f"selfcol_{body_names[i]}_{body_names[j]}"
                exclude.bodyname1 = body_names[i]
                exclude.bodyname2 = body_names[j]


def add_mujoco_assets_to_spec(
    spec: mujoco.MjSpec,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
) -> None:
    """Attach mesh and texture assets to the MuJoCo spec."""
    for mesh_name, mesh_filename in mesh_assets.items():
        mesh = spec.add_mesh(name=mesh_name)
        mesh.file = mesh_filename

    for texture_name, texture_filename in texture_assets.items():
        texture = spec.add_texture(name=texture_name)
        texture.file = texture_filename
        texture.type = mujoco.mjtTexture.mjTEXTURE_2D

        material = spec.add_material(name=texture_name)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name

    console_logger.info(f"Total textures: {len(texture_assets)}")


def compile_and_write_mjcf(
    spec: mujoco.MjSpec,
    output_path: Path,
    meshes_dir: Path,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
) -> Path:
    """Compile an MjSpec and write the final XML next to its mesh assets."""
    original_cwd = os.getcwd()
    try:
        os.chdir(meshes_dir)
        dropped_meshes: set[str] = set()
        while True:
            try:
                spec.compile()
                break
            except ValueError as e:
                err_str = str(e)
                bad_mesh = get_bad_mesh_name_from_compile_error(err_str)
                if bad_mesh is None or bad_mesh in dropped_meshes:
                    raise
                if not drop_bad_collision_mesh_from_spec(spec, bad_mesh, mesh_assets):
                    raise
                dropped_meshes.add(bad_mesh)
        xml_string = spec.to_xml()
    finally:
        os.chdir(original_cwd)

    xml_string = re.sub(
        r"<compiler([^/]*)/\s*>",
        r'<compiler\1 meshdir="meshes" texturedir="meshes"/>',
        xml_string,
    )
    xml_string = re.sub(
        r"<compiler([^/>]*)>",
        r'<compiler\1 meshdir="meshes" texturedir="meshes">',
        xml_string,
    )

    with open(output_path, "w") as f:
        f.write(xml_string)

    console_logger.info(f"Exported MJCF to: {output_path}")
    console_logger.info(f"Mesh assets in: {meshes_dir}")
    console_logger.info(f"Total meshes: {len(mesh_assets)}")
    console_logger.info(f"Total textures: {len(texture_assets)}")

    return output_path


def parse_pose(pose_elem: ET.Element | None) -> tuple[list[float], list[float]]:
    """Parse SDF pose element into position and quaternion."""
    if pose_elem is None or pose_elem.text is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

    values = [float(v) for v in pose_elem.text.split()]
    pos = values[:3]

    if len(values) >= 6:
        roll, pitch, yaw = values[3:6]
        quat = rpy_to_quat(roll, pitch, yaw)
    else:
        quat = [1.0, 0.0, 0.0, 0.0]

    return pos, quat


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert roll-pitch-yaw to quaternion (w-first)."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [float(w), float(x), float(y), float(z)]


def quat_conjugate(q: list[float]) -> list[float]:
    """Return conjugate of quaternion (w, x, y, z)."""
    return [q[0], -q[1], -q[2], -q[3]]


def quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_rotate_vector(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by quaternion q."""
    # v' = q * v * q^-1 where v is treated as quaternion (0, vx, vy, vz).
    v_quat = [0.0, v[0], v[1], v[2]]
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return [result[1], result[2], result[3]]


def quat_to_rotation_matrix(q: list[float]) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def compute_relative_pose(
    parent_pos: list[float],
    parent_quat: list[float],
    child_pos: list[float],
    child_quat: list[float],
) -> tuple[list[float], list[float]]:
    """Compute child pose relative to parent.

    Given parent and child poses in world/model frame, compute the child's
    pose relative to the parent frame.

    Args:
        parent_pos: Parent position in model frame [x, y, z].
        parent_quat: Parent quaternion in model frame [w, x, y, z].
        child_pos: Child position in model frame [x, y, z].
        child_quat: Child quaternion in model frame [w, x, y, z].

    Returns:
        (relative_pos, relative_quat): Child pose in parent frame.
    """
    # Relative position: rotate (child_pos - parent_pos) by inverse of parent rotation.
    delta_pos = [
        child_pos[0] - parent_pos[0],
        child_pos[1] - parent_pos[1],
        child_pos[2] - parent_pos[2],
    ]
    parent_quat_inv = quat_conjugate(parent_quat)
    rel_pos = quat_rotate_vector(parent_quat_inv, delta_pos)

    # Relative rotation: q_rel = q_parent^-1 * q_child.
    rel_quat = quat_multiply(parent_quat_inv, child_quat)

    return rel_pos, rel_quat


def apply_inertial(body: mujoco._specs.MjsBody, inertial_elem: ET.Element) -> None:
    """Apply inertial properties from SDF to MuJoCo body.

    Preserves off-diagonal inertia terms using MuJoCo's fullinertia attribute
    when needed. Falls back to diagonal inertia when off-diagonal terms are zero
    and the inertial frame has identity rotation.
    """
    mass_elem = inertial_elem.find("mass")
    if mass_elem is not None and mass_elem.text:
        body.mass = float(mass_elem.text)

    pose_elem = inertial_elem.find("pose")
    pos = [0.0, 0.0, 0.0]
    quat = [1.0, 0.0, 0.0, 0.0]
    if pose_elem is not None:
        pos, quat = parse_pose(pose_elem)
    body.ipos = pos

    inertia_elem = inertial_elem.find("inertia")
    if inertia_elem is not None:
        ixx = get_float(inertia_elem, "ixx", 0.0)
        iyy = get_float(inertia_elem, "iyy", 0.0)
        izz = get_float(inertia_elem, "izz", 0.0)
        ixy = get_float(inertia_elem, "ixy", 0.0)
        ixz = get_float(inertia_elem, "ixz", 0.0)
        iyz = get_float(inertia_elem, "iyz", 0.0)

        has_off_diagonal = ixy != 0.0 or ixz != 0.0 or iyz != 0.0
        has_rotation = quat != [1.0, 0.0, 0.0, 0.0]

        if has_off_diagonal or has_rotation:
            # Build full 3x3 symmetric inertia tensor.
            I_local = np.array(
                [
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz],
                ]
            )

            if has_rotation:
                # Transform inertia tensor from inertial frame to body frame.
                R = quat_to_rotation_matrix(quat)
                I_body = R @ I_local @ R.T
            else:
                I_body = I_local

            # fullinertia expects [ixx, iyy, izz, ixy, ixz, iyz].
            body.fullinertia = [
                I_body[0, 0],
                I_body[1, 1],
                I_body[2, 2],
                I_body[0, 1],
                I_body[0, 2],
                I_body[1, 2],
            ]
            # Do not set iquat; MuJoCo compiler sets it from eigendecomposition.
        else:
            body.inertia = [ixx, iyy, izz]
            body.iquat = quat

    body.explicitinertial = True


def get_float(parent: ET.Element, tag: str, default: float) -> float:
    """Get float value from child element."""
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return float(elem.text)
    return default


def add_joint_from_sdf(
    body: mujoco._specs.MjsBody,
    joint_elem: ET.Element,
    child_abs_quat: list[float] | None = None,
    joint_pos: list[float] | None = None,
    name_prefix: str = "",
) -> None:
    """Add joint to body based on SDF joint element.

    Args:
        body: MuJoCo body to add joint to.
        joint_elem: SDF <joint> element.
        child_abs_quat: Child link's absolute quaternion in model frame [w, x, y, z].
            Used to transform axis from model frame to child link frame.
        name_prefix: Prefix for joint name to ensure uniqueness across rooms.
        joint_pos: Joint anchor position [x, y, z] from SDF joint pose.
    """
    raw_joint_name = joint_elem.get("name", "joint")
    # Prefix with name_prefix (model_name + link_name) to ensure uniqueness
    # across rooms that may have the same articulated furniture.
    joint_name = f"{name_prefix}_{raw_joint_name}" if name_prefix else raw_joint_name
    joint_type_str = joint_elem.get("type", "revolute")

    # Get MuJoCo joint type.
    mj_joint_type = SDF_TO_MJCF_JOINT_TYPE.get(joint_type_str)
    if mj_joint_type is None:
        console_logger.warning(f"Unsupported joint type: {joint_type_str}")
        return

    # Create joint.
    joint = body.add_joint(name=joint_name)
    joint.type = mj_joint_type

    # Apply joint position (anchor point) if provided.
    if joint_pos is not None:
        joint.pos = joint_pos

    # Parse axis.
    axis_elem = joint_elem.find("axis")
    if axis_elem is not None:
        xyz_elem = axis_elem.find("xyz")
        if xyz_elem is not None and xyz_elem.text:
            axis_values = [float(v) for v in xyz_elem.text.split()]

            # Check if axis is expressed in model frame.
            # If so, transform it to the child link frame.
            expressed_in = xyz_elem.get("expressed_in", "")
            if expressed_in == "__model__" and child_abs_quat is not None:
                # Transform axis from model frame to child link frame.
                # axis_in_child = q_child^-1 * axis_in_model.
                child_quat_inv = quat_conjugate(child_abs_quat)
                axis_values = quat_rotate_vector(child_quat_inv, axis_values)

            joint.axis = axis_values

        # Parse limits.
        limit_elem = axis_elem.find("limit")
        if limit_elem is not None:
            lower = get_float(limit_elem, "lower", -np.inf)
            upper = get_float(limit_elem, "upper", np.inf)
            if np.isfinite(lower) and np.isfinite(upper):
                joint.limited = True
                joint.range = [lower, upper]

        # Parse dynamics (damping).
        dynamics_elem = axis_elem.find("dynamics")
        if dynamics_elem is not None:
            damping = get_float(dynamics_elem, "damping", 0.0)
            if damping > 0:
                joint.damping = damping

            friction = get_float(dynamics_elem, "friction", 0.0)
            if friction > 0:
                joint.frictionloss = friction


def add_geom_from_sdf(
    spec: mujoco.MjSpec,
    body: mujoco._specs.MjsBody,
    geom_elem: ET.Element,
    sdf_dir: Path,
    meshes_dir: Path,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
    color_assets: dict[str, list[float]],
    is_collision: bool,
    name_prefix: str,
    room_id: str = "",
) -> None:
    """Add geometry to body from SDF visual or collision element."""
    base_name = geom_elem.get("name", "geom")
    geom_kind = "collision" if is_collision else "visual"
    geom_name = f"{name_prefix}_{base_name}_{geom_kind}"

    geometry_elem = geom_elem.find("geometry")
    if geometry_elem is None:
        return

    pos, quat = parse_pose(geom_elem.find("pose"))

    geom = body.add_geom(name=geom_name)
    geom.pos = pos
    geom.quat = quat

    if is_collision:
        geom.contype = 1
        geom.conaffinity = 1
        geom.group = 3  # Collision geoms in group 3 (toggle with key 3 in viewer)

        surface_elem = geom_elem.find("surface")
        if surface_elem is not None:
            friction_elem = surface_elem.find("friction")
            if friction_elem is not None:
                ode_elem = friction_elem.find("ode")
                if ode_elem is not None:
                    mu = get_float(ode_elem, "mu", 1.0)
                    geom.friction = [mu, 0.005, 0.0001]
    else:
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 0  # Visual geoms in group 0 (toggle with key 0 in viewer)

    # Handle geometry types.
    box_elem = geometry_elem.find("box")
    if box_elem is not None:
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        size_elem = box_elem.find("size")
        if size_elem is not None and size_elem.text:
            sizes = [float(v) for v in size_elem.text.split()]
            geom.size = [s / 2 for s in sizes]
        return

    sphere_elem = geometry_elem.find("sphere")
    if sphere_elem is not None:
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        radius_elem = sphere_elem.find("radius")
        if radius_elem is not None and radius_elem.text:
            geom.size = [float(radius_elem.text), 0, 0]
        return

    cylinder_elem = geometry_elem.find("cylinder")
    if cylinder_elem is not None:
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        radius_elem = cylinder_elem.find("radius")
        length_elem = cylinder_elem.find("length")
        if radius_elem is not None and length_elem is not None:
            r = float(radius_elem.text) if radius_elem.text else 0.5
            h = float(length_elem.text) / 2 if length_elem.text else 0.5
            geom.size = [r, h, 0]
        return

    mesh_elem = geometry_elem.find("mesh")
    if mesh_elem is not None:
        uri_elem = mesh_elem.find("uri")
        scale_elem = mesh_elem.find("scale")
        mesh_scale = [1.0, 1.0, 1.0]
        base_color_name = None
        if scale_elem is not None and scale_elem.text:
            mesh_scale = parse_scale(scale_elem.text)
        if uri_elem is not None and uri_elem.text:
            mesh_uri = uri_elem.text
            # Resolve package:// URIs or regular paths.
            mesh_path = resolve_package_uri(mesh_uri, sdf_dir)
            mesh_name = f"{geom_name}_mesh"
            texture_name = None
            base_color = None

            if mesh_path is not None and mesh_path.exists():
                # Convert GLTF to OBJ if necessary.
                if mesh_path.suffix.lower() in (".gltf", ".glb"):
                    # Include parent directory to avoid filename collisions.
                    # e.g., "north_wall/wall.gltf" → "north_wall_wall.obj"
                    # Include room_id prefix to avoid collisions across rooms.
                    parent_prefix = mesh_path.parent.name
                    room_prefix = f"{room_id}_" if room_id else ""
                    # Include scale in filename to cache scaled variants separately.
                    scale_suffix = ""
                    if mesh_scale != [1.0, 1.0, 1.0]:
                        scale_suffix = f"_s{'_'.join(f'{s:.3g}' for s in mesh_scale)}"
                    obj_filename = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}.obj"
                    obj_path = meshes_dir / obj_filename
                    # Use consistent names based on mesh file, not geom.
                    # Include scale suffix to differentiate scaled variants.
                    base_texture_name = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}_tex"
                    base_color_name = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}_color"
                    expected_texture_file = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}_texture.png"

                    if not obj_path.exists():
                        success, texture_path, base_color = convert_gltf_to_obj(
                            gltf_path=mesh_path,
                            obj_path=obj_path,
                            texture_dir=meshes_dir,
                            scale=mesh_scale,
                        )
                        if not success:
                            console_logger.warning(
                                f"Skipping mesh {mesh_name}: GLTF conversion failed"
                            )
                            if is_collision:
                                spec.delete(geom)
                            else:
                                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                                geom.size = [0.1, 0.1, 0.1]
                            return
                        # Track texture if extracted.
                        if texture_path and texture_path.exists():
                            texture_name = base_texture_name
                            texture_assets[texture_name] = texture_path.name
                        # Track base color if no texture.
                        elif base_color:
                            color_assets[base_color_name] = base_color
                    else:
                        # OBJ already exists - check if texture was previously extracted.
                        existing_texture = meshes_dir / expected_texture_file
                        if existing_texture.exists():
                            texture_name = base_texture_name
                            # Ensure texture is in assets (may already be there).
                            if texture_name not in texture_assets:
                                texture_assets[texture_name] = expected_texture_file
                        elif base_color_name not in color_assets:
                            # Try to get base color from GLTF.
                            base_color = get_gltf_base_color(mesh_path)
                            if base_color:
                                color_assets[base_color_name] = base_color

                    if is_collision:
                        try:
                            collision_mesh = trimesh.load(obj_path, force="mesh")
                        except Exception as e:
                            console_logger.warning(
                                f"Failed to validate converted collision mesh {obj_path}: {e}"
                            )
                            spec.delete(geom)
                            return
                        if maybe_drop_degenerate_collision_geom(
                            spec=spec,
                            geom=geom,
                            geom_name=geom_name,
                            mesh=collision_mesh,
                            mesh_path=obj_path,
                        ):
                            return

                    mesh_assets[mesh_name] = obj_filename
                else:
                    # OBJ/STL mesh - validate and optionally scale.
                    try:
                        existing_mesh = trimesh.load(mesh_path, force="mesh")
                    except Exception as e:
                        console_logger.warning(
                            f"Failed to validate mesh {mesh_path}: {e}"
                        )
                        if is_collision:
                            spec.delete(geom)
                        else:
                            geom.type = mujoco.mjtGeom.mjGEOM_BOX
                            geom.size = [0.05, 0.05, 0.05]
                        return

                    if mesh_scale != [1.0, 1.0, 1.0]:
                        apply_scale_to_trimesh(existing_mesh, mesh_scale)

                    if is_collision and maybe_drop_degenerate_collision_geom(
                        spec=spec,
                        geom=geom,
                        geom_name=geom_name,
                        mesh=existing_mesh,
                        mesh_path=mesh_path,
                    ):
                        return

                    dest_filename = build_mesh_asset_filename(
                        mesh_path=mesh_path,
                        sdf_dir=sdf_dir,
                        room_id=room_id,
                        scale=mesh_scale,
                    )
                    dest_path = meshes_dir / dest_filename
                    if not dest_path.exists():
                        if mesh_scale != [1.0, 1.0, 1.0]:
                            # The mesh already has the export scale applied in-memory.
                            existing_mesh.export(dest_path)
                            console_logger.info(
                                f"Scaled mesh {mesh_path.name} -> {dest_filename}"
                            )
                        else:
                            shutil.copy(mesh_path, dest_path)
                    mesh_assets[mesh_name] = dest_filename
            else:
                console_logger.warning(f"Mesh not found: {mesh_uri}")
                # Use default box as fallback.
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = [0.1, 0.1, 0.1]
                return

            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh_name

            # Apply material/color (visual geoms only).
            if not is_collision:
                if texture_name:
                    geom.material = texture_name
                elif base_color_name and base_color_name in color_assets:
                    # Apply base color directly to geom.
                    geom.rgba = color_assets[base_color_name]
        return

    # Default fallback.
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.1, 0.1, 0.1]


def export_sdf_to_mujoco(
    sdf_path: Path,
    output_dir: Path,
    is_static: bool = False,
) -> Path:
    """Export a single SDF file to MuJoCo MJCF format.

    This is a standalone export mode for testing articulated models directly,
    without needing a full scene directory structure.

    Args:
        sdf_path: Path to SDF file.
        output_dir: Output directory for MJCF and meshes.
        is_static: Whether the model is static (no freejoint).

    Returns:
        Path to the exported scene.xml file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    output_path = output_dir / "scene.xml"

    # Create MuJoCo spec with environment (skybox, ground with collision).
    spec = create_mujoco_spec_with_environment(
        model_name=sdf_path.stem, ground_collides=True
    )

    # Track mesh, texture, and color assets.
    mesh_assets: dict[str, str] = {}
    texture_assets: dict[str, str] = {}
    color_assets: dict[str, list[float]] = {}

    # Parse SDF to get model name.
    tree = parse_sdf_with_drake_namespace(sdf_path)
    model_elem = tree.getroot().find(".//model")
    if model_elem is None:
        raise ValueError(f"No model element in SDF: {sdf_path}")

    model_name = model_elem.get("name", sdf_path.stem)

    # Process SDF model with identity quaternion, raised 0.5m for testing.
    # room_id is empty since this is a single-model export, not a multi-room scene.
    process_sdf_model(
        sdf_path=sdf_path,
        sdf_dir=sdf_path.parent,
        model_name=model_name,
        transform_pos=[0.0, 0.0, 0.5],
        transform_quat=[1.0, 0.0, 0.0, 0.0],
        is_static=is_static,
        spec=spec,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
        color_assets=color_assets,
        room_id="",
    )

    # Add mesh assets to spec.
    for mesh_name, mesh_filename in mesh_assets.items():
        mesh = spec.add_mesh(name=mesh_name)
        mesh.file = mesh_filename

    # Add texture and material assets.
    for texture_name, texture_filename in texture_assets.items():
        texture = spec.add_texture(name=texture_name)
        texture.file = texture_filename
        texture.type = mujoco.mjtTexture.mjTEXTURE_2D
        material = spec.add_material(name=texture_name)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name

    console_logger.info(f"Processed model: {model_name}")

    return compile_and_write_mjcf(
        spec=spec,
        output_path=output_path,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
    )


def fix_usd_texture_wrap_modes(materials_lib_path: Path) -> None:
    """Add repeat wrap mode to all UsdUVTexture shaders.

    The mujoco_usd_converter doesn't set wrapS/wrapT on texture shaders,
    causing UVs outside 0-1 to be clamped instead of tiled. Floor and wall
    meshes use UV coordinates > 1.0 for texture tiling, so they need repeat
    wrap mode to display correctly in usdview and Isaac Sim.

    Args:
        materials_lib_path: Path to MaterialsLibrary.usdc file.
    """
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.Open(str(materials_lib_path))

    textures_fixed = 0
    for prim in stage.TraverseAll():
        if prim.GetTypeName() == "Shader":
            shader = UsdShade.Shader(prim)
            shader_id = shader.GetIdAttr().Get()
            if shader_id == "UsdUVTexture":
                # Add wrap mode inputs for texture tiling.
                wrap_s = shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token)
                wrap_s.Set("repeat")
                wrap_t = shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token)
                wrap_t.Set("repeat")
                textures_fixed += 1

    stage.Save()
    console_logger.info(f"  Fixed wrap mode on {textures_fixed} texture shaders")


def export_to_usd(
    output_path: Path,
    output_dir: Path,
    apply_isaac_sim_fix: bool = True,
) -> None:
    """Export MuJoCo scene to USD format.

    Args:
        output_path: Path to scene.xml file.
        output_dir: Output directory containing meshes.

    Note:
        USD export is incompatible with bpy (Blender Python) in the same
        environment. Both packages install conflicting versions of the pxr
        (OpenUSD) library. If you need USD export, use a separate venv
        without bpy installed.
    """
    console_logger.info("\nExporting to USD format...")

    # Check for bpy/pxr conflict.
    try:
        import bpy  # noqa: F401

        console_logger.error(
            "USD export is incompatible with bpy (Blender) in the same environment.\n"
            "Both packages install conflicting versions of the pxr (OpenUSD) library.\n"
            "To use USD export, run the setup script to create a separate venv:\n"
            "  ./scripts/setup_mujoco_export.sh\n"
            "  source .mujoco_venv/bin/activate\n"
            "  python scripts/export_scene_to_mujoco.py --sdf ... --usd"
        )
        return
    except ImportError:
        pass  # bpy not installed, safe to proceed.

    try:
        import mujoco_usd_converter
        import usdex.core

        from pxr import Usd

        usd_dir = output_dir / "usd"
        usd_dir.mkdir(exist_ok=True)

        # Create a modified XML without checker texture (not supported in USD).
        # Remove grid texture, material, and ground plane geom.
        with open(output_path, "r") as f:
            xml_for_usd = f.read()

        # Remove grid texture (builtin checker).
        xml_for_usd = re.sub(r'<texture[^>]*name="grid"[^/]*/>\s*', "", xml_for_usd)
        # Remove grid material.
        xml_for_usd = re.sub(r'<material[^>]*name="grid"[^/]*/>\s*', "", xml_for_usd)
        # Remove ground plane geom that uses grid material.
        xml_for_usd = re.sub(
            r'<geom[^>]*name="ground_plane"[^/]*/>\s*', "", xml_for_usd
        )

        # Update meshdir/texturedir to point to parent directory's meshes folder.
        xml_for_usd = re.sub(r'meshdir="meshes"', 'meshdir="../meshes"', xml_for_usd)
        xml_for_usd = re.sub(
            r'texturedir="meshes"', 'texturedir="../meshes"', xml_for_usd
        )

        # Write temporary XML for USD conversion.
        usd_xml_path = usd_dir / "scene_for_usd.xml"
        with open(usd_xml_path, "w") as f:
            f.write(xml_for_usd)

        # Use mujoco-usd-converter for proper USD export with meshes.
        # Requires mujoco==3.3.5 and mujoco-usd-converter==0.1.0a3.
        converter = mujoco_usd_converter.Converter()
        asset = converter.convert(str(usd_xml_path), str(usd_dir))

        # Open stage and save with comment.
        stage: Usd.Stage = Usd.Stage.Open(asset.path)
        usdex.core.saveStage(
            stage, comment="Exported from scenecode via mujoco-usd-converter"
        )

        # Fix texture wrap modes for proper tiling support.
        materials_lib_path = usd_dir / "Payload" / "MaterialsLibrary.usdc"
        if materials_lib_path.exists():
            fix_usd_texture_wrap_modes(materials_lib_path)

        # Fix physics for Isaac Sim compatibility.
        physics_path = usd_dir / "Payload" / "Physics.usda"
        if apply_isaac_sim_fix and physics_path.exists():
            from fix_usd_isaac_sim import fix_physics_layer

            fix_physics_layer(physics_path)

        # Clean up temporary XML.
        usd_xml_path.unlink()

        console_logger.info(f"  USD exported to: {asset.path}")
        console_logger.info(f"  USD payloads in: {usd_dir / 'Payload'}")
    except ImportError as e:
        console_logger.error(f"USD export requires mujoco-usd-converter: {e}")
    except Exception as e:
        console_logger.error(f"USD export failed: {e}")


def validate_mujoco_export(output_path: Path) -> bool:
    """Validate that the exported MJCF loads successfully in MuJoCo.

    Args:
        output_path: Path to scene.xml file.

    Returns:
        True if validation passed.
    """
    try:
        model = mujoco.MjModel.from_xml_path(str(output_path))
        data = mujoco.MjData(model)

        # Run a few simulation steps.
        for _ in range(10):
            mujoco.mj_step(model, data)

        # Check for NaN.
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            console_logger.error("Simulation produced NaN values")
            return False

        console_logger.info(
            f"Validation passed: {model.nbody} bodies, {model.njnt} joints, "
            f"{model.ngeom} geoms"
        )
        return True
    except Exception as e:
        console_logger.error(f"Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export an existing scene to self-contained MuJoCo MJCF format"
    )
    parser.add_argument(
        "scene_path",
        type=Path,
        nargs="?",
        help="Path to scene directory (e.g., outputs/2025-12-05/13-39-27/scene_039)",
    )
    parser.add_argument(
        "--sdf",
        type=Path,
        default=None,
        help="Convert a single SDF file directly (for testing articulated models)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for MuJoCo export (default: scene_path/mujoco)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Treat model as static (no freejoint at base link). Only used with --sdf.",
    )
    parser.add_argument(
        "--no-floor-plan",
        action="store_true",
        help="Exclude floor plan (floor, walls) from export",
    )
    parser.add_argument(
        "--weld-furniture",
        action="store_true",
        help="Make furniture static (no freejoint). Default: furniture has freejoint.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip MuJoCo validation after export",
    )
    parser.add_argument(
        "--usd",
        action="store_true",
        help="Also export to USD format (OpenUSD/Universal Scene Description)",
    )
    parser.add_argument(
        "--skip-isaac-sim-fix",
        action="store_true",
        help=(
            "When exporting USD, skip the Isaac Sim compatibility fixer and "
            "leave the raw mujoco-usd-converter output untouched"
        ),
    )

    args = parser.parse_args()

    if args.scene_path is None and args.sdf is None:
        parser.error("Either scene_path or --sdf must be specified")

    # Handle standalone SDF conversion mode.
    if args.sdf:
        sdf_path = args.sdf.resolve()
        if not sdf_path.exists():
            console_logger.error(f"SDF file does not exist: {sdf_path}")
            sys.exit(1)

        output_dir = args.output or Path(f"/tmp/mujoco_{sdf_path.stem}")
        console_logger.info(f"Converting SDF to MuJoCo: {sdf_path}")

        output_path = export_sdf_to_mujoco(
            sdf_path=sdf_path, output_dir=output_dir, is_static=args.static
        )

        if not args.skip_validation:
            console_logger.info("Validating export...")
            if not validate_mujoco_export(output_path):
                console_logger.error("Export validation failed")
                sys.exit(1)

        console_logger.info(f"\nExport complete!")
        console_logger.info(f"  Scene file: {output_path}")
        console_logger.info(f"  Meshes dir: {output_dir / 'meshes'}")
        console_logger.info(f"\nTo view in MuJoCo:")
        console_logger.info(f"  python -m mujoco.viewer --mjcf={output_path}")

        # Export to USD if requested.
        if args.usd:
            export_to_usd(
                output_path,
                output_dir,
                apply_isaac_sim_fix=not args.skip_isaac_sim_fix,
            )

        return

    # Regular scene export mode.
    if not args.scene_path:
        parser.error("scene_path is required unless --sdf is specified")

    scene_path = args.scene_path.resolve()
    if not scene_path.exists():
        console_logger.error(f"Scene path does not exist: {scene_path}")
        sys.exit(1)

    output_dir = args.output or scene_path / "mujoco"

    console_logger.info(f"Loading scene from: {scene_path}")

    house_state_path = scene_path / "combined_house" / "house_state.json"
    dmd_path = scene_path / "combined_house" / "house.dmd.yaml"

    console_logger.info(f"Exporting to: {output_dir}")
    if house_state_path.exists():
        house = load_house_from_directory(scene_path)
        output_path = export_scene_to_mujoco(
            house=house,
            output_dir=output_dir,
            include_floor_plan=not args.no_floor_plan,
            weld_furniture=args.weld_furniture,
        )
    elif dmd_path.exists():
        console_logger.info(
            "house_state.json missing; exporting directly from house.dmd.yaml"
        )
        output_path = export_dmd_scene_to_mujoco(
            scene_dir=scene_path,
            dmd_path=dmd_path,
            output_dir=output_dir,
            include_floor_plan=not args.no_floor_plan,
            weld_furniture=args.weld_furniture,
        )
    else:
        console_logger.error(
            f"Missing scene metadata: expected {house_state_path} or {dmd_path}"
        )
        sys.exit(1)

    # Validate export.
    if not args.skip_validation:
        console_logger.info("Validating export...")
        if not validate_mujoco_export(output_path):
            console_logger.error("Export validation failed")
            sys.exit(1)

    console_logger.info(f"\nExport complete!")
    console_logger.info(f"  Scene file: {output_path}")
    console_logger.info(f"  Meshes dir: {output_dir / 'meshes'}")
    console_logger.info(f"\nTo load in MuJoCo:")
    console_logger.info(f"  import mujoco")
    console_logger.info(f"  model = mujoco.MjModel.from_xml_path('{output_path}')")

    # Export to USD if requested.
    if args.usd:
        export_to_usd(
            output_path,
            output_dir,
            apply_isaac_sim_fix=not args.skip_isaac_sim_fix,
        )


if __name__ == "__main__":
    main()
