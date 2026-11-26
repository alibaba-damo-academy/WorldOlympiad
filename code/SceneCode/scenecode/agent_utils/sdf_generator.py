"""Drake SDF file generation for simulation assets.

This module generates Drake-compatible SDF files with:
- Visual geometry (GLTF mesh with separate textures - Drake requirement)
- Collision geometry (convex decomposition pieces)
- Inertial properties (mass, center of mass, inertia tensor)
- Material properties (friction coefficients)

Also provides rescale_sdf() for in-place uniform scaling of SDF files.
"""

import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import trimesh

from scenecode.agent_utils.materials import get_friction
from scenecode.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenecode.utils.inertia_utils import fix_sdf_file_inertia
from scenecode.utils.sdf_utils import (
    parse_pose,
    parse_scale,
    pose_to_string,
    scale_inertia,
    scale_pose_translation,
    scale_to_string,
)

console_logger = logging.getLogger(__name__)


# Y-up to Z-up coordinate transformation (90° rotation around X-axis).
# Drake auto-converts Y-up GLTF visual meshes to Z-up on load.
YUP_TO_ZUP_TRANSFORM = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
)


def generate_drake_sdf(
    visual_mesh_path: Path,
    collision_pieces: list[trimesh.Trimesh],
    physics_analysis: MeshPhysicsAnalysis,
    output_path: Path,
    asset_name: str | None = None,
) -> Path:
    """Generate Drake SDF file for a simulation asset with visual geometry, collision
    geometry, and physics properties.

    Args:
        visual_mesh_path: Path to the visual mesh file (GLTF with separate textures,
            required by Drake). GLB files with embedded textures must be converted
            first using convert_glb_to_gltf().
        collision_pieces: List of convex mesh pieces from CoACD decomposition.
        physics_analysis: Physics properties from VLM analysis (mass, material).
        output_path: Path where SDF file will be saved.
        asset_name: Optional name for the asset (defaults to visual mesh stem).

    Returns:
        Path to the generated SDF file.

    Raises:
        FileNotFoundError: If visual mesh file doesn't exist.
        ValueError: If collision pieces list is empty or mesh has invalid volume.
    """
    if not visual_mesh_path.exists():
        raise FileNotFoundError(f"Visual mesh not found: {visual_mesh_path}")

    if not collision_pieces:
        raise ValueError("collision_pieces cannot be empty")

    asset_name = asset_name or visual_mesh_path.stem

    console_logger.info(
        f"Generating Drake SDF for '{asset_name}' with {len(collision_pieces)} "
        f"collision pieces"
    )

    # Load visual mesh to compute inertia.
    visual_mesh = trimesh.load(visual_mesh_path, force="mesh")

    # Handle Scene objects.
    if isinstance(visual_mesh, trimesh.Scene):
        meshes = [
            geom
            for geom in visual_mesh.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]
        visual_mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()

    # Transform visual mesh from Y-up to Z-up for inertia computation.
    # Drake auto-converts Y-up GLTF visual meshes to Z-up on load, so inertial
    # properties must also be in Z-up to match.
    visual_mesh.apply_transform(YUP_TO_ZUP_TRANSFORM)

    # Calculate inertial properties.
    mass = physics_analysis.mass_kg

    # Validate mass is positive.
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")

    # Get volume and validate.
    volume = visual_mesh.volume
    if volume <= 0:
        # Try fixing normals - some assets have inverted normals causing negative volume.
        console_logger.warning(
            f"Mesh '{asset_name}' has negative volume ({volume:.6f}), "
            f"attempting to fix normals..."
        )
        visual_mesh.fix_normals()
        volume = visual_mesh.volume
        if volume <= 0:
            # Still negative - use absolute value. Mass from VLM is correct,
            # we just need the magnitude for density calculation.
            console_logger.warning(
                f"Mesh '{asset_name}' still has negative volume ({volume:.6f}) "
                f"after normal fix. Using absolute value."
            )
            volume = abs(volume)

    density = mass / volume

    # Get inertia tensor from trimesh (assumes uniform density).
    # trimesh returns moment of inertia; we need to scale by density.
    inertia_tensor = visual_mesh.moment_inertia * density

    # Validate inertia tensor has positive eigenvalues.
    # If invalid (e.g., from inverted mesh normals), set to None to omit from SDF.
    eigenvalues = np.linalg.eigvals(inertia_tensor)
    if np.any(eigenvalues < 0):
        console_logger.warning(
            f"Computed inertia tensor for '{asset_name}' has negative eigenvalues "
            f"[{eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}, {eigenvalues[2]:.3f}]. "
            f"This indicates inverted mesh geometry. "
            f"Using mass={mass:.3f}kg but omitting inertia tensor."
        )
        inertia_tensor = None  # Signal to omit tensor in SDF.

    # Get center of mass.
    center_of_mass = visual_mesh.center_mass

    # Get friction coefficient for material.
    friction = get_friction(physics_analysis.material)

    # Create SDF XML structure.
    sdf = ET.Element("sdf", version="1.7")
    model = ET.SubElement(sdf, "model", name=asset_name)

    # Add single link (simple rigid body).
    link = ET.SubElement(model, "link", name="base_link")

    # Inertial properties.
    inertial = ET.SubElement(link, "inertial")

    mass_elem = ET.SubElement(inertial, "mass")
    mass_elem.text = f"{mass:.6f}"

    # Center of mass pose.
    com_pose = ET.SubElement(inertial, "pose")
    com_pose.text = (
        f"{center_of_mass[0]:.6f} {center_of_mass[1]:.6f} "
        f"{center_of_mass[2]:.6f} 0 0 0"
    )

    # Inertia tensor (only include if valid).
    if inertia_tensor is not None:
        inertia = ET.SubElement(inertial, "inertia")
        ET.SubElement(inertia, "ixx").text = f"{inertia_tensor[0, 0]:.6e}"
        ET.SubElement(inertia, "iyy").text = f"{inertia_tensor[1, 1]:.6e}"
        ET.SubElement(inertia, "izz").text = f"{inertia_tensor[2, 2]:.6e}"
        ET.SubElement(inertia, "ixy").text = f"{inertia_tensor[0, 1]:.6e}"
        ET.SubElement(inertia, "ixz").text = f"{inertia_tensor[0, 2]:.6e}"
        ET.SubElement(inertia, "iyz").text = f"{inertia_tensor[1, 2]:.6e}"
    # If inertia_tensor is None, omit the <inertia> tag entirely.
    # Drake will use default values (I_xx=I_yy=I_zz=1.0, products=0.0).

    # Visual geometry (external mesh reference).
    visual = ET.SubElement(link, "visual", name="visual")
    visual_geom = ET.SubElement(visual, "geometry")
    visual_mesh_elem = ET.SubElement(visual_geom, "mesh")

    # Use relative URI for mesh (assumes mesh is in same directory as SDF).
    mesh_filename = visual_mesh_path.name
    visual_uri = ET.SubElement(visual_mesh_elem, "uri")
    visual_uri.text = mesh_filename

    # Collision geometry (convex pieces).
    for i, piece in enumerate(collision_pieces):
        collision = ET.SubElement(link, "collision", name=f"collision_{i}")

        # Add friction properties.
        surface = ET.SubElement(collision, "surface")
        friction_elem = ET.SubElement(surface, "friction")
        ode = ET.SubElement(friction_elem, "ode")
        mu = ET.SubElement(ode, "mu")
        mu.text = f"{friction:.3f}"
        mu2 = ET.SubElement(ode, "mu2")
        mu2.text = f"{friction:.3f}"

        # Collision geometry (convex mesh).
        collision_geom = ET.SubElement(collision, "geometry")
        collision_mesh_elem = ET.SubElement(collision_geom, "mesh")

        # Save collision piece as separate OBJ file.
        collision_mesh_filename = f"{asset_name}_collision_{i}.obj"
        collision_mesh_path = output_path.parent / collision_mesh_filename

        # Transform collision piece from Y-up to Z-up before exporting.
        # Drake automatically converts Y-up GLTF visual meshes to Z-up on load,
        # but does NOT convert OBJ collision meshes. We must export OBJ in Z-up
        # to match the visual mesh coordinate system.
        piece.apply_transform(YUP_TO_ZUP_TRANSFORM)

        # Export collision piece.
        piece.export(collision_mesh_path)

        # Reference in SDF.
        collision_uri = ET.SubElement(collision_mesh_elem, "uri")
        collision_uri.text = collision_mesh_filename

        # Declare mesh as convex for Drake.
        ET.SubElement(collision_mesh_elem, "{drake.mit.edu}declare_convex")

    # Format XML with indentation.
    ET.indent(sdf, space="  ", level=0)

    # Create ElementTree and write to file.
    tree = ET.ElementTree(sdf)

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with XML declaration.
    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Fix any inertia tensors that violate the triangle inequality.
    fix_sdf_file_inertia(output_path)

    console_logger.info(f"Generated Drake SDF: {output_path}")

    return output_path


def rescale_sdf(sdf_path: Path, scale_factor: float) -> None:
    """Apply uniform scale to an existing SDF file in-place.

    CRITICAL: Does NOT scale model pose translation (world position).
    The model pose represents WHERE the object is placed in the scene.
    Scaling it would MOVE the object, not resize it in-place.

    This function ONLY scales internal structure to resize the mesh geometry
    while keeping the object anchored at its current position.

    Args:
        sdf_path: Path to SDF file to modify in-place.
        scale_factor: Uniform scale multiplier (e.g., 1.5 = 50% larger).

    Raises:
        FileNotFoundError: If SDF file doesn't exist.
        ValueError: If scale_factor is not positive.
    """
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    if scale_factor == 1.0:
        return  # No-op for identity scale.

    console_logger.info(f"Rescaling SDF '{sdf_path.name}' by factor {scale_factor:.3f}")

    tree = ET.parse(sdf_path)
    root = tree.getroot()

    # Process each model in the SDF.
    for model in root.findall(".//model"):
        # NOTE: We do NOT scale model pose - that's world position.

        # Process each link.
        for link in model.findall("link"):
            _scale_link(link, scale_factor)

        # Process each joint.
        for joint in model.findall("joint"):
            _scale_joint_pose(joint, scale_factor)

    # Format and write back.
    ET.indent(root, space="  ", level=0)
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)

    # Fix any inertia tensors that violate the triangle inequality.
    fix_sdf_file_inertia(sdf_path)

    console_logger.info(f"Rescaled SDF saved: {sdf_path}")


def _scale_link(link: ET.Element, scale_factor: float) -> None:
    """Scale all components within a link element."""
    # Scale link pose translation (relative position within model).
    link_pose = link.find("pose")
    if link_pose is not None and link_pose.text:
        xyz, rpy = parse_pose(link_pose.text)
        scaled_xyz = scale_pose_translation(xyz, scale_factor)
        link_pose.text = pose_to_string(scaled_xyz, rpy)

    # Scale inertial properties.
    inertial = link.find("inertial")
    if inertial is not None:
        _scale_inertial(inertial, scale_factor)

    # Scale visual geometries.
    for visual in link.findall("visual"):
        _scale_geometry_element(visual, scale_factor)

    # Scale collision geometries.
    for collision in link.findall("collision"):
        _scale_geometry_element(collision, scale_factor)


def _scale_inertial(inertial: ET.Element, scale_factor: float) -> None:
    """Scale inertial properties: CoM pose and inertia tensor."""
    # Scale center of mass pose translation.
    inertial_pose = inertial.find("pose")
    if inertial_pose is not None and inertial_pose.text:
        xyz, rpy = parse_pose(inertial_pose.text)
        scaled_xyz = scale_pose_translation(xyz, scale_factor)
        inertial_pose.text = pose_to_string(scaled_xyz, rpy)

    # Scale inertia tensor (scales as s^2 for constant mass).
    inertia = inertial.find("inertia")
    if inertia is not None:
        ixx = float(inertia.findtext("ixx", "0"))
        iyy = float(inertia.findtext("iyy", "0"))
        izz = float(inertia.findtext("izz", "0"))
        ixy = float(inertia.findtext("ixy", "0"))
        ixz = float(inertia.findtext("ixz", "0"))
        iyz = float(inertia.findtext("iyz", "0"))

        scaled = scale_inertia(ixx, iyy, izz, ixy, ixz, iyz, scale_factor)

        _set_or_create_text(inertia, "ixx", f"{scaled[0]:.6e}")
        _set_or_create_text(inertia, "iyy", f"{scaled[1]:.6e}")
        _set_or_create_text(inertia, "izz", f"{scaled[2]:.6e}")
        _set_or_create_text(inertia, "ixy", f"{scaled[3]:.6e}")
        _set_or_create_text(inertia, "ixz", f"{scaled[4]:.6e}")
        _set_or_create_text(inertia, "iyz", f"{scaled[5]:.6e}")


def _scale_geometry_element(element: ET.Element, scale_factor: float) -> None:
    """Scale a visual or collision element's pose and geometry."""
    # Scale element pose translation.
    pose = element.find("pose")
    if pose is not None and pose.text:
        xyz, rpy = parse_pose(pose.text)
        scaled_xyz = scale_pose_translation(xyz, scale_factor)
        pose.text = pose_to_string(scaled_xyz, rpy)

    geometry = element.find("geometry")
    if geometry is None:
        return

    # Scale mesh geometry via <scale> element.
    mesh = geometry.find("mesh")
    if mesh is not None:
        scale_elem = mesh.find("scale")
        if scale_elem is not None and scale_elem.text:
            # Multiply existing scale.
            existing = parse_scale(scale_elem.text)
            new_scale = [v * scale_factor for v in existing]
            scale_elem.text = scale_to_string(new_scale)
        else:
            # Add new scale element.
            scale_elem = ET.SubElement(mesh, "scale")
            scale_elem.text = scale_to_string(
                [scale_factor, scale_factor, scale_factor]
            )
        return

    # Scale primitive geometries.
    box = geometry.find("box")
    if box is not None:
        size_elem = box.find("size")
        if size_elem is not None and size_elem.text:
            sizes = [float(v) for v in size_elem.text.strip().split()]
            scaled_sizes = [v * scale_factor for v in sizes]
            size_elem.text = " ".join(f"{v:.8f}" for v in scaled_sizes)
        return

    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius_elem = cylinder.find("radius")
        if radius_elem is not None and radius_elem.text:
            radius = float(radius_elem.text)
            radius_elem.text = f"{radius * scale_factor:.8f}"
        length_elem = cylinder.find("length")
        if length_elem is not None and length_elem.text:
            length = float(length_elem.text)
            length_elem.text = f"{length * scale_factor:.8f}"
        return

    sphere = geometry.find("sphere")
    if sphere is not None:
        radius_elem = sphere.find("radius")
        if radius_elem is not None and radius_elem.text:
            radius = float(radius_elem.text)
            radius_elem.text = f"{radius * scale_factor:.8f}"
        return


def _scale_joint_pose(joint: ET.Element, scale_factor: float) -> None:
    """Scale joint pose translation."""
    pose = joint.find("pose")
    if pose is not None and pose.text:
        xyz, rpy = parse_pose(pose.text)
        scaled_xyz = scale_pose_translation(xyz, scale_factor)
        pose.text = pose_to_string(scaled_xyz, rpy)


def _set_or_create_text(parent: ET.Element, tag: str, text: str) -> None:
    """Set text of existing element or create new one."""
    elem = parent.find(tag)
    if elem is None:
        elem = ET.SubElement(parent, tag)
    elem.text = text


def add_self_collision_filter(sdf_path: Path) -> None:
    """Add drake:collision_filter_group to prevent self-collision in articulated models.

    Adds a collision filter group that prevents links within the same
    articulated model from colliding with each other, while still allowing
    collisions with other objects in the scene.

    This is idempotent: if a collision_filter_group already exists in the
    model, the function returns without modification.

    Args:
        sdf_path: Path to the SDF file to modify in-place.
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    # Find the <model> element (handles both <sdf><model> and
    # <sdf><world><model> structures).
    model = root.find("model")
    if model is None:
        model = root.find("world/model")
    if model is None:
        console_logger.warning(
            f"No <model> element found in {sdf_path}, "
            f"skipping self-collision filter."
        )
        return

    # Extract all link names.
    links = model.findall("link")
    link_names = [link.get("name") for link in links if link.get("name")]

    # Skip single-body objects (no self-collision possible).
    if len(link_names) < 2:
        console_logger.debug(
            f"Only {len(link_names)} link(s) in {sdf_path}, "
            f"skipping self-collision filter."
        )
        return

    # Skip if a collision_filter_group already exists (idempotent).
    drake_ns = "drake.mit.edu"
    existing = model.find(f"{{{drake_ns}}}collision_filter_group")
    if existing is not None:
        console_logger.debug(
            f"collision_filter_group already exists in {sdf_path}, skipping."
        )
        return

    # Register Drake namespace for clean prefix output.
    ET.register_namespace("drake", drake_ns)

    # Build the collision filter group element.
    group = ET.SubElement(
        model, f"{{{drake_ns}}}collision_filter_group", name="no_self_collision"
    )
    for name in link_names:
        ET.SubElement(group, f"{{{drake_ns}}}member").text = name
    ignored = ET.SubElement(group, f"{{{drake_ns}}}ignored_collision_filter_group")
    ignored.text = "no_self_collision"

    # Re-indent for readable output.
    ET.indent(tree, space="  ", level=0)

    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)

    console_logger.info(
        f"Added self-collision filter to {sdf_path} " f"({len(link_names)} links)."
    )
