"""URDF to SDF converter for articulated objects.

This module converts URDF files (specifically from PartNet-Mobility dataset) to
Drake-compatible SDF format. It handles the quirks of PartNet-Mobility URDFs:
- Missing <inertial> elements (adds defaults)
- Empty 'base' link as root
- Relative mesh paths

Key transformations:
- <robot name> → <sdf><model name>
- <link> → <link> with <pose>
- <joint origin xyz rpy> → joint pose computation
- <joint axis> → <axis><xyz>
- <joint limit> → <axis><limit>
"""

import logging
import tempfile
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from scipy.spatial.transform import Rotation

from scenecode.agent_utils.convex_decomposition_server import ConvexDecompositionClient
from scenecode.agent_utils.mesh_utils import load_mesh_as_trimesh, merge_objs_to_gltf
from scenecode.agent_utils.physics_tools import compute_inertia_from_mesh
from scenecode.utils.inertia_utils import fix_sdf_file_inertia
from scenecode.utils.sdf_utils import pose_to_string

console_logger = logging.getLogger(__name__)

# Default joint properties.
DEFAULT_JOINT_DAMPING = 0.05  # Nm/(rad/s) for revolute, N/(m/s) for prismatic
DEFAULT_JOINT_FRICTION = 0.05  # Nm for revolute, N for prismatic

SUPPORTED_MESH_EXTENSIONS = {".obj", ".gltf", ".glb"}


@dataclass
class LinkPhysics:
    """Physics properties for a link."""

    mass: float
    inertia_ixx: float
    inertia_iyy: float
    inertia_izz: float
    inertia_ixy: float = 0.0
    inertia_ixz: float = 0.0
    inertia_iyz: float = 0.0
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class URDFParseResult:
    """Result of parsing a URDF file."""

    robot_name: str
    """Name of the robot from the URDF."""

    links: dict[str, ET.Element]
    """Mapping of link name to link element."""

    joints: dict[str, ET.Element]
    """Mapping of joint name to joint element."""

    parent_map: dict[str, str]
    """Mapping of child link to parent link."""

    root_link: str | None
    """Name of root link (no parent)."""


@dataclass
class URDFLinkMeshInfo:
    """Mesh information for a single URDF link."""

    link_name: str
    """Name of the link."""

    mesh_paths: list[Path]
    """Paths to mesh files (OBJ, GLTF, GLB) for visual geometry."""

    origins: list[tuple[float, float, float]]
    """Origin offsets for each mesh file (xyz in meters)."""

    world_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World position of the link from joint chain (xyz in meters)."""

    world_rotation: tuple[tuple[float, ...], ...] | None = None
    """World rotation matrix of the link from joint chain (3x3)."""


def parse_origin(origin_elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Parse URDF origin element into position and RPY.

    Args:
        origin_elem: URDF <origin> element or None.

    Returns:
        Tuple of (xyz position array, rpy rotation array in radians).
    """
    if origin_elem is None:
        return np.zeros(3), np.zeros(3)

    xyz_str = origin_elem.get("xyz", "0 0 0")
    rpy_str = origin_elem.get("rpy", "0 0 0")

    xyz = np.array([float(x) for x in xyz_str.split()])
    rpy = np.array([float(x) for x in rpy_str.split()])

    return xyz, rpy


def parse_urdf(urdf_path: Path) -> URDFParseResult:
    """Parse URDF file and extract structure.

    Args:
        urdf_path: Path to URDF file.

    Returns:
        URDFParseResult with parsed data.

    Raises:
        ValueError: If URDF structure is invalid.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError(f"Expected <robot> root element, got <{root.tag}>")

    robot_name = root.get("name", "unnamed_robot")

    # Collect links.
    links = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        if link_name:
            links[link_name] = link_elem

    # Collect joints and build parent map.
    joints = {}
    parent_map = {}
    child_links = set()

    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.get("name")
        if joint_name:
            joints[joint_name] = joint_elem

            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")

            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.get("link")
                child_link = child_elem.get("link")

                if parent_link and child_link:
                    parent_map[child_link] = parent_link
                    child_links.add(child_link)

    # Find root link (link with no parent).
    root_link = None
    for link_name in links:
        if link_name not in child_links:
            root_link = link_name
            break

    return URDFParseResult(
        robot_name=robot_name,
        links=links,
        joints=joints,
        parent_map=parent_map,
        root_link=root_link,
    )


def extract_link_meshes(urdf_path: Path) -> list[URDFLinkMeshInfo]:
    """Extract link-to-mesh mappings from URDF with position offsets.

    Parses a URDF file and extracts the visual mesh files for each link,
    along with their origin offsets and position offsets from the joint chain.
    Supports OBJ, GLTF, and GLB formats.

    Args:
        urdf_path: Path to URDF file.

    Returns:
        List of URDFLinkMeshInfo with link names, mesh paths, and position offsets.
    """
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    urdf_dir = urdf_path.parent

    # Build link-to-joint mapping for kinematic chain traversal.
    link_to_parent_joint: dict[str, ET.Element] = {}
    for joint in robot.findall("joint"):
        child = joint.find("child")
        if child is not None:
            child_link = child.get("link")
            if child_link:
                link_to_parent_joint[child_link] = joint

    def get_link_visual_position(link_name: str) -> np.ndarray:
        """Compute position offset for VLM visualization.

        For VLM visualization, we only accumulate position offsets (xyz) from
        joints, NOT coordinate system rotations (rpy). The rotations are for
        simulation coordinate transforms, not visual assembly.
        """
        # Build chain from link to root.
        chain = []
        current_link = link_name
        while current_link in link_to_parent_joint:
            joint = link_to_parent_joint[current_link]
            parent = joint.find("parent")
            if parent is None:
                break
            chain.append(joint)
            current_link = parent.get("link", "")

        # Accumulate position offsets only (ignore rotations for visualization).
        world_pos = np.zeros(3)
        for joint in reversed(chain):
            origin = joint.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                joint_xyz = np.array([float(v) for v in xyz_str.split()])
                world_pos = world_pos + joint_xyz

        return world_pos

    link_meshes = []

    for link in robot.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue

        mesh_paths = []
        origins = []

        for visual in link.findall("visual"):
            mesh = visual.find(".//mesh")
            if mesh is None:
                continue

            filename = mesh.get("filename")
            if not filename:
                continue

            # Check if file extension is supported.
            file_ext = Path(filename).suffix.lower()
            if file_ext not in SUPPORTED_MESH_EXTENSIONS:
                continue

            mesh_path = urdf_dir / filename
            if not mesh_path.exists():
                continue

            # Parse origin offset.
            origin = visual.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                xyz = tuple(float(v) for v in xyz_str.split())
            else:
                xyz = (0.0, 0.0, 0.0)

            mesh_paths.append(mesh_path)
            origins.append(xyz)

        # Only include links with visual geometry.
        if mesh_paths:
            # Compute position offset from joint chain (no rotation for VLM).
            world_pos = get_link_visual_position(link_name)

            link_meshes.append(
                URDFLinkMeshInfo(
                    link_name=link_name,
                    mesh_paths=mesh_paths,
                    origins=origins,
                    world_position=tuple(world_pos),
                    world_rotation=None,  # No rotation for VLM visualization.
                )
            )

    return link_meshes


def validate_urdf_meshes(
    urdf_path: Path, urdf_result: URDFParseResult
) -> tuple[list[str], list[str]]:
    """Validate that all mesh files referenced in URDF exist.

    Args:
        urdf_path: Path to URDF file.
        urdf_result: Parsed URDF result.

    Returns:
        Tuple of (list of valid mesh paths, list of missing mesh paths).
    """
    urdf_dir = urdf_path.parent
    valid_meshes = []
    missing_meshes = []

    for link_elem in urdf_result.links.values():
        for visual_or_collision in link_elem.findall("visual") + link_elem.findall(
            "collision"
        ):
            geometry = visual_or_collision.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename:
                        mesh_path = urdf_dir / filename
                        if mesh_path.exists():
                            if filename not in valid_meshes:
                                valid_meshes.append(filename)
                        else:
                            if filename not in missing_meshes:
                                missing_meshes.append(filename)

    return valid_meshes, missing_meshes


def repair_urdf_missing_meshes(
    urdf_path: Path, output_path: Path
) -> tuple[Path, list[str]]:
    """Repair URDF by removing references to missing mesh files.

    Args:
        urdf_path: Path to input URDF file.
        output_path: Path to write repaired URDF.

    Returns:
        Tuple of (path to repaired URDF, list of removed mesh references).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    removed_meshes = []

    for link_elem in root.findall("link"):
        elements_to_remove = []

        for visual_or_collision in link_elem.findall("visual") + link_elem.findall(
            "collision"
        ):
            geometry = visual_or_collision.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename:
                        mesh_path = urdf_dir / filename
                        if not mesh_path.exists():
                            elements_to_remove.append(visual_or_collision)
                            if filename not in removed_meshes:
                                removed_meshes.append(filename)

        for elem in elements_to_remove:
            link_elem.remove(elem)

    # Write repaired URDF.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return output_path, removed_meshes


def convert_urdf_geometry_to_sdf(
    urdf_geometry: ET.Element,
    urdf_dir: Path,
    use_gltf: bool = False,
    scale_factor: float = 1.0,
) -> ET.Element | None:
    """Convert URDF geometry element to SDF format.

    Args:
        urdf_geometry: URDF <geometry> element.
        urdf_dir: Directory containing URDF (for resolving relative paths).
        use_gltf: If True, use .gltf extension for mesh files (for visuals).
            If False, use original .obj extension (for collisions).
        scale_factor: Uniform scale factor to apply to geometry.

    Returns:
        SDF <geometry> element or None if geometry is invalid.
    """
    sdf_geometry = ET.Element("geometry")

    mesh = urdf_geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename")
        if filename:
            # Verify mesh exists in URDF directory.
            mesh_path = urdf_dir / filename
            if mesh_path.exists():
                # Use original URDF relative path.
                # Assumes mesh directories (visual/) are copied to SDF location.
                sdf_mesh = ET.SubElement(sdf_geometry, "mesh")
                uri = ET.SubElement(sdf_mesh, "uri")
                # Use GLTF for visuals (Drake Meshcat textures), OBJ for collisions.
                if use_gltf and filename.endswith(".obj"):
                    uri.text = filename[:-4] + ".gltf"
                else:
                    uri.text = filename
                # Add scale element if not 1.0.
                if scale_factor != 1.0:
                    scale_elem = ET.SubElement(sdf_mesh, "scale")
                    scale_elem.text = f"{scale_factor} {scale_factor} {scale_factor}"
                return sdf_geometry

    # Handle primitive shapes (scale dimensions directly).
    box = urdf_geometry.find("box")
    if box is not None:
        size_str = box.get("size", "1 1 1")
        sizes = [float(s) * scale_factor for s in size_str.split()]
        sdf_box = ET.SubElement(sdf_geometry, "box")
        sdf_size = ET.SubElement(sdf_box, "size")
        sdf_size.text = f"{sizes[0]} {sizes[1]} {sizes[2]}"
        return sdf_geometry

    cylinder = urdf_geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.get("radius", "1")) * scale_factor
        length = float(cylinder.get("length", "1")) * scale_factor
        sdf_cylinder = ET.SubElement(sdf_geometry, "cylinder")
        ET.SubElement(sdf_cylinder, "radius").text = str(radius)
        ET.SubElement(sdf_cylinder, "length").text = str(length)
        return sdf_geometry

    sphere = urdf_geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius", "1")) * scale_factor
        sdf_sphere = ET.SubElement(sdf_geometry, "sphere")
        ET.SubElement(sdf_sphere, "radius").text = str(radius)
        return sdf_geometry

    return None


def convert_urdf_visual_to_sdf(
    urdf_visual: ET.Element,
    urdf_dir: Path,
    visual_index: int,
    link_name: str,
    scale_factor: float = 1.0,
) -> ET.Element | None:
    """Convert URDF visual element to SDF format.

    Args:
        urdf_visual: URDF <visual> element.
        urdf_dir: Directory containing URDF.
        visual_index: Index for unique naming.
        link_name: Name of parent link (for unique naming).
        scale_factor: Uniform scale factor to apply to geometry.

    Returns:
        SDF <visual> element or None if invalid.
    """
    # Always use unique name to avoid Drake conflicts with duplicate URDF names.
    name = f"{link_name}_visual_{visual_index}"
    sdf_visual = ET.Element("visual", name=name)

    # Convert origin to pose (scale the position).
    origin = urdf_visual.find("origin")
    xyz, rpy = parse_origin(origin)
    scaled_xyz = [v * scale_factor for v in xyz]
    pose = ET.SubElement(sdf_visual, "pose")
    pose.text = pose_to_string(scaled_xyz, rpy)

    # Convert geometry (use GLTF for visual meshes to support textures in Drake).
    geometry = urdf_visual.find("geometry")
    if geometry is not None:
        sdf_geometry = convert_urdf_geometry_to_sdf(
            geometry, urdf_dir, use_gltf=True, scale_factor=scale_factor
        )
        if sdf_geometry is not None:
            sdf_visual.append(sdf_geometry)
        else:
            return None

    return sdf_visual


def convert_urdf_collision_to_sdf(
    urdf_collision: ET.Element,
    urdf_dir: Path,
    collision_index: int,
    link_name: str,
    friction: float = 0.5,
) -> ET.Element | None:
    """Convert URDF collision element to SDF format.

    Args:
        urdf_collision: URDF <collision> element.
        urdf_dir: Directory containing URDF.
        collision_index: Index for unique naming.
        link_name: Name of parent link (for unique naming).
        friction: Friction coefficient for surface.

    Returns:
        SDF <collision> element or None if invalid.
    """
    # Always use unique name to avoid Drake conflicts with duplicate URDF names.
    name = f"{link_name}_collision_{collision_index}"
    sdf_collision = ET.Element("collision", name=name)

    # Convert origin to pose.
    origin = urdf_collision.find("origin")
    xyz, rpy = parse_origin(origin)
    pose = ET.SubElement(sdf_collision, "pose")
    pose.text = pose_to_string(xyz, rpy)

    # Convert geometry.
    geometry = urdf_collision.find("geometry")
    if geometry is not None:
        sdf_geometry = convert_urdf_geometry_to_sdf(geometry, urdf_dir)
        if sdf_geometry is not None:
            sdf_collision.append(sdf_geometry)
        else:
            return None

    # Add surface friction.
    surface = ET.SubElement(sdf_collision, "surface")
    friction_elem = ET.SubElement(surface, "friction")
    ode = ET.SubElement(friction_elem, "ode")
    ET.SubElement(ode, "mu").text = f"{friction:.3f}"
    ET.SubElement(ode, "mu2").text = f"{friction:.3f}"

    return sdf_collision


def merge_link_visual_meshes_for_sdf(
    urdf_link: ET.Element, urdf_dir: Path, sdf_dir: Path, link_name: str
) -> Path | None:
    """Merge all visual meshes for a link into a single GLTF file."""
    obj_paths_with_transforms: list[
        tuple[Path, tuple[float, float, float], tuple[float, float, float]]
    ] = []
    for visual in urdf_link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is None:
            continue

        mesh = geometry.find("mesh")
        if mesh is None:
            continue

        filename = mesh.get("filename")
        if not filename or not filename.endswith(".obj"):
            continue

        mesh_path = urdf_dir / filename
        if not mesh_path.exists():
            continue

        xyz, rpy = parse_origin(visual.find("origin"))
        obj_paths_with_transforms.append(
            (
                mesh_path,
                tuple(float(v) for v in xyz),
                tuple(float(v) for v in rpy),
            )
        )

    if not obj_paths_with_transforms:
        return None

    output_dir = sdf_dir / "visual"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gltf = output_dir / f"{link_name}_visual.gltf"

    try:
        merge_objs_to_gltf(
            obj_paths_with_offsets=obj_paths_with_transforms,
            output_path=output_gltf,
        )
        console_logger.info(
            f"Merged {len(obj_paths_with_transforms)} visual meshes for {link_name}"
        )
        return output_gltf
    except Exception as e:
        console_logger.warning(f"Failed to merge visual meshes for {link_name}: {e}")
        return None


def generate_link_collision_geometry_for_sdf(
    urdf_link: ET.Element,
    urdf_dir: Path,
    sdf_dir: Path,
    link_name: str,
    collision_client: ConvexDecompositionClient,
    collision_threshold: float = 0.05,
) -> list[Path]:
    """Generate convex collision geometry for a link using CoACD.

    Combines all visual meshes for the link, runs CoACD convex decomposition
    via the convex decomposition server, and saves the resulting convex pieces
    as OBJ files. Note: This function always uses CoACD since articulated assets
    benefit from simpler collision geometry.

    Args:
        urdf_link: URDF <link> element.
        urdf_dir: Directory containing URDF (for resolving mesh paths).
        sdf_dir: Directory where SDF will be written (for saving collision meshes).
        link_name: Name of the link.
        collision_client: Convex decomposition client for collision geometry.
        collision_threshold: CoACD approximation threshold (0.01-0.1 typical).

    Returns:
        List of paths to generated collision OBJ files.
    """
    # Collect visual mesh files and their origins.
    mesh_infos: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for visual in urdf_link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is not None:
            mesh = geometry.find("mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if filename:
                    mesh_path = urdf_dir / filename
                    if mesh_path.exists():
                        origin = visual.find("origin")
                        xyz, rpy = parse_origin(origin)
                        rot = Rotation.from_euler("xyz", rpy).as_matrix()
                        mesh_infos.append((mesh_path, xyz, rot))

    if not mesh_infos:
        return []

    # Load and combine meshes, applying visual origins.
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0

    for mesh_path, visual_xyz, visual_rot in mesh_infos:
        try:
            mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)
            if mesh is not None:
                # Apply visual origin transform to vertices.
                vertices = mesh.vertices @ visual_rot.T + visual_xyz
                combined_vertices.append(vertices)
                combined_faces.append(mesh.faces + vertex_offset)
                vertex_offset += len(mesh.vertices)
        except Exception as e:
            console_logger.warning(f"Failed to load mesh {mesh_path}: {e}")

    if not combined_vertices:
        return []

    # Create combined mesh.
    all_vertices = np.vstack(combined_vertices)
    all_faces = np.vstack(combined_faces)
    combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    # Generate convex decomposition using convex decomposition server.
    # Save mesh to temp file since client requires a file path.
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            temp_mesh_path = Path(f.name)
        combined_mesh.export(temp_mesh_path)

        try:
            convex_pieces = collision_client.generate_collision_geometry(
                mesh_path=temp_mesh_path,
                method="coacd",
                threshold=collision_threshold,
            )
        finally:
            # Clean up temp file.
            temp_mesh_path.unlink(missing_ok=True)
    except Exception as e:
        console_logger.warning(
            f"Convex decomposition failed for {link_name}, falling back to convex "
            f"hull: {e}"
        )
        # Fallback to single convex hull.
        convex_pieces = [combined_mesh.convex_hull]

    # Save collision pieces as OBJ files.
    collision_dir = sdf_dir / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)

    collision_paths = []
    for i, piece in enumerate(convex_pieces):
        collision_path = collision_dir / f"{link_name}_collision_{i}.obj"
        piece.export(collision_path)
        collision_paths.append(collision_path)

    console_logger.debug(
        f"Generated {len(collision_paths)} collision pieces for {link_name}"
    )
    return collision_paths


def create_sdf_collision_elements_from_paths(
    collision_paths: list[Path],
    sdf_dir: Path,
    link_name: str,
    friction: float = 0.5,
    scale_factor: float = 1.0,
) -> list[ET.Element]:
    """Create SDF collision elements from generated collision mesh paths.

    Args:
        collision_paths: Paths to collision OBJ files.
        sdf_dir: Directory where SDF will be written (for relative paths).
        link_name: Name of parent link (for unique naming).
        friction: Friction coefficient for surfaces.
        scale_factor: Uniform scale factor to apply to collision geometry.

    Returns:
        List of SDF <collision> elements.
    """
    collision_elements = []
    for i, collision_path in enumerate(collision_paths):
        name = f"{link_name}_collision_{i}"
        sdf_collision = ET.Element("collision", name=name)

        # Identity pose (collision geometry already in link frame).
        pose = ET.SubElement(sdf_collision, "pose")
        pose.text = "0 0 0 0 0 0"

        # Geometry with mesh reference.
        sdf_geometry = ET.SubElement(sdf_collision, "geometry")
        sdf_mesh = ET.SubElement(sdf_geometry, "mesh")
        uri = ET.SubElement(sdf_mesh, "uri")
        rel_path = collision_path.relative_to(sdf_dir)
        uri.text = str(rel_path)
        # Add scale element if not 1.0.
        if scale_factor != 1.0:
            scale_elem = ET.SubElement(sdf_mesh, "scale")
            scale_elem.text = f"{scale_factor} {scale_factor} {scale_factor}"

        # Add surface friction.
        surface = ET.SubElement(sdf_collision, "surface")
        friction_elem = ET.SubElement(surface, "friction")
        ode = ET.SubElement(friction_elem, "ode")
        ET.SubElement(ode, "mu").text = f"{friction:.3f}"
        ET.SubElement(ode, "mu2").text = f"{friction:.3f}"

        collision_elements.append(sdf_collision)

    return collision_elements


def convert_urdf_link_to_sdf(
    urdf_link: ET.Element,
    urdf_dir: Path,
    sdf_dir: Path,
    physics: LinkPhysics | None = None,
    friction: float = 0.5,
    link_pose: tuple[list[float], list[float], str] | None = None,
    generated_collision_paths: list[Path] | None = None,
    merged_visual_path: Path | None = None,
    scale_factor: float = 1.0,
) -> ET.Element:
    """Convert URDF link element to SDF format.

    Args:
        urdf_link: URDF <link> element.
        urdf_dir: Directory containing URDF.
        sdf_dir: Directory where SDF will be written.
        physics: Physics properties for the link.
        friction: Friction coefficient for collisions.
        link_pose: Optional tuple of (xyz, rpy, relative_to_frame) for link pose.
            In URDF, child links are positioned by joint origins. In SDF, we need
            to explicitly set link poses relative to parent links.
        generated_collision_paths: Optional list of paths to pre-generated convex
            decomposition collision meshes. If provided, these are used instead
            of converting URDF collision elements.
        merged_visual_path: Optional path to pre-merged GLTF visual mesh. If
            provided, a single visual element is created using this mesh instead
            of converting individual URDF visual elements.
        scale_factor: Uniform scale factor to apply to geometry and positions.

    Returns:
        SDF <link> element.
    """
    link_name = urdf_link.get("name", "unnamed_link")
    sdf_link = ET.Element("link", name=link_name)

    # Set link pose (from joint origin in URDF semantics). Scale positions.
    if link_pose is not None:
        xyz, rpy, relative_to = link_pose
        scaled_xyz = [v * scale_factor for v in xyz]
        pose = ET.SubElement(sdf_link, "pose")
        pose.set("relative_to", relative_to)
        pose.text = pose_to_string(scaled_xyz, rpy)

    # Check if link has any geometry (visual or collision).
    has_geometry = bool(urdf_link.findall("visual") or urdf_link.findall("collision"))

    # Add inertial properties.
    # Mass is from VLM (already appropriate for target scale).
    # Center of mass and inertia are computed from mesh geometry (need scaling).
    if physics is not None:
        inertial = ET.SubElement(sdf_link, "inertial")
        ET.SubElement(inertial, "mass").text = f"{physics.mass:.6f}"

        # Center of mass pose (scale position, keep orientation).
        com_pose = ET.SubElement(inertial, "pose")
        com = physics.center_of_mass
        scaled_com = [v * scale_factor for v in com]
        com_pose.text = (
            f"{scaled_com[0]:.6f} {scaled_com[1]:.6f} {scaled_com[2]:.6f} 0 0 0"
        )

        # Inertia tensor (scales as scale_factor^2 for same mass at smaller distances).
        inertia_scale = scale_factor * scale_factor
        inertia = ET.SubElement(inertial, "inertia")
        ET.SubElement(inertia, "ixx").text = (
            f"{physics.inertia_ixx * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "iyy").text = (
            f"{physics.inertia_iyy * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "izz").text = (
            f"{physics.inertia_izz * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "ixy").text = (
            f"{physics.inertia_ixy * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "ixz").text = (
            f"{physics.inertia_ixz * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "iyz").text = (
            f"{physics.inertia_iyz * inertia_scale:.6e}"
        )
    elif has_geometry:
        raise ValueError(
            f"Link '{link_name}' has geometry but no physics properties provided. "
            f"VLM analysis should provide physics for all links with geometry."
        )
    else:
        # Add explicit zero inertia for massless links (e.g., 'base' anchor frame).
        # Without this, libsdformat assigns default mass=1kg which affects physics.
        inertial = ET.SubElement(sdf_link, "inertial")
        ET.SubElement(inertial, "mass").text = "0.0"
        inertia = ET.SubElement(inertial, "inertia")
        for comp in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]:
            ET.SubElement(inertia, comp).text = "0.0"

    # Convert visual elements.
    # Use merged visual path if provided, else convert from URDF.
    if merged_visual_path:
        # Create single visual element for merged GLTF.
        sdf_visual = ET.Element("visual", name=f"{link_name}_visual")
        pose = ET.SubElement(sdf_visual, "pose")
        pose.text = "0 0 0 0 0 0"  # Identity pose (offsets baked into GLTF).
        sdf_geometry = ET.SubElement(sdf_visual, "geometry")
        sdf_mesh = ET.SubElement(sdf_geometry, "mesh")
        uri = ET.SubElement(sdf_mesh, "uri")
        rel_path = merged_visual_path.relative_to(sdf_dir)
        uri.text = str(rel_path)
        # Add scale element if not 1.0.
        if scale_factor != 1.0:
            scale_elem = ET.SubElement(sdf_mesh, "scale")
            scale_elem.text = f"{scale_factor} {scale_factor} {scale_factor}"
        sdf_link.append(sdf_visual)
    else:
        for i, urdf_visual in enumerate(urdf_link.findall("visual")):
            sdf_visual = convert_urdf_visual_to_sdf(
                urdf_visual, urdf_dir, i, link_name, scale_factor=scale_factor
            )
            if sdf_visual is not None:
                sdf_link.append(sdf_visual)

    # Convert collision elements.
    # Use generated collision paths if provided, else convert from URDF.
    if generated_collision_paths:
        collision_elements = create_sdf_collision_elements_from_paths(
            collision_paths=generated_collision_paths,
            sdf_dir=sdf_dir,
            link_name=link_name,
            friction=friction,
            scale_factor=scale_factor,
        )
        for sdf_collision in collision_elements:
            sdf_link.append(sdf_collision)
    else:
        for i, urdf_collision in enumerate(urdf_link.findall("collision")):
            sdf_collision = convert_urdf_collision_to_sdf(
                urdf_collision, urdf_dir, i, link_name, friction
            )
            if sdf_collision is not None:
                sdf_link.append(sdf_collision)

    return sdf_link


def convert_urdf_joint_to_sdf(
    urdf_joint: ET.Element, include_dynamics: bool = True
) -> ET.Element:
    """Convert URDF joint element to SDF format.

    In URDF, joint origin positions the child link relative to parent. In SDF,
    we handle this by setting link poses (relative_to parent link) in
    convert_urdf_link_to_sdf. The joint pose defaults to identity relative to
    child, which is correct since in URDF the child link frame coincides with
    the joint frame at q=0.

    Args:
        urdf_joint: URDF <joint> element.
        include_dynamics: Whether to include damping/friction dynamics.

    Returns:
        SDF <joint> element.
    """
    joint_name = urdf_joint.get("name", "unnamed_joint")
    joint_type = urdf_joint.get("type", "fixed")

    # Map URDF joint types to SDF.
    type_map = {
        "revolute": "revolute",
        "continuous": "revolute",  # SDF uses revolute with no limits.
        "prismatic": "prismatic",
        "fixed": "fixed",
        "floating": "ball",  # Approximate.
        "planar": "universal",  # Approximate.
    }
    sdf_type = type_map.get(joint_type, "fixed")

    sdf_joint = ET.Element("joint", name=joint_name, type=sdf_type)

    # Parent and child.
    parent_elem = urdf_joint.find("parent")
    child_elem = urdf_joint.find("child")

    if parent_elem is not None:
        parent = ET.SubElement(sdf_joint, "parent")
        parent.text = parent_elem.get("link", "")

    if child_elem is not None:
        child = ET.SubElement(sdf_joint, "child")
        child.text = child_elem.get("link", "")

    # Axis (for revolute/prismatic).
    if sdf_type in ("revolute", "prismatic"):
        axis_elem = urdf_joint.find("axis")
        sdf_axis = ET.SubElement(sdf_joint, "axis")

        if axis_elem is not None:
            axis_xyz = axis_elem.get("xyz", "0 0 1")
        else:
            axis_xyz = "0 0 1"  # Default Z-axis.

        xyz_elem = ET.SubElement(sdf_axis, "xyz")
        xyz_elem.text = axis_xyz

        # Limits.
        limit_elem = urdf_joint.find("limit")
        if limit_elem is not None:
            sdf_limit = ET.SubElement(sdf_axis, "limit")

            lower = limit_elem.get("lower")
            upper = limit_elem.get("upper")
            effort = limit_elem.get("effort")
            velocity = limit_elem.get("velocity")

            if lower is not None:
                ET.SubElement(sdf_limit, "lower").text = lower
            if upper is not None:
                ET.SubElement(sdf_limit, "upper").text = upper
            if effort is not None:
                ET.SubElement(sdf_limit, "effort").text = effort
            if velocity is not None:
                ET.SubElement(sdf_limit, "velocity").text = velocity
        elif joint_type == "continuous":
            # No limits for continuous joints.
            pass

        # Dynamics (damping, friction).
        if include_dynamics:
            dynamics = ET.SubElement(sdf_axis, "dynamics")
            ET.SubElement(dynamics, "damping").text = f"{DEFAULT_JOINT_DAMPING}"
            ET.SubElement(dynamics, "friction").text = f"{DEFAULT_JOINT_FRICTION}"

    return sdf_joint


def convert_urdf_to_sdf(
    urdf_path: Path,
    output_path: Path,
    link_physics: dict[str, LinkPhysics] | None = None,
    link_friction: dict[str, float] | None = None,
    model_name: str | None = None,
    repair_missing_meshes: bool = True,
    model_pose: tuple[float, float, float, float, float, float] | None = None,
    generate_collision: bool = False,
    collision_client: ConvexDecompositionClient | None = None,
    collision_threshold: float = 0.05,
    merge_visuals: bool = False,
    scale_factor: float = 1.0,
) -> Path:
    """Convert URDF file to Drake-compatible SDF format.

    Args:
        urdf_path: Path to input URDF file.
        output_path: Path to output SDF file.
        link_physics: Optional dict mapping link names to physics properties.
        link_friction: Optional dict mapping link names to friction coefficients.
        model_name: Optional model name (defaults to URDF robot name).
        repair_missing_meshes: Whether to remove references to missing mesh files.
        model_pose: Optional model-level pose (x, y, z, roll, pitch, yaw) for
            canonicalization. Applied to the entire model as a single transform.
            Use this instead of transforming individual mesh poses.
        generate_collision: Whether to generate convex collision geometry using
            CoACD. If True, collision meshes are generated from visual geometry
            and saved to the output directory. If False, existing URDF collision
            elements are converted as-is.
        collision_client: Convex decomposition client for collision geometry
            generation. Required when generate_collision is True.
        collision_threshold: CoACD approximation threshold (0.01-0.1 typical).
            Lower values produce more convex pieces with higher fidelity.
        merge_visuals: Whether to merge all visual meshes per link into a single
            GLTF file. Reduces draw calls and simplifies file structure.
        scale_factor: Uniform scale factor to apply to all geometry and positions.
            Used to correct unit conversion issues (e.g., 0.1 to convert from
            decimeters to meters).

    Returns:
        Path to generated SDF file.

    Raises:
        FileNotFoundError: If URDF file doesn't exist.
        ValueError: If URDF structure is invalid or generate_collision is True
            but collision_client is None.
    """
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    if generate_collision and collision_client is None:
        raise ValueError(
            "collision_client is required when generate_collision is True. "
            "Start a convex decomposition server and pass the client."
        )

    urdf_dir = urdf_path.parent
    sdf_dir = output_path.parent

    # Parse URDF.
    urdf_result = parse_urdf(urdf_path)

    # Validate and optionally repair missing meshes.
    _, missing_meshes = validate_urdf_meshes(urdf_path, urdf_result)

    if missing_meshes:
        console_logger.warning(
            f"URDF '{urdf_path.name}' references {len(missing_meshes)} missing mesh "
            f"files: {missing_meshes[:5]}{'...' if len(missing_meshes) > 5 else ''}"
        )

        if repair_missing_meshes:
            # Create repaired URDF in temp location and re-parse.
            repaired_path = output_path.parent / f"{urdf_path.stem}_repaired.urdf"
            repaired_path, removed = repair_urdf_missing_meshes(
                urdf_path, repaired_path
            )
            console_logger.info(
                f"Repaired URDF by removing {len(removed)} missing mesh references"
            )
            urdf_result = parse_urdf(repaired_path)

    # Create SDF structure.
    sdf = ET.Element("sdf", version="1.7")
    model = ET.SubElement(sdf, "model", name=model_name or urdf_result.robot_name)

    # Add model-level pose for canonicalization (applied to entire model).
    # Scale position components but not rotation.
    if model_pose is not None:
        scaled_pose = (
            model_pose[0] * scale_factor,
            model_pose[1] * scale_factor,
            model_pose[2] * scale_factor,
            model_pose[3],
            model_pose[4],
            model_pose[5],
        )
        pose_elem = ET.SubElement(model, "pose")
        pose_elem.text = (
            f"{scaled_pose[0]:.8f} {scaled_pose[1]:.8f} {scaled_pose[2]:.8f} "
            f"{scaled_pose[3]:.8f} {scaled_pose[4]:.8f} {scaled_pose[5]:.8f}"
        )

    # Build mapping of child_link -> (parent_link, joint_origin).
    # In URDF, joint origin positions the child relative to parent.
    # In SDF, we express this as link pose relative to parent.
    link_poses: dict[str, tuple[list[float], list[float], str]] = {}
    for urdf_joint in urdf_result.joints.values():
        child_elem = urdf_joint.find("child")
        parent_elem = urdf_joint.find("parent")
        if child_elem is not None and parent_elem is not None:
            child_link = child_elem.get("link")
            parent_link = parent_elem.get("link")
            if child_link and parent_link:
                origin = urdf_joint.find("origin")
                xyz, rpy = parse_origin(origin)
                link_poses[child_link] = (xyz, rpy, parent_link)

    # Generate collision geometry for all links if requested.
    # This is done before link conversion to have collision paths available.
    link_collision_paths: dict[str, list[Path]] = {}
    if generate_collision:
        console_logger.info(
            f"Generating collision geometry (threshold={collision_threshold})"
        )
        for link_name, urdf_link in urdf_result.links.items():
            # Only generate for links with visual geometry.
            if urdf_link.findall("visual"):
                collision_paths = generate_link_collision_geometry_for_sdf(
                    urdf_link=urdf_link,
                    urdf_dir=urdf_dir,
                    sdf_dir=sdf_dir,
                    link_name=link_name,
                    collision_client=collision_client,  # type: ignore[arg-type]
                    collision_threshold=collision_threshold,
                )
                if collision_paths:
                    link_collision_paths[link_name] = collision_paths
                    console_logger.info(
                        f"Generated {len(collision_paths)} collision meshes "
                        f"for link '{link_name}'"
                    )

    # Merge visual meshes for all links if requested.
    # This is done before link conversion to have merged paths available.
    link_merged_visuals: dict[str, Path] = {}
    if merge_visuals:
        console_logger.info("Merging visual meshes per link")
        for link_name, urdf_link in urdf_result.links.items():
            # Only merge for links with visual geometry.
            if urdf_link.findall("visual"):
                merged_path = merge_link_visual_meshes_for_sdf(
                    urdf_link=urdf_link,
                    urdf_dir=urdf_dir,
                    sdf_dir=sdf_dir,
                    link_name=link_name,
                )
                if merged_path:
                    link_merged_visuals[link_name] = merged_path

    # Convert links.
    for link_name, urdf_link in urdf_result.links.items():
        physics = link_physics.get(link_name) if link_physics else None
        friction = link_friction.get(link_name, 0.5) if link_friction else 0.5
        pose = link_poses.get(link_name)
        collision_paths = link_collision_paths.get(link_name)
        merged_visual = link_merged_visuals.get(link_name)

        sdf_link = convert_urdf_link_to_sdf(
            urdf_link=urdf_link,
            urdf_dir=urdf_dir,
            sdf_dir=sdf_dir,
            physics=physics,
            friction=friction,
            link_pose=pose,
            generated_collision_paths=collision_paths,
            merged_visual_path=merged_visual,
            scale_factor=scale_factor,
        )
        model.append(sdf_link)

    # Convert joints.
    for urdf_joint in urdf_result.joints.values():
        sdf_joint = convert_urdf_joint_to_sdf(urdf_joint)
        model.append(sdf_joint)

    # Format XML with indentation.
    ET.indent(sdf, space="  ", level=0)

    # Write SDF file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(sdf)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    # Fix any inertia tensors that violate the triangle inequality.
    fix_sdf_file_inertia(output_path)

    console_logger.info(
        f"Converted URDF to SDF: {urdf_path.name} -> {output_path.name}"
    )

    return output_path


def compute_forward_kinematics(
    urdf_result: URDFParseResult,
    joint_positions: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute link transforms at given joint positions.

    Args:
        urdf_result: Parsed URDF result.
        joint_positions: Dict mapping joint names to positions (default: all zeros).

    Returns:
        Dict mapping link names to 4x4 homogeneous transform matrices.
    """
    if joint_positions is None:
        joint_positions = {}

    # Build joint info for quick lookup.
    joint_info = {}  # child_link -> (joint_elem, parent_link)
    for joint_elem in urdf_result.joints.values():
        child_elem = joint_elem.find("child")
        parent_elem = joint_elem.find("parent")
        if child_elem is not None and parent_elem is not None:
            child_link = child_elem.get("link")
            parent_link = parent_elem.get("link")
            if child_link and parent_link:
                joint_info[child_link] = (joint_elem, parent_link)

    # Initialize transforms.
    transforms = {}

    def compute_transform(link_name: str) -> np.ndarray:
        """Recursively compute transform for a link."""
        if link_name in transforms:
            return transforms[link_name]

        if link_name not in joint_info:
            # Root link, identity transform.
            transforms[link_name] = np.eye(4)
            return transforms[link_name]

        joint_elem, parent_link = joint_info[link_name]

        # Get parent transform.
        parent_transform = compute_transform(parent_link)

        # Get joint origin transform.
        origin = joint_elem.find("origin")
        xyz, rpy = parse_origin(origin)

        # Build transform from origin.
        R = Rotation.from_euler("xyz", rpy).as_matrix()
        T_origin = np.eye(4)
        T_origin[:3, :3] = R
        T_origin[:3, 3] = xyz

        # Apply joint motion if applicable.
        joint_type = joint_elem.get("type", "fixed")
        joint_name = joint_elem.get("name", "")
        q = joint_positions.get(joint_name, 0.0)

        T_joint = np.eye(4)
        if joint_type in ("revolute", "continuous") and q != 0.0:
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_xyz = axis_elem.get("xyz", "0 0 1")
                axis = np.array([float(x) for x in axis_xyz.split()])
                axis = axis / np.linalg.norm(axis)
                T_joint[:3, :3] = Rotation.from_rotvec(q * axis).as_matrix()
        elif joint_type == "prismatic" and q != 0.0:
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_xyz = axis_elem.get("xyz", "0 0 1")
                axis = np.array([float(x) for x in axis_xyz.split()])
                axis = axis / np.linalg.norm(axis)
                T_joint[:3, 3] = q * axis

        # Combine transforms.
        transforms[link_name] = parent_transform @ T_origin @ T_joint
        return transforms[link_name]

    # Compute transforms for all links.
    for link_name in urdf_result.links:
        compute_transform(link_name)

    return transforms


def get_link_meshes(
    urdf_path: Path, urdf_result: URDFParseResult
) -> dict[str, list[Path]]:
    """Get mesh file paths for each link.

    Args:
        urdf_path: Path to URDF file.
        urdf_result: Parsed URDF result.

    Returns:
        Dict mapping link names to lists of mesh file paths.
    """
    urdf_dir = urdf_path.parent
    link_meshes = {}

    for link_name, link_elem in urdf_result.links.items():
        meshes = []

        for visual_or_collision in link_elem.findall("visual") + link_elem.findall(
            "collision"
        ):
            geometry = visual_or_collision.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename:
                        mesh_path = urdf_dir / filename
                        if mesh_path.exists() and mesh_path not in meshes:
                            meshes.append(mesh_path)

        if meshes:
            link_meshes[link_name] = meshes

    return link_meshes


def compute_articulated_bounding_box(
    urdf_path: Path, joint_positions: dict[str, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bounding box of articulated model at given joint positions.

    This computes the axis-aligned bounding box of all mesh vertices
    transformed to world space using forward kinematics.

    Args:
        urdf_path: Path to URDF file.
        joint_positions: Dict mapping joint names to positions (default: all zeros).

    Returns:
        Tuple of (min_xyz, max_xyz, center) arrays.
    """
    # Parse URDF.
    urdf_result = parse_urdf(urdf_path)
    urdf_dir = urdf_path.parent

    # Compute link transforms.
    link_transforms = compute_forward_kinematics(urdf_result, joint_positions)

    # Transform all mesh vertices to world space.
    all_verts_world = []
    for link_name, link_elem in urdf_result.links.items():
        if link_name not in link_transforms:
            continue

        T = link_transforms[link_name]

        # Get all visual/collision meshes with their origin transforms.
        for geom_type in ["visual", "collision"]:
            for geom in link_elem.findall(geom_type):
                geometry = geom.find("geometry")
                if geometry is not None:
                    mesh_elem = geometry.find("mesh")
                    if mesh_elem is not None:
                        filename = mesh_elem.get("filename")
                        if filename:
                            mesh_path = urdf_dir / filename
                            if not mesh_path.exists():
                                continue

                            try:
                                mesh = trimesh.load(mesh_path, force="mesh")
                                if isinstance(mesh, trimesh.Scene):
                                    meshes = [
                                        g
                                        for g in mesh.geometry.values()
                                        if isinstance(g, trimesh.Trimesh)
                                    ]
                                    if meshes:
                                        mesh = trimesh.util.concatenate(meshes)
                                    else:
                                        continue

                                # Apply visual/collision origin transform.
                                geom_xyz, geom_rot = _parse_geom_origin(geom)
                                verts = mesh.vertices @ geom_rot.T + geom_xyz

                                # Then apply link transform.
                                verts_homogeneous = np.hstack(
                                    [verts, np.ones((len(verts), 1))]
                                )
                                verts_world = (T @ verts_homogeneous.T).T[:, :3]
                                all_verts_world.extend(verts_world)

                            except Exception as e:
                                console_logger.warning(
                                    f"Failed to load mesh {mesh_path}: {e}"
                                )
                                continue

    if not all_verts_world:
        console_logger.warning("No mesh vertices found for bounding box computation")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    all_verts_world = np.array(all_verts_world)
    min_xyz = np.min(all_verts_world, axis=0)
    max_xyz = np.max(all_verts_world, axis=0)
    center = (min_xyz + max_xyz) / 2

    return min_xyz, max_xyz, center


def compute_sdf_bounding_box(
    sdf_path: Path,
    scale_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bounding box from SDF visual meshes with transforms applied.

    This computes the bounding box by loading GLTF meshes from the SDF's visual
    directory and applying the link pose transforms defined in the SDF. This
    accounts for coordinate frame changes introduced by the GLTF Y-up export.

    Note: The bounding box is computed in model frame (without model-level pose),
    suitable for computing the canonicalization pose.

    Args:
        sdf_path: Path to SDF file.
        scale_factor: Scale factor applied to mesh geometry (from <scale> element).

    Returns:
        Tuple of (min_xyz, max_xyz, center) arrays in model frame.
    """
    sdf_dir = sdf_path.parent
    visual_dir = sdf_dir / "visual"

    if not visual_dir.exists():
        console_logger.warning(f"Visual directory not found: {visual_dir}")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    # Parse SDF to get link poses.
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model = root.find("model")
    if model is None:
        console_logger.warning("No model element found in SDF")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    # First pass: collect raw poses and relative_to info.
    # SDF uses relative_to for pose inheritance, so we need to resolve the chain.
    raw_poses: dict[str, tuple[np.ndarray, np.ndarray, str | None]] = {}

    for link in model.findall("link"):
        link_name = link.get("name", "")
        pose_elem = link.find("pose")

        if pose_elem is not None and pose_elem.text:
            values = [float(v) for v in pose_elem.text.strip().split()]
            xyz = np.array(values[:3])
            rpy = values[3:6]
            rot = Rotation.from_euler("xyz", rpy).as_matrix()
            relative_to = pose_elem.get("relative_to", None)
        else:
            xyz = np.zeros(3)
            rot = np.eye(3)
            relative_to = None

        raw_poses[link_name] = (xyz, rot, relative_to)

    # Second pass: resolve relative_to chain to get model-frame poses.
    link_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def resolve_pose(
        name: str, visited: set | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recursively resolve pose chain to get model-frame pose."""
        if visited is None:
            visited = set()

        if name in link_poses:
            return link_poses[name]

        if name in visited:
            console_logger.warning(f"Circular reference in pose chain: {name}")
            return np.zeros(3), np.eye(3)
        visited.add(name)

        if name not in raw_poses:
            # Unknown link or "base" - return identity (model frame origin).
            return np.zeros(3), np.eye(3)

        xyz, rot, relative_to = raw_poses[name]

        if relative_to is None or relative_to == "__model__" or relative_to == "base":
            # Directly relative to model frame.
            model_xyz, model_rot = xyz, rot
        else:
            # Relative to another link - compose transforms.
            parent_xyz, parent_rot = resolve_pose(relative_to, visited)
            # T_model_link = T_model_parent * T_parent_link
            model_xyz = parent_rot @ xyz + parent_xyz
            model_rot = parent_rot @ rot

        link_poses[name] = (model_xyz, model_rot)
        return model_xyz, model_rot

    for name in raw_poses:
        resolve_pose(name)

    # Load GLTF meshes and apply transforms.
    all_verts_world = []

    for gltf_path in visual_dir.glob("*.gltf"):
        # Extract link name from filename (e.g., "link_0_visual.gltf" -> "link_0").
        link_name = gltf_path.stem.replace("_visual", "")

        if link_name not in link_poses:
            continue

        try:
            mesh = trimesh.load(gltf_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                meshes = [
                    g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    continue

            # R_GF converts: file Y → geometry Z, file Z → geometry -Y
            # This matches Drake's MakeFromOrthonormalColumns(UnitX, UnitZ, -UnitY).
            R_GF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

            # Transform chain: R_GF (Y-up to Z-up) → scale → link pose
            verts = mesh.vertices @ R_GF.T  # Apply Y-up to Z-up first
            verts = verts * scale_factor  # Then scale

            # Apply link pose transform.
            # Note: Link translations from URDF are in original (unscaled) coordinates,
            # so we must scale them to match the scaled mesh geometry.
            xyz, rot = link_poses[link_name]
            xyz_scaled = xyz * scale_factor
            verts_world = verts @ rot.T + xyz_scaled
            all_verts_world.extend(verts_world)

        except Exception as e:
            console_logger.warning(f"Failed to load GLTF {gltf_path}: {e}")
            continue

    if not all_verts_world:
        console_logger.warning("No mesh vertices found for SDF bounding box")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    all_verts_world = np.array(all_verts_world)
    min_xyz = np.min(all_verts_world, axis=0)
    max_xyz = np.max(all_verts_world, axis=0)
    center = (min_xyz + max_xyz) / 2

    return min_xyz, max_xyz, center


def update_sdf_model_pose(
    sdf_path: Path,
    model_pose: tuple[float, float, float, float, float, float],
) -> None:
    """Update the model-level pose in an existing SDF file.

    Args:
        sdf_path: Path to SDF file to update.
        model_pose: New model pose (x, y, z, roll, pitch, yaw).
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model = root.find("model")

    if model is None:
        raise ValueError(f"No model element found in SDF: {sdf_path}")

    # Find or create pose element (should be first child after model name).
    pose_elem = model.find("pose")
    if pose_elem is None:
        # Insert pose as first child.
        pose_elem = ET.Element("pose")
        model.insert(0, pose_elem)

    # Update pose text.
    pose_elem.text = (
        f"{model_pose[0]:.8f} {model_pose[1]:.8f} {model_pose[2]:.8f} "
        f"{model_pose[3]:.8f} {model_pose[4]:.8f} {model_pose[5]:.8f}"
    )

    # Re-indent and write.
    ET.indent(root, space="  ", level=0)
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
    console_logger.info(f"Updated model pose in {sdf_path.name}")


def _parse_geom_origin(geom_elem: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """Parse geometry origin into translation and rotation matrix."""
    origin = geom_elem.find("origin")
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz_str = origin.get("xyz", "0 0 0")
    xyz = np.array([float(v) for v in xyz_str.split()])
    rpy_str = origin.get("rpy", "0 0 0")
    rpy = np.array([float(v) for v in rpy_str.split()])
    rot = Rotation.from_euler("xyz", rpy).as_matrix()
    return xyz, rot


def compute_link_physics_from_meshes(
    urdf_path: Path, link_masses: dict[str, float]
) -> dict[str, LinkPhysics]:
    """Compute physics properties for each link from mesh geometry and given masses.

    This function loads all meshes for each link, combines them into a single mesh,
    and computes inertial properties using the specified mass.

    Args:
        urdf_path: Path to URDF file.
        link_masses: Dict mapping link names to masses in kg.

    Returns:
        Dict mapping link names to LinkPhysics objects.
    """
    urdf_result = parse_urdf(urdf_path)
    urdf_dir = urdf_path.parent

    physics_dict = {}
    for link_name, link_elem in urdf_result.links.items():
        if link_name not in link_masses:
            # Skip links without mass (e.g., 'base' link with no geometry).
            continue
        mass = link_masses[link_name]

        try:
            # Collect mesh files and their geometry origins.
            mesh_infos: list[tuple[Path, np.ndarray, np.ndarray]] = []
            for geom_type in ["visual", "collision"]:
                for geom in link_elem.findall(geom_type):
                    geometry = geom.find("geometry")
                    if geometry is not None:
                        mesh = geometry.find("mesh")
                        if mesh is not None:
                            filename = mesh.get("filename")
                            if filename:
                                mesh_path = urdf_dir / filename
                                if mesh_path.exists():
                                    xyz, rot = _parse_geom_origin(geom)
                                    # Avoid duplicates (same mesh in visual and collision).
                                    if not any(
                                        p == mesh_path for p, _, _ in mesh_infos
                                    ):
                                        mesh_infos.append((mesh_path, xyz, rot))

            if not mesh_infos:
                continue

            # Load and combine all meshes, applying geometry origins.
            combined_vertices = []
            combined_faces = []
            vertex_offset = 0

            for mesh_path, geom_xyz, geom_rot in mesh_infos:
                mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)
                if mesh is not None:
                    # Apply geometry origin transform.
                    vertices = mesh.vertices @ geom_rot.T + geom_xyz
                    combined_vertices.append(vertices)
                    combined_faces.append(mesh.faces + vertex_offset)
                    vertex_offset += len(vertices)

            if not combined_vertices:
                continue

            # Create combined mesh.
            all_vertices = np.vstack(combined_vertices)
            all_faces = np.vstack(combined_faces)
            combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

            # Compute inertia.
            inertial = compute_inertia_from_mesh(combined_mesh, mass)

            # Convert to LinkPhysics.
            if inertial.inertia_tensor is None:
                raise ValueError(
                    f"Inertia computation failed for link '{link_name}'. "
                    f"Mesh may have invalid geometry."
                )

            physics_dict[link_name] = LinkPhysics(
                mass=inertial.mass,
                inertia_ixx=inertial.inertia_tensor[0, 0],
                inertia_iyy=inertial.inertia_tensor[1, 1],
                inertia_izz=inertial.inertia_tensor[2, 2],
                inertia_ixy=inertial.inertia_tensor[0, 1],
                inertia_ixz=inertial.inertia_tensor[0, 2],
                inertia_iyz=inertial.inertia_tensor[1, 2],
                center_of_mass=tuple(inertial.center_of_mass),
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute physics for link '{link_name}': {e}"
            ) from e

    return physics_dict
