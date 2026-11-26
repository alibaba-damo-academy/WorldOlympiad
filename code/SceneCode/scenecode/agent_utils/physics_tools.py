"""Physics computation utilities for inertia and mass properties."""

import logging

from pathlib import Path

import numpy as np
import trimesh

from omegaconf import DictConfig

from scenecode.agent_utils.clearance_zones import (
    compute_door_clearance_violations,
    compute_open_connection_blocked_violations,
    compute_wall_height_violations,
    compute_window_clearance_violations,
)
from scenecode.agent_utils.physics_validation import (
    compute_scene_collisions,
    compute_thin_covering_boundary_violations,
    compute_thin_covering_overlaps,
    filter_collisions_by_agent,
    filter_door_violations_by_agent,
    filter_open_connection_violations_by_agent,
    filter_thin_covering_boundary_violations_by_agent,
    filter_thin_covering_overlaps_by_agent,
    filter_wall_height_violations_by_agent,
    filter_window_violations_by_agent,
)
from scenecode.agent_utils.room import AgentType, RoomScene, UniqueID

console_logger = logging.getLogger(__name__)


def _format_violations(violations: list, header: str, log_header: str) -> list[str]:
    """Format a list of violations into message lines with logging.

    Args:
        violations: List of violation objects with to_description() method.
        header: Header for the messages (e.g., "Collisions (5):").
        log_header: Header for console logging (e.g., "=== Collisions (5) ===").

    Returns:
        List of formatted message strings, empty if no violations.
    """
    if not violations:
        return []

    console_logger.info(log_header)
    for i, v in enumerate(violations, 1):
        console_logger.info(f"  {i}. {v.to_description()}")

    messages = [header]
    for v in violations:
        messages.append(f"- {v.to_description()}")

    return messages


def _build_violation_message(
    collisions: list,
    thin_covering_overlaps: list,
    thin_covering_boundary_violations: list,
    door_violations: list,
    open_violations: list,
    height_violations: list,
    window_violations: list,
) -> str:
    """Build the final violation message string.

    Args:
        collisions: Object collision violations.
        thin_covering_overlaps: Thin covering overlap violations.
        thin_covering_boundary_violations: Thin covering boundary violations.
        door_violations: Door clearance violations.
        open_violations: Open connection blocked violations.
        height_violations: Wall height exceeded violations.
        window_violations: Window access violations.

    Returns:
        Final formatted message string describing all violations,
        or a success message if no violations exist.
    """
    messages = []
    messages.extend(
        _format_violations(
            collisions,
            f"Collisions ({len(collisions)}):",
            f"=== Collisions ({len(collisions)} total) ===",
        )
    )
    messages.extend(
        _format_violations(
            thin_covering_overlaps,
            f"Thin covering overlaps ({len(thin_covering_overlaps)}):",
            f"=== Thin Covering Overlaps ({len(thin_covering_overlaps)} total) ===",
        )
    )
    messages.extend(
        _format_violations(
            thin_covering_boundary_violations,
            f"Thin covering boundary violations "
            f"({len(thin_covering_boundary_violations)}):",
            f"=== Thin Covering Boundary Violations "
            f"({len(thin_covering_boundary_violations)} total) ===",
        )
    )
    messages.extend(
        _format_violations(
            door_violations,
            f"Door clearance violations ({len(door_violations)}):",
            f"=== Door Clearance Violations ({len(door_violations)} total) ===",
        )
    )
    messages.extend(
        _format_violations(
            open_violations,
            f"Open connection blocked ({len(open_violations)}):",
            f"=== Open Connection Blocked ({len(open_violations)} total) ===",
        )
    )
    messages.extend(
        _format_violations(
            height_violations,
            f"Wall height exceeded ({len(height_violations)}):",
            f"=== Wall Height Exceeded ({len(height_violations)} total) ===",
        )
    )
    messages.extend(
        _format_violations(
            window_violations,
            "Window access warnings:",
            f"=== Window Access Warnings ({len(window_violations)} total) ===",
        )
    )

    total = (
        len(collisions)
        + len(thin_covering_overlaps)
        + len(thin_covering_boundary_violations)
        + len(door_violations)
        + len(open_violations)
        + len(height_violations)
        + len(window_violations)
    )

    if total == 0:
        return "No physics violations detected. All objects are properly placed."

    return (
        f"Physics violations detected ({total} issue(s)):\n"
        f"{chr(10).join(messages)}\n\n"
        f"Please resolve these issues before concluding the design."
    )


def check_physics_violations(
    scene: RoomScene,
    cfg: DictConfig,
    current_furniture_id: UniqueID | None = None,
    agent_type: AgentType | None = None,
) -> str:
    """Check for physics violations (collisions, penetrations) in the scene.

    Args:
        scene: RoomScene object to validate.
        cfg: Configuration containing physics_validation settings with:
            - object_penetration_threshold_m: Threshold for object-object collisions.
            - floor_penetration_tolerance_m: Tolerance for furniture-floor penetration.
        current_furniture_id: Optional ID of furniture currently being populated by
            manipuland agent. When provided, filters out collisions involving
            manipulands from other furniture.
        agent_type: Optional agent type for filtering. When provided, only shows
            violations involving objects the agent can modify.

    Returns:
        String describing collision status. Either "No physics violations detected..."
        or a detailed description of all detected collisions.
    """
    console_logger.info("Checking physics violations")
    cfg_physics = cfg.physics_validation

    # Compute all violation types upfront.
    collisions = compute_scene_collisions(
        scene=scene,
        penetration_threshold=cfg_physics.object_penetration_threshold_m,
        floor_penetration_tolerance=cfg_physics.floor_penetration_tolerance_m,
        current_furniture_id=current_furniture_id,
        manipuland_furniture_tolerance_m=cfg_physics.manipuland_furniture_tolerance_m,
    )
    thin_covering_overlaps = compute_thin_covering_overlaps(scene)
    thin_covering_boundary_violations = compute_thin_covering_boundary_violations(
        scene=scene, wall_thickness=cfg_physics.wall_thickness
    )

    # Clearance zone violations (require room geometry with openings).
    room_geom = scene.room_geometry
    door_violations = []
    open_violations = []
    height_violations = []
    window_violations = []
    if room_geom and room_geom.openings:
        door_violations = compute_door_clearance_violations(scene=scene)
        open_violations = compute_open_connection_blocked_violations(
            scene=scene,
            passage_size=cfg.clearance_zones.passage_size,
            open_connection_clearance=cfg.clearance_zones.open_connection_clearance,
        )
        height_violations = compute_wall_height_violations(scene=scene)
        window_violations = compute_window_clearance_violations(scene=scene)

    # Apply agent-type filtering if specified.
    if agent_type is not None:
        collisions = filter_collisions_by_agent(
            collisions=collisions,
            scene=scene,
            agent_type=agent_type,
            current_furniture_id=current_furniture_id,
        )
        thin_covering_overlaps = filter_thin_covering_overlaps_by_agent(
            overlaps=thin_covering_overlaps,
            scene=scene,
            agent_type=agent_type,
            current_furniture_id=current_furniture_id,
        )
        thin_covering_boundary_violations = (
            filter_thin_covering_boundary_violations_by_agent(
                violations=thin_covering_boundary_violations,
                agent_type=agent_type,
            )
        )
        door_violations = filter_door_violations_by_agent(
            violations=door_violations, scene=scene, agent_type=agent_type
        )
        open_violations = filter_open_connection_violations_by_agent(
            violations=open_violations, scene=scene, agent_type=agent_type
        )
        height_violations = filter_wall_height_violations_by_agent(
            violations=height_violations, scene=scene, agent_type=agent_type
        )
        window_violations = filter_window_violations_by_agent(
            violations=window_violations, scene=scene, agent_type=agent_type
        )

    return _build_violation_message(
        collisions=collisions,
        thin_covering_overlaps=thin_covering_overlaps,
        thin_covering_boundary_violations=thin_covering_boundary_violations,
        door_violations=door_violations,
        open_violations=open_violations,
        height_violations=height_violations,
        window_violations=window_violations,
    )


class InertialProperties:
    """Inertial properties computed from a mesh.

    Attributes:
        mass: Mass in kg.
        center_of_mass: Center of mass [x, y, z] in meters.
        inertia_tensor: 3x3 inertia tensor in kg*m^2, or None if invalid.
        is_valid: Whether the computed inertia tensor is valid (positive eigenvalues).
    """

    def __init__(
        self,
        mass: float,
        center_of_mass: np.ndarray,
        inertia_tensor: np.ndarray | None,
        is_valid: bool,
    ):
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia_tensor = inertia_tensor
        self.is_valid = is_valid


def compute_inertia_from_bounding_box(
    mesh: trimesh.Trimesh, mass: float
) -> InertialProperties:
    """Compute inertial properties from mesh bounding box.

    This is a fallback for non-watertight meshes where volume-based inertia
    computation fails. Uses the solid box inertia formula:
        Ixx = (1/12) * m * (h^2 + d^2)
        Iyy = (1/12) * m * (w^2 + d^2)
        Izz = (1/12) * m * (w^2 + h^2)

    Args:
        mesh: Trimesh object to compute inertia for.
        mass: Target mass in kg. Must be positive.

    Returns:
        InertialProperties with computed mass, center of mass, and inertia tensor.

    Raises:
        ValueError: If mass is not positive.
    """
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")

    # Get bounding box dimensions.
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    w, h, d = extents[0], extents[1], extents[2]

    # Compute solid box inertia.
    ixx = (1.0 / 12.0) * mass * (h**2 + d**2)
    iyy = (1.0 / 12.0) * mass * (w**2 + d**2)
    izz = (1.0 / 12.0) * mass * (w**2 + h**2)

    inertia_tensor = np.array(
        [
            [ixx, 0.0, 0.0],
            [0.0, iyy, 0.0],
            [0.0, 0.0, izz],
        ]
    )

    # Center of mass is bounding box center.
    center_of_mass = mesh.centroid

    return InertialProperties(
        mass=mass,
        center_of_mass=center_of_mass,
        inertia_tensor=inertia_tensor,
        is_valid=True,
    )


def compute_inertia_from_mesh(
    mesh: trimesh.Trimesh,
    mass: float,
    validate_eigenvalues: bool = True,
    fallback_to_bounding_box: bool = True,
) -> InertialProperties:
    """Compute inertial properties from mesh geometry and specified mass.

    This function computes the inertia tensor assuming uniform density throughout
    the mesh volume. The density is calculated from the given mass and mesh volume,
    then applied to trimesh's moment of inertia computation.

    For non-watertight meshes, the function can optionally fall back to bounding
    box inertia computation, which is less accurate but always works.

    Args:
        mesh: Trimesh object to compute inertia for. Should be watertight for
            accurate volume-based computation.
        mass: Target mass in kg. Must be positive.
        validate_eigenvalues: If True, validate that the computed inertia tensor has
            positive eigenvalues (physically valid). If validation fails and
            fallback_to_bounding_box is True, uses bounding box inertia. Default: True.
        fallback_to_bounding_box: If True, use bounding box inertia when volume-based
            computation fails (non-watertight mesh or invalid eigenvalues).
            Default: True.

    Returns:
        InertialProperties with computed mass, center of mass, inertia tensor,
        and validity flag.

    Raises:
        ValueError: If mass is not positive, or if mesh has invalid volume and
            fallback_to_bounding_box is False.
    """
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")

    # Get volume and validate.
    volume = mesh.volume
    if volume <= 0:
        if fallback_to_bounding_box:
            console_logger.debug(
                f"Mesh has invalid volume ({volume:.6f}), using bounding box inertia."
            )
            return compute_inertia_from_bounding_box(mesh, mass)
        raise ValueError(
            f"Mesh has invalid volume: {volume}. Mesh may be non-watertight or "
            f"have inverted normals."
        )

    # Compute density.
    density = mass / volume

    # Compute inertia tensor from trimesh (assumes uniform density).
    # trimesh.moment_inertia returns the moment of inertia for unit density,
    # so we scale by actual density.
    inertia_tensor = mesh.moment_inertia * density

    # Validate inertia tensor has positive eigenvalues.
    is_valid = True
    if validate_eigenvalues:
        eigenvalues = np.linalg.eigvals(inertia_tensor)
        if np.any(eigenvalues < 0):
            if fallback_to_bounding_box:
                console_logger.debug(
                    f"Inertia tensor has negative eigenvalues "
                    f"[{eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}, "
                    f"{eigenvalues[2]:.3f}], using bounding box inertia."
                )
                return compute_inertia_from_bounding_box(mesh, mass)
            console_logger.warning(
                f"Computed inertia tensor has negative eigenvalues "
                f"[{eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}, {eigenvalues[2]:.3f}]. "
                f"This indicates inverted mesh geometry."
            )
            is_valid = False
            inertia_tensor = None

    # Get center of mass.
    center_of_mass = mesh.center_mass

    return InertialProperties(
        mass=mass,
        center_of_mass=center_of_mass,
        inertia_tensor=inertia_tensor,
        is_valid=is_valid,
    )


def compute_inertia_from_mesh_path(
    mesh_path: Path,
    mass: float,
    validate_eigenvalues: bool = True,
    fallback_to_bounding_box: bool = True,
) -> InertialProperties:
    """Compute inertial properties from mesh file and specified mass.

    Convenience wrapper around compute_inertia_from_mesh that handles loading
    the mesh from a file path.

    Args:
        mesh_path: Path to mesh file (GLTF, GLB, OBJ, STL, etc.). Must exist.
        mass: Target mass in kg. Must be positive.
        validate_eigenvalues: If True, validate that the computed inertia tensor has
            positive eigenvalues. Default: True.
        fallback_to_bounding_box: If True, use bounding box inertia when volume-based
            computation fails. Default: True.

    Returns:
        InertialProperties with computed mass, center of mass, inertia tensor,
        and validity flag.

    Raises:
        FileNotFoundError: If mesh_path does not exist.
        ValueError: If mass is not positive or mesh has invalid volume and
            fallback_to_bounding_box is False.
    """
    # Late import to avoid circular dependency with mesh_utils.
    from scenecode.agent_utils.mesh_utils import load_mesh_as_trimesh

    mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)
    return compute_inertia_from_mesh(
        mesh=mesh,
        mass=mass,
        validate_eigenvalues=validate_eigenvalues,
        fallback_to_bounding_box=fallback_to_bounding_box,
    )
