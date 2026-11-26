import logging
import tempfile
import time
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Context,
    DiagramBuilder,
    LoadModelDirectives,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    SceneGraph,
)
from pydrake.multibody.tree import JointIndex

from scenecode.agent_utils.room import RoomScene

console_logger = logging.getLogger(__name__)


def create_drake_plant_and_scene_graph_from_scene(
    scene: RoomScene,
    builder: DiagramBuilder | None = None,
    include_objects: list | None = None,
    exclude_room_geometry: bool = False,
    weld_furniture: bool = True,
    free_mounted_objects_for_collision: bool = False,
) -> tuple[MultibodyPlant, SceneGraph]:
    """
    Create a MultibodyPlant and SceneGraph with the given scene loaded.

    Args:
        scene (RoomScene): The scene to load into the plant.
        builder (DiagramBuilder | None): Optional diagram builder to use. If None,
            creates a new one. When provided, the plant and scene_graph will be
            created in this builder's context.
        include_objects (list | None): Optional list of UniqueID objects to include.
            If provided, only these objects will be rendered. Useful for focused
            rendering (e.g., manipuland agent viewing only current furniture).
        exclude_room_geometry (bool): If True, completely exclude the floor plan from
            the scene. Useful for focused rendering of furniture + manipulands only.
        weld_furniture (bool): If True (default), weld furniture to world frame.
            If False, add furniture as free bodies. Use False for collision checking
            to enable broadphase query (which filters welded body collisions).
        free_mounted_objects_for_collision (bool): If True, wall-mounted and
            ceiling-mounted objects are treated as free bodies instead of welded.
            Used for collision checking where Drake's broadphase needs free bodies
            to detect collisions between them.

    Returns:
        tuple[MultibodyPlant, SceneGraph]: The configured plant and scene graph.
    """
    start_time = time.time()

    if builder is None:
        builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    # Load the scene using Drake directives.
    directive_yaml = scene.to_drake_directive(
        include_objects=include_objects,
        exclude_room_geometry=exclude_room_geometry,
        weld_furniture=weld_furniture,
        free_mounted_objects_for_collision=free_mounted_objects_for_collision,
    )

    # Write directive to temporary file and load it.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(directive_yaml)
        temp_directive_path = f.name

    try:
        directives = LoadModelDirectives(temp_directive_path)
        ProcessModelDirectives(directives, plant, parser=None)
    finally:
        # Clean up temporary file.
        Path(temp_directive_path).unlink(missing_ok=True)

    # Finalize the plant.
    plant.Finalize()

    end_time = time.time()
    console_logger.info(
        f"Created Drake plant and scene graph in {end_time - start_time:.2f} seconds."
    )

    return plant, scene_graph


def create_plant_from_dmd(
    directive_path: Path,
    scene_dir: Path | None = None,
) -> tuple[DiagramBuilder, MultibodyPlant, SceneGraph]:
    """Create Drake plant from a model directive file.

    Args:
        directive_path: Path to the Drake model directive YAML file.
        scene_dir: Optional scene root directory for package:// URI resolution.
            If not provided, searches parent directories for package.xml.

    Returns:
        Tuple of (builder, plant, scene_graph). The plant is finalized but
        the diagram is NOT built, allowing the caller to add camera configs.

    Raises:
        FileNotFoundError: If directive_path does not exist.
    """
    if not directive_path.exists():
        raise FileNotFoundError(f"Directive file not found: {directive_path}")

    # Create Drake diagram components.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    parser = Parser(plant)
    # Enable auto-renaming to handle duplicate model names in SDF files.
    parser.SetAutoRenaming(True)

    # Register "scene" package for package://scene/ URI resolution.
    # Find scene root by looking for package.xml in parent directories.
    if scene_dir is None:
        for parent in directive_path.parents:
            if (parent / "package.xml").exists():
                scene_dir = parent
                break

    if scene_dir is not None:
        parser.package_map().Add("scene", str(scene_dir))

    # Load and process directives using Drake's native machinery.
    directives = LoadModelDirectives(directive_path)
    ProcessModelDirectives(directives, parser)

    plant.Finalize()

    return builder, plant, scene_graph


def set_articulated_joints_to_max(plant: MultibodyPlant, context: Context) -> None:
    """Set all articulated joints to their upper limits (open state).

    For doors, drawers, and other articulated parts, this opens them to their
    maximum extent. This is useful for rendering furniture with internal surfaces
    visible for manipuland placement.

    NOTE: This works well for ArtVIP assets but might be less meaningful for
    PartNet-Mobility assets.

    Args:
        plant: Finalized MultibodyPlant containing articulated objects.
        context: Plant context to modify joint positions in.
    """
    positions = plant.GetPositions(context).copy()
    joints_modified = 0

    for joint_idx in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(joint_idx))

        # Skip fixed joints (welded).
        if joint.num_positions() == 0:
            continue

        # Get joint limits.
        lower = joint.position_lower_limits()
        upper = joint.position_upper_limits()

        if len(upper) == 0:
            continue

        q_start = joint.position_start()
        for i in range(len(upper)):
            lo = lower[i] if i < len(lower) else 0.0
            hi = upper[i]

            # Use whichever limit has larger absolute value.
            if np.isfinite(lo) and np.isfinite(hi):
                limit = hi if abs(hi) >= abs(lo) else lo
            elif np.isfinite(hi):
                limit = hi
            elif np.isfinite(lo):
                limit = lo
            else:
                continue

            positions[q_start + i] = limit
            joints_modified += 1

    plant.SetPositions(context, positions)

    if joints_modified > 0:
        console_logger.info(
            f"Set {joints_modified} joint position(s) to max values (open state)"
        )


def parse_joint_child_links(sdf_path: Path) -> dict[str, str]:
    """Parse SDF to get child link → joint name mapping.

    For articulated objects, each joint connects a parent link to a child link.
    This function builds a mapping from child link names to their controlling
    joint names, used for surface classification and FK transforms.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        Mapping from child link name to joint name. Links not in this map
        are either the base link or don't move (fixed joints).
    """
    tree = ET.parse(sdf_path)
    link_to_joint: dict[str, str] = {}

    for joint in tree.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type", "")

        # Skip fixed joints - they don't move.
        if joint_type == "fixed":
            continue

        child = joint.find("child")
        if child is not None and joint_name:
            link_to_joint[child.text] = joint_name

    console_logger.debug(f"Parsed {len(link_to_joint)} joints from {sdf_path.name}")
    return link_to_joint


def get_open_position(lower: float, upper: float) -> float:
    """Get 'open' position: whichever limit has larger absolute value.

    For doors/drawers, the 'open' position is typically the limit furthest from
    zero. This handles both positive ranges (0 to 1.57) and negative ranges
    (-1.57 to 0).

    Args:
        lower: Lower joint limit.
        upper: Upper joint limit.

    Returns:
        The joint position that represents "open" state.
    """
    if np.isfinite(lower) and np.isfinite(upper):
        return upper if abs(upper) >= abs(lower) else lower
    elif np.isfinite(upper):
        return upper
    elif np.isfinite(lower):
        return lower
    return 0.0


def get_closed_position(lower: float, upper: float) -> float:
    """Get 'closed' position: whichever limit has smaller absolute value.

    For doors/drawers, the 'closed' position is typically the limit closest to
    zero (rest position).

    Args:
        lower: Lower joint limit.
        upper: Upper joint limit.

    Returns:
        The joint position that represents "closed" state.
    """
    if np.isfinite(lower) and np.isfinite(upper):
        return lower if abs(lower) <= abs(upper) else upper
    return 0.0


def set_joints_to_config(
    plant: MultibodyPlant, context: Context, joint_config: dict[str, float]
) -> None:
    """Set specific joints to given positions.

    Unlike `set_articulated_joints_to_max()` which opens ALL joints, this
    function allows fine-grained control over individual joints. Used for
    per-drawer rendering where only specific drawers should be open.

    Args:
        plant: Finalized MultibodyPlant containing articulated objects.
        context: Plant context to modify joint positions in.
        joint_config: Mapping from joint name to desired position.

    Note:
        Joints not in joint_config are left unchanged.
    """
    for joint_name, position in joint_config.items():
        try:
            joint = plant.GetJointByName(joint_name)

            # Skip joints with no positions (fixed joints).
            if joint.num_positions() == 0:
                continue

            # Use the joint's set method for single-DOF joints.
            if joint.num_positions() == 1:
                q_start = joint.position_start()
                positions = plant.GetPositions(context).copy()
                positions[q_start] = position
                plant.SetPositions(context, positions)
            else:
                console_logger.warning(
                    f"Joint {joint_name} has {joint.num_positions()} DOFs, "
                    f"only single-DOF joints supported by set_joints_to_config"
                )
        except RuntimeError as e:
            console_logger.warning(f"Failed to set joint {joint_name}: {e}")


def get_all_link_transforms(
    plant: MultibodyPlant, context: Context
) -> dict[str, RigidTransform]:
    """Get world-frame transforms for all links in the plant.

    Used to compute FK deltas between rest and open joint positions.
    By querying transforms before and after joint modification, we can
    compute the delta transform to apply to support surfaces.

    Args:
        plant: Finalized MultibodyPlant.
        context: Plant context with current joint positions.

    Returns:
        Mapping from link (body) name to its world-frame transform.
    """
    transforms: dict[str, RigidTransform] = {}
    for model_idx in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(model_idx)
        for body_index in plant.GetBodyIndices(model_instance):
            body = plant.get_body(body_index)

            # Skip the world body.
            if body.name() == "world":
                continue

            transforms[body.name()] = body.EvalPoseInWorld(context)

    return transforms


def get_joint_limits(plant: MultibodyPlant) -> dict[str, tuple[float, float]]:
    """Get position limits for all joints in the plant.

    Args:
        plant: Finalized MultibodyPlant.

    Returns:
        Mapping from joint name to (lower_limit, upper_limit) tuple.
        Only includes joints with exactly one position DOF.
    """
    joint_limits: dict[str, tuple[float, float]] = {}

    for joint_idx in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(joint_idx))

        # Skip fixed joints or multi-DOF joints.
        if joint.num_positions() != 1:
            continue

        lower = joint.position_lower_limits()
        upper = joint.position_upper_limits()

        if len(lower) > 0 and len(upper) > 0:
            joint_limits[joint.name()] = (float(lower[0]), float(upper[0]))

    return joint_limits
