"""Collision detection utilities using Drake signed distance queries.

Provides pairwise collision detection for objects with SDF files using
Drake's ComputeSignedDistancePairwiseClosestPoints() for accurate mesh-based
collision detection.
"""

import logging
import tempfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    QueryObject,
    RigidTransform,
)

from scenecode.agent_utils.room import SceneObject
from scenecode.utils.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)


@dataclass
class CollisionInfo:
    """Information about a collision between two objects.

    Attributes:
        obj_a_idx: Index of first object in input list.
        obj_b_idx: Index of second object in input list.
        penetration_m: Penetration depth in meters (positive = overlap).
    """

    obj_a_idx: int
    obj_b_idx: int
    penetration_m: float


def compute_pairwise_collisions(
    objects: list[SceneObject], transforms: list[RigidTransform]
) -> list[CollisionInfo]:
    """Compute collisions between objects using Drake SDF queries.

    Uses ComputeSignedDistancePairwiseClosestPoints() for accurate collision
    detection based on actual mesh geometry (not OBBs or spheres).

    Args:
        objects: List of SceneObjects with SDF paths.
        transforms: World transforms for each object (same length as objects).

    Returns:
        List of CollisionInfo for each colliding pair. Empty if no collisions.
        Sorted by penetration depth (largest first).

    Raises:
        ValueError: If objects and transforms have different lengths.
        RuntimeError: If Drake setup or query fails.
    """
    if len(objects) != len(transforms):
        raise ValueError(
            f"Objects ({len(objects)}) and transforms ({len(transforms)}) "
            f"must have same length"
        )

    if len(objects) < 2:
        return []

    # Filter objects with valid SDF paths.
    valid_indices: list[int] = []
    valid_objects: list[SceneObject] = []
    valid_transforms: list[RigidTransform] = []
    for i, (obj, transform) in enumerate(zip(objects, transforms)):
        if obj.sdf_path and obj.sdf_path.exists():
            valid_indices.append(i)
            valid_objects.append(obj)
            valid_transforms.append(transform)
        else:
            console_logger.debug(f"Skipping object {i} ({obj.name}): no valid SDF path")

    if len(valid_objects) < 2:
        return []

    try:
        # Build Drake scene with objects at specified transforms.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder=builder,
            time_step=0.0,  # Continuous - no dynamics needed.
        )

        # Build directive YAML to load all objects.
        directive_parts = ["directives:"]
        model_name_to_original_idx: dict[str, int] = {}

        for i, (obj, transform) in enumerate(zip(valid_objects, valid_transforms)):
            model_name = f"obj_{i}"
            model_name_to_original_idx[model_name] = valid_indices[i]

            # Get base link name from SDF.
            try:
                base_link = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError:
                base_link = "base_link"

            # Convert transform to angle-axis for Drake directive.
            translation = transform.translation()
            angle_axis = transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      {base_link}:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        directive_yaml = "\n".join(directive_parts)

        # Write directive to temp file and load.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as directive_file:
            directive_file.write(directive_yaml)
            directive_path = Path(directive_file.name)

        try:
            directives = LoadModelDirectives(str(directive_path))
            ProcessModelDirectives(directives, plant, parser=None)
            plant.Finalize()

            diagram = builder.Build()
            context = diagram.CreateDefaultContext()

            # Query signed distances.
            scene_graph_context = scene_graph.GetMyContextFromRoot(context)
            query_object: QueryObject = scene_graph.get_query_output_port().Eval(
                scene_graph_context
            )

            # Get all penetrating pairs (distance < 0).
            all_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
                max_distance=0.0  # Only penetrating pairs.
            )

            # Build geometry ID to model name mapping.
            inspector = query_object.inspector()
            geometry_id_to_model: dict[int, str] = {}
            for model_name in model_name_to_original_idx:
                model_instance = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_instance)
                for body_idx in body_indices:
                    body = plant.get_body(body_idx)
                    frame_id = plant.GetBodyFrameIdOrThrow(body.index())
                    for geom_id in inspector.GetGeometries(frame_id):
                        geometry_id_to_model[int(geom_id.get_value())] = model_name

            # Convert pairs to CollisionInfo.
            collisions: list[CollisionInfo] = []
            seen_pairs: set[tuple[int, int]] = set()

            for pair in all_pairs:
                if pair.distance >= 0:
                    continue

                # Map geometry IDs to model names.
                id_a = int(pair.id_A.get_value())
                id_b = int(pair.id_B.get_value())

                model_a = geometry_id_to_model.get(id_a)
                model_b = geometry_id_to_model.get(id_b)

                if model_a is None or model_b is None:
                    continue
                if model_a == model_b:
                    # Skip self-collisions.
                    continue

                # Map to original indices.
                orig_idx_a = model_name_to_original_idx[model_a]
                orig_idx_b = model_name_to_original_idx[model_b]

                # Deduplicate (same pair may have multiple geometry collisions).
                pair_key = (min(orig_idx_a, orig_idx_b), max(orig_idx_a, orig_idx_b))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                collisions.append(
                    CollisionInfo(
                        obj_a_idx=orig_idx_a,
                        obj_b_idx=orig_idx_b,
                        penetration_m=abs(pair.distance),
                    )
                )

            # Sort by penetration (largest first).
            collisions.sort(key=lambda c: c.penetration_m, reverse=True)

            if collisions:
                console_logger.debug(
                    f"Found {len(collisions)} collisions, max penetration: "
                    f"{collisions[0].penetration_m * 1000:.1f}mm"
                )

            return collisions

        finally:
            directive_path.unlink(missing_ok=True)

    except Exception as e:
        console_logger.error(f"Collision detection failed: {e}")
        raise RuntimeError(f"Collision detection failed: {e}") from e
