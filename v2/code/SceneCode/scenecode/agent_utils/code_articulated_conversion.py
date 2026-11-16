"""Helpers for converting generated articulated URDF assets into packaged SDFs."""

from __future__ import annotations

import json
import logging
import shutil

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from omegaconf import DictConfig
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from scipy.spatial.transform import Rotation

from scenecode.agent_utils.articulated_physics_analyzer import (
    PlacementOptions,
    analyze_articulated_physics,
)
from scenecode.agent_utils.convex_decomposition_server import ConvexDecompositionClient
from scenecode.agent_utils.urdf_to_sdf import (
    compute_articulated_bounding_box,
    compute_link_physics_from_meshes,
    compute_sdf_bounding_box,
    convert_urdf_to_sdf,
    extract_link_meshes,
    parse_urdf,
    update_sdf_model_pose,
    validate_urdf_meshes,
)
from scenecode.agent_utils.vlm_service import VLMService

if TYPE_CHECKING:
    from scenecode.agent_utils.blender import BlenderServer

console_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeArticulatedConversionResult:
    """Packaged articulated conversion result for runtime scene integration."""

    sdf_path: Path
    output_dir: Path
    analysis_path: Path
    bounding_box_min: list[float]
    bounding_box_max: list[float]
    model_pose: tuple[float, float, float, float, float, float]
    scale_factor: float
    placement_type: str = "floor"


def validate_with_drake(model_path: Path) -> bool:
    """Validate that Drake can parse the generated model."""
    try:
        plant = MultibodyPlant(time_step=0.0)
        parser = Parser(plant)
        parser.AddModels(str(model_path))
        plant.Finalize()
        return True
    except Exception as exc:
        console_logger.error("Drake validation failed for %s: %s", model_path, exc)
        return False


def compute_front_rotation_matrix(front_axis: str) -> np.ndarray:
    """Compute rotation matrix to rotate the detected front axis to +Y."""
    rotations = {
        "+Y": 0.0,
        "-Y": np.pi,
        "+X": -np.pi / 2,
        "-X": np.pi / 2,
    }
    angle = rotations.get(front_axis, 0.0)
    return Rotation.from_euler("z", angle).as_matrix()


def get_placement_type_from_options(placement_options: PlacementOptions) -> str:
    """Convert VLM placement options to a canonical placement type string."""
    if placement_options.on_floor:
        return "floor"
    if placement_options.on_wall:
        return "wall"
    if placement_options.on_ceiling:
        return "ceiling"
    if placement_options.on_object:
        return "on_object"
    return "floor"


def compute_canonical_pose(
    placement_type: str,
    front_axis: str,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    center: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Compute canonical pose using the VLM-predicted placement type."""
    del center  # Center is derivable from the rotated bounds below.
    front_rotation = compute_front_rotation_matrix(front_axis)
    yaw = Rotation.from_matrix(front_rotation).as_euler("xyz")[2]

    corners = np.array(
        [
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ]
    )
    rotated_corners = corners @ front_rotation.T
    rot_min = rotated_corners.min(axis=0)
    rot_max = rotated_corners.max(axis=0)
    rot_center = (rot_min + rot_max) / 2

    if placement_type == "wall":
        tx = -rot_center[0]
        ty = -rot_min[1]
        tz = -rot_center[2]
    elif placement_type == "ceiling":
        tx = -rot_center[0]
        ty = -rot_center[1]
        tz = -rot_max[2]
    else:
        tx = -rot_center[0]
        ty = -rot_center[1]
        tz = -rot_min[2]

    return tx, ty, tz, 0.0, 0.0, yaw


def compute_agent_scale_factor(
    current_dimensions: np.ndarray,
    desired_dimensions: list[float] | tuple[float, float, float] | None,
) -> float:
    """Compute a uniform scale factor from agent dimensions."""
    if desired_dimensions is None:
        return 1.0

    desired = np.asarray(desired_dimensions, dtype=float)
    if desired.shape != (3,):
        raise ValueError(
            f"desired_dimensions must have shape (3,), got {desired_dimensions}"
        )

    valid_mask = (current_dimensions > 0) & (desired > 0)
    if not np.any(valid_mask):
        return 1.0

    return float(np.median(desired[valid_mask] / current_dimensions[valid_mask]))


def convert_generated_articulated_urdf(
    *,
    urdf_path: Path,
    collision_client: ConvexDecompositionClient,
    vlm_service: VLMService,
    cfg: DictConfig,
    blender_server: "BlenderServer",
    desired_dimensions: list[float] | tuple[float, float, float] | None,
    output_path: Path | None = None,
    debug_output_dir: Path | None = None,
    model_name: str | None = None,
    collision_threshold: float = 0.05,
) -> CodeArticulatedConversionResult:
    """Convert a generated articulated URDF into a packaged SDF using VLM placement semantics."""
    urdf_path = urdf_path.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    if output_path is None:
        output_path = urdf_path.with_suffix('.sdf')
    output_path = output_path.resolve()
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if debug_output_dir is None:
        debug_output_dir = output_dir / 'vlm_images'
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    packaged_urdf_path = output_dir / urdf_path.name
    if packaged_urdf_path.resolve() != urdf_path:
        shutil.copy2(urdf_path, packaged_urdf_path)

    urdf_result = parse_urdf(urdf_path)
    valid_meshes, missing_meshes = validate_urdf_meshes(
        urdf_path=urdf_path,
        urdf_result=urdf_result,
    )
    if not valid_meshes:
        raise ValueError(f"No valid meshes found in {urdf_path}")
    if missing_meshes:
        console_logger.warning(
            "URDF references %d missing meshes; converter will repair them: %s",
            len(missing_meshes),
            missing_meshes[:5],
        )

    bbox_min_raw, bbox_max_raw, _ = compute_articulated_bounding_box(urdf_path)
    raw_dimensions = bbox_max_raw - bbox_min_raw
    scale_factor = compute_agent_scale_factor(raw_dimensions, desired_dimensions)

    link_meshes = extract_link_meshes(urdf_path)
    link_names_with_geometry = [link_mesh.link_name for link_mesh in link_meshes]
    analysis = analyze_articulated_physics(
        urdf_path=urdf_path,
        link_names=link_names_with_geometry,
        bounding_box={
            'min': bbox_min_raw.tolist(),
            'max': bbox_max_raw.tolist(),
        },
        vlm_service=vlm_service,
        cfg=cfg,
        blender_server=blender_server,
        category=model_name or urdf_path.stem,
        debug_output_dir=debug_output_dir,
    )

    link_physics = compute_link_physics_from_meshes(
        urdf_path=urdf_path,
        link_masses=analysis.link_masses,
    )

    sdf_path = convert_urdf_to_sdf(
        urdf_path=urdf_path,
        output_path=output_path,
        link_physics=link_physics,
        model_name=model_name or urdf_path.stem,
        repair_missing_meshes=True,
        model_pose=None,
        generate_collision=True,
        collision_client=collision_client,
        collision_threshold=collision_threshold,
        merge_visuals=True,
        scale_factor=scale_factor,
    )

    bbox_min, bbox_max, center = compute_sdf_bounding_box(
        sdf_path=sdf_path,
        scale_factor=scale_factor,
    )
    placement_type = get_placement_type_from_options(analysis.placement_options)
    model_pose = compute_canonical_pose(
        placement_type=placement_type,
        front_axis=analysis.front_axis,
        min_xyz=bbox_min,
        max_xyz=bbox_max,
        center=center,
    )
    update_sdf_model_pose(sdf_path, model_pose)

    if not validate_with_drake(sdf_path):
        raise RuntimeError(f"Generated SDF failed Drake validation: {sdf_path}")

    merged_visuals = list((output_dir / 'visual').glob('*_visual.gltf'))
    analysis_path = output_dir / 'analysis.json'
    analysis_dict = {
        'front_axis': analysis.front_axis,
        'placement_options': asdict(analysis.placement_options),
        'raw_vlm_placement_options': asdict(analysis.placement_options),
        'scale_correct': True,
        'scale_factor': scale_factor,
        'scale_source': 'agent_dimensions',
        'desired_dimensions': list(desired_dimensions) if desired_dimensions else None,
        'link_materials': analysis.link_materials,
        'link_masses': analysis.link_masses,
        'total_mass_kg': analysis.total_mass_kg,
        'category': model_name or urdf_path.stem,
        'model_id': urdf_path.stem,
        'bounding_box': {
            'min': bbox_min.tolist(),
            'max': bbox_max.tolist(),
            'center': center.tolist(),
        },
        'placement_type': placement_type,
        'model_pose': list(model_pose),
        'format': 'sdf',
        'asset_source': 'code_articulated',
        'urdf_path': str(urdf_path),
        'sdf_path': str(sdf_path),
        'visual_merge_succeeded': bool(merged_visuals),
        'visual_merge_outputs': [str(path) for path in merged_visuals],
        'collision_dir': str(output_dir / 'collision'),
    }
    if analysis.link_descriptions:
        analysis_dict['link_descriptions'] = analysis.link_descriptions
    if analysis.front_view_image_index is not None:
        analysis_dict['front_view_image_index'] = analysis.front_view_image_index
    if analysis.object_description:
        analysis_dict['object_description'] = analysis.object_description
    analysis_path.write_text(json.dumps(analysis_dict, indent=2), encoding='utf-8')

    return CodeArticulatedConversionResult(
        sdf_path=sdf_path,
        output_dir=output_dir,
        analysis_path=analysis_path,
        bounding_box_min=bbox_min.tolist(),
        bounding_box_max=bbox_max.tolist(),
        model_pose=model_pose,
        scale_factor=scale_factor,
        placement_type=placement_type,
    )
