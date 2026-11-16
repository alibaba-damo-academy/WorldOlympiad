#!/usr/bin/env python3
"""Convert PartNet-Mobility dataset to simulation-ready SDF format.

This script processes articulated URDF assets from PartNet-Mobility into
Drake-compatible SDF files with:
- Physics properties (mass, inertia from VLM-estimated masses)
- Canonicalized orientation (front-facing +Y, z-up)
- Corrected scale (VLM-validated dimensions)
- Placement type metadata (floor/wall/ceiling)

Usage:
    # Process single asset or small batch
    python scripts/convert_partnet_mobility.py \\
        --input /path/to/partnet-mobility-v0 \\
        --output /path/to/processed_output
"""

import argparse
import json
import logging
import shutil
import traceback
import sys

from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_API_BASE = "https://api.openai.com/v1"

import numpy as np

from omegaconf import OmegaConf
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from scenecode.agent_utils.articulated_physics_analyzer import (
    PlacementOptions,
    analyze_articulated_physics,
)
from scenecode.agent_utils.convex_decomposition_server import (
    ConvexDecompositionClient,
    ConvexDecompositionServer,
)
from scenecode.agent_utils.materials import get_friction
from scenecode.agent_utils.mesh_utils import convert_objs_to_gltf
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

console_logger = logging.getLogger(__name__)


def get_placement_type(opts: PlacementOptions) -> str:
    """Get canonical placement type from placement options.

    Floor takes precedence if model can be placed on floor or wall/ceiling.
    """
    if opts.on_floor:
        return "floor"
    elif opts.on_wall:
        return "wall"
    elif opts.on_ceiling:
        return "ceiling"
    return "floor"  # Default to floor.


@dataclass
class AnalysisResult:
    """Result of VLM analysis for an articulated asset."""

    # Canonicalization.
    front_axis: str  # "+X", "-X", "+Y", "-Y" (Z is always up).

    # Placement.
    placement_options: PlacementOptions

    # Scale.
    scale_correct: bool
    corrected_dimensions_m: tuple[float, float, float] | None  # (W, D, H)
    scale_factor: float  # Applied scale factor (1.0 if unchanged).

    # Physics per link.
    link_materials: dict[str, str]  # link_name -> material
    link_masses: dict[str, float]  # link_name -> mass_kg

    # Metadata.
    category: str
    model_id: str

    # Total mass (sum of link masses).
    total_mass_kg: float = 0.0

    # Link descriptions (from VLM analysis).
    link_descriptions: dict[str, str] | None = None

    # VLM images directory (if saved).
    vlm_images_dir: Path | None = None

    # Front view image index (for debugging).
    front_view_image_index: int | None = None

    # Whether this object is a manipuland (can be picked up/manipulated).
    is_manipuland: bool = False

    # Overall description of the object (from VLM analysis).
    object_description: str | None = None


def load_asset_metadata(asset_dir: Path) -> dict:
    """Load asset metadata from meta.json.

    Args:
        asset_dir: Path to asset directory.

    Returns:
        Metadata dict with at least 'model_cat' and 'model_id' keys.
    """
    meta_path = asset_dir / "meta.json"
    if not meta_path.exists():
        return {"model_cat": "Unknown", "model_id": asset_dir.name}

    with open(meta_path) as f:
        return json.load(f)


def compute_front_rotation_matrix(front_axis: str) -> np.ndarray:
    """Compute rotation matrix to rotate front axis to +Y.

    Args:
        front_axis: Current front axis ("+X", "-X", "+Y", "-Y").

    Returns:
        3x3 rotation matrix.
    """
    # Rotation around Z axis to align front to +Y.
    rotations = {
        "+Y": 0.0,  # Already facing +Y.
        "-Y": np.pi,  # 180 degrees.
        "+X": -np.pi / 2,  # 90 degrees CCW.
        "-X": np.pi / 2,  # 90 degrees CW.
    }

    angle = rotations.get(front_axis, 0.0)
    return Rotation.from_euler("z", angle).as_matrix()


def compute_canonical_pose(
    placement_type: str,
    front_axis: str,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    center: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Compute canonical pose (x, y, z, roll, pitch, yaw) based on placement type.

    Different placement types require different canonicalization:
    - Floor/on_object: Bottom at Z=0, center XY, front faces +Y
    - Wall: Back at Y=0 (against wall), center X, center Z
    - Ceiling: Top at Z=0, center XY (no flip - objects already modeled correctly)

    Args:
        placement_type: One of "floor", "wall", "ceiling", or default "floor".
        front_axis: Front axis for rotation ("+X", "-X", "+Y", "-Y").
        min_xyz: Minimum bounding box coordinates.
        max_xyz: Maximum bounding box coordinates.
        center: Center of bounding box.

    Returns:
        Pose tuple (tx, ty, tz, roll, pitch, yaw).
    """
    # Rotation around Z to align front to +Y.
    front_rotation = compute_front_rotation_matrix(front_axis)
    yaw = Rotation.from_matrix(front_rotation).as_euler("xyz")[2]

    # SDF pose semantics: p_world = R @ p_model + t
    # Since rotation is applied first, we must compute translation based on the
    # ROTATED bounding box, not the original. Otherwise centering will be wrong.
    # Build all 8 corners of the bounding box.
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
    # Rotate corners by front_rotation (yaw around Z).
    rotated_corners = corners @ front_rotation.T
    # Compute rotated bounding box.
    rot_min = rotated_corners.min(axis=0)
    rot_max = rotated_corners.max(axis=0)
    rot_center = (rot_min + rot_max) / 2

    if placement_type == "wall":
        # Wall objects: back at Y=0 (against wall), center X, center Z.
        # After rotation, object faces +Y, so back is at rotated min Y.
        tx = -rot_center[0]
        ty = -rot_min[1]  # Put back at Y=0.
        tz = -rot_center[2]  # Center vertically.
    elif placement_type == "ceiling":
        # Ceiling objects: top at Z=0, center XY.
        # No flip needed - ceiling objects (lights, fans) are modeled with
        # their active side facing down already.
        tx = -rot_center[0]
        ty = -rot_center[1]
        tz = -rot_max[2]  # Put top at Z=0.
    else:
        # Floor and on_object: bottom at Z=0, center XY.
        tx = -rot_center[0]
        ty = -rot_center[1]
        tz = -rot_min[2]  # Put bottom at Z=0.

    # No roll/pitch needed - only yaw for front axis alignment.
    roll, pitch = 0.0, 0.0

    return tx, ty, tz, roll, pitch, yaw


def analyze_asset_with_vlm(
    asset_dir: Path,
    metadata: dict,
    vlm_images_output_dir: Path | None = None,
    api_base: str | None = None,
) -> AnalysisResult:
    """Analyze asset using VLM for physics and canonicalization.

    Args:
        asset_dir: Path to asset directory.
        metadata: Asset metadata.
        vlm_images_output_dir: Directory to save VLM analysis images (if provided).
        api_base: OpenAI-compatible API base URL for VLM requests.

    Returns:
        AnalysisResult with canonicalization and physics info.
    """
    category = metadata.get("model_cat", "Unknown")
    model_id = metadata.get("model_id", asset_dir.name)

    # Parse URDF to get link names.
    urdf_path = asset_dir / "mobility.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Load VLM configuration.
    config_path = (
        Path(__file__).parent.parent
        / "configs/furniture_agent/base_furniture_agent.yaml"
    )
    cfg = OmegaConf.load(config_path)

    # Create VLM service.
    vlm_service = VLMService(api_base=api_base)

    # Compute bounding box for VLM analysis.
    bbox_min, bbox_max, _ = compute_articulated_bounding_box(urdf_path)

    # Get link names with visual geometry (excludes empty links like "base").
    link_meshes = extract_link_meshes(urdf_path)
    link_names_with_geometry = [lm.link_name for lm in link_meshes]

    # Analyze with VLM using per-link rendering.
    vlm_analysis = analyze_articulated_physics(
        urdf_path=urdf_path,
        link_names=link_names_with_geometry,
        bounding_box={"min": bbox_min.tolist(), "max": bbox_max.tolist()},
        vlm_service=vlm_service,
        cfg=cfg,
        category=category,
        debug_output_dir=vlm_images_output_dir,
    )

    # Build result from VLM analysis.
    placement = PlacementOptions(
        on_floor=vlm_analysis.placement_options.on_floor,
        on_wall=vlm_analysis.placement_options.on_wall,
        on_ceiling=vlm_analysis.placement_options.on_ceiling,
        on_object=vlm_analysis.placement_options.on_object,
    )

    return AnalysisResult(
        front_axis=vlm_analysis.front_axis,
        placement_options=placement,
        scale_correct=vlm_analysis.scale_correct,
        corrected_dimensions_m=None,
        scale_factor=vlm_analysis.scale_factor,
        link_materials=vlm_analysis.link_materials,
        link_masses=vlm_analysis.link_masses,
        category=category,
        model_id=model_id,
        total_mass_kg=vlm_analysis.total_mass_kg,
        link_descriptions=vlm_analysis.link_descriptions,
        vlm_images_dir=vlm_analysis.vlm_images_dir,
        front_view_image_index=vlm_analysis.front_view_image_index,
        object_description=vlm_analysis.object_description,
    )


def validate_with_drake(model_path: Path) -> bool:
    """Validate URDF or SDF file can be parsed by Drake.

    Args:
        model_path: Path to URDF or SDF file.

    Returns:
        True if validation passes.
    """
    try:
        plant = MultibodyPlant(time_step=0.0)
        parser = Parser(plant)
        parser.AddModels(str(model_path))
        plant.Finalize()
        return True
    except Exception as e:
        console_logger.error(f"Drake validation failed for {model_path}: {e}")
        return False


def process_single_asset(
    asset_dir: Path,
    output_dir: Path,
    collision_client: ConvexDecompositionClient,
    api_base: str | None = None,
) -> Path | None:
    """Process a single PartNet-Mobility asset.

    Args:
        asset_dir: Path to asset directory.
        output_dir: Path to output directory.
        collision_client: Convex decomposition client for collision geometry generation.
        api_base: OpenAI-compatible API base URL for VLM requests.

    Returns:
        Path to generated SDF, or None if processing failed.
    """
    try:
        # Load metadata.
        metadata = load_asset_metadata(asset_dir)
        model_id = asset_dir.name
        category = metadata.get("model_cat", "Unknown")

        console_logger.info(f"Processing {model_id} ({category})")

        # Check URDF exists.
        urdf_path = asset_dir / "mobility.urdf"
        if not urdf_path.exists():
            console_logger.warning(f"No URDF found in {asset_dir}")
            return None

        # Validate URDF has valid meshes.
        urdf_result = parse_urdf(urdf_path)
        valid_meshes, _ = validate_urdf_meshes(
            urdf_path=urdf_path, urdf_result=urdf_result
        )
        if not valid_meshes:
            console_logger.warning(f"No valid meshes in {model_id}")
            return None

        # Create output directory for this asset.
        asset_output_dir = output_dir / model_id
        asset_output_dir.mkdir(parents=True, exist_ok=True)

        # Create VLM images directory.
        vlm_images_dir = asset_output_dir / "vlm_images"
        vlm_images_dir.mkdir(parents=True, exist_ok=True)

        # Analyze asset with VLM.
        analysis = analyze_asset_with_vlm(
            asset_dir=asset_dir,
            metadata=metadata,
            vlm_images_output_dir=vlm_images_dir,
            api_base=api_base,
        )

        # Compute link physics from meshes.
        link_physics = compute_link_physics_from_meshes(
            urdf_path=urdf_path, link_masses=analysis.link_masses
        )

        # Compute link friction from materials.
        link_friction = {
            link_name: get_friction(material)
            for link_name, material in analysis.link_materials.items()
        }

        # Two-pass conversion to handle GLTF coordinate frame changes:
        # Pass 1: Convert URDF to SDF WITHOUT model pose (exports GLTFs).
        sdf_path = convert_urdf_to_sdf(
            urdf_path=urdf_path,
            output_path=asset_output_dir / "mobility.sdf",
            link_physics=link_physics,
            link_friction=link_friction,
            model_name=f"partnet_{model_id}",
            repair_missing_meshes=True,
            model_pose=None,  # No pose yet - will add after computing from GLTFs.
            generate_collision=True,
            collision_client=collision_client,
            collision_threshold=0.05,
            merge_visuals=True,
            scale_factor=analysis.scale_factor,
        )

        # Pass 2: Compute bounding box from output GLTFs (accounts for Y-up export).
        # This gives us the correct coordinate frame for the rendered geometry.
        min_xyz, max_xyz, center = compute_sdf_bounding_box(
            sdf_path=sdf_path,
            scale_factor=analysis.scale_factor,
        )

        # Compute model-level pose for canonicalization.
        placement_type = get_placement_type(analysis.placement_options)
        model_pose = compute_canonical_pose(
            placement_type=placement_type,
            front_axis=analysis.front_axis,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            center=center,
        )

        # Update SDF with the correct model pose.
        # Note: model_pose is already in scaled coordinates (from GLTF bounding box).
        update_sdf_model_pose(sdf_path, model_pose)

        # Validate with Drake.
        if not validate_with_drake(sdf_path):
            console_logger.error(f"Drake validation failed for {model_id}")
            return None

        # Save analysis results.
        # Bounding box and model_pose are already in scaled coordinates:
        # - compute_sdf_bounding_box scales vertices and uses SDF link poses (already scaled)
        # - compute_canonical_pose derives model_pose from scaled bounding box
        analysis_path = asset_output_dir / "analysis.json"
        analysis_dict = {
            "front_axis": analysis.front_axis,
            "placement_options": asdict(analysis.placement_options),
            "scale_correct": analysis.scale_correct,
            "scale_factor": analysis.scale_factor,
            "link_materials": analysis.link_materials,
            "link_masses": analysis.link_masses,
            "total_mass_kg": analysis.total_mass_kg,
            "is_manipuland": analysis.is_manipuland,
            "category": analysis.category,
            "model_id": analysis.model_id,
            "bounding_box": {
                "min": min_xyz.tolist(),
                "max": max_xyz.tolist(),
                "center": center.tolist(),
            },
            "placement_type": placement_type,
            "model_pose": list(model_pose),
            "format": "sdf",
        }
        # Add optional fields if present.
        if analysis.link_descriptions:
            analysis_dict["link_descriptions"] = analysis.link_descriptions
        if analysis.front_view_image_index is not None:
            analysis_dict["front_view_image_index"] = analysis.front_view_image_index
        if analysis.object_description:
            analysis_dict["object_description"] = analysis.object_description
        with open(analysis_path, "w") as f:
            json.dump(analysis_dict, f, indent=2)

        # Copy original files.
        for orig_file in ["meta.json", "bounding_box.json"]:
            src = asset_dir / orig_file
            dst = asset_output_dir / orig_file
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text())

        # Check if merge succeeded by looking for merged GLTFs.
        # If merge failed, fall back to copying OBJs and converting them.
        visual_dst = asset_output_dir / "visual"
        merged_gltfs = list(visual_dst.glob("*_visual.gltf"))
        if not merged_gltfs:
            console_logger.info("Merge failed, falling back to individual GLTFs")
            # Copy textured_objs directory from source.
            textured_objs_src = asset_dir / "textured_objs"
            if textured_objs_src.exists():
                if visual_dst.exists():
                    shutil.rmtree(visual_dst)
                shutil.copytree(src=textured_objs_src, dst=visual_dst)
            # Convert OBJ files to GLTF.
            if visual_dst.exists():
                convert_objs_to_gltf(visual_dst)

        console_logger.info(f"Successfully processed {model_id}")
        return sdf_path

    except Exception as e:
        console_logger.error(f"Failed to process {asset_dir.name}: {e}")
        console_logger.debug(traceback.format_exc())
        return None


def find_partnet_assets(input_path: Path) -> list[Path]:
    """Find PartNet-Mobility asset directories.

    Args:
        input_path: Path to PartNet-Mobility dataset root.

    Returns:
        List of asset directory paths.
    """
    assets = []
    for child in sorted(input_path.iterdir()):
        if child.is_dir() and (child / "mobility.urdf").exists():
            assets.append(child)
    return assets


def main():
    parser = argparse.ArgumentParser(
        description="Convert PartNet-Mobility dataset to simulation-ready SDF."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to PartNet-Mobility dataset root.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of assets to process (for testing).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip assets that already have output.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only process assets of this category.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible API base URL for VLM requests.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of asset IDs to process.",
    )
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s"
    )
    console_logger.info(f"Using OpenAI-compatible API base: {args.api_base}")

    # Find assets.
    console_logger.info(f"Scanning {args.input} for PartNet-Mobility assets...")
    assets = find_partnet_assets(args.input)
    console_logger.info(f"Found {len(assets)} assets")

    # Filter by specific IDs if specified.
    if args.ids:
        target_ids = set(args.ids.split(","))
        filtered = [a for a in assets if a.name in target_ids]
        assets = filtered
        console_logger.info(f"Filtered to {len(assets)} assets by ID: {args.ids}")

    # Filter by category if specified.
    if args.category:
        filtered = []
        for asset_dir in assets:
            metadata = load_asset_metadata(asset_dir)
            if metadata.get("model_cat") == args.category:
                filtered.append(asset_dir)
        assets = filtered
        console_logger.info(f"Filtered to {len(assets)} {args.category} assets")

    # Skip existing if requested.
    if args.skip_existing:
        filtered = []
        for asset_dir in assets:
            output_sdf = args.output / asset_dir.name / "mobility.sdf"
            if not output_sdf.exists():
                filtered.append(asset_dir)
        skipped = len(assets) - len(filtered)
        assets = filtered
        console_logger.info(
            f"Skipping {skipped} existing assets, {len(assets)} remaining"
        )

    # Apply limit.
    if args.limit:
        assets = assets[: args.limit]
        console_logger.info(f"Limited to {len(assets)} assets")

    if not assets:
        console_logger.info("No assets to process")
        return

    # Create output directory.
    args.output.mkdir(parents=True, exist_ok=True)

    # Start convex decomposition server for collision geometry generation.
    console_logger.info("Starting convex decomposition server")
    collision_server = ConvexDecompositionServer(port_range=(7100, 7150))
    collision_server.start()
    collision_server.wait_until_ready()
    collision_client = collision_server.get_client()

    # Process assets sequentially.
    # For parallel processing, use scripts/convert_partnet_parallel.sh instead.
    success_count = 0
    fail_count = 0
    try:
        for asset_dir in tqdm(assets, desc="Processing"):
            result = process_single_asset(
                asset_dir=asset_dir,
                output_dir=args.output,
                collision_client=collision_client,
                api_base=args.api_base,
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
    finally:
        # Stop convex decomposition server.
        console_logger.info("Stopping convex decomposition server")
        collision_server.stop()

    console_logger.info(f"Done! Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
