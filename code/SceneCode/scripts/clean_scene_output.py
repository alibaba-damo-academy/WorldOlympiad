#!/usr/bin/env python3
"""Clean scene output directories for portability.

Removes non-essential files from scene directories, reducing size significantly.
Preserves combined_house/ and all its dependencies (room geometry, floor plan meshes,
used materials, and used generated assets).

Supports both single scene directories and experiment directories containing multiple
scenes. When given an experiment directory, cleans all scene_* subdirectories.
"""

import argparse
import json
import logging
import os
import re
import shutil

from pathlib import Path
from typing import Callable

from scenecode.utils.parallel import run_parallel_isolated

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def collect_used_sdf_directories(scene_dir: Path) -> set[Path]:
    """Extract SDF directory paths from house.dmd.yaml.

    The house.dmd.yaml file is the authoritative source for assets used in the
    final scene. It uses package://scene/ URIs that map to relative paths.

    Args:
        scene_dir: Path to the scene directory.

    Returns:
        Set of absolute paths to SDF directories used in the final scene.
    """
    house_dmd_path = scene_dir / "combined_house" / "house.dmd.yaml"
    if not house_dmd_path.exists():
        raise FileNotFoundError(f"Missing: {house_dmd_path}")

    with open(house_dmd_path) as f:
        content = f.read()

    # Extract package://scene/ URIs from house.dmd.yaml.
    # Pattern matches: package://scene/room_xxx/generated_assets/.../xxx.sdf
    pattern = (
        r"package://scene/(room_[^/]+/generated_assets/[^/]+/sdf/[^/]+)/[^/]+\.sdf"
    )
    matches = re.findall(pattern, content)

    used_dirs: set[Path] = set()
    for match in matches:
        full_path = scene_dir / match
        used_dirs.add(full_path.resolve())

    return used_dirs


def _extract_materials_from_gltf(gltf_path: Path) -> dict[str, Path | None]:
    """Extract material names and resolve their source paths from GLTF image URIs.

    Args:
        gltf_path: Path to a GLTF file.

    Returns:
        Dict mapping material names to their resolved source directory paths.
        If a material's source path cannot be resolved, it maps to None.
    """
    materials: dict[str, Path | None] = {}
    try:
        with open(gltf_path) as f:
            gltf = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse GLTF {gltf_path}: {e}")
        return materials

    for img in gltf.get("images", []):
        uri = img.get("uri", "")
        # URIs look like: ../../../materials/Tiles050/Tiles050_2K-JPG_Color.jpg
        if "materials/" in uri:
            # Extract material name (directory name after "materials/").
            parts = uri.split("materials/")
            if len(parts) > 1:
                material_name = parts[1].split("/")[0]
                # Resolve the actual path relative to GLTF location.
                resolved_path = (gltf_path.parent / uri).resolve()
                material_dir = (
                    resolved_path.parent
                )  # Material directory (contains textures).
                if material_dir.exists():
                    materials[material_name] = material_dir
                elif material_name not in materials:
                    materials[material_name] = None

    return materials


def collect_referenced_paths_in_asset(asset_dir: Path) -> tuple[set[str], set[str]]:
    """Find all files and directories referenced by SDF and GLTF in an asset directory.

    Parses the SDF to find visual/collision meshes, then parses any GLTF files
    to find their buffer dependencies.

    Args:
        asset_dir: Path to an asset directory containing SDF and geometry files.

    Returns:
        Tuple of (referenced_files, referenced_dirs) where:
        - referenced_files: Set of filenames directly in asset_dir that are referenced.
        - referenced_dirs: Set of subdirectory names that contain referenced files.
    """
    referenced_uris: set[str] = set()

    # Find and parse SDF files.
    for sdf_path in asset_dir.glob("*.sdf"):
        referenced_uris.add(sdf_path.name)
        try:
            with open(sdf_path) as f:
                content = f.read()
            # Extract mesh URIs from SDF.
            for match in re.findall(r"<uri>([^<]+)</uri>", content):
                referenced_uris.add(match)
        except OSError:
            continue

    # Parse any referenced GLTF files to find their buffers and images.
    # Handle both top-level GLTFs and GLTFs in subdirectories.
    gltf_uris = [uri for uri in referenced_uris if uri.endswith(".gltf")]
    for gltf_uri in gltf_uris:
        gltf_path = asset_dir / gltf_uri
        if not gltf_path.exists():
            continue
        try:
            with open(gltf_path) as f:
                gltf = json.load(f)
            # Get the directory containing this GLTF (for relative URI resolution).
            gltf_dir = Path(gltf_uri).parent
            # Extract buffer URIs.
            for buf in gltf.get("buffers", []):
                if uri := buf.get("uri"):
                    # Resolve relative to GLTF location.
                    if gltf_dir != Path("."):
                        referenced_uris.add(str(gltf_dir / uri))
                    else:
                        referenced_uris.add(uri)
            # Extract image URIs (external images only, not embedded).
            for img in gltf.get("images", []):
                if uri := img.get("uri"):
                    if gltf_dir != Path("."):
                        referenced_uris.add(str(gltf_dir / uri))
                    else:
                        referenced_uris.add(uri)
        except (json.JSONDecodeError, OSError):
            continue

    # Separate into files (no path separator) and directories (has path separator).
    referenced_files: set[str] = set()
    referenced_dirs: set[str] = set()
    for uri in referenced_uris:
        if "/" in uri:
            # Extract top-level directory name.
            dir_name = uri.split("/")[0]
            referenced_dirs.add(dir_name)
        else:
            referenced_files.add(uri)

    return referenced_files, referenced_dirs


def collect_used_materials(scene_dir: Path) -> dict[str, Path | None]:
    """Parse floor_plans GLTF files to find referenced materials and their source paths.

    Only floor_plans GLTFs reference materials/ - generated assets have embedded
    textures.

    Args:
        scene_dir: Path to the scene directory.

    Returns:
        Dict mapping material names to their resolved source directory paths.
    """
    used_materials: dict[str, Path | None] = {}

    floor_plans = scene_dir / "floor_plans"
    if not floor_plans.exists():
        return used_materials

    # Scan all GLTF files in floor_plans/[room]/{floors,walls,windows}/.
    for pattern in [
        "*/floors/*.gltf",
        "*/walls/*/*.gltf",
        "*/windows/*/*.gltf",
    ]:
        for gltf_path in floor_plans.glob(pattern):
            materials = _extract_materials_from_gltf(gltf_path)
            for name, path in materials.items():
                # Prefer resolved paths over None.
                if name not in used_materials or (
                    path and not used_materials.get(name)
                ):
                    used_materials[name] = path

    return used_materials


def _rewrite_gltf_texture_uris(gltf_path: Path, scene_root: Path) -> None:
    """Rewrite texture URIs in a GLTF to use scene-local materials directory.

    Args:
        gltf_path: Path to GLTF file to modify.
        scene_root: Root of the scene directory (where materials/ is).
    """
    try:
        with open(gltf_path) as f:
            gltf = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    modified = False
    for img in gltf.get("images", []):
        uri = img.get("uri", "")
        if "materials/" in uri:
            # Extract the path after "materials/".
            parts = uri.split("materials/")
            if len(parts) > 1:
                # e.g., "Wood094_1K-JPG/Wood094_1K-JPG_Color.jpg"
                material_rel_path = parts[1]
                # Compute correct relative path from GLTF to scene_root/materials/.
                rel_to_scene = os.path.relpath(
                    scene_root / "materials", gltf_path.parent
                )
                new_uri = f"{rel_to_scene}/{material_rel_path}"
                img["uri"] = new_uri.replace("\\", "/")  # Ensure forward slashes.
                modified = True

    if modified:
        with open(gltf_path, "w") as f:
            json.dump(gltf, f, indent=2)


def copy_essential_files(
    scene_dir: Path,
    output_dir: Path,
    used_sdf_dirs: set[Path],
    used_materials: dict[str, Path | None],
    verbose: bool = False,
) -> None:
    """Copy only essential files to output directory.

    Args:
        scene_dir: Source scene directory.
        output_dir: Destination directory.
        used_sdf_dirs: Set of SDF directories used in the scene.
        used_materials: Dict mapping material names to their resolved source paths.
        verbose: Whether to print verbose output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy essential top-level directories/files.
    for item in ["combined_house", "room_geometry", "package.xml"]:
        src = scene_dir / item
        if src.exists():
            dst = output_dir / item
            if verbose:
                logger.info(f"Copying {src} -> {dst}")
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # Copy only used materials (from resolved source paths).
    materials_dst = output_dir / "materials"
    materials_dst.mkdir(parents=True, exist_ok=True)
    # Fallback: look for materials in current working directory (project root).
    cwd_materials = Path.cwd() / "materials"
    for mat_name, mat_src in used_materials.items():
        dst = materials_dst / mat_name
        if dst.exists():
            continue  # Already copied.
        if mat_src and mat_src.exists():
            if verbose:
                logger.info(f"Copying material {mat_name} from {mat_src}")
            shutil.copytree(mat_src, dst)
        else:
            # Fallback to scene_dir/materials.
            mat_src_fallback = scene_dir / "materials" / mat_name
            if mat_src_fallback.exists():
                if verbose:
                    logger.info(f"Copying material {mat_name} from scene fallback")
                shutil.copytree(mat_src_fallback, dst)
            else:
                # Fallback to project root materials.
                mat_src_cwd = cwd_materials / mat_name
                if mat_src_cwd.exists():
                    if verbose:
                        logger.info(
                            f"Copying material {mat_name} from project root fallback"
                        )
                    shutil.copytree(mat_src_cwd, dst)

    # Copy floor_plans GLTF directories (floors, walls, windows only).
    floor_plans_src = scene_dir / "floor_plans"
    if floor_plans_src.exists():
        for room_dir in floor_plans_src.iterdir():
            if room_dir.is_dir() and room_dir.name not in [
                "final_floor_plan",
                "floor_plan_renders",
            ]:
                for subdir in ["floors", "walls", "windows"]:
                    src = room_dir / subdir
                    if src.exists():
                        dst = output_dir / "floor_plans" / room_dir.name / subdir
                        if verbose:
                            logger.info(f"Copying {src} -> {dst}")
                        shutil.copytree(src, dst)

    # Rewrite GLTF texture URIs to point to scene-local materials.
    floor_plans_dst = output_dir / "floor_plans"
    if floor_plans_dst.exists():
        for pattern in ["*/floors/*.gltf", "*/walls/*/*.gltf", "*/windows/*/*.gltf"]:
            for gltf_path in floor_plans_dst.glob(pattern):
                _rewrite_gltf_texture_uris(gltf_path, output_dir)

    # Copy only used SDF directories from each room (with orphaned file cleanup).
    for room_dir in scene_dir.glob("room_*"):
        room_name = room_dir.name
        for asset_type in [
            "furniture",
            "manipuland",
            "wall_mounted",
            "ceiling_mounted",
        ]:
            sdf_dir = room_dir / "generated_assets" / asset_type / "sdf"
            if not sdf_dir.exists():
                continue
            for asset_dir in sdf_dir.iterdir():
                if asset_dir.is_dir() and asset_dir.resolve() in used_sdf_dirs:
                    dst_dir = (
                        output_dir
                        / room_name
                        / "generated_assets"
                        / asset_type
                        / "sdf"
                        / asset_dir.name
                    )
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    # Only copy referenced files and directories (skip orphaned).
                    ref_files, ref_dirs = collect_referenced_paths_in_asset(asset_dir)
                    for item in asset_dir.iterdir():
                        if item.is_file() and item.name in ref_files:
                            if verbose:
                                logger.info(f"Copying {item.name}")
                            shutil.copy2(item, dst_dir / item.name)
                        elif item.is_dir() and item.name in ref_dirs:
                            if verbose:
                                logger.info(f"Copying {item.name}/")
                            shutil.copytree(item, dst_dir / item.name)


def clean_scene(
    scene_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Clean a scene directory by removing non-essential files.

    Args:
        scene_dir: Path to the scene directory.
        output_dir: If provided, copy essential files here instead of modifying in place.
        dry_run: If True, only print what would be deleted without actually deleting.
        verbose: If True, print verbose output.

    Raises:
        FileNotFoundError: If scene_dir is not a valid scene directory.
    """
    # Validate scene directory.
    house_dmd_path = scene_dir / "combined_house" / "house.dmd.yaml"
    if not house_dmd_path.exists():
        raise FileNotFoundError(
            f"Not a valid scene directory: {scene_dir}\n"
            f"Missing: combined_house/house.dmd.yaml"
        )

    # Collect dependencies.
    logger.info("Analyzing scene dependencies...")
    used_sdf_dirs = collect_used_sdf_directories(scene_dir)
    used_materials = collect_used_materials(scene_dir)

    if verbose:
        logger.info(f"Found {len(used_sdf_dirs)} used SDF directories")
        logger.info(f"Found {len(used_materials)} used materials")

    # If output_dir specified, copy essential files only.
    if output_dir:
        if dry_run:
            logger.info(f"Would copy essential files to: {output_dir}")
            logger.info(f"  - combined_house/, room_geometry/, package.xml")
            logger.info(f"  - {len(used_materials)} materials")
            logger.info(f"  - floor_plans (floors, walls, windows only)")
            logger.info(f"  - {len(used_sdf_dirs)} used SDF directories")
            return

        logger.info(f"Copying essential files to: {output_dir}")
        copy_essential_files(
            scene_dir=scene_dir,
            output_dir=output_dir,
            used_sdf_dirs=used_sdf_dirs,
            used_materials=used_materials,
            verbose=verbose,
        )
        logger.info("Done!")
        return

    # Otherwise, delete non-essential in place.
    to_delete: list[Path] = []

    # Intermediate checkpoints.
    for checkpoint in scene_dir.glob("combined_house_after_*"):
        to_delete.append(checkpoint)

    # final_floor_plan at scene root (separate from floor_plans/).
    final_floor_plan = scene_dir / "final_floor_plan"
    if final_floor_plan.exists():
        to_delete.append(final_floor_plan)

    # Floor plans intermediate (keep room GLTF dirs).
    floor_plans = scene_dir / "floor_plans"
    if floor_plans.exists():
        for subdir in ["final_floor_plan", "floor_plan_renders"]:
            path = floor_plans / subdir
            if path.exists():
                to_delete.append(path)

    # house_layout.json (not referenced).
    house_layout = scene_dir / "house_layout.json"
    if house_layout.exists():
        to_delete.append(house_layout)

    # SQLite databases (scene root).
    for db in scene_dir.glob("*.db*"):
        to_delete.append(db)

    # Room-level databases (24+ per room: furniture agent, manipuland agent, etc.).
    for db in scene_dir.glob("room_*/*.db"):
        to_delete.append(db)

    # Log files.
    for log in scene_dir.glob("*.log"):
        to_delete.append(log)
    for log in scene_dir.glob("room_*/room.log"):
        to_delete.append(log)

    # Scene renders and states.
    for room_dir in scene_dir.glob("room_*"):
        for subdir in ["scene_renders", "scene_states"]:
            path = room_dir / subdir
            if path.exists():
                to_delete.append(path)

        # Action logs.
        action_log = room_dir / "action_log.json"
        if action_log.exists():
            to_delete.append(action_log)

        # Debug/images/geometry/registry in generated_assets.
        for asset_type in [
            "furniture",
            "manipuland",
            "wall_mounted",
            "ceiling_mounted",
        ]:
            assets_dir = room_dir / "generated_assets" / asset_type
            if not assets_dir.exists():
                continue

            for subdir in ["debug", "images", "geometry"]:
                path = assets_dir / subdir
                if path.exists():
                    to_delete.append(path)

            # Asset registry files.
            registry = assets_dir / "asset_registry.json"
            if registry.exists():
                to_delete.append(registry)

            # Unused SDF directories.
            sdf_dir = assets_dir / "sdf"
            if sdf_dir.exists():
                for asset_dir in sdf_dir.iterdir():
                    if asset_dir.is_dir() and asset_dir.resolve() not in used_sdf_dirs:
                        to_delete.append(asset_dir)
                    elif asset_dir.is_dir() and asset_dir.resolve() in used_sdf_dirs:
                        # Delete orphaned files/dirs within used SDF directories.
                        ref_files, ref_dirs = collect_referenced_paths_in_asset(
                            asset_dir
                        )
                        for item in asset_dir.iterdir():
                            if item.is_file() and item.name not in ref_files:
                                to_delete.append(item)
                            elif item.is_dir() and item.name not in ref_dirs:
                                to_delete.append(item)

    # Unused materials.
    materials_dir = scene_dir / "materials"
    if materials_dir.exists():
        for mat_dir in materials_dir.iterdir():
            if mat_dir.is_dir() and mat_dir.name not in used_materials:
                to_delete.append(mat_dir)

    # Execute deletions.
    if dry_run:
        logger.info("Dry run - would delete:")
        for path in sorted(to_delete):
            rel_path = path.relative_to(scene_dir)
            logger.info(f"  {rel_path}")
        logger.info(f"\nTotal: {len(to_delete)} items")
        # Show what materials would be copied.
        materials_to_copy = [
            name
            for name, src in used_materials.items()
            if src and not (scene_dir / "materials" / name).exists()
        ]
        if materials_to_copy:
            logger.info(
                f"Would copy {len(materials_to_copy)} materials from project root"
            )
    else:
        logger.info(f"Deleting {len(to_delete)} non-essential items...")
        for path in to_delete:
            if verbose:
                rel_path = path.relative_to(scene_dir)
                logger.info(f"Deleting: {rel_path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        # Copy materials from resolved source paths to scene_dir/materials.
        materials_dir = scene_dir / "materials"
        materials_dir.mkdir(parents=True, exist_ok=True)
        # Fallback: look for materials in current working directory (project root).
        cwd_materials = Path.cwd() / "materials"
        for mat_name, mat_src in used_materials.items():
            dst = materials_dir / mat_name
            if dst.exists():
                continue  # Already exists in scene.
            if mat_src and mat_src.exists():
                if verbose:
                    logger.info(f"Copying material {mat_name} from {mat_src}")
                shutil.copytree(mat_src, dst)
            else:
                # Fallback to project root materials.
                mat_src_fallback = cwd_materials / mat_name
                if mat_src_fallback.exists():
                    if verbose:
                        logger.info(
                            f"Copying material {mat_name} from project root fallback"
                        )
                    shutil.copytree(mat_src_fallback, dst)

        # Rewrite GLTF texture URIs to point to scene-local materials.
        floor_plans = scene_dir / "floor_plans"
        if floor_plans.exists():
            for pattern in [
                "*/floors/*.gltf",
                "*/walls/*/*.gltf",
                "*/windows/*/*.gltf",
            ]:
                for gltf_path in floor_plans.glob(pattern):
                    _rewrite_gltf_texture_uris(gltf_path, scene_dir)

        logger.info("Done!")


def is_scene_directory(path: Path) -> bool:
    """Check if path is a scene directory (has combined_house/house.dmd.yaml)."""
    return (path / "combined_house" / "house.dmd.yaml").exists()


def is_experiment_directory(path: Path) -> bool:
    """Check if path is an experiment directory (has scene_* subdirectories)."""
    return any(path.glob("scene_*"))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean scene output directories for portability"
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to scene or experiment directory "
            "(e.g., outputs/2025-12-31/23-25-50 or outputs/.../scene_039)"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for cleaned scene(s) (default: modify in place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers for experiment directories (default: 1)",
    )

    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")

    input_path = args.path.resolve()

    # Determine if input is a scene or experiment directory.
    if is_scene_directory(input_path):
        # Single scene.
        clean_scene(
            scene_dir=input_path,
            output_dir=args.output.resolve() if args.output else None,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    elif is_experiment_directory(input_path):
        # Experiment with multiple scenes.
        scene_dirs = sorted(input_path.glob("scene_*"))
        logger.info(f"Found {len(scene_dirs)} scenes in experiment directory")

        if args.workers > 1:
            # Parallel processing mode.
            tasks: list[tuple[str, Callable, dict]] = []
            for scene_dir in scene_dirs:
                if not is_scene_directory(scene_dir):
                    logger.warning(f"Skipping {scene_dir.name}: not a valid scene")
                    continue

                output_dir = None
                if args.output:
                    output_dir = args.output.resolve() / scene_dir.name

                tasks.append(
                    (
                        scene_dir.name,
                        clean_scene,
                        {
                            "scene_dir": scene_dir,
                            "output_dir": output_dir,
                            "dry_run": args.dry_run,
                            "verbose": args.verbose,
                        },
                    )
                )

            if tasks:
                logger.info(
                    f"Processing {len(tasks)} scenes with {args.workers} workers"
                )
                results = run_parallel_isolated(
                    tasks=tasks,
                    max_workers=args.workers,
                )

                successes = sum(1 for ok, _ in results.values() if ok)
                failures = len(results) - successes
                logger.info(f"\nCompleted: {successes} succeeded, {failures} failed")
        else:
            # Sequential processing mode.
            for scene_dir in scene_dirs:
                if not is_scene_directory(scene_dir):
                    logger.warning(f"Skipping {scene_dir.name}: not a valid scene")
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {scene_dir.name}")
                logger.info(f"{'='*60}")

                output_dir = None
                if args.output:
                    output_dir = args.output.resolve() / scene_dir.name

                clean_scene(
                    scene_dir=scene_dir,
                    output_dir=output_dir,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )

            logger.info(f"\n{'='*60}")
            logger.info(f"Completed processing {len(scene_dirs)} scenes")
    else:
        parser.error(
            f"Not a valid scene or experiment directory: {input_path}\n"
            f"Expected combined_house/house.dmd.yaml or scene_* subdirectories"
        )


if __name__ == "__main__":
    main()
