#!/usr/bin/env python3
"""Fix corrupted joint poses in ArtVIP SDF files.

The original ArtVIP USD data has incorrect localPose0 values (3-8 meters instead
of ~0.15m), causing articulated parts to fly away when joints are opened.

This script computes correct joint poses from mesh geometry:
- Prismatic joints (drawers): back center of child mesh
- Revolute joints (doors): hinge edge of child mesh

Usage:
    python scripts/fix_artvip_joint_poses.py [--dry-run]
"""

import argparse
import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Threshold for detecting corrupt poses (values > 1m are likely wrong).
CORRUPT_THRESHOLD_M = 1.0


def load_mesh_bounds(path: Path) -> np.ndarray:
    """Load mesh and return bounds, handling gltf Scene objects."""
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Empty scene: {path}")
        combined = trimesh.util.concatenate(list(mesh.geometry.values()))
        return combined.bounds
    return mesh.bounds


def get_child_mesh_path(
    sdf_path: Path, child_link_name: str, root: ET.Element
) -> Path | None:
    """Find mesh path from link's visual geometry URI."""
    link = root.find(f".//link[@name='{child_link_name}']")
    if link is None:
        return None

    uri_elem = link.find(".//visual/geometry/mesh/uri")
    if uri_elem is None:
        return None

    mesh_uri = uri_elem.text
    sdf_dir = sdf_path.parent

    # Prefer .obj over .gltf (trimesh loads obj faster).
    obj_path = sdf_dir / mesh_uri.replace(".gltf", ".obj")
    if obj_path.exists():
        return obj_path

    gltf_path = sdf_dir / mesh_uri
    if gltf_path.exists():
        return gltf_path

    return None


def compute_prismatic_pose(bounds: np.ndarray) -> tuple[float, float, float]:
    """Compute joint pose at mesh back center (for drawers)."""
    center_x = (bounds[0][0] + bounds[1][0]) / 2
    back_y = bounds[0][1]  # min Y = back
    center_z = (bounds[0][2] + bounds[1][2]) / 2
    return (center_x, back_y, center_z)


def compute_revolute_pose(
    bounds: np.ndarray, hinge_side: str
) -> tuple[float, float, float]:
    """Compute joint pose at mesh hinge edge (for doors)."""
    if hinge_side == "left":
        hinge_x = bounds[0][0]  # min X
    else:
        hinge_x = bounds[1][0]  # max X

    front_y = bounds[1][1]  # max Y = front
    center_z = (bounds[0][2] + bounds[1][2]) / 2
    return (hinge_x, front_y, center_z)


def is_pose_corrupt(pose_text: str) -> bool:
    """Check if pose has X or Y values > threshold (likely corrupt).

    Only checks X and Y, not Z, since tall objects can have Z > 1m legitimately.
    """
    try:
        values = [float(x) for x in pose_text.split()[:3]]
        # Only check X and Y - tall furniture can have Z > 1m.
        return max(abs(values[0]), abs(values[1])) > CORRUPT_THRESHOLD_M
    except (ValueError, IndexError):
        return False


def fix_sdf_joint_poses(sdf_path: Path, dry_run: bool = False) -> int:
    """Fix joint poses in an SDF file. Returns number of joints fixed."""
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    fixed_count = 0
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")

        # Skip non-movable joints.
        if joint_type not in ("prismatic", "revolute"):
            continue

        pose_elem = joint.find("pose")
        if pose_elem is None or pose_elem.text is None:
            continue

        # Skip if not corrupt.
        if not is_pose_corrupt(pose_elem.text):
            continue

        child_elem = joint.find("child")
        if child_elem is None:
            continue
        child_link = child_elem.text

        # Find mesh path.
        mesh_path = get_child_mesh_path(
            sdf_path=sdf_path, child_link_name=child_link, root=root
        )
        if mesh_path is None:
            logger.warning(f"  No mesh found for {child_link} in {sdf_path.name}")
            continue

        try:
            bounds = load_mesh_bounds(mesh_path)
        except Exception as e:
            logger.warning(f"  Failed to load mesh {mesh_path}: {e}")
            continue

        # Parse old pose to preserve rotation and detect hinge side.
        old_values = pose_elem.text.split()
        old_xyz = [float(v) for v in old_values[:3]]
        old_rotation = " ".join(old_values[3:]) if len(old_values) > 3 else "0 0 0"

        # Compute new pose.
        if joint_type == "prismatic":
            new_xyz = compute_prismatic_pose(bounds)
        else:  # revolute
            # Detect hinge side from existing pose X sign.
            hinge_side = "left" if old_xyz[0] < 0 else "right"
            new_xyz = compute_revolute_pose(bounds=bounds, hinge_side=hinge_side)

        # Validate new pose X/Y are reasonable (Z can be > 1m for tall objects).
        if max(abs(new_xyz[0]), abs(new_xyz[1])) > CORRUPT_THRESHOLD_M:
            logger.warning(
                f"  Computed pose X/Y still large for {joint_name}: {new_xyz}"
            )
            continue

        new_pose_text = (
            f"{new_xyz[0]:.6f} {new_xyz[1]:.6f} {new_xyz[2]:.6f} {old_rotation}"
        )

        if dry_run:
            logger.info(f"  Would fix {joint_name}: {old_xyz} -> {new_xyz}")
        else:
            pose_elem.text = new_pose_text
            logger.info(f"  Fixed {joint_name}: {old_xyz} -> {new_xyz}")

        fixed_count += 1

    if fixed_count > 0 and not dry_run:
        # Write with XML declaration.
        tree.write(sdf_path, encoding="unicode", xml_declaration=True)

    return fixed_count


def main():
    parser = argparse.ArgumentParser(
        description="Fix corrupted joint poses in ArtVIP SDFs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without modifying files",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/artvip_sdf"),
        help="Path to artvip_sdf directory",
    )
    args = parser.parse_args()

    if not args.path.exists():
        logger.error(f"Path does not exist: {args.path}")
        return

    total_fixed = 0
    files_fixed = 0

    for sdf_path in sorted(args.path.rglob("*.sdf")):
        # Skip non-articulated SDFs.
        content = sdf_path.read_text()
        if "<joint" not in content:
            continue

        fixed = fix_sdf_joint_poses(sdf_path, dry_run=args.dry_run)
        if fixed > 0:
            logger.info(f"{sdf_path.name}: fixed {fixed} joint(s)")
            total_fixed += fixed
            files_fixed += 1

    action = "Would fix" if args.dry_run else "Fixed"
    logger.info(f"\n{action} {total_fixed} joints in {files_fixed} files")


if __name__ == "__main__":
    main()
