"""Placement noise utilities for adding realistic variation to object placement.

This module provides deterministic Gaussian noise application to object transforms
to simulate human placement imperfections. Noise is applied to X-Y position and
yaw rotation while preserving Z coordinate and roll/pitch orientation.
"""

import logging
import math

from enum import Enum

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

console_logger = logging.getLogger(__name__)


class PlacementNoiseMode(Enum):
    """Modes for controlling placement noise behavior."""

    OFF = "off"
    NATURAL = "natural"
    PERFECT = "perfect"
    AUTO = "auto"


def apply_placement_noise(
    transform: RigidTransform,
    position_xy_std_meters: float,
    rotation_yaw_std_degrees: float,
) -> RigidTransform:
    """Apply Gaussian noise to furniture placement for realistic variation.

    Adds human-like imperfections to placement by applying noise to X-Y position
    and yaw rotation. Preserves Z coordinate (floor placement) and roll/pitch
    (upright orientation).

    The noise is deterministic - seeded from the input transform to ensure
    consistent results during scene replay.

    Args:
        transform: Original placement transform.
        position_xy_std_meters: Standard deviation for X-Y position noise (meters).
        rotation_yaw_std_degrees: Standard deviation for yaw rotation noise (degrees).

    Returns:
        Modified transform with noise applied.
    """
    if position_xy_std_meters == 0.0 and rotation_yaw_std_degrees == 0.0:
        return transform

    position = transform.translation()
    rpy = RollPitchYaw(transform.rotation())

    # Create deterministic seed from transform to ensure reproducibility.
    # Use position and yaw to generate a unique but deterministic seed.
    seed = hash(
        (
            round(position[0], 6),  # Round to avoid floating point drift.
            round(position[1], 6),
            round(position[2], 6),
            round(rpy.yaw_angle(), 6),
        )
    )
    # Ensure seed is within valid range for numpy (0 to 2^32-1).
    seed = abs(seed) & 0xFFFFFFFF

    # Create deterministic RNG with this seed.
    rng = np.random.default_rng(seed)

    noise_x = rng.normal(loc=0.0, scale=position_xy_std_meters)
    noise_y = rng.normal(loc=0.0, scale=position_xy_std_meters)
    noise_yaw_deg = rng.normal(loc=0.0, scale=rotation_yaw_std_degrees)

    new_position = [
        position[0] + noise_x,
        position[1] + noise_y,
        position[2],  # Preserve Z
    ]

    new_rpy = RollPitchYaw(
        rpy.roll_angle(),  # Preserve roll
        rpy.pitch_angle(),  # Preserve pitch
        rpy.yaw_angle() + math.radians(noise_yaw_deg),  # Add noise to yaw
    )

    console_logger.info(
        f"Applied placement noise: {noise_x:.3f}m, {noise_y:.3f}m, {noise_yaw_deg:.3f}°"
    )

    return RigidTransform(rpy=new_rpy, p=new_position)


def apply_wall_placement_noise(
    position_x: float,
    position_z: float,
    rotation_deg: float,
    position_along_wall_std_meters: float,
    position_height_std_meters: float,
    rotation_std_degrees: float,
) -> tuple[float, float, float]:
    """Apply Gaussian noise to wall object placement for realistic variation.

    Adds human-like imperfections to wall placement by applying noise to the
    position along wall (X), height (Z), and rotation around wall normal.

    Wall coordinate system:
        X: along wall (from start to end)
        Z: vertical height
        Rotation: around wall normal (into room)

    The noise is deterministic - seeded from the input values to ensure
    consistent results during scene replay.

    Args:
        position_x: Position along wall (meters from wall start).
        position_z: Height on wall (meters from floor).
        rotation_deg: Rotation around wall normal (degrees).
        position_along_wall_std_meters: Std dev for X position noise (meters).
        position_height_std_meters: Std dev for Z height noise (meters).
        rotation_std_degrees: Std dev for rotation noise (degrees).

    Returns:
        Tuple of (noisy_x, noisy_z, noisy_rotation_deg).
    """
    if (
        position_along_wall_std_meters == 0.0
        and position_height_std_meters == 0.0
        and rotation_std_degrees == 0.0
    ):
        return position_x, position_z, rotation_deg

    # Create deterministic seed from input values to ensure reproducibility.
    seed = hash((round(position_x, 6), round(position_z, 6), round(rotation_deg, 6)))
    # Ensure seed is within valid range for numpy (0 to 2^32-1).
    seed = abs(seed) & 0xFFFFFFFF

    # Create deterministic RNG with this seed.
    rng = np.random.default_rng(seed)

    noise_x = rng.normal(loc=0.0, scale=position_along_wall_std_meters)
    noise_z = rng.normal(loc=0.0, scale=position_height_std_meters)
    noise_rotation = rng.normal(loc=0.0, scale=rotation_std_degrees)

    noisy_x = position_x + noise_x
    noisy_z = position_z + noise_z
    noisy_rotation = rotation_deg + noise_rotation

    console_logger.info(
        f"Applied wall placement noise: "
        f"x={noise_x:.3f}m, z={noise_z:.3f}m, rot={noise_rotation:.3f}°"
    )

    return noisy_x, noisy_z, noisy_rotation


def apply_ceiling_placement_noise(
    position_x: float,
    position_y: float,
    rotation_deg: float,
    position_xy_std_meters: float,
    rotation_yaw_std_degrees: float,
) -> tuple[float, float, float]:
    """Apply Gaussian noise to ceiling object placement for realistic variation.

    Adds human-like imperfections to ceiling placement by applying noise to the
    X, Y position on the ceiling plane and rotation around the Z-axis (yaw).

    Ceiling coordinate system (room-frame):
        X: room X-axis
        Y: room Y-axis
        Rotation: around Z-axis (yaw, looking down from above)

    The noise is deterministic - seeded from the input values to ensure
    consistent results during scene replay.

    Args:
        position_x: X position on ceiling (meters in room coords).
        position_y: Y position on ceiling (meters in room coords).
        rotation_deg: Rotation around Z-axis (degrees).
        position_xy_std_meters: Std dev for XY position noise (meters).
        rotation_yaw_std_degrees: Std dev for rotation noise (degrees).

    Returns:
        Tuple of (noisy_x, noisy_y, noisy_rotation_deg).
    """
    if position_xy_std_meters == 0.0 and rotation_yaw_std_degrees == 0.0:
        return position_x, position_y, rotation_deg

    # Create deterministic seed from input values to ensure reproducibility.
    seed = hash((round(position_x, 6), round(position_y, 6), round(rotation_deg, 6)))
    # Ensure seed is within valid range for numpy (0 to 2^32-1).
    seed = abs(seed) & 0xFFFFFFFF

    # Create deterministic RNG with this seed.
    rng = np.random.default_rng(seed)

    noise_x = rng.normal(loc=0.0, scale=position_xy_std_meters)
    noise_y = rng.normal(loc=0.0, scale=position_xy_std_meters)
    noise_rotation = rng.normal(loc=0.0, scale=rotation_yaw_std_degrees)

    noisy_x = position_x + noise_x
    noisy_y = position_y + noise_y
    noisy_rotation = rotation_deg + noise_rotation

    console_logger.info(
        f"Applied ceiling placement noise: "
        f"x={noise_x:.3f}m, y={noise_y:.3f}m, rot={noise_rotation:.3f}°"
    )

    return noisy_x, noisy_y, noisy_rotation
