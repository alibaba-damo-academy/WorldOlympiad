"""Unit tests for placement noise utilities."""

import math

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.placement_noise import (
    PlacementNoiseMode,
    apply_placement_noise,
)


def test_noise_preserves_z_coordinate():
    """Test that noise application preserves Z coordinate for floor placement."""
    original_transform = RigidTransform(p=[1.0, 2.0, 0.0])

    noised_transform = apply_placement_noise(
        transform=original_transform,
        position_xy_std_meters=0.05,
        rotation_yaw_std_degrees=10.0,
    )

    assert noised_transform.translation()[2] == 0.0


def test_noise_preserves_roll_pitch():
    """Test that noise application preserves roll and pitch for upright furniture."""
    original_rpy = RollPitchYaw(0.1, 0.2, 0.3)
    original_transform = RigidTransform(rpy=original_rpy, p=[1.0, 2.0, 0.0])

    noised_transform = apply_placement_noise(
        transform=original_transform,
        position_xy_std_meters=0.05,
        rotation_yaw_std_degrees=10.0,
    )

    noised_rpy = RollPitchYaw(noised_transform.rotation())
    assert math.isclose(noised_rpy.roll_angle(), 0.1, abs_tol=1e-10)
    assert math.isclose(noised_rpy.pitch_angle(), 0.2, abs_tol=1e-10)


def test_zero_std_returns_original_transform():
    """Test that zero standard deviation returns unchanged transform."""
    original_transform = RigidTransform(p=[1.0, 2.0, 0.0])

    noised_transform = apply_placement_noise(
        transform=original_transform,
        position_xy_std_meters=0.0,
        rotation_yaw_std_degrees=0.0,
    )

    original_pos = original_transform.translation()
    noised_pos = noised_transform.translation()

    assert np.allclose(original_pos, noised_pos)

    original_rpy = RollPitchYaw(original_transform.rotation())
    noised_rpy = RollPitchYaw(noised_transform.rotation())

    assert math.isclose(original_rpy.yaw_angle(), noised_rpy.yaw_angle(), abs_tol=1e-10)


def test_enum_string_values():
    """Test that PlacementNoiseMode enum has expected string values."""
    assert PlacementNoiseMode.OFF.value == "off"
    assert PlacementNoiseMode.NATURAL.value == "natural"
    assert PlacementNoiseMode.PERFECT.value == "perfect"
    assert PlacementNoiseMode.AUTO.value == "auto"


def test_noise_actually_adds_variation():
    """Test that noise with non-zero std actually modifies the transform."""
    original_transform = RigidTransform(p=[1.0, 2.0, 0.0])

    # Apply noise multiple times and check at least one differs.
    transforms_differ = False
    for _ in range(10):
        noised_transform = apply_placement_noise(
            transform=original_transform,
            position_xy_std_meters=0.05,
            rotation_yaw_std_degrees=10.0,
        )

        original_pos = original_transform.translation()
        noised_pos = noised_transform.translation()

        if not np.allclose(original_pos[:2], noised_pos[:2], atol=1e-6):
            transforms_differ = True
            break

    assert transforms_differ, "Noise should add variation to at least one transform"
