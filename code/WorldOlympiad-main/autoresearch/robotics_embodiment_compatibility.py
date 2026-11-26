"""Fail-closed geometric compatibility checks for robotics asset adapters."""

from __future__ import annotations

from typing import Any


def validate_parallel_gripper_cylinder(
    *,
    radius_m: float,
    max_aperture_m: float,
    minimum_closing_travel_m: float,
) -> dict[str, Any]:
    """Validate that a cylindrical proxy leaves useful gripper closing travel.

    Touching an object at the gripper's maximum aperture is not a grasp: the
    fingers need inward travel to establish force closure and tolerate pose
    error.  This check intentionally rejects zero-margin proxy geometry before
    simulator materialization or policy inference.
    """

    values = {
        "radius_m": float(radius_m),
        "max_aperture_m": float(max_aperture_m),
        "minimum_closing_travel_m": float(minimum_closing_travel_m),
    }
    if values["radius_m"] <= 0.0:
        raise ValueError("cylindrical grasp-proxy radius must be positive")
    if values["max_aperture_m"] <= 0.0:
        raise ValueError("parallel-gripper maximum aperture must be positive")
    if values["minimum_closing_travel_m"] < 0.0:
        raise ValueError("minimum closing travel cannot be negative")

    diameter = 2.0 * values["radius_m"]
    closing_travel = values["max_aperture_m"] - diameter
    if closing_travel < values["minimum_closing_travel_m"]:
        raise ValueError(
            "grasp proxy is incompatible with the parallel gripper: "
            f"diameter={diameter:.6f} m, max_aperture={values['max_aperture_m']:.6f} m, "
            f"available_closing_travel={closing_travel:.6f} m, required="
            f"{values['minimum_closing_travel_m']:.6f} m"
        )

    return {
        "schema_version": "parallel_gripper_asset_compatibility.v1",
        "status": "pass",
        "geometry": "cylinder",
        "proxy_radius_m": values["radius_m"],
        "proxy_diameter_m": diameter,
        "gripper_max_aperture_m": values["max_aperture_m"],
        "available_closing_travel_m": closing_travel,
        "required_closing_travel_m": values["minimum_closing_travel_m"],
    }
