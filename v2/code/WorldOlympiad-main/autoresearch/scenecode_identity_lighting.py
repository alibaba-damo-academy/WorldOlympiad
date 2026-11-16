"""Colour-faithful lighting contract for the SceneCode identity render.

The identity gate exists to compare one rendered task asset with its declared
``visual_description``.  That comparison is only meaningful when the render
displays the asset's *declared material* faithfully.  Two renderer defaults in
this pipeline previously broke that property and produced colour-identity
failures for assets whose declared material colour was correct:

* the CPU path-tracer used Blender's ``AgX`` filmic view transform, a
  display-referred look that desaturates and darkens mid-tones, so a declared
  albedo of 0.86 was displayed near 0.74; and
* the fallback Drake/VTK path uses an uncalibrated default rig and displayed the
  same albedo near 0.46.

The measured consequence was a *non-deterministic* verdict: the identical
declared material (baseColorFactor 0.82/0.66/0.45 versus 0.88/0.72/0.50)
rendered at a foreground mean of 0.45 in three tasks and passed the identity
gate in two of them while failing in the third.  A gate whose verdict depends on
the display transform rather than on the asset is not an identity gate.

This module owns the calibrated, asset-independent lighting used by the identity
renderer: a neutral white uniform environment as the dominant illuminant plus
two bounded area lights that add form without pushing radiance above the
declared albedo, rendered through the ``Standard`` view transform so the
displayed value tracks the declared material.  It contains no task, asset or
case data.

Calibration evidence (Blender 4.4.3, CPU Cycles, 24 samples, five views, the real
published assets of this suite):

===================================  ==========================  =================
declared baseColorFactor             calibrated foreground mean  sRGB(declared)
===================================  ==========================  =================
0.88 / 0.72 / 0.50 (light wood)      0.949 / 0.876 / 0.759       0.95 / 0.87 / 0.74
0.86 / 0.86 / 0.84 (light gray)      0.956 / 0.956 / 0.947       0.94 / 0.94 / 0.93
(textured medium-dark brown)         0.568 / 0.413 / 0.314       (stays dark)
===================================  ==========================  =================

A declared light surface therefore renders light and a genuinely dark surface
still renders dark, so a colour contradiction reported by the reviewer is now
attributable to the asset instead of to the renderer.
"""

from __future__ import annotations

from typing import Any

# ``Standard`` is Blender's linear-scene-referred-to-sRGB transform.  Filmic
# looks (``AgX``, ``Filmic``) are deliberately rejected: they are aesthetic
# display transforms, not material measurements.
NEUTRAL_VIEW_TRANSFORM = "Standard"
NEUTRAL_LOOK = "None"
NEUTRAL_EXPOSURE = 0.0

# A neutral white uniform environment dominates the illumination so a diffuse
# surface returns approximately its own albedo independent of viewing angle.
NEUTRAL_WORLD_COLOR: tuple[float, float, float] = (1.0, 1.0, 1.0)
NEUTRAL_WORLD_STRENGTH = 0.75

# Bounded directional lights add shading and highlight cues so a reviewer can
# still judge three-dimensional form.  Energy is declared relative to the
# assembly radius squared, so illumination is independent of asset size.
NEUTRAL_AREA_LIGHTS: tuple[tuple[str, tuple[float, float, float], float], ...] = (
    ("Key", (2.0, -3.0, 4.0), 80.0),
    ("Fill", (-3.0, 1.0, 2.0), 40.0),
)

FORBIDDEN_VIEW_TRANSFORMS = frozenset(
    {"AgX", "Filmic", "Filmic Log", "False Color", "Khronos PBR Neutral"}
)


def identity_lighting_plan() -> dict[str, Any]:
    """Return the calibrated lighting settings applied by the Blender renderer."""

    if NEUTRAL_VIEW_TRANSFORM in FORBIDDEN_VIEW_TRANSFORMS:
        raise ValueError("identity render must not apply a display-referred look")
    return {
        "schema_version": "scenecode_identity_lighting.v1",
        "view_transform": NEUTRAL_VIEW_TRANSFORM,
        "look": NEUTRAL_LOOK,
        "exposure": NEUTRAL_EXPOSURE,
        "world_color": list(NEUTRAL_WORLD_COLOR),
        "world_strength": NEUTRAL_WORLD_STRENGTH,
        "area_lights": [
            [name, list(offset), float(energy)] for name, offset, energy in NEUTRAL_AREA_LIGHTS
        ],
        "faithful_albedo_display": True,
        "purpose": "display the declared material colour of one isolated task asset",
    }


def validate_identity_lighting(lighting: dict[str, Any]) -> None:
    """Fail closed when a plan would render through a display-referred look."""

    view_transform = str(lighting.get("view_transform", ""))
    if view_transform != NEUTRAL_VIEW_TRANSFORM:
        raise ValueError(
            f"identity render must use the {NEUTRAL_VIEW_TRANSFORM!r} view transform, "
            f"got {view_transform!r}"
        )
    color = lighting.get("world_color")
    if not isinstance(color, list) or len(color) != 3 or any(
        not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
        for value in color
    ):
        raise ValueError("identity world colour must be three values inside [0, 1]")
    strength = lighting.get("world_strength")
    if not isinstance(strength, (int, float)) or not 0.0 < float(strength) <= 1.0:
        raise ValueError("identity world strength must lie inside (0, 1]")
    lights = lighting.get("area_lights")
    if not isinstance(lights, list) or not lights:
        raise ValueError("identity lighting needs at least one bounded area light")
    for item in lights:
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError("identity area lights are (name, offset, relative energy)")
        offset = item[1]
        energy = item[2]
        if not isinstance(offset, list) or len(offset) != 3:
            raise ValueError("identity area light offsets are three values")
        if not isinstance(energy, (int, float)) or float(energy) < 0.0:
            raise ValueError("identity area light energy must be nonnegative")


def faithful_display_value(albedo: float) -> float:
    """Return the sRGB display value of a perfectly diffuse declared albedo.

    The calibrated rig is built so this value is what the identity render shows
    for an unoccluded surface facing the camera.  Tests use it as the expected
    relationship instead of a hard-coded per-asset pixel target.
    """

    value = float(albedo)
    if not 0.0 <= value <= 1.0:
        raise ValueError("albedo must lie inside [0, 1]")
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1.0 / 2.4) - 0.055
