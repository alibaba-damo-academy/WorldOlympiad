"""Auditable pilot material priors."""

from __future__ import annotations

from .models import ContactMaterial


MATERIALS: dict[str, ContactMaterial] = {
    "rubber_ball": ContactMaterial(
        name="rubber_ball", lateral_friction=0.8, rolling_friction=0.002,
        restitution=0.6, linear_damping=0.04, angular_damping=0.04, density_kg_m3=1100,
    ),
    "ceramic": ContactMaterial(
        # The generated cup has an unglazed/non-slip foot.  PyBullet combines
        # the cup and wooden support coefficients multiplicatively; 0.85 x
        # 0.60 keeps the cup planted long enough for a high-side robot push to
        # create the expected overturning moment instead of pure sliding.
        name="ceramic", lateral_friction=0.85, restitution=0.05,
        linear_damping=0.04, angular_damping=0.04, density_kg_m3=2400,
    ),
    "wood_block": ContactMaterial(
        name="wood_block", lateral_friction=0.45, restitution=0.1,
        linear_damping=0.04, angular_damping=0.04, density_kg_m3=650,
    ),
    "wood_support": ContactMaterial(
        name="wood_support", lateral_friction=0.6, restitution=0.05,
        linear_damping=0.04, angular_damping=0.04, density_kg_m3=650,
    ),
    "room": ContactMaterial(
        name="room", lateral_friction=0.7, restitution=0.05,
        linear_damping=0.04, angular_damping=0.04,
    ),
    "robot": ContactMaterial(
        name="robot", lateral_friction=0.8, restitution=0.0,
        linear_damping=0.02, angular_damping=0.02,
    ),
}

# Category intervals are deliberately broad enough to preserve plausible
# SceneCode/VLM estimates, but reject the upstream generic 5 kg fallback and
# obviously implausible toy-object masses.  Every selected value is copied into
# the event body's metadata and therefore into WorldIR/physics provenance.
MASS_PRIORS_KG: dict[str, dict[str, float]] = {
    "ball": {"minimum": 0.05, "maximum": 1.50, "default": 0.25},
    "cup": {"minimum": 0.10, "maximum": 1.00, "default": 0.35},
    "slider": {"minimum": 0.05, "maximum": 0.80, "default": 0.30},
    "block": {"minimum": 0.01, "maximum": 0.50, "default": 0.18},
}


CATEGORY_COLORS: dict[str, tuple[float, float, float, float]] = {
    "ball": (0.9, 0.05, 0.05, 1.0),
    "cup": (0.05, 0.25, 0.9, 1.0),
    "slider": (0.62, 0.32, 0.12, 1.0),
    "block_red": (0.9, 0.15, 0.1, 1.0),
    "block_green": (0.1, 0.75, 0.25, 1.0),
    "block_yellow": (0.95, 0.7, 0.05, 1.0),
    "table": (0.48, 0.24, 0.08, 1.0),
    "room": (0.82, 0.82, 0.78, 1.0),
}
