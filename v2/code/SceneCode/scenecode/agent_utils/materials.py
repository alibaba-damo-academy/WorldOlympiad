"""Material properties utilities.

Loads material properties from materials.yaml and provides accessor functions.
"""

import logging

from functools import lru_cache
from pathlib import Path

import yaml

console_logger = logging.getLogger(__name__)

# Default friction for unknown materials.
DEFAULT_FRICTION = 0.5

# Path to materials data file.
MATERIALS_YAML_PATH = Path(__file__).parent / "data" / "materials.yaml"


@lru_cache(maxsize=1)
def load_materials() -> dict[str, dict]:
    """Load material properties from materials.yaml.

    Returns:
        Dictionary mapping material names to their properties.
        Each material has: youngs_modulus, density, friction, material_modulus.
    """
    if not MATERIALS_YAML_PATH.exists():
        console_logger.warning(
            f"Materials file not found at {MATERIALS_YAML_PATH}, "
            f"using default friction={DEFAULT_FRICTION}"
        )
        return {}

    with open(MATERIALS_YAML_PATH) as f:
        return yaml.safe_load(f)


def get_friction(material: str) -> float:
    """Get friction coefficient for a material.

    Args:
        material: Material name (case-insensitive).

    Returns:
        Friction coefficient. Returns DEFAULT_FRICTION if material not found.
    """
    materials = load_materials()
    material_lower = material.lower()

    if material_lower in materials:
        return materials[material_lower].get("friction", DEFAULT_FRICTION)

    console_logger.debug(
        f"Unknown material '{material}', using default friction={DEFAULT_FRICTION}"
    )
    return DEFAULT_FRICTION


def get_density(material: str) -> float:
    """Get density for a material in kg/m^3.

    Args:
        material: Material name (case-insensitive).

    Returns:
        Density in kg/m^3. Returns 1000.0 (water density) if material not found.
    """
    materials = load_materials()
    material_lower = material.lower()

    default_density = 1000.0
    if material_lower in materials:
        return materials[material_lower].get("density", default_density)

    console_logger.debug(
        f"Unknown material '{material}', using default density={default_density}"
    )
    return default_density
