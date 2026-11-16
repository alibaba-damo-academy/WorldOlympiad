"""Materials resolver for floor plan generation.

Abstracts material selection from generated PBR textures or local defaults.
"""

import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from scenecode.agent_utils.material_generator import (
    MaterialGenerator,
    MaterialGeneratorConfig,
)
from scenecode.utils.material import Material

if TYPE_CHECKING:
    from scenecode.agent_utils.image_generation import BaseImageGenerator

console_logger = logging.getLogger(__name__)


# Default materials directory at repository root.
DEFAULT_MATERIALS_DIR = Path(__file__).parent.parent.parent.parent / "materials"


@dataclass
class MaterialsConfig:
    """Configuration for materials resolution."""

    use_retrieval_server: bool = False
    """Deprecated compatibility flag. Retrieval is no longer used in main flow."""

    generator: MaterialGeneratorConfig = field(default_factory=MaterialGeneratorConfig)
    """Code-generated material configuration."""

    default_wall_material: str = "Plaster001_1K-JPG"
    """Default wall material name."""

    default_floor_material: str = "Wood094_1K-JPG"
    """Default floor material name."""

    materials_dir: Path | None = None
    """Override materials directory (default: materials/ at repo root)."""

    output_dir: Path | None = None
    """Output directory for scene. Generated materials are written below it."""

    server_host: str = "127.0.0.1"
    """Deprecated compatibility host; retrieval is not used in main flow."""

    server_port: int = 7008
    """Deprecated compatibility port; retrieval is not used in main flow."""


class MaterialsResolver:
    """Resolves material requests to paths.

    Supports two modes:
    1. Generated: Creates PBR material folders from code-generated textures.
    2. Local fallback: Returns default wall/floor materials if generation fails.
    """

    def __init__(
        self,
        config: MaterialsConfig | None = None,
        image_generator: "BaseImageGenerator | None" = None,
    ):
        """Initialize materials resolver.

        Args:
            config: Materials configuration. If None, uses defaults.
            image_generator: Optional image generator for code-generated materials.
        """
        self.config = config or MaterialsConfig()
        self.materials_dir = self.config.materials_dir or DEFAULT_MATERIALS_DIR
        self.output_dir = self.config.output_dir
        self.image_generator = image_generator
        self._material_generator: MaterialGenerator | None = None

        # Cache discovered materials from local directory.
        self._available_materials: dict[str, Path] | None = None

        # Cache materials generated during this resolver lifetime.
        self._generated_materials_cache: dict[str, Material] = {}

    def get_material(self, description: str) -> Material | None:
        """Get material matching description.

        Args:
            description: Description of desired material
                (e.g., "warm oak hardwood", "white painted drywall").

        Returns:
            Material if found, None otherwise.
        """
        generated_material = self._get_generated_material(description)
        if generated_material is not None:
            return generated_material

        # Local fallback: only 2 materials exist, pick based on simple check.
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["floor", "wood", "tile", "carpet", "parquet"]):
            return self.get_default_floor_material()
        return self.get_default_wall_material()

    def get_default_wall_material(self) -> Material | None:
        """Get default wall material."""
        return self._resolve_material_id(self.config.default_wall_material)

    def get_default_floor_material(self) -> Material | None:
        """Get default floor material."""
        return self._resolve_material_id(self.config.default_floor_material)

    def get_material_by_id(self, material_id: str) -> Material | None:
        """Get material by exact ID.

        Args:
            material_id: Material identifier (e.g., 'Plaster001_1K-JPG').

        Returns:
            Material if found, None otherwise.
        """
        return self._resolve_material_id(material_id)

    def _resolve_material_id(self, material_id: str) -> Material | None:
        """Resolve a material ID to Material.

        Args:
            material_id: Material identifier.

        Returns:
            Material if found, None otherwise.
        """
        # First check generated cache.
        if material_id in self._generated_materials_cache:
            return self._generated_materials_cache[material_id]

        generated_material = self._resolve_generated_material_id(material_id)
        if generated_material is not None:
            return generated_material

        # Then check local materials directory.
        self._discover_materials()

        if material_id in self._available_materials:
            # Local materials use folder name as ID.
            return Material.from_path(self._available_materials[material_id])

        console_logger.warning(f"Material not found: {material_id}")
        return None

    def _get_generated_material(self, description: str) -> Material | None:
        """Generate a PBR material for the requested description.

        Generation is intentionally best-effort. If no image generator is available
        or generation fails, callers fall back to local default materials.
        """
        if not self.config.generator.enabled:
            return None
        if self.image_generator is None:
            console_logger.warning(
                "Material generator enabled but no image generator is available; "
                "falling back to local default material."
            )
            return None
        if self.output_dir is None:
            console_logger.warning(
                "Material generator enabled but output_dir is not set; "
                "falling back to local default material."
            )
            return None

        generator = self._get_material_generator()
        if generator is None:
            return None

        material = generator.generate_material(description=description)
        if material is None:
            console_logger.warning(
                f"Generated material failed for '{description}'; "
                "falling back to local default material."
            )
            return None

        self._generated_materials_cache[material.material_id] = material
        return material

    def _get_material_generator(self) -> MaterialGenerator | None:
        """Create or return the code material generator."""
        if self.output_dir is None or self.image_generator is None:
            return None
        if self._material_generator is None:
            generated_materials_dir = self._generated_materials_dir()
            generated_materials_dir.mkdir(parents=True, exist_ok=True)
            self._material_generator = MaterialGenerator(
                config=self.config.generator,
                output_dir=generated_materials_dir,
                image_generator=self.image_generator,
            )
        return self._material_generator

    def _generated_materials_dir(self) -> Path:
        """Return the scene-local directory for generated material folders."""
        if self.output_dir is None:
            raise ValueError("output_dir is required for generated materials")
        return self.output_dir / "materials" / "generated_materials"

    def _resolve_generated_material_id(self, material_id: str) -> Material | None:
        """Resolve generated material folders from this scene output directory."""
        if self.output_dir is None:
            return None

        material_path = self._generated_materials_dir() / material_id
        if not material_path.is_dir():
            return None

        material = Material(
            path=material_path,
            material_id=material_id,
            texture_scale=self.config.generator.texture_scale,
        )
        self._generated_materials_cache[material_id] = material
        return material

    def _discover_materials(self) -> None:
        """Discover available materials in materials directory."""
        if self._available_materials is not None:
            return

        self._available_materials = {}

        if not self.materials_dir.exists():
            console_logger.warning(
                f"Materials directory not found: {self.materials_dir}"
            )
            return

        # Find all material directories (contain texture files).
        for path in self.materials_dir.iterdir():
            if path.is_dir():
                # Check if it contains texture files.
                has_textures = any(
                    f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for f in path.iterdir()
                    if f.is_file()
                )
                if has_textures:
                    self._available_materials[path.name] = path

        console_logger.debug(
            f"Discovered {len(self._available_materials)} materials in "
            f"{self.materials_dir}"
        )
