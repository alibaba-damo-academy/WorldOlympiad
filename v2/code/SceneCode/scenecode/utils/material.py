"""Material domain model for PBR materials."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Material:
    """Represents a PBR material folder with texture files.

    A material folder typically contains PBR texture files:
    - *_Color.jpg - Base color/albedo
    - *_NormalGL.jpg or *_Normal.jpg - Normal map
    - *_Roughness.jpg - Roughness map
    - *_Displacement.jpg - Displacement map (optional)

    Use Material.from_path() for convenience when material_id equals folder name.
    """

    path: Path
    """Path to material folder containing PBR textures."""

    material_id: str
    """Material ID for lookup (typically folder name like 'Wood094_1K-JPG')."""

    texture_scale: float | None = None
    """Texture scale in meters per tile. None means 'cover' mode (no tiling)."""

    def get_texture(self, texture_type: str) -> Path | None:
        """Find texture file by type.

        Args:
            texture_type: Texture type suffix (e.g., 'Color', 'Normal', 'Roughness',
                'NormalGL', 'Displacement').

        Returns:
            Path to texture file if found, None otherwise.
        """
        # Primary pattern: *_{texture_type}.jpg (standard AmbientCG format).
        for ext in [".jpg", ".JPG", ".png", ".PNG"]:
            pattern = f"*_{texture_type}{ext}"
            matches = list(self.path.glob(pattern))
            if matches:
                return matches[0]

        # Fallback: case-insensitive search with keyword anywhere in name.
        for pattern in [
            f"*{texture_type}*",
            f"*{texture_type.lower()}*",
            f"*{texture_type.upper()}*",
        ]:
            matches = [
                f
                for f in self.path.glob(pattern)
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if matches:
                return matches[0]

        return None

    def get_all_textures(self) -> dict[str, Path]:
        """Get all available PBR textures.

        Returns:
            Dictionary with keys: 'color', 'normal', 'roughness' (if found).
            Values are Paths to the corresponding texture files.

        Raises:
            FileNotFoundError: If required textures (color, normal, roughness)
                are not found.
        """
        textures: dict[str, Path] = {}

        # Find color texture.
        color = self.get_texture("Color")
        if color:
            textures["color"] = color

        # Find normal texture (prefer NormalGL for OpenGL-style normals).
        normal = self.get_texture("NormalGL")
        if not normal:
            normal = self.get_texture("Normal")
        if normal:
            textures["normal"] = normal

        # Find roughness texture.
        roughness = self.get_texture("Roughness")
        if roughness:
            textures["roughness"] = roughness

        # Validate required textures.
        required = ["color", "normal", "roughness"]
        missing = [t for t in required if t not in textures]
        if missing:
            raise FileNotFoundError(
                f"Missing required PBR textures in {self.path}: {missing}. "
                f"Expected files: *_Color.jpg, *_NormalGL.jpg, *_Roughness.jpg"
            )

        return textures

    @property
    def name(self) -> str:
        """Material folder name (e.g., 'Wood094_1K-JPG')."""
        return self.path.name

    def to_dict(self) -> dict:
        """Serialize material to dictionary."""
        return {
            "path": str(self.path),
            "material_id": self.material_id,
            "texture_scale": self.texture_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Material":
        """Deserialize material from dictionary."""
        path = Path(data["path"])
        # Backward compatibility: use path.name if material_id missing.
        material_id = data.get("material_id", path.name)
        # Backward compatibility: default to None if texture_scale missing.
        texture_scale = data.get("texture_scale")
        return cls(path=path, material_id=material_id, texture_scale=texture_scale)

    @classmethod
    def from_path(cls, path: Path | str) -> "Material":
        """Create Material from path (derives material_id from folder name)."""
        p = Path(path) if isinstance(path, str) else path
        return cls(path=p, material_id=p.name)
