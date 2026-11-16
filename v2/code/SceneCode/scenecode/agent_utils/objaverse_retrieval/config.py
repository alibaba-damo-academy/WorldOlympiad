"""Configuration for Objaverse (ObjectThor) retrieval system."""

import logging

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


@dataclass
class ObjaverseConfig:
    """Configuration for Objaverse asset retrieval."""

    data_path: Path
    """Path to ObjectThor assets directory (containing {uid}/{uid}.glb subdirectories)."""

    preprocessed_path: Path
    """Path to preprocessed data (indices, embeddings)."""

    use_top_k: int = 5
    """Number of top CLIP candidates to consider before size ranking."""

    object_type_mapping: dict[str, str] | None = None
    """Map scenecode ObjectType to Objaverse categories."""

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        self.data_path = Path(self.data_path)
        self.preprocessed_path = Path(self.preprocessed_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Objaverse data path does not exist: {self.data_path}"
            )

        if not self.preprocessed_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data path does not exist: {self.preprocessed_path}"
            )

        if self.object_type_mapping is None:
            # Map scenecode ObjectType to Objaverse placement-based categories.
            self.object_type_mapping = {
                "FURNITURE": "large_objects",
                "MANIPULAND": "small_objects",
                "WALL_MOUNTED": "wall_objects",
                "CEILING_MOUNTED": "ceiling_objects",
            }

        console_logger.info(
            f"Objaverse config initialized:\n"
            f"  data_path: {self.data_path}\n"
            f"  preprocessed_path: {self.preprocessed_path}\n"
            f"  top_k: {self.use_top_k}"
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ObjaverseConfig":
        """Create config from Hydra/OmegaConf nested structure.

        Args:
            cfg: Objaverse config subtree (cfg.asset_manager.objaverse).

        Returns:
            ObjaverseConfig instance.
        """
        return cls(
            data_path=Path(cfg.data_path),
            preprocessed_path=Path(cfg.preprocessed_path),
            use_top_k=cfg.use_top_k,
            object_type_mapping=dict(cfg.object_type_mapping),
        )
