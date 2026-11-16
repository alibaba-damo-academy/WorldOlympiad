"""Configuration for materials retrieval system."""

import logging

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


@dataclass
class MaterialsConfig:
    """Configuration for materials retrieval.

    Materials are stored in a single directory with embeddings precomputed.
    Unlike articulated objects, there is only one source (AmbientCG).
    """

    data_path: Path
    """Path to materials directory (contains material subdirectories)."""

    embeddings_path: Path
    """Path to embeddings directory (clip_embeddings.npy, embedding_index.yaml, etc.)."""

    use_top_k: int = 5
    """Number of top CLIP candidates to return."""

    enabled: bool = True
    """Whether materials retrieval is enabled."""

    def __post_init__(self) -> None:
        """Validate configuration and convert paths."""
        self.data_path = Path(self.data_path)
        self.embeddings_path = Path(self.embeddings_path)

        if self.enabled:
            if not self.data_path.exists():
                console_logger.error(
                    f"Materials data path does not exist: {self.data_path}. "
                    "Materials retrieval will be disabled."
                )
                self.enabled = False
            elif not self.embeddings_path.exists():
                console_logger.error(
                    f"Materials embeddings path does not exist: {self.embeddings_path}. "
                    "Materials retrieval will be disabled."
                )
                self.enabled = False
            else:
                # Validate required embedding files exist.
                required_files = [
                    "clip_embeddings.npy",
                    "embedding_index.yaml",
                    "metadata_index.yaml",
                ]
                for filename in required_files:
                    if not (self.embeddings_path / filename).exists():
                        console_logger.error(
                            f"Required embedding file not found: "
                            f"{self.embeddings_path / filename}. "
                            "Materials retrieval will be disabled."
                        )
                        self.enabled = False
                        break

        if self.enabled:
            console_logger.info(
                f"Materials config initialized:\n"
                f"  data_path: {self.data_path}\n"
                f"  embeddings_path: {self.embeddings_path}\n"
                f"  use_top_k: {self.use_top_k}"
            )
        else:
            console_logger.warning("Materials retrieval is disabled.")

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MaterialsConfig":
        """Create config from Hydra/OmegaConf structure.

        Args:
            cfg: Materials config subtree (cfg.materials_retrieval_server).

        Returns:
            MaterialsConfig instance.
        """
        return cls(
            data_path=Path(cfg.data_path),
            embeddings_path=Path(cfg.embeddings_path),
            use_top_k=cfg.use_top_k,
            enabled=cfg.get("enabled", True),
        )
