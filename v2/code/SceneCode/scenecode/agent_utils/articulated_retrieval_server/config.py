"""Configuration for articulated object retrieval system."""

import logging

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


@dataclass
class ArticulatedSourceConfig:
    """Configuration for a single articulated data source."""

    name: str
    """Source name (e.g., 'partnet_mobility', 'artvip')."""

    enabled: bool
    """Whether this source is enabled for retrieval."""

    data_path: Path
    """Path to processed SDF assets directory."""

    embeddings_path: Path
    """Path to precomputed CLIP embeddings directory."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.data_path = Path(self.data_path)
        self.embeddings_path = Path(self.embeddings_path)

        if self.enabled:
            if not self.data_path.exists():
                console_logger.error(
                    f"Articulated source '{self.name}' data path does not exist: "
                    f"{self.data_path}. Source will be skipped."
                )
                self.enabled = False
            elif not self.embeddings_path.exists():
                console_logger.error(
                    f"Articulated source '{self.name}' embeddings path does not exist: "
                    f"{self.embeddings_path}. Source will be skipped."
                )
                self.enabled = False

    @classmethod
    def from_config(cls, name: str, cfg: DictConfig) -> "ArticulatedSourceConfig":
        """Create config from Hydra/OmegaConf nested structure.

        Args:
            name: Source name.
            cfg: Source config subtree.

        Returns:
            ArticulatedSourceConfig instance.
        """
        return cls(
            name=name,
            enabled=cfg.enabled,
            data_path=Path(cfg.data_path),
            embeddings_path=Path(cfg.embeddings_path),
        )


@dataclass
class ArticulatedConfig:
    """Configuration for articulated object retrieval."""

    sources: dict[str, ArticulatedSourceConfig]
    """Map of source name to source config."""

    use_top_k: int = 5
    """Number of top CLIP candidates before bbox ranking."""

    def __post_init__(self) -> None:
        """Log configuration summary."""
        enabled_sources = [name for name, cfg in self.sources.items() if cfg.enabled]
        disabled_sources = [
            name for name, cfg in self.sources.items() if not cfg.enabled
        ]

        console_logger.info(
            f"Articulated config initialized:\n"
            f"  enabled sources: {enabled_sources}\n"
            f"  disabled sources: {disabled_sources}"
        )

    @property
    def enabled_sources(self) -> dict[str, ArticulatedSourceConfig]:
        """Get only enabled sources."""
        return {name: cfg for name, cfg in self.sources.items() if cfg.enabled}

    @property
    def has_enabled_sources(self) -> bool:
        """Check if any sources are enabled."""
        return len(self.enabled_sources) > 0

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ArticulatedConfig":
        """Create config from Hydra/OmegaConf nested structure.

        Args:
            cfg: Articulated config subtree (cfg.asset_manager.articulated).

        Returns:
            ArticulatedConfig instance.
        """
        sources = {}
        for source_name in cfg.sources:
            source_cfg = getattr(cfg.sources, source_name)
            sources[source_name] = ArticulatedSourceConfig.from_config(
                name=source_name, cfg=source_cfg
            )
        return cls(sources=sources, use_top_k=cfg.use_top_k)
