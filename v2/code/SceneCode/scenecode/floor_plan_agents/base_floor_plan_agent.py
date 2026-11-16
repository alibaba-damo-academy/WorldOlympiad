"""Base class for floor plan agents.

Provides shared functionality for floor plan design and geometry generation.
"""

import logging

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from scenecode.agent_utils.house import HouseLayout
from scenecode.agent_utils.image_generation import create_image_generator
from scenecode.agent_utils.material_generator import MaterialGeneratorConfig
from scenecode.floor_plan_agents.tools.floor_plan_tools import DoorWindowConfig
from scenecode.floor_plan_agents.tools.materials_resolver import MaterialsConfig
from scenecode.floor_plan_agents.tools.room_placement import ScoringWeights
from scenecode.prompts import prompt_registry
from scenecode.utils.logging import BaseLogger

if TYPE_CHECKING:
    from scenecode.agent_utils.image_generation import BaseImageGenerator

console_logger = logging.getLogger(__name__)


class BaseFloorPlanAgent(ABC):
    """Base class with shared functionality for floor plan agents.

    NOTE: A new FloorPlanAgent instance is created for each house/scene.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
    ):
        """Initialize base floor plan agent.

        Args:
            cfg: Configuration for floor plan generation.
            logger: Scene-specific logger from experiment.
            materials_server_host: Deprecated compatibility parameter; ignored.
            materials_server_port: Deprecated compatibility parameter; ignored.
        """
        self.cfg = cfg
        self.logger = logger
        self.materials_server_host = materials_server_host
        self.materials_server_port = materials_server_port
        self._material_image_generator: "BaseImageGenerator | None" = None

        # Floor plan mode: "room" (single room) or "house" (multi-room).
        self.mode = cfg.mode

        # Prompt registry for agent prompts.
        self.prompt_registry = prompt_registry

        # Layout being designed (set in subclass).
        self.layout: HouseLayout | None = None

    def _create_materials_config(self) -> MaterialsConfig:
        """Create materials configuration from config.

        Returns:
            MaterialsConfig with settings from cfg.
        """
        generator_config = MaterialGeneratorConfig(
            enabled=bool(
                OmegaConf.select(self.cfg, "materials.generator.enabled", default=False)
            ),
            backend=str(
                OmegaConf.select(
                    self.cfg, "materials.generator.backend", default="openai"
                )
            ),
            max_retries=int(
                OmegaConf.select(self.cfg, "materials.generator.max_retries", default=2)
            ),
            default_roughness=int(
                OmegaConf.select(
                    self.cfg, "materials.generator.default_roughness", default=128
                )
            ),
            texture_scale=float(
                OmegaConf.select(
                    self.cfg, "materials.generator.texture_scale", default=0.5
                )
            ),
        )

        # Get output directory from layout if available.
        output_dir = None
        if self.layout and self.layout.house_dir:
            output_dir = self.layout.house_dir

        return MaterialsConfig(
            use_retrieval_server=bool(
                OmegaConf.select(
                    self.cfg, "materials.use_retrieval_server", default=False
                )
            ),
            generator=generator_config,
            default_wall_material=str(
                OmegaConf.select(
                    self.cfg,
                    "materials.default_wall_material",
                    default="Plaster001_1K-JPG",
                )
            ),
            default_floor_material=str(
                OmegaConf.select(
                    self.cfg,
                    "materials.default_floor_material",
                    default="Wood094_1K-JPG",
                )
            ),
            output_dir=output_dir,
            server_host=self.materials_server_host,
            server_port=self.materials_server_port,
        )

    def _get_material_image_generator(self) -> "BaseImageGenerator | None":
        """Create the image generator used for generated floor/wall materials."""
        if not bool(
            OmegaConf.select(self.cfg, "materials.generator.enabled", default=False)
        ):
            return None

        image_generation_cfg = OmegaConf.select(
            self.cfg, "image_generation", default=None
        )
        if image_generation_cfg is None:
            console_logger.warning(
                "materials.generator.enabled=True but floor plan image_generation "
                "config is missing; generated materials will use local fallback."
            )
            return None

        if self._material_image_generator is None:
            try:
                api_base = OmegaConf.select(self.cfg, "openai.api_base", default=None)
            except Exception:
                api_base = None
            self._material_image_generator = create_image_generator(
                backend=image_generation_cfg.backend,
                config=image_generation_cfg,
                api_base=api_base,
            )
        return self._material_image_generator

    def _create_door_window_config(self) -> DoorWindowConfig:
        """Create door/window configuration from config.

        Returns:
            DoorWindowConfig with constraints from cfg.
        """
        doors_cfg = self.cfg.doors
        windows_cfg = self.cfg.windows

        return DoorWindowConfig(
            door_width_min=doors_cfg.width_range[0],
            door_width_max=doors_cfg.width_range[1],
            door_height_min=doors_cfg.height_range[0],
            door_height_max=doors_cfg.height_range[1],
            door_default_width=doors_cfg.default_width,
            door_default_height=doors_cfg.default_height,
            window_width_min=windows_cfg.width_range[0],
            window_width_max=windows_cfg.width_range[1],
            window_height_min=windows_cfg.height_range[0],
            window_height_max=windows_cfg.height_range[1],
            window_default_width=windows_cfg.default_width,
            window_default_height=windows_cfg.default_height,
            window_default_sill_height=windows_cfg.default_sill_height,
            window_segment_margin=windows_cfg.segment_margin,
            exterior_door_clearance_m=doors_cfg.exterior_clearance,
        )

    def _create_scoring_weights(self) -> ScoringWeights:
        """Create scoring weights for room placement from config.

        Returns:
            ScoringWeights with compactness and stability weights.
        """
        weights_cfg = self.cfg.room_placement.scoring_weights
        return ScoringWeights(
            compactness=weights_cfg.compactness,
            stability=weights_cfg.stability,
        )

    def cleanup(self) -> None:
        """Cleanup resources held by the agent.

        Override in subclass if there are resources to clean up.
        """

    @abstractmethod
    async def generate_house_layout(self, prompt: str, output_dir: Path) -> HouseLayout:
        """Generate a house layout with floor plan geometry.

        This is the main entry point for floor plan generation. It runs the agent trio
        to design the layout, then generates geometry for all rooms.

        Args:
            prompt: Description of the house/room to design.
            output_dir: Directory to save generated geometry files.

        Returns:
            HouseLayout with designed layout and generated RoomGeometry.
        """
        raise NotImplementedError
