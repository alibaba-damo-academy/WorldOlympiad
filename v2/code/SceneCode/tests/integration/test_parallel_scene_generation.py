import logging
import shutil
import tempfile
import unittest

from pathlib import Path

from omegaconf import OmegaConf

from scenecode.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
)
from tests.integration.common import (
    has_gpu_available,
    has_hunyuan3d_installed,
    has_openai_key,
    is_github_actions,
)


@unittest.skipIf(
    not has_openai_key()
    or not has_gpu_available()
    or not has_hunyuan3d_installed()
    or is_github_actions(),
    "Requires OpenAI API key, GPU, Hunyuan3D-2, and non-CI environment",
)
class TestParallelSceneGeneration(unittest.TestCase):
    """Integration test for parallel scene generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "parallel_test"
        self.output_dir.mkdir(exist_ok=True)

        # Print output directory for progress tracking.
        print(f"\n{'='*60}")
        print(f"Test output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # Load base experiment configuration.
        experiment_config_path = (
            Path(__file__).parent.parent.parent
            / "configs/experiment/base_experiment.yaml"
        )
        base_experiment_config = OmegaConf.load(experiment_config_path)

        # Add required agent and generator configs.
        # Load base furniture agent config first.
        base_furniture_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )

        # Load stateful furniture agent config.
        stateful_furniture_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/stateful_furniture_agent.yaml"
        )

        # Merge furniture agent configs.
        furniture_agent_config = OmegaConf.merge(
            base_furniture_agent_config, stateful_furniture_agent_config
        )
        # Load floor plan agent configs.
        base_floor_plan_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/floor_plan_agent/base_floor_plan_agent.yaml"
        )
        stateful_floor_plan_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/floor_plan_agent/stateful_floor_plan_agent.yaml"
        )

        # Merge floor plan agent configs.
        floor_plan_config = OmegaConf.merge(
            base_floor_plan_agent_config, stateful_floor_plan_agent_config
        )
        base_manipuland_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/base_manipuland_agent.yaml"
        )
        stateful_manipuland_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/stateful_manipuland_agent.yaml"
        )
        manipuland_agent_config = OmegaConf.merge(
            base_manipuland_agent_config, stateful_manipuland_agent_config
        )

        # Load wall agent configs.
        base_wall_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/wall_agent/base_wall_agent.yaml"
        )
        stateful_wall_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/wall_agent/stateful_wall_agent.yaml"
        )
        wall_agent_config = OmegaConf.merge(
            base_wall_agent_config, stateful_wall_agent_config
        )

        # Load ceiling agent configs.
        base_ceiling_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/ceiling_agent/base_ceiling_agent.yaml"
        )
        stateful_ceiling_agent_config = OmegaConf.load(
            Path(__file__).parent.parent.parent
            / "configs/ceiling_agent/stateful_ceiling_agent.yaml"
        )
        ceiling_agent_config = OmegaConf.merge(
            base_ceiling_agent_config, stateful_ceiling_agent_config
        )

        # Add _name fields to configs as Hydra would.
        furniture_agent_config._name = "stateful_furniture_agent"
        floor_plan_config._name = "stateful_floor_plan_agent"
        manipuland_agent_config._name = "stateful_manipuland_agent"
        wall_agent_config._name = "stateful_wall_agent"
        ceiling_agent_config._name = "stateful_ceiling_agent"

        # Create complete base config structure with proper nesting.
        self.base_config = OmegaConf.create(
            {
                "experiment": base_experiment_config,
                "furniture_agent": furniture_agent_config,
                "floor_plan_agent": floor_plan_config,
                "manipuland_agent": manipuland_agent_config,
                "wall_agent": wall_agent_config,
                "ceiling_agent": ceiling_agent_config,
            }
        )

    def _dump_scene_diagnostics(self, scene_dir: Path) -> str:
        """Get diagnostic info about a scene directory for debugging.

        Only called on test failures to provide context about what went wrong.

        Args:
            scene_dir: Path to the scene directory to diagnose.

        Returns:
            Formatted diagnostic string with directory contents and log tail.
        """
        lines = [f"\n=== Scene Diagnostics: {scene_dir.name} ==="]

        # List top-level contents.
        if scene_dir.exists():
            lines.append(f"Contents: {[f.name for f in scene_dir.iterdir()]}")

            # Check generated_assets subdirs.
            assets_dir = scene_dir / "generated_assets"
            if assets_dir.exists():
                for subdir in ["images", "geometry", "sdf", "debug"]:
                    subdir_path = assets_dir / subdir
                    count = (
                        len(list(subdir_path.iterdir())) if subdir_path.exists() else 0
                    )
                    lines.append(f"  {subdir}: {count} files")

            # Show last 10 lines of scene.log if it exists.
            log_file = scene_dir / "scene.log"
            if log_file.exists():
                lines.append("\nLast 10 lines of scene.log:")
                lines.extend(log_file.read_text().splitlines()[-10:])

        return "\n".join(lines)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_parallel_scene_generation(self):
        """Test generating 2 scenes in parallel.

        Verifies that multiple scenes can be generated concurrently with:
        - Proper scene isolation (separate directories and logs)
        - Asset server communication working correctly
        - Expected output files created for each scene
        """

        # Define test overrides.
        test_overrides = {
            "name": "test_parallel_scene_generation",
            "experiment": {
                "name": "test_parallel_scene_generation",  # Required by experiment
                "num_workers": 2,  # Test parallel execution
                "prompts": [
                    "A room with exactly one chair and one table. No other objects.",
                    "A room with exactly one bed and one nightstand. No other objects.",
                ],
                "output_dir": self.output_dir,
            },
            "max_critique_rounds": 1,  # Faster for testing
            "openai": {
                "model": "gpt-4o-mini",  # Cheaper model for testing
                "service_tier": "default",
                "reasoning_effort": {
                    "planner": "low",  # Faster for tests
                    "designer": "low",
                    "critic": "low",
                },
                "verbosity": {
                    "planner": "low",
                    "designer": "low",
                    "critic": "low",
                },
            },
            "rendering": {
                "image_size": 256,  # Smaller for faster testing
            },
        }

        # Merge configs (base config provides all other values).
        test_config = OmegaConf.merge(self.base_config, test_overrides)

        # Run experiment.
        experiment = IndoorSceneGenerationExperiment(cfg=test_config)
        experiment.generate_scenes()

        # Log generation completion summary.
        scene_count = len(list(self.output_dir.glob("scene_*")))
        logging.info(f"Generation complete. Found {scene_count} scene directories")

        # Verify results - should have 2 scene directories.
        scene_dirs = list(self.output_dir.glob("scene_*"))
        self.assertEqual(len(scene_dirs), 2, "Should generate 2 scenes")

        # Check each scene has required files.
        for scene_dir in scene_dirs:
            self.assertTrue(
                scene_dir.is_dir(), f"Scene directory should exist: {scene_dir}"
            )

            # Check log file exists and has content.
            log_file = scene_dir / "scene.log"
            self.assertTrue(log_file.exists(), f"Scene log should exist: {scene_dir}")

            # Check floor plan files exist.
            floor_plan_file = scene_dir / "room_geometry.sdf"
            self.assertTrue(
                floor_plan_file.exists(),
                f"Floor plan SDF should exist: {scene_dir}",
            )

            # Check generated assets directory structure (required for successful scenes).
            generated_assets_dir = scene_dir / "generated_assets"
            self.assertTrue(
                generated_assets_dir.exists() and generated_assets_dir.is_dir(),
                f"Generated assets directory should exist: {scene_dir}",
            )

            # Check expected subdirectories.
            for subdir in ["images", "geometry", "sdf", "debug"]:
                subdir_path = generated_assets_dir / subdir
                # Check that the subdirectory exists and is a directory.
                self.assertTrue(
                    subdir_path.exists() and subdir_path.is_dir(),
                    f"Generated assets/{subdir} should exist and be directory: "
                    f"{scene_dir}",
                )
                # Check that the subdirectory is not empty.
                try:
                    self.assertTrue(
                        len(list(subdir_path.iterdir())) > 0,
                        f"Generated assets/{subdir} should not be empty: {scene_dir}",
                    )
                except AssertionError:
                    # Dump diagnostics and re-raise with more context.
                    print(self._dump_scene_diagnostics(scene_dir))
                    raise

            # Check scene renders directory.
            scene_renders_dir = scene_dir / "scene_renders"
            self.assertTrue(
                scene_renders_dir.exists() and scene_renders_dir.is_dir(),
                f"Scene renders directory should exist: {scene_dir}",
            )

            # Check final scene state directory.
            scene_states_dir = scene_dir / "scene_states"
            self.assertTrue(
                scene_states_dir.exists() and scene_states_dir.is_dir(),
                f"Scene states directory should exist: {scene_dir}",
            )

            final_scene_dir = scene_states_dir / "final_scene"
            self.assertTrue(
                final_scene_dir.exists() and final_scene_dir.is_dir(),
                f"Final scene directory should exist: {scene_dir}",
            )

            # Check final scene files.
            final_scene_state = final_scene_dir / "scene_state.json"
            final_scene_directive = final_scene_dir / "scene.dmd.yaml"
            self.assertTrue(
                final_scene_state.exists(),
                f"Final scene state should exist: {scene_dir}",
            )
            self.assertTrue(
                final_scene_directive.exists(),
                f"Final scene directive should exist: {scene_dir}",
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
