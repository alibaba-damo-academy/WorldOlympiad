import json
import sys
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenecode.agent_utils.articulated_physics_analyzer import (
    analyze_articulated_physics,
    get_front_axis_from_image_number,
)


class TestArticulatedFrontAxisMapping(unittest.TestCase):
    def test_get_front_axis_with_vertical_views(self) -> None:
        self.assertEqual(get_front_axis_from_image_number(0), "+Z")
        self.assertEqual(get_front_axis_from_image_number(1), "-Z")
        self.assertEqual(get_front_axis_from_image_number(2), "+X")
        self.assertEqual(get_front_axis_from_image_number(3), "+Y")
        self.assertEqual(get_front_axis_from_image_number(4), "-X")
        self.assertEqual(get_front_axis_from_image_number(5), "-Y")


class TestAnalyzeArticulatedPhysicsFallback(unittest.TestCase):
    def test_missing_front_view_image_index_defaults_to_image_5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            urdf_path = tmp_path / "cabinet.urdf"
            urdf_path.write_text('<robot name="cabinet"/>', encoding="utf-8")

            combined_image = tmp_path / "combined_0.png"
            combined_image.write_bytes(b"png")
            link_image = tmp_path / "drawer_0.png"
            link_image.write_bytes(b"png")

            render_result = SimpleNamespace(
                combined_image_paths=[combined_image],
                link_image_paths={"drawer": [link_image]},
                link_dimensions={"drawer": (1.0, 0.5, 0.4)},
            )
            blender_server = MagicMock()
            blender_server.is_running.return_value = True
            blender_server.render_multiview_articulated.return_value = render_result

            response_json = {
                "placement_options": {
                    "on_floor": True,
                    "on_wall": False,
                    "on_ceiling": False,
                    "on_object": False,
                },
                "scale_correct": True,
                "scale_factor": 1.0,
                "total_mass_kg": 3.0,
                "link_analysis": {
                    "drawer": {
                        "material": "wood",
                        "mass_kg": 3.0,
                        "is_static": False,
                        "description": "drawer",
                    }
                },
            }
            cfg = OmegaConf.create(
                {
                    "openai": {
                        "model": "gpt-4o-mini",
                        "reasoning_effort": {"mesh_analysis": "low"},
                        "verbosity": {"mesh_analysis": "low"},
                        "vision_detail": "low",
                    }
                }
            )
            vlm_service = MagicMock()
            vlm_service.create_completion.return_value = json.dumps(response_json)

            with (
                patch(
                    "scenecode.agent_utils.articulated_physics_analyzer.extract_link_meshes",
                    return_value=[
                        SimpleNamespace(
                            link_name="drawer",
                            mesh_paths=[tmp_path / "drawer.obj"],
                            origins=[(0.0, 0.0, 0.0)],
                            world_position=(0.0, 0.0, 0.0),
                            world_rotation=None,
                        )
                    ],
                ),
                patch(
                    "scenecode.agent_utils.articulated_physics_analyzer.prompt_manager.get_prompt",
                    return_value="prompt",
                ),
                patch(
                    "scenecode.agent_utils.articulated_physics_analyzer.encode_image_to_base64",
                    return_value="encoded",
                ),
            ):
                result = analyze_articulated_physics(
                    urdf_path=urdf_path,
                    link_names=["drawer"],
                    bounding_box={"min": [0.0, 0.0, 0.0], "max": [1.0, 0.5, 0.4]},
                    vlm_service=vlm_service,
                    cfg=cfg,
                    blender_server=blender_server,
                    category="cabinet",
                    debug_output_dir=tmp_path / "vlm_images",
                )

        self.assertEqual(result.front_view_image_index, 5)
        self.assertEqual(result.front_axis, "-Y")
        blender_server.render_multiview_articulated.assert_called_once()

    def test_render_multiview_articulated_uses_blender_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            urdf_path = tmp_path / "cabinet.urdf"
            urdf_path.write_text('<robot name="cabinet"/>', encoding="utf-8")

            combined_image = tmp_path / "combined_0.png"
            combined_image.write_bytes(b"png")
            link_image = tmp_path / "drawer_0.png"
            link_image.write_bytes(b"png")

            render_result = SimpleNamespace(
                combined_image_paths=[combined_image],
                link_image_paths={"drawer": [link_image]},
                link_dimensions={"drawer": (1.0, 0.5, 0.4)},
            )

            response_json = {
                "placement_options": {
                    "on_floor": True,
                    "on_wall": False,
                    "on_ceiling": False,
                    "on_object": False,
                },
                "front_view_image_index": 3,
                "scale_correct": True,
                "scale_factor": 1.0,
                "total_mass_kg": 3.0,
                "link_analysis": {
                    "drawer": {
                        "material": "wood",
                        "mass_kg": 3.0,
                        "is_static": False,
                        "description": "drawer",
                    }
                },
            }
            cfg = OmegaConf.create(
                {
                    "openai": {
                        "model": "gpt-4o-mini",
                        "reasoning_effort": {"mesh_analysis": "low"},
                        "verbosity": {"mesh_analysis": "low"},
                        "vision_detail": "low",
                    }
                }
            )
            vlm_service = MagicMock()
            vlm_service.create_completion.return_value = json.dumps(response_json)
            blender_server = MagicMock()
            blender_server.is_running.return_value = True

            def fake_render_multiview_articulated(**kwargs):
                self.assertEqual(kwargs["num_combined_side_views"], 4)
                self.assertEqual(kwargs["num_link_side_views"], 4)
                return render_result

            blender_server.render_multiview_articulated.side_effect = (
                fake_render_multiview_articulated
            )

            with (
                patch(
                    "scenecode.agent_utils.articulated_physics_analyzer.extract_link_meshes",
                    return_value=[
                        SimpleNamespace(
                            link_name="drawer",
                            mesh_paths=[tmp_path / "drawer.obj"],
                            origins=[(0.0, 0.0, 0.0)],
                            world_position=(0.0, 0.0, 0.0),
                            world_rotation=None,
                        )
                    ],
                ),
                patch(
                    "scenecode.agent_utils.blender.renderer.BlenderRenderer",
                    side_effect=AssertionError("embedded BlenderRenderer must not be used"),
                ),
                patch(
                    "scenecode.agent_utils.articulated_physics_analyzer.prompt_manager.get_prompt",
                    return_value="prompt",
                ),
                patch(
                    "scenecode.agent_utils.articulated_physics_analyzer.encode_image_to_base64",
                    return_value="encoded",
                ),
            ):
                result = analyze_articulated_physics(
                    urdf_path=urdf_path,
                    link_names=["drawer"],
                    bounding_box={"min": [0.0, 0.0, 0.0], "max": [1.0, 0.5, 0.4]},
                    vlm_service=vlm_service,
                    cfg=cfg,
                    blender_server=blender_server,
                    category="cabinet",
                    debug_output_dir=tmp_path / "vlm_images",
                )

        self.assertEqual(result.front_view_image_index, 3)
        self.assertEqual(result.front_axis, "+Y")
        blender_server.render_multiview_articulated.assert_called_once()

    def test_requires_running_blender_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            urdf_path = tmp_path / "cabinet.urdf"
            urdf_path.write_text('<robot name="cabinet"/>', encoding="utf-8")
            cfg = OmegaConf.create({"openai": {"model": "gpt-4o-mini"}})
            blender_server = MagicMock()
            blender_server.is_running.return_value = False

            with self.assertRaisesRegex(
                RuntimeError,
                "Articulated physics analysis requires a running BlenderServer",
            ):
                analyze_articulated_physics(
                    urdf_path=urdf_path,
                    link_names=["drawer"],
                    bounding_box={
                        "min": [0.0, 0.0, 0.0],
                        "max": [1.0, 0.5, 0.4],
                    },
                    vlm_service=MagicMock(),
                    cfg=cfg,
                    blender_server=blender_server,
                )


if __name__ == "__main__":
    unittest.main()
