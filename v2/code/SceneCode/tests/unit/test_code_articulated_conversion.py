import json
import shutil
import sys
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenecode.agent_utils.code_articulated_conversion import (
    compute_agent_scale_factor,
    convert_generated_articulated_urdf,
)
from scenecode.agent_utils.articulated_physics_analyzer import PlacementOptions


class TestCodeArticulatedConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.urdf_path = self.temp_dir / 'cabinet.urdf'
        self.urdf_path.write_text('<robot name="cabinet"/>', encoding='utf-8')
        self.cfg = OmegaConf.create({'openai': {'model': 'gpt-4o-mini'}})

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_compute_agent_scale_factor_uses_desired_dimensions(self) -> None:
        scale = compute_agent_scale_factor(
            np.array([1.0, 2.0, 3.0]),
            [2.0, 4.0, 6.0],
        )
        self.assertEqual(scale, 2.0)

    def test_convert_generated_articulated_urdf_uses_vlm_placement_and_agent_scale(self) -> None:
        analysis = SimpleNamespace(
            front_axis='-Y',
            placement_options=PlacementOptions(on_floor=False, on_wall=True),
            link_materials={'body': 'wood'},
            link_masses={'body': 5.0},
            total_mass_kg=5.0,
            link_descriptions={'body': 'cabinet body'},
            front_view_image_index=5,
            object_description='storage cabinet',
        )

        output_path = self.temp_dir / 'cabinet.sdf'
        output_path.write_text('<sdf version="1.7"><model name="cabinet"/></sdf>', encoding='utf-8')

        blender_server = MagicMock()
        with (
            patch(
                'scenecode.agent_utils.code_articulated_conversion.validate_urdf_meshes',
                return_value=([Path('mesh.obj')], []),
            ),
            patch(
                'scenecode.agent_utils.code_articulated_conversion.parse_urdf',
                return_value=SimpleNamespace(),
            ),
            patch(
                'scenecode.agent_utils.code_articulated_conversion.compute_articulated_bounding_box',
                return_value=(
                    np.array([0.0, 0.0, 0.0]),
                    np.array([1.0, 0.5, 0.5]),
                    np.array([0.5, 0.25, 0.25]),
                ),
            ),
            patch(
                'scenecode.agent_utils.code_articulated_conversion.extract_link_meshes',
                return_value=[SimpleNamespace(link_name='body')],
            ),
            patch(
                'scenecode.agent_utils.code_articulated_conversion.analyze_articulated_physics',
                return_value=analysis,
            ) as mock_analyze,
            patch(
                'scenecode.agent_utils.code_articulated_conversion.compute_link_physics_from_meshes',
                return_value={'body': MagicMock()},
            ),
            patch(
                'scenecode.agent_utils.code_articulated_conversion.convert_urdf_to_sdf',
                return_value=output_path,
            ) as mock_convert,
            patch(
                'scenecode.agent_utils.code_articulated_conversion.compute_sdf_bounding_box',
                return_value=(
                    np.array([-0.5, -0.25, 0.0]),
                    np.array([0.5, 0.25, 1.0]),
                    np.array([0.0, 0.0, 0.5]),
                ),
            ),
            patch(
                'scenecode.agent_utils.code_articulated_conversion.update_sdf_model_pose'
            ) as mock_update_pose,
            patch(
                'scenecode.agent_utils.code_articulated_conversion.validate_with_drake',
                return_value=True,
            ),
        ):
            result = convert_generated_articulated_urdf(
                urdf_path=self.urdf_path,
                collision_client=MagicMock(),
                vlm_service=MagicMock(),
                cfg=self.cfg,
                blender_server=blender_server,
                desired_dimensions=[2.0, 1.0, 1.0],
                output_path=output_path,
                model_name='cabinet',
            )

        self.assertEqual(result.placement_type, 'wall')
        self.assertAlmostEqual(result.scale_factor, 2.0)
        self.assertEqual(result.sdf_path, output_path)
        mock_analyze.assert_called_once()
        self.assertIs(mock_analyze.call_args.kwargs['blender_server'], blender_server)
        mock_convert.assert_called_once()
        self.assertAlmostEqual(mock_convert.call_args.kwargs['scale_factor'], 2.0)
        mock_update_pose.assert_called_once()
        model_pose = mock_update_pose.call_args.args[1]
        self.assertAlmostEqual(model_pose[5], np.pi)

        analysis_json = json.loads((self.temp_dir / 'analysis.json').read_text(encoding='utf-8'))
        self.assertEqual(analysis_json['placement_type'], 'wall')
        self.assertEqual(analysis_json['placement_options']['on_floor'], False)
        self.assertEqual(analysis_json['placement_options']['on_wall'], True)
        self.assertEqual(analysis_json['raw_vlm_placement_options']['on_wall'], True)
        self.assertEqual(analysis_json['scale_source'], 'agent_dimensions')
        self.assertEqual(analysis_json['desired_dimensions'], [2.0, 1.0, 1.0])
        self.assertEqual(analysis_json['front_axis'], '-Y')
        self.assertEqual(analysis_json['front_view_image_index'], 5)


if __name__ == '__main__':
    unittest.main()
