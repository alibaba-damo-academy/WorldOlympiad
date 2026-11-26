"""Native SceneCode/Drake checks; run with its Python and staged runtime on PYTHONPATH."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import bpy  # Initialize mathutils before the SceneCode imports.
    import trimesh
    from scenecode.agent_utils import support_surface_extraction as extraction
except ModuleNotFoundError:
    extraction = None


@unittest.skipIf(extraction is None, "requires the native SceneCode runtime")
class SdfSupportVisualMeshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def mesh(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        box = trimesh.creation.box(extents=[.8,.6,.2])
        box.apply_translation([0,0,.1])
        box.apply_transform([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
        for name, data in trimesh.exchange.gltf.export_gltf(trimesh.Scene(box)).items():
            (folder/name).write_bytes(data)
        return folder/'model.gltf'

    def sdf(self, uri, second=False):
        def link(name, x):
            return f'<link name="{name}"><pose>{x} 0 .3 0 0 0</pose><visual name="visual"><pose>0 0 .1 0 0 0</pose><geometry><mesh><uri>{uri}</uri><scale>2 2 2</scale></mesh></geometry></visual></link>'
        text = '<sdf version="1.7"><model name="fixture"><static>true</static><pose>1 2 3 0 0 .4</pose>'
        text += link('top', 0) + (link('other', 2) if second else '')
        path = self.root/'fixture.sdf'
        path.write_text(text+'</model></sdf>')
        return path

    def test_sdf_scale_and_link_visual_frames_are_model_local(self):
        mesh = self.mesh(self.root/'blender_output/parts')
        sdf = self.sdf(mesh.relative_to(self.root))
        rows = extraction._load_sdf_visual_link_meshes(sdf)
        self.assertEqual([row[1] for row in rows], ['top'])
        self.assertAlmostEqual(rows[0][2].bounds[1,2], .8, places=5)
        surfaces = extraction.extract_support_surfaces_articulated(self.root, sdf_path=sdf)
        self.assertEqual(len(surfaces), 1)
        self.assertEqual(surfaces[0].link_name, 'top')
        self.assertAlmostEqual(surfaces[0].transform.translation()[2], .81, places=5)

    def test_shared_mesh_uri_keeps_distinct_link_poses(self):
        mesh = self.mesh(self.root/'blender_output/parts')
        sdf = self.sdf(mesh.relative_to(self.root), second=True)
        rows = extraction._load_sdf_visual_link_meshes(sdf)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1][2].centroid[0]-rows[0][2].centroid[0], 2, places=5)
        surfaces = extraction.extract_support_surfaces_articulated(self.root, sdf_path=sdf)
        self.assertEqual({s.link_name for s in surfaces}, {'top','other'})

    def test_legacy_mesh_layout_keeps_existing_extraction_path(self):
        mesh = self.mesh(self.root)
        mesh.rename(self.root/'legacy_combined.gltf')
        with patch.object(extraction, '_load_sdf_visual_link_meshes', side_effect=AssertionError('legacy path changed')):
            surfaces = extraction.extract_support_surfaces_articulated(self.root)
        self.assertEqual(len(surfaces), 1)
        self.assertEqual(surfaces[0].link_name, 'legacy')
        self.assertAlmostEqual(surfaces[0].transform.translation()[2], .21, places=5)

    def test_missing_declared_mesh_stays_a_real_input_error(self):
        sdf = self.sdf('blender_output/parts/missing.gltf')
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            extraction.extract_support_surfaces_articulated(self.root, sdf_path=sdf)


    def test_existing_combined_mesh_fallback_is_preserved(self):
        mesh = self.mesh(self.root)
        mesh.rename(self.root/'combined_scene.gltf')
        sdf = self.sdf('combined_scene.gltf')
        with patch.object(extraction, '_load_sdf_visual_link_meshes', side_effect=AssertionError('combined fallback changed')):
            surfaces = extraction.extract_support_surfaces_articulated(self.root, sdf_path=sdf)
        self.assertEqual(len(surfaces), 1)
        self.assertIsNone(surfaces[0].link_name)
        self.assertAlmostEqual(surfaces[0].transform.translation()[2], .21, places=5)


    def test_codeobject_bbox_uses_declared_scale_and_model_frame(self):
        from scenecode.agent_utils.urdf_to_sdf import compute_sdf_bounding_box
        mesh = self.mesh(self.root/'blender_output/parts')
        sdf = self.sdf(mesh.relative_to(self.root))
        lo, hi, centre = compute_sdf_bounding_box(sdf, scale_factor=2)
        self.assertAlmostEqual(lo[2], .4, places=5)
        self.assertAlmostEqual(hi[2], .8, places=5)
        self.assertAlmostEqual(centre[2], .6, places=5)





    def test_codeobject_bbox_with_empty_legacy_visual_directory(self):
        from scenecode.agent_utils.urdf_to_sdf import compute_sdf_bounding_box
        mesh = self.mesh(self.root/'blender_output/parts')
        (self.root/'visual').mkdir()
        sdf = self.sdf(mesh.relative_to(self.root))
        lo, hi, _ = compute_sdf_bounding_box(sdf)
        self.assertAlmostEqual(lo[2], .4, places=5)
        self.assertAlmostEqual(hi[2], .8, places=5)


    def scene_object(self, sdf, lo, hi):
        import numpy as np
        from pydrake.all import RigidTransform
        from scenecode.agent_utils.room import SceneObject, ObjectType, UniqueID
        return SceneObject(UniqueID('fixture_0'), ObjectType.FURNITURE, 'fixture',
            'fixture', RigidTransform(), sdf_path=sdf,
            metadata={'asset_source':'code_articulated'},
            bbox_min=np.array(lo), bbox_max=np.array(hi))

    def test_zero_bounds_are_recovered_at_object_construction(self):
        mesh = self.mesh(self.root/'blender_output/parts')
        sdf = self.sdf(mesh.relative_to(self.root))
        obj = self.scene_object(sdf, [0,0,0], [0,0,0])
        self.assertAlmostEqual(obj.bbox_min[2], .4, places=5)
        self.assertAlmostEqual(obj.bbox_max[2], .8, places=5)

    def test_valid_cached_bounds_keep_existing_path(self):
        obj = self.scene_object(self.root/'unread.sdf', [-1,-1,0], [1,1,2])
        self.assertEqual(obj.bbox_max.tolist(), [1,1,2])

    def test_loading_zero_bound_checkpoint_recovers_metadata_without_changing_record(self):
        from scenecode.agent_utils.room import SceneObject
        mesh = self.mesh(self.root/'blender_output/parts')
        sdf = self.sdf(mesh.relative_to(self.root))
        obj = self.scene_object(sdf, [0,0,0], [0,0,0])
        record = obj.to_dict(self.root)
        record['bbox_min'] = [0,0,0]
        record['bbox_max'] = [0,0,0]
        restored = SceneObject.from_dict(record, scene_dir=self.root)
        self.assertAlmostEqual(restored.bbox_min[2], .4, places=5)
        self.assertAlmostEqual(restored.bbox_max[2], .8, places=5)
        self.assertEqual(record['bbox_max'], [0,0,0])
