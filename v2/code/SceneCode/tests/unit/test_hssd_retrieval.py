import json
import shutil
import tempfile
import unittest

from pathlib import Path

import numpy as np
import trimesh
import yaml

from scenecode.agent_utils.hssd_retrieval.alignment import (
    apply_hssd_alignment_transform,
    compute_rotation_matrix,
)
from scenecode.agent_utils.hssd_retrieval.clip_similarity import (
    compute_clip_similarities,
    filter_meshes_by_category,
)
from scenecode.agent_utils.hssd_retrieval.config import HssdConfig
from scenecode.agent_utils.hssd_retrieval.data_loader import (
    HssdMeshMetadata,
    HssdPreprocessedData,
    construct_hssd_mesh_path,
    load_preprocessed_data,
)


class TestHssdConfig(unittest.TestCase):
    """Test HSSD configuration validation logic."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_missing_data_path(self):
        """Test config validation fails when data path doesn't exist."""
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()

        with self.assertRaises(FileNotFoundError) as cm:
            HssdConfig(
                data_path=self.tmp_path / "nonexistent",
                preprocessed_path=preprocessed_path,
            )
        self.assertIn("HSSD data path does not exist", str(cm.exception))

    def test_config_missing_preprocessed_path(self):
        """Test config validation fails when preprocessed path doesn't exist."""
        data_path = self.tmp_path / "hssd-models"
        data_path.mkdir()

        with self.assertRaises(FileNotFoundError) as cm:
            HssdConfig(
                data_path=data_path,
                preprocessed_path=self.tmp_path / "nonexistent",
            )
        self.assertIn("Preprocessed data path does not exist", str(cm.exception))


class TestMeshPaths(unittest.TestCase):
    """Test HSSD mesh path construction logic (regular vs decomposed)."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_construct_regular_mesh_path(self):
        """Test path construction uses first character for regular meshes."""
        hssd_dir = self.tmp_path / "hssd-models"
        mesh_id = "abc123def456"

        expected_path = hssd_dir / "objects" / "a" / f"{mesh_id}.glb"
        expected_path.parent.mkdir(parents=True)
        expected_path.touch()

        path = construct_hssd_mesh_path(hssd_dir, mesh_id)
        self.assertEqual(path, expected_path)

    def test_construct_mesh_path_not_found(self):
        """Test validation fails when mesh file doesn't exist."""
        hssd_dir = self.tmp_path / "hssd-models"
        mesh_id = "nonexistent123"

        with self.assertRaises(FileNotFoundError) as cm:
            construct_hssd_mesh_path(hssd_dir, mesh_id)
        self.assertIn("HSSD mesh not found", str(cm.exception))


class TestAlignment(unittest.TestCase):
    """Test mesh alignment algorithms and coordinate system transforms."""

    def test_compute_rotation_matrix_identity(self):
        """Test rotation matrix for already-canonical orientation."""
        source_up = np.array([0.0, 1.0, 0.0])
        source_front = np.array([0.0, 0.0, 1.0])

        rotation = compute_rotation_matrix(source_up, source_front)

        np.testing.assert_array_almost_equal(rotation, np.eye(3), decimal=6)

    def test_compute_rotation_matrix_90deg(self):
        """Test rotation matrix for 90-degree rotation."""
        source_up = np.array([1.0, 0.0, 0.0])
        source_front = np.array([0.0, 1.0, 0.0])

        rotation = compute_rotation_matrix(source_up, source_front)

        test_vector = source_up
        rotated = rotation @ test_vector
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=6)

    def test_apply_hssd_alignment_skips_when_already_canonical(self):
        """Test alignment logic skips transform when mesh is already canonical."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        metadata = HssdMeshMetadata(
            mesh_id="test123",
            name="Test Box",
            up="0,1,0",
            front="0,0,1",
            wordnet_key="box.n.01",
        )

        aligned = apply_hssd_alignment_transform(mesh, metadata)

        np.testing.assert_array_almost_equal(mesh.vertices, aligned.vertices, decimal=6)

    def test_apply_hssd_alignment_rotates_when_needed(self):
        """Test alignment applies rotation when orientation differs from canonical."""
        mesh = trimesh.creation.box(extents=[2, 1, 1])
        metadata = HssdMeshMetadata(
            mesh_id="test456",
            name="Test Box",
            up="1,0,0",
            front="0,1,0",
            wordnet_key="box.n.01",
        )

        aligned = apply_hssd_alignment_transform(mesh, metadata)

        self.assertFalse(np.allclose(mesh.vertices, aligned.vertices))

    def test_apply_hssd_alignment_skips_when_empty_vectors(self):
        """Test alignment skips when metadata has empty up/front vectors."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        original_vertices = mesh.vertices.copy()

        # Test with empty strings (79.8% of HSSD dataset).
        metadata_empty = HssdMeshMetadata(
            mesh_id="test789",
            name="Test Box",
            up="",
            front="",
            wordnet_key="box.n.01",
        )

        aligned = apply_hssd_alignment_transform(mesh, metadata_empty)

        # Mesh should be returned unchanged.
        np.testing.assert_array_almost_equal(
            original_vertices, aligned.vertices, decimal=6
        )


class TestClipSimilarity(unittest.TestCase):
    """Test CLIP similarity computation and filtering algorithms."""

    def test_compute_clip_similarities(self):
        """Test cosine similarity computation ranks embeddings correctly."""
        text_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        mesh_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.7, 0.7, 0.0, 0.0],
            ]
        )
        mesh_indices = [0, 1, 2]

        similarities = compute_clip_similarities(
            text_embedding, mesh_embeddings, mesh_indices
        )

        self.assertGreater(similarities[0], similarities[1])
        self.assertGreater(similarities[0], similarities[2])
        self.assertGreater(similarities[2], similarities[1])

    def test_filter_meshes_by_category(self):
        """Test filtering returns correct mesh indices for object category."""
        preprocessed_data = HssdPreprocessedData(
            metadata_by_wordnet={
                "chair.n.01": [
                    HssdMeshMetadata(
                        mesh_id="mesh1",
                        name="Chair 1",
                        up="0,1,0",
                        front="0,0,1",
                        wordnet_key="chair.n.01",
                    )
                ],
                "table.n.01": [
                    HssdMeshMetadata(
                        mesh_id="mesh2",
                        name="Table 1",
                        up="0,1,0",
                        front="0,0,1",
                        wordnet_key="table.n.01",
                    )
                ],
            },
            clip_embeddings=np.zeros((2, 512)),
            embedding_index=["mesh1", "mesh2"],
            object_categories={
                "large_objects": ["chair.n.01", "table.n.01"],
                "small_objects": [],
            },
        )

        indices = filter_meshes_by_category(preprocessed_data, "large_objects")

        self.assertEqual(len(indices), 2)
        self.assertIn(0, indices)
        self.assertIn(1, indices)

    def test_filter_meshes_invalid_category(self):
        """Test filtering returns empty list for invalid category."""
        preprocessed_data = HssdPreprocessedData(
            metadata_by_wordnet={},
            clip_embeddings=np.zeros((0, 512)),
            embedding_index=[],
            object_categories={"large_objects": []},
        )

        indices = filter_meshes_by_category(preprocessed_data, "invalid_category")

        self.assertEqual(len(indices), 0)


class TestDataLoader(unittest.TestCase):
    """Test file I/O and data loading logic."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_preprocessed_data(self):
        """Test loading preprocessed data from files."""
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()

        index_data = {
            "chair.n.01": [
                {
                    "id": "mesh1",
                    "name": "Chair",
                    "up": "0,1,0",
                    "front": "0,0,1",
                }
            ]
        }
        with open(preprocessed_path / "hssd_wnsynsetkey_index.json", "w") as f:
            json.dump(index_data, f)

        embeddings = np.random.rand(1, 512)
        np.save(preprocessed_path / "clip_hssd_embeddings.npy", embeddings)

        embedding_index = ["mesh1"]
        with open(preprocessed_path / "clip_hssd_embeddings_index.yaml", "w") as f:
            yaml.dump(embedding_index, f)

        categories = {
            "available_categories": ["large_objects"],
            "large_objects": ["chair.n.01"],
        }
        with open(preprocessed_path / "object_categories.json", "w") as f:
            json.dump(categories, f)

        data = load_preprocessed_data(preprocessed_path)

        self.assertIn("chair.n.01", data.metadata_by_wordnet)
        self.assertEqual(len(data.metadata_by_wordnet["chair.n.01"]), 1)
        self.assertEqual(data.clip_embeddings.shape, (1, 512))
        self.assertEqual(data.embedding_index, ["mesh1"])
        self.assertIn("large_objects", data.object_categories)

    def test_load_preprocessed_data_missing_files(self):
        """Test validation fails when required files are missing."""
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()

        with self.assertRaises(FileNotFoundError):
            load_preprocessed_data(preprocessed_path)


if __name__ == "__main__":
    unittest.main()
