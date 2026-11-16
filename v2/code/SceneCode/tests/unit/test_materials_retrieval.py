"""Unit tests for materials retrieval server."""

import shutil
import tempfile
import unittest

from pathlib import Path

import numpy as np
import yaml

from scenecode.agent_utils.materials_retrieval_server.config import MaterialsConfig
from scenecode.agent_utils.materials_retrieval_server.data_loader import (
    MaterialMetadata,
    MaterialsPreprocessedData,
    load_preprocessed_data,
)
from scenecode.agent_utils.materials_retrieval_server.dataclasses import (
    MaterialRetrievalResult,
    MaterialsRetrievalServerRequest,
    MaterialsRetrievalServerResponse,
)


class TestMaterialsConfig(unittest.TestCase):
    """Test MaterialsConfig validation logic."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_missing_data_path_disables(self):
        """Test config validation disables when data path doesn't exist."""
        embeddings_path = self.tmp_path / "embeddings"
        embeddings_path.mkdir()

        config = MaterialsConfig(
            data_path=self.tmp_path / "nonexistent",
            embeddings_path=embeddings_path,
        )
        self.assertFalse(config.enabled)

    def test_config_missing_embeddings_path_disables(self):
        """Test config validation disables when embeddings path doesn't exist."""
        data_path = self.tmp_path / "materials"
        data_path.mkdir()

        config = MaterialsConfig(
            data_path=data_path,
            embeddings_path=self.tmp_path / "nonexistent",
        )
        self.assertFalse(config.enabled)

    def test_config_missing_embedding_files_disables(self):
        """Test config validation disables when required files are missing."""
        data_path = self.tmp_path / "materials"
        data_path.mkdir()
        embeddings_path = self.tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Create only some of the required files.
        np.save(embeddings_path / "clip_embeddings.npy", np.zeros((10, 1024)))

        config = MaterialsConfig(
            data_path=data_path,
            embeddings_path=embeddings_path,
        )
        self.assertFalse(config.enabled)

    def test_config_valid_paths_enabled(self):
        """Test config validation enables when all paths and files exist."""
        data_path = self.tmp_path / "materials"
        data_path.mkdir()
        embeddings_path = self.tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Create all required files.
        np.save(embeddings_path / "clip_embeddings.npy", np.zeros((10, 1024)))
        with open(embeddings_path / "embedding_index.yaml", "w") as f:
            yaml.dump(["Material001"], f)
        with open(embeddings_path / "metadata_index.yaml", "w") as f:
            yaml.dump({"Material001": {"category": "Test", "tags": []}}, f)

        config = MaterialsConfig(
            data_path=data_path,
            embeddings_path=embeddings_path,
        )
        self.assertTrue(config.enabled)


class TestMaterialsPreprocessedData(unittest.TestCase):
    """Test MaterialsPreprocessedData functionality."""

    def test_get_metadata(self):
        """Test metadata lookup returns correct value."""
        metadata = MaterialMetadata(
            material_id="Bricks001",
            category="Bricks",
            tags=["red", "rough"],
        )
        data = MaterialsPreprocessedData(
            metadata_by_id={"Bricks001": metadata},
            clip_embeddings=np.zeros((1, 1024)),
            embedding_index=["Bricks001"],
        )

        result = data.get_metadata("Bricks001")
        self.assertEqual(result.material_id, "Bricks001")
        self.assertEqual(result.category, "Bricks")

    def test_get_metadata_not_found(self):
        """Test metadata lookup returns None for missing material."""
        data = MaterialsPreprocessedData(
            metadata_by_id={},
            clip_embeddings=np.zeros((0, 1024)),
            embedding_index=[],
        )

        result = data.get_metadata("nonexistent")
        self.assertIsNone(result)

    def test_get_embedding_index(self):
        """Test embedding index lookup returns correct value."""
        data = MaterialsPreprocessedData(
            metadata_by_id={},
            clip_embeddings=np.zeros((2, 1024)),
            embedding_index=["Material001", "Material002"],
        )

        self.assertEqual(data.get_embedding_index("Material001"), 0)
        self.assertEqual(data.get_embedding_index("Material002"), 1)
        self.assertIsNone(data.get_embedding_index("nonexistent"))


class TestDataLoader(unittest.TestCase):
    """Test data loading logic."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_preprocessed_data(self):
        """Test loading preprocessed data from files."""
        data_path = self.tmp_path / "materials"
        data_path.mkdir()
        embeddings_path = self.tmp_path / "embeddings"
        embeddings_path.mkdir()

        # Create test data.
        embeddings = np.random.rand(2, 1024).astype(np.float32)
        np.save(embeddings_path / "clip_embeddings.npy", embeddings)

        embedding_index = ["Bricks001", "Wood094"]
        with open(embeddings_path / "embedding_index.yaml", "w") as f:
            yaml.dump(embedding_index, f)

        metadata_index = {
            "Bricks001": {"category": "Bricks", "tags": ["red", "rough"]},
            "Wood094": {"category": "Wood Floor", "tags": ["brown", "smooth"]},
        }
        with open(embeddings_path / "metadata_index.yaml", "w") as f:
            yaml.dump(metadata_index, f)

        config = MaterialsConfig(data_path=data_path, embeddings_path=embeddings_path)

        data = load_preprocessed_data(config)

        self.assertIsNotNone(data)
        self.assertEqual(len(data.metadata_by_id), 2)
        self.assertEqual(data.clip_embeddings.shape, (2, 1024))
        self.assertEqual(data.embedding_index, ["Bricks001", "Wood094"])
        self.assertEqual(data.get_metadata("Bricks001").category, "Bricks")

    def test_load_preprocessed_data_disabled_config(self):
        """Test loading returns None when config is disabled."""
        config = MaterialsConfig(
            data_path=self.tmp_path / "nonexistent",
            embeddings_path=self.tmp_path / "nonexistent",
        )
        self.assertFalse(config.enabled)

        data = load_preprocessed_data(config)
        self.assertIsNone(data)


class TestDataclasses(unittest.TestCase):
    """Test request/response dataclass serialization."""

    def test_request_to_dict(self):
        """Test request serialization to dictionary."""
        request = MaterialsRetrievalServerRequest(
            material_description="warm hardwood floor",
            output_dir="/tmp/output",
            scene_id="scene_001",
            num_candidates=3,
        )

        result = request.to_dict()

        self.assertEqual(result["material_description"], "warm hardwood floor")
        self.assertEqual(result["output_dir"], "/tmp/output")
        self.assertEqual(result["scene_id"], "scene_001")
        self.assertEqual(result["num_candidates"], 3)

    def test_request_to_json(self):
        """Test request serialization to JSON."""
        request = MaterialsRetrievalServerRequest(
            material_description="red brick wall",
            output_dir="/tmp/output",
        )

        json_str = request.to_json()

        self.assertIn("red brick wall", json_str)
        self.assertIn("/tmp/output", json_str)

    def test_result_from_dict(self):
        """Test result deserialization from dictionary."""
        data = {
            "material_path": "/tmp/output/Bricks001",
            "material_id": "Bricks001",
            "similarity_score": 0.85,
            "category": "Bricks",
            "color_texture": "/tmp/output/Bricks001/Bricks001_2K-JPG_Color.jpg",
            "normal_texture": "/tmp/output/Bricks001/Bricks001_2K-JPG_NormalGL.jpg",
            "roughness_texture": "/tmp/output/Bricks001/Bricks001_2K-JPG_Roughness.jpg",
        }

        result = MaterialRetrievalResult.from_dict(data)

        self.assertEqual(result.material_id, "Bricks001")
        self.assertEqual(result.similarity_score, 0.85)
        self.assertEqual(result.category, "Bricks")

    def test_response_serialization_roundtrip(self):
        """Test response serializes and deserializes correctly."""
        result = MaterialRetrievalResult(
            material_path="/tmp/output/Wood094",
            material_id="Wood094",
            similarity_score=0.92,
            category="Wood Floor",
            color_texture="/tmp/output/Wood094/Wood094_2K-JPG_Color.jpg",
            normal_texture="/tmp/output/Wood094/Wood094_2K-JPG_NormalGL.jpg",
            roughness_texture="/tmp/output/Wood094/Wood094_2K-JPG_Roughness.jpg",
        )
        response = MaterialsRetrievalServerResponse(
            results=[result],
            query_description="smooth wooden floor",
        )

        # Serialize and deserialize.
        response_dict = response.to_dict()
        restored = MaterialsRetrievalServerResponse.from_dict(response_dict)

        self.assertEqual(len(restored.results), 1)
        self.assertEqual(restored.results[0].material_id, "Wood094")
        self.assertEqual(restored.query_description, "smooth wooden floor")


if __name__ == "__main__":
    unittest.main()
