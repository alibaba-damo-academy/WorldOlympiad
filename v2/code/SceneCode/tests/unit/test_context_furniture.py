"""Unit tests for context furniture selection.

These tests focus on key contracts:
1. AABB edge distance computation works correctly
2. VLM response validation filters out invalid IDs
"""

import unittest

import numpy as np

from scenecode.agent_utils.room import UniqueID
from scenecode.agent_utils.scene_analyzer import (
    SceneAnalyzer,
    _compute_aabb_edge_distance,
)


class TestComputeAABBEdgeDistance(unittest.TestCase):
    """Tests for _compute_aabb_edge_distance function."""

    def test_separated_boxes_positive_distance(self):
        """Two separated boxes should have positive distance."""
        # Box A: [0,0,0] to [1,1,1].
        bounds_a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        # Box B: [3,0,0] to [4,1,1] - 2 meters gap in X.
        bounds_b = (np.array([3.0, 0.0, 0.0]), np.array([4.0, 1.0, 1.0]))

        distance = _compute_aabb_edge_distance(bounds_a, bounds_b)
        self.assertAlmostEqual(distance, 2.0)

    def test_overlapping_boxes_zero_distance(self):
        """Overlapping boxes should have zero distance."""
        # Box A: [0,0,0] to [2,2,2].
        bounds_a = (np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))
        # Box B: [1,1,0] to [3,3,2] - overlaps in XY.
        bounds_b = (np.array([1.0, 1.0, 0.0]), np.array([3.0, 3.0, 2.0]))

        distance = _compute_aabb_edge_distance(bounds_a, bounds_b)
        assert distance == 0.0

    def test_adjacent_boxes_zero_distance(self):
        """Adjacent boxes (touching edges) should have zero distance."""
        # Box A: [0,0,0] to [1,1,1].
        bounds_a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        # Box B: [1,0,0] to [2,1,1] - touching at X=1.
        bounds_b = (np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))

        distance = _compute_aabb_edge_distance(bounds_a, bounds_b)
        assert distance == 0.0

    def test_diagonal_separation(self):
        """Boxes separated diagonally should have Euclidean distance."""
        # Box A: [0,0,0] to [1,1,1].
        bounds_a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        # Box B: [2,2,0] to [3,3,1] - 1m gap in both X and Y.
        bounds_b = (np.array([2.0, 2.0, 0.0]), np.array([3.0, 3.0, 1.0]))

        distance = _compute_aabb_edge_distance(bounds_a, bounds_b)
        # Distance = sqrt(1^2 + 1^2) = sqrt(2).
        self.assertAlmostEqual(distance, np.sqrt(2.0))

    def test_z_axis_ignored(self):
        """Z-axis separation should be ignored (XY plane only)."""
        # Box A: [0,0,0] to [1,1,1].
        bounds_a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        # Box B: [0,0,10] to [1,1,11] - same XY, 9m Z gap.
        bounds_b = (np.array([0.0, 0.0, 10.0]), np.array([1.0, 1.0, 11.0]))

        distance = _compute_aabb_edge_distance(bounds_a, bounds_b)
        # Z is ignored, so boxes overlap in XY -> distance = 0.
        assert distance == 0.0


class TestParseContextSelectionResponse(unittest.TestCase):
    """Tests for SceneAnalyzer._parse_context_selection_response method.

    This tests the key contract: VLM responses are validated against the
    geometric candidates, filtering out any hallucinated or invalid IDs.
    """

    def _call_parse_response(self, vlm_response: dict, all_candidates: dict) -> dict:
        """Helper to call the private method without needing a full instance."""
        # The method only uses self for logging, so we can call with None.
        return SceneAnalyzer._parse_context_selection_response(
            None, vlm_response, all_candidates
        )

    def test_valid_ids_returned(self):
        """Valid context IDs should be returned."""
        vlm_response = {
            "context_selections": [
                {
                    "furniture_id": "dining_table_0",
                    "context_furniture_ids": ["chair_0", "chair_1"],
                    "reasoning": "Chairs indicate place settings",
                }
            ]
        }
        all_candidates = {
            "dining_table_0": [
                {"furniture_id": "chair_0", "name": "Chair", "distance_m": 0.5},
                {"furniture_id": "chair_1", "name": "Chair", "distance_m": 0.5},
                {"furniture_id": "chair_2", "name": "Chair", "distance_m": 0.5},
            ]
        }

        result = self._call_parse_response(vlm_response, all_candidates)

        assert UniqueID("dining_table_0") in result
        assert len(result[UniqueID("dining_table_0")]) == 2
        assert UniqueID("chair_0") in result[UniqueID("dining_table_0")]
        assert UniqueID("chair_1") in result[UniqueID("dining_table_0")]

    def test_invalid_ids_filtered_out(self):
        """Invalid context IDs not in candidates should be filtered out."""
        vlm_response = {
            "context_selections": [
                {
                    "furniture_id": "dining_table_0",
                    "context_furniture_ids": [
                        "chair_0",
                        "hallucinated_chair",
                        "nonexistent_sofa",
                    ],
                    "reasoning": "VLM hallucinated some furniture",
                }
            ]
        }
        # Only chair_0 is a valid candidate.
        all_candidates = {
            "dining_table_0": [
                {"furniture_id": "chair_0", "name": "Chair", "distance_m": 0.5},
            ]
        }

        result = self._call_parse_response(vlm_response, all_candidates)

        # Only valid ID (chair_0) should be in result.
        assert UniqueID("dining_table_0") in result
        assert len(result[UniqueID("dining_table_0")]) == 1
        assert UniqueID("chair_0") in result[UniqueID("dining_table_0")]

    def test_empty_context_not_added(self):
        """Furniture with empty context after filtering should not be added."""
        vlm_response = {
            "context_selections": [
                {
                    "furniture_id": "bookshelf_0",
                    "context_furniture_ids": ["hallucinated_chair"],
                    "reasoning": "No valid context",
                }
            ]
        }
        # No valid candidates for bookshelf_0.
        all_candidates = {
            "bookshelf_0": [
                {"furniture_id": "real_chair", "name": "Chair", "distance_m": 0.5},
            ]
        }

        result = self._call_parse_response(vlm_response, all_candidates)

        # bookshelf_0 should not be in result (empty context after filtering).
        assert UniqueID("bookshelf_0") not in result

    def test_missing_furniture_id_skipped(self):
        """Entries without furniture_id should be skipped."""
        vlm_response = {
            "context_selections": [
                {
                    # Missing furniture_id.
                    "context_furniture_ids": ["chair_0"],
                    "reasoning": "Missing ID",
                }
            ]
        }
        all_candidates = {
            "dining_table_0": [
                {"furniture_id": "chair_0", "name": "Chair", "distance_m": 0.5},
            ]
        }

        result = self._call_parse_response(vlm_response, all_candidates)

        # Should return empty dict since entry is invalid.
        assert len(result) == 0


if __name__ == "__main__":
    unittest.main()
