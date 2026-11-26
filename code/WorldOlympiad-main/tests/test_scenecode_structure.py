from __future__ import annotations

import unittest

import numpy as np

from autoresearch.scenecode_structure import (
    evaluate_open_vessel,
    evaluate_through_hole,
    structure_contract,
)


def _box(low: tuple[float, float, float], high: tuple[float, float, float]):
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _combine(*meshes: tuple[np.ndarray, np.ndarray]):
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh_vertices, mesh_faces in meshes:
        vertices.append(mesh_vertices)
        faces.append(mesh_faces + offset)
        offset += len(mesh_vertices)
    return np.concatenate(vertices), np.concatenate(faces)


def _open_box(*, include_front: bool = True):
    panels = [
        _box((-1.0, -0.7, 0.0), (1.0, 0.7, 0.08)),
        _box((-1.0, -0.7, 0.0), (-0.92, 0.7, 0.8)),
        _box((0.92, -0.7, 0.0), (1.0, 0.7, 0.8)),
        _box((-1.0, 0.62, 0.0), (1.0, 0.7, 0.8)),
    ]
    if include_front:
        panels.append(_box((-1.0, -0.7, 0.0), (1.0, -0.62, 0.8)))
    return _combine(*panels)


class SceneCodeStructureGateTest(unittest.TestCase):
    def test_contracts_are_derived_from_visual_identity_not_case_names(self) -> None:
        self.assertEqual(
            structure_contract("open-top rectangular matte box"), "open_vessel"
        )
        self.assertEqual(
            structure_contract("upright cup with an open top rim"), "open_vessel"
        )
        self.assertEqual(
            structure_contract("white block with one vertical hole"), "through_hole"
        )
        self.assertIsNone(structure_contract("solid matte blue cube"))

    def test_closed_cube_cannot_pass_as_open_container(self) -> None:
        vertices, faces = _box((-1.0, -0.7, 0.0), (1.0, 0.7, 0.8))
        result = evaluate_open_vessel(vertices, faces)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["top_open"])

    def test_bottom_and_four_side_directions_pass_open_container(self) -> None:
        vertices, faces = _open_box()
        result = evaluate_open_vessel(vertices, faces)
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()))

    def test_missing_side_fails_open_container(self) -> None:
        vertices, faces = _open_box(include_front=False)
        result = evaluate_open_vessel(vertices, faces)
        self.assertFalse(result["passed"])
        side_checks = [
            result["checks"]["two_sided_enclosure_axis_0"],
            result["checks"]["two_sided_enclosure_axis_1"],
        ]
        self.assertIn(False, side_checks)

    def test_open_center_with_surrounding_body_passes_through_hole(self) -> None:
        vertices, faces = _combine(
            _box((-1.0, -1.0, 0.0), (-0.3, 1.0, 0.8)),
            _box((0.3, -1.0, 0.0), (1.0, 1.0, 0.8)),
            _box((-0.3, -1.0, 0.0), (0.3, -0.3, 0.8)),
            _box((-0.3, 0.3, 0.0), (0.3, 1.0, 0.8)),
        )
        result = evaluate_through_hole(vertices, faces)
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
