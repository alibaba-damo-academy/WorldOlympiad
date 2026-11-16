from __future__ import annotations

import numpy as np
import pytest

from autoresearch.compiler import gltf_y_up_to_worldir_z_up_transform


def test_gltf_y_up_becomes_worldir_z_up() -> None:
    transform = gltf_y_up_to_worldir_z_up_transform()
    source = np.asarray(
        [
            [0.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 3.0, 1.0],
        ]
    )
    converted = (transform @ source.T).T
    assert converted[0, :3] == pytest.approx([0.0, 0.0, 2.0])
    assert converted[1, :3] == pytest.approx([0.0, -3.0, 0.0])
    rotation = transform[:3, :3]
    assert rotation @ rotation.T == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
