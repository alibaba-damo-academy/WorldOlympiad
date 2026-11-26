import numpy as np

from autoresearch.scenecode_primitives import (
    primitive_color,
    primitive_mesh,
    solid_primitive_contract,
)


def test_contract_is_narrow_and_rejects_open_shapes() -> None:
    assert solid_primitive_contract("solid orange short cylinder with no opening") == "cylinder"
    assert solid_primitive_contract("single-piece blue cube") == "cuboid"
    assert solid_primitive_contract("solid red sphere") == "sphere"
    assert solid_primitive_contract("orange cylinder") is None
    assert solid_primitive_contract("solid hollow cylinder tube") is None
    assert solid_primitive_contract("solid-looking open-top box") is None


def test_extended_color_vocabulary() -> None:
    assert primitive_color("matte vivid-teal solid cylinder") == ("teal", (25, 155, 150))
    assert primitive_color("uncolored solid cube")[0] == "gray"


def test_cylinder_has_requested_bounds_and_closed_caps() -> None:
    vertices, normals, uvs, indices = primitive_mesh("cylinder", [0.10, 0.08, 0.06])
    assert np.allclose(vertices.min(axis=0), [-0.05, -0.04, 0.0], atol=1e-6)
    assert np.allclose(vertices.max(axis=0), [0.05, 0.04, 0.06], atol=1e-6)
    assert np.any(np.all(normals == [0, 0, 1], axis=1))
    assert np.any(np.all(normals == [0, 0, -1], axis=1))
    assert indices.max() < len(vertices)
    assert len(normals) == len(vertices) == len(uvs)


def test_other_primitives_keep_bottom_on_support_plane() -> None:
    for kind in ("cuboid", "sphere"):
        vertices, _, _, indices = primitive_mesh(kind, [0.12, 0.10, 0.08])
        assert np.isclose(vertices[:, 2].min(), 0.0, atol=1e-6)
        assert np.isclose(vertices[:, 2].max(), 0.08, atol=1e-6)
        assert indices.max() < len(vertices)
