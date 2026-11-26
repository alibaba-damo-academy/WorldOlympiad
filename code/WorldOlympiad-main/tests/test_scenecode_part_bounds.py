import ast

import pytest

from autoresearch.scenecode_part_bounds import append_part_bounds_contract, check_part_bounds, MARKER


def _spec():
    return {"name": "contents", "shape": {"dimensions": {"width": 0.29, "depth": 0.29, "height": 0.68}},
            "position": {"x": 0, "y": 0, "z": 0.38}, "rotation": {"x": 0, "y": 0, "z": 0}}


def test_centered_shape_matches_plan():
    assert check_part_bounds(_spec(), [-0.145, -0.145, -0.34], [0.145, 0.145, 0.34])["passed"]


def test_base_origin_misused_as_center_is_rejected_with_measured_offset():
    result = check_part_bounds(_spec(), [-0.145, -0.145, 0], [0.145, 0.145, 0.68])
    assert not result["passed"]
    assert result["center_offset_in_planned_axes"] == [0, 0, 0.34]


def test_wrong_dimensions_and_nonfinite_bounds_fail():
    assert not check_part_bounds(_spec(), [-1, -1, -1], [1, 1, 1])["passed"]
    with pytest.raises(ValueError):
        check_part_bounds(_spec(), [float("nan"), 0, 0], [1, 1, 1])


def test_every_revision_gets_one_fresh_immutable_guard():
    code = append_part_bounds_contract("print('generated part')", _spec())
    repeated = append_part_bounds_contract(code, _spec())
    assert repeated.count(MARKER) == 1
    assert repeated.startswith("print('generated part')")
    ast.parse(repeated)
