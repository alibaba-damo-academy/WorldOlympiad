from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from autoresearch.scenecode_visual_collision_alignment import (
    derive_common_visual_shift,
    measure_visual_collision_alignment,
)


def _asset(path, offsets):
    links = []
    for index, offset in enumerate(offsets):
        links.append(f"""
        <link name="part_{index}">
          <pose>{index} 0 0 0 0 0</pose>
          <inertial><mass>1</mass></inertial>
          <visual name="visible"><pose>0 0 {offset} 0 0 0</pose>
            <geometry><box><size>1 1 1</size></box></geometry></visual>
          <collision name="physical"><pose>0 0 {offset} 0 0 0</pose>
            <geometry><box><size>1 1 1</size></box></geometry></collision>
        </link>""")
    path.write_text('<sdf version="1.7"><model name="asset">' + "".join(links) + '</model></sdf>')


def _runtime(offsets):
    states = {
        f"part_{index}": {"position": [float(index), 0.0, 0.0],
                          "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        for index in range(len(offsets))
    }
    bounds = {
        f"part_{index}": ([index - 0.5, -0.5, -0.5],
                          [index + 0.5, 0.5, 0.5])
        for index in range(len(offsets))
    }
    return states, bounds


def test_common_shift_repairs_visuals_without_changing_physics(tmp_path):
    source = tmp_path / "source" / "asset.sdf"
    source.parent.mkdir()
    _asset(source, [0.375, 0.375])
    states, bounds = _runtime([0.375, 0.375])
    before = measure_visual_collision_alignment(
        source, link_states=states, physics_aabbs=bounds)
    assert before["max_center_error_m"] == pytest.approx(0.375)
    candidate = derive_common_visual_shift(
        source, tmp_path / "candidate", link_states=states, physics_aabbs=bounds)
    after = measure_visual_collision_alignment(
        candidate, link_states=states, physics_aabbs=bounds)
    assert after["max_center_error_m"] < 1e-8
    old = ET.parse(source)
    new = ET.parse(candidate)
    assert [ET.tostring(node) for node in old.findall(".//collision")] == [
        ET.tostring(node) for node in new.findall(".//collision")]
    assert [ET.tostring(node) for node in old.findall(".//inertial")] == [
        ET.tostring(node) for node in new.findall(".//inertial")]
    with pytest.raises(FileExistsError):
        derive_common_visual_shift(
            source, tmp_path / "candidate", link_states=states, physics_aabbs=bounds)


def test_inconsistent_link_offsets_are_not_auto_repaired(tmp_path):
    source = tmp_path / "source" / "asset.sdf"
    source.parent.mkdir()
    _asset(source, [0.375, 0.125])
    states, bounds = _runtime([0.375, 0.125])
    with pytest.raises(ValueError, match="no safe shared"):
        derive_common_visual_shift(
            source, tmp_path / "candidate", link_states=states, physics_aabbs=bounds)
    assert not (tmp_path / "candidate").exists()


def test_matching_centres_do_not_hide_a_large_geometry_gap(tmp_path):
    source = tmp_path / "source" / "asset.sdf"
    source.parent.mkdir()
    _asset(source, [0.375, 0.375])
    states, bounds = _runtime([0.375, 0.375])
    expanded = {
        key: ([value - 0.05 for value in low],
              [value + 0.05 for value in high])
        for key, (low, high) in bounds.items()
    }
    with pytest.raises(ValueError, match="failed runtime alignment"):
        derive_common_visual_shift(
            source, tmp_path / "candidate",
            link_states=states, physics_aabbs=expanded)
    assert not (tmp_path / "candidate").exists()
