"""Focused tests for the generic SDF rigid joint-lock operator."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from autoresearch.sdf_rigid_joint_lock import rigidify_sdf_joints

SDF = """<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="kettle">
    <link name="body">
      <inertial><mass>1.0</mass></inertial>
      <visual name="body_visual"><geometry><mesh><uri>visual/body.gltf</uri></mesh></geometry></visual>
      <collision name="body_collision"><geometry><box><size>1 1 1</size></box></geometry></collision>
    </link>
    <link name="lid">
      <inertial><mass>0.1</mass></inertial>
      <visual name="lid_visual"><geometry><mesh><uri>visual/lid.gltf</uri></mesh></geometry></visual>
      <collision name="lid_collision"><geometry><box><size>.5 .5 .5</size></box></geometry></collision>
    </link>
    <link name="base">
      <inertial><mass>0.2</mass></inertial>
    </link>
    <joint name="lid_hinge" type="revolute">
      <parent>body</parent>
      <child>lid</child>
      <pose relative_to="body">0 0 1 0 0 0</pose>
      <axis>
        <xyz>1 0 0</xyz>
        <limit><lower>-1.5707963</lower><upper>0</upper><effort>10</effort><velocity>2</velocity></limit>
        <dynamics><damping>0.05</damping><friction>0.05</friction></dynamics>
      </axis>
    </joint>
    <joint name="base_fixed" type="fixed">
      <parent>body</parent>
      <child>base</child>
    </joint>
  </model>
</sdf>
"""


def _write(tmp_path: Path) -> Path:
    (tmp_path / "visual").mkdir(parents=True)
    (tmp_path / "visual/body.gltf").write_text("body", encoding="utf-8")
    (tmp_path / "visual/lid.gltf").write_text("lid", encoding="utf-8")
    path = tmp_path / "asset.sdf"
    path.write_text(SDF, encoding="utf-8")
    return path


def test_lock_converts_hinge_and_preserves_link_content(tmp_path):
    source = _write(tmp_path)
    output = tmp_path / "derived/asset_rigid.sdf"
    report = rigidify_sdf_joints(
        source,
        output,
        joints=["lid_hinge"],
        provenance={"task_object_id": "kettle", "authority": "task never manipulates the lid"},
    )
    assert report["passed"] is True
    assert report["checks"]["declared_joints_are_fixed"] is True
    assert report["remaining_non_fixed_joints"] == []
    assert report["link_count"] == 3
    assert all(report["checks"][name] for name in report["checks"])
    lock = report["locked_joints"][0]
    assert lock["original_type"] == "revolute"
    assert lock["original_limits"] == pytest.approx([-1.5707963, 0.0])
    assert lock["locked_value"] == 0.0
    root = ET.parse(output).getroot()
    joint = next(j for j in root.iter("joint") if j.get("name") == "lid_hinge")
    assert joint.get("type") == "fixed"
    assert joint.find("axis") is None
    assert joint.findtext("pose") == "0 0 1 0 0 0"
    # Mesh uris are absolutised so the derived file is self-contained.
    uris = {uri.text for uri in root.iter("uri")}
    assert all(str(uri).startswith("/") for uri in uris)
    assert report["source_sdf_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_already_fixed_or_missing_joint_fails_closed(tmp_path):
    source = _write(tmp_path)
    with pytest.raises(ValueError):
        rigidify_sdf_joints(source, tmp_path / "a.sdf", joints=["base_fixed"], provenance={})
    with pytest.raises(ValueError):
        rigidify_sdf_joints(source, tmp_path / "b.sdf", joints=["not_a_joint"], provenance={})
    with pytest.raises(ValueError):
        rigidify_sdf_joints(source, tmp_path / "c.sdf", joints=[], provenance={})


def test_zero_outside_limits_is_refused(tmp_path):
    source = _write(tmp_path)
    source.write_text(
        SDF.replace("<lower>-1.5707963</lower>", "<lower>0.2</lower>"), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        rigidify_sdf_joints(source, tmp_path / "d.sdf", joints=["lid_hinge"], provenance={})
