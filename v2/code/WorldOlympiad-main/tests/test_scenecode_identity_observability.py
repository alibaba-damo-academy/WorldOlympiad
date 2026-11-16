from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from autoresearch.scenecode_identity_render_contract import (
    asset_material_profile, read_gltf, source_digests, validate_render_sources,
)
from autoresearch.scenecode_identity_lighting import (
    FORBIDDEN_VIEW_TRANSFORMS,
    faithful_display_value,
    identity_lighting_plan,
    validate_identity_lighting,
)
from scripts.render_scenecode_asset_identity import (
    identity_render_path,
    native_visual_plan,
    render_asset,
)


def _asset(tmp_path: Path, extension: bool = True, absolute: bool = False):
    mesh = tmp_path / "shared" / "shell.gltf"
    mesh.parent.mkdir()
    texture = mesh.parent / "color.png"
    texture.write_bytes(b"original texture bytes")
    material = {"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.1, 0.1, 1]}}
    if extension:
        material["extensions"] = {"KHR_materials_transmission": {"transmissionFactor": 1}}
    mesh.write_text(json.dumps({"asset": {"version": "2.0"}, "materials": [material],
                               "images": [{"uri": "color.png"}]}))
    sdf = tmp_path / "sdf" / "asset.sdf"
    sdf.parent.mkdir()
    uri = str(mesh) if absolute else "../shared/shell.gltf"
    sdf.write_text(f'<sdf version="1.9"><model name="asset"><link name="shell">'
                   '<pose>0.1 0.2 0.3 0 0 0</pose><visual name="v">'
                   f'<geometry><mesh><uri>{uri}</uri><scale>1 2 3</scale></mesh></geometry>'
                   '</visual></link></model></sdf>')
    return sdf, mesh, texture


@pytest.mark.parametrize("absolute", [True, False])
def test_original_material_and_external_dependencies_are_never_rewritten(tmp_path, absolute):
    sdf, mesh, texture = _asset(tmp_path, absolute=absolute)
    before = {p: p.read_bytes() for p in (sdf, mesh, texture)}
    profile = asset_material_profile(sdf)
    plan = native_visual_plan(sdf)
    assert profile["requires_native_material_renderer"] is True
    assert profile["material_extensions"] == ["KHR_materials_transmission"]
    assert set(profile["dependencies"]) == {str(p) for p in before}
    assert plan[0]["mesh_path"] == str(mesh)
    assert plan[0]["scale"] == [1, 2, 3]
    assert [row[3] for row in plan[0]["transform"][:3]] == [0.1, 0.2, 0.3]
    assert {p: p.read_bytes() for p in before} == before


def test_opaque_material_keeps_vtk_path(tmp_path):
    sdf, _, _ = _asset(tmp_path, extension=False)
    assert asset_material_profile(sdf)["requires_native_material_renderer"] is False


def test_glb_material_extensions_are_detected(tmp_path):
    payload = json.dumps({"materials": [{"extensions": {"KHR_materials_ior": {"ior": 1.33}}}]}).encode()
    payload += b" " * (-len(payload) % 4)
    path = tmp_path / "asset.glb"
    path.write_bytes(struct.pack("<4sIIII", b"glTF", 2, 20 + len(payload), len(payload), 0x4E4F534A) + payload)
    assert read_gltf(path)["materials"][0]["extensions"]["KHR_materials_ior"]["ior"] == 1.33
    path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="truncated"):
        read_gltf(path)


def test_native_material_dispatch_precedes_drake_loading(tmp_path, monkeypatch):
    from scripts import render_scenecode_asset_identity as renderer
    sdf, mesh, _ = _asset(tmp_path)
    original = mesh.read_bytes()
    captured = {}
    def native(**kwargs):
        captured.update(kwargs)
        return {"effective_renderer": "blender_native_cpu"}
    monkeypatch.setattr(renderer, "render_native_material_asset", native)
    result = render_asset(sdf_path=sdf, output_dir=tmp_path / "render", object_id="bottle",
                          width=320, height=320, views=[(0, 25), (90, 25), (180, 25)],
                          renderer="vtk", visual_description="clear bottle")
    assert result["effective_renderer"] == "blender_native_cpu"
    assert captured["sdf_path"] == sdf
    assert mesh.read_bytes() == original


def _report(tmp_path):
    sdf, mesh, texture = _asset(tmp_path)
    repo = tmp_path / "repo"
    for name in ("scripts/render_scenecode_asset_identity.py", "scripts/render_scenecode_identity_blender.py",
                 "autoresearch/scenecode_identity_lighting.py",
                 "autoresearch/scenecode_identity_render_contract.py"):
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original renderer source")
    report = {"render_source_digests": source_digests(repo), "results": {"bottle": {
        "sdf_path": str(sdf), "effective_renderer": "blender_native_cpu",
        "material_profile": asset_material_profile(sdf)}}}
    return repo, report, mesh, texture


@pytest.mark.parametrize("changed", ["mesh", "texture", "renderer", "appearance_copy", "unsupported_renderer"])
def test_cached_identity_rejects_changed_or_unobservable_inputs(tmp_path, changed):
    repo, report, mesh, texture = _report(tmp_path)
    validate_render_sources(report, repo)
    if changed in {"mesh", "texture"}:
        path = mesh if changed == "mesh" else texture
        path.write_bytes(path.read_bytes() + b" ")
    elif changed == "renderer":
        (repo / "scripts/render_scenecode_identity_blender.py").write_text("changed")
    elif changed == "appearance_copy":
        report["results"]["bottle"]["observable_material_copy"] = "modified material"
    else:
        report["results"]["bottle"]["effective_renderer"] = "vtk"
    with pytest.raises(ValueError):
        validate_render_sources(report, repo)


def test_render_refuses_source_changes_during_execution(tmp_path, monkeypatch):
    from scripts import render_scenecode_asset_identity as renderer
    from autoresearch.scenecode_identity_render_contract import sha256
    import sys
    sdf, _, _ = _asset(tmp_path)
    marker = tmp_path / "renderer.py"
    marker.write_text("original source")
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"objects": [{"object_id": "bottle", "visual_description": "clear bottle"}]}))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"matches": [{"status": "matched", "task_object_id": "bottle", "sdf_path": str(sdf)}]}))
    monkeypatch.setattr(renderer, "source_digests", lambda root: {str(marker): sha256(marker)})
    def changed_renderer(**kwargs):
        marker.write_text("changed source")
        return {"material_profile": asset_material_profile(sdf), "structure_gate": {"passed": True},
                "render_gate_passed": True, "effective_renderer": "blender_native_cpu"}
    monkeypatch.setattr(renderer, "render_asset", changed_renderer)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["render", "--task-json", str(task), "--inventory-certificate", str(inventory),
                                      "--output-dir", str(output), "--renderer", "vtk"])
    with pytest.raises(ValueError, match="dependency changed"):
        renderer.main()
    assert not (output / "asset_identity_report.json").exists()
def test_named_frame_pose_coordinates_take_precedence_over_attachment(tmp_path):
    import numpy as np
    from scripts.render_scenecode_asset_identity import _sdf_frame_transforms
    sdf = tmp_path / "explicit_pose_frame.sdf"
    sdf.write_text('<sdf version="1.7"><model name="assembly"><pose>1 2 3 0 0 0</pose>'
                   '<frame name="origin" attached_to="body"><pose relative_to="__model__">0 0 .2 0 0 0</pose></frame>'
                   '<link name="body"><pose relative_to="origin">0 0 .4 0 0 0</pose></link>'
                   '</model></sdf>')
    _, frames = _sdf_frame_transforms(sdf)
    np.testing.assert_allclose(frames["body"][:3, 3], [1, 2, 3.6])


def test_identity_lighting_displays_declared_albedo_without_a_filmic_look():
    lighting = identity_lighting_plan()
    validate_identity_lighting(lighting)
    assert lighting["view_transform"] == "Standard"
    assert lighting["view_transform"] not in FORBIDDEN_VIEW_TRANSFORMS
    assert lighting["world_color"] == [1.0, 1.0, 1.0]
    assert 0.0 < lighting["world_strength"] <= 1.0
    # A declared light surface must display light: the calibrated rig renders a
    # diffuse albedo as its own sRGB value instead of an aesthetic tone map.
    assert faithful_display_value(0.86) == pytest.approx(0.940, abs=0.01)
    assert faithful_display_value(0.50) == pytest.approx(0.735, abs=0.01)


def test_identity_lighting_rejects_a_display_referred_view_transform():
    lighting = identity_lighting_plan()
    lighting["view_transform"] = "AgX"
    with pytest.raises(ValueError, match="view transform"):
        validate_identity_lighting(lighting)


def test_identity_render_path_prefers_the_calibrated_native_renderer():
    opaque = {"requires_native_material_renderer": False}
    exotic = {"requires_native_material_renderer": True}
    # An opaque asset still uses the calibrated native path whenever available.
    assert identity_render_path(
        material_profile=opaque, renderer="vtk", native_available=True
    ) == "blender_native_cpu"
    # Without the native renderer the uncalibrated VTK path remains the fallback.
    assert identity_render_path(
        material_profile=opaque, renderer="vtk", native_available=False
    ) == "vtk"
    # A declared exotic material always requires the native renderer.
    assert identity_render_path(
        material_profile=exotic, renderer="vtk", native_available=False
    ) == "blender_native_cpu"
    assert identity_render_path(
        material_profile=opaque, renderer="blender", native_available=False
    ) == "blender_native_cpu"


def test_opaque_asset_uses_the_calibrated_native_path_when_available(
    tmp_path, monkeypatch
):
    from scripts import render_scenecode_asset_identity as renderer
    sdf, _, _ = _asset(tmp_path, extension=False)
    captured = {}

    def native(**kwargs):
        captured.update(kwargs)
        return {"effective_renderer": "blender_native_cpu"}

    monkeypatch.setattr(renderer, "render_native_material_asset", native)
    monkeypatch.setattr(renderer, "blender_executable", lambda: "/usr/bin/blender")
    result = render_asset(sdf_path=sdf, output_dir=tmp_path / "render", object_id="shell",
                          width=160, height=160, views=[(0, 25)], renderer="vtk",
                          visual_description="matte opaque shell")
    assert result["effective_renderer"] == "blender_native_cpu"
    assert captured["visual_description"] == "matte opaque shell"
