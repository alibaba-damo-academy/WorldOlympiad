import hashlib
import json
import shutil
import pytest

from scripts.derive_scenecode_component_materials import derive
from autoresearch.robotics_visual_revision_verifier import verify_visual_revision
from scripts.package_robotics_reference import _validate_visual_variant_manifest


def test_multi_component_material_revision_preserves_physics_and_geometry(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sdf = source / "cabinet.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="cabinet">
      <link name="body"><visual><geometry><mesh><uri>body.gltf</uri></mesh></geometry></visual></link>
      <link name="door"><visual><geometry><mesh><uri>door.gltf</uri></mesh></geometry></visual></link>
    </model></sdf>""")
    gltf = {"materials": [{"name": "plain", "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}]}
    for name in ("body.gltf", "door.gltf"):
        (source / name).write_text(json.dumps(gltf))
    (source / "collision.obj").write_text("immutable collision")
    original = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("cabinet.sdf", "body.gltf", "door.gltf", "collision.obj")}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema_version": "worldolympiad_component_material_plan.v1",
        "variant_id": "contrast_v1",
        "edits": [
            {"link_name": "body", "base_color_rgba": [.9, .88, .84, 1.],
             "metallic_factor": 0., "roughness_factor": .8},
            {"link_name": "door", "base_color_rgba": [.2, .35, .5, 1.],
             "metallic_factor": 0., "roughness_factor": .8},
        ],
    }))
    output = tmp_path / "variant"
    assert derive(source_sdf=sdf, plan_json=plan, output_dir=output) == output / sdf.name
    manifest = json.loads((output / "visual_variant_manifest.json").read_text())
    assert manifest["changed_files"] == ["body.gltf", "door.gltf"]
    assert manifest["physics_geometry_unchanged"] is True
    assert (output / "cabinet.sdf").read_bytes() == sdf.read_bytes()
    assert (output / "collision.obj").read_bytes() == (source / "collision.obj").read_bytes()
    assert all(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
               for name, digest in original.items())
    verdict = verify_visual_revision(
        manifest_path=output / "visual_variant_manifest.json",
        output=tmp_path / "visual_verdict.json", expected_source_sdf=sdf)
    assert verdict["passed"] is True
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copytree(source, bundle / "visual_variant_source_asset")
    shutil.copytree(output, bundle / "visual_variant")
    packed_manifest = bundle / "visual_variant/visual_variant_manifest.json"
    packed = json.loads(packed_manifest.read_text())
    packed["source_sdf"] = "visual_variant_source_asset/cabinet.sdf"
    packed_manifest.write_text(json.dumps(packed))
    assert _validate_visual_variant_manifest(packed_manifest, bundle)[
        "physics_geometry_unchanged"] is True
    (output / "door.gltf").write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        verify_visual_revision(
            manifest_path=output / "visual_variant_manifest.json",
            output=tmp_path / "visual_verdict.json", expected_source_sdf=sdf)


def test_visual_revision_rejects_changed_collision_resource(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sdf = source / "item.sdf"
    sdf.write_text('<sdf version="1.9"><model name="item"><link name="body"><visual><geometry><mesh><uri>body.gltf</uri></mesh></geometry></visual><collision><geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision></link></model></sdf>')
    (source / "body.gltf").write_text(json.dumps({"materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}], "meshes": []}))
    (source / "collision.obj").write_text("fixed")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": "worldolympiad_component_material_plan.v1", "variant_id": "v", "edits": [{"link_name": "body", "base_color_rgba": [.1, .2, .3, 1], "metallic_factor": 0, "roughness_factor": .8}]}))
    output = tmp_path / "variant"
    derive(source_sdf=sdf, plan_json=plan, output_dir=output)
    (output / "collision.obj").write_text("changed")
    manifest = json.loads((output / "visual_variant_manifest.json").read_text())
    manifest["output_file_sha256"]["collision.obj"] = hashlib.sha256((output / "collision.obj").read_bytes()).hexdigest()
    manifest["changed_files"] = ["body.gltf", "collision.obj"]
    (output / "visual_variant_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="non-material"):
        verify_visual_revision(manifest_path=output / "visual_variant_manifest.json", output=tmp_path / "v.json", expected_source_sdf=sdf)


def test_visual_revision_rejects_changed_gltf_graph(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sdf = source / "item.sdf"
    sdf.write_text('<sdf version="1.9"><model name="item"><link name="body"><visual><geometry><mesh><uri>body.gltf</uri></mesh></geometry></visual></link></model></sdf>')
    base = {"materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}], "meshes": []}
    (source / "body.gltf").write_text(json.dumps(base))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": "worldolympiad_component_material_plan.v1", "variant_id": "v", "edits": [{"link_name": "body", "base_color_rgba": [.1, .2, .3, 1], "metallic_factor": 0, "roughness_factor": .8}]}))
    output = tmp_path / "variant"
    derive(source_sdf=sdf, plan_json=plan, output_dir=output)
    changed = json.loads((output / "body.gltf").read_text())
    changed["meshes"] = [{"primitives": [{"attributes": {"POSITION": 0}}]}]
    (output / "body.gltf").write_text(json.dumps(changed))
    manifest = json.loads((output / "visual_variant_manifest.json").read_text())
    manifest["output_file_sha256"]["body.gltf"] = hashlib.sha256((output / "body.gltf").read_bytes()).hexdigest()
    (output / "visual_variant_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="geometry"):
        verify_visual_revision(manifest_path=output / "visual_variant_manifest.json", output=tmp_path / "v.json", expected_source_sdf=sdf)


def test_multi_material_cabinet_frame_can_be_repaired_without_mesh_edit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sdf = source / "cabinet.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="cabinet"><link name="frame">
      <visual><geometry><mesh><uri>frame.gltf</uri></mesh></geometry></visual>
      </link></model></sdf>""")
    gltf = {"materials": [
        {"name": "outer", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
        {"name": "inner", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}},
    ], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}]}
    (source / "frame.gltf").write_text(json.dumps(gltf))
    plan = tmp_path / "plan.json"
    def edit(name, colour):
        return {"link_name": "frame", "material_name": name,
                "base_color_rgba": [*colour, 1.0],
                "metallic_factor": 0.0, "roughness_factor": 0.85}
    plan.write_text(json.dumps({
        "schema_version": "worldolympiad_component_material_plan.v1",
        "variant_id": "frame_repair", "edits": [
            edit("outer", [0.98, 0.96, 0.92]),
            edit("inner", [0.70, 0.72, 0.74]),
        ]}))
    output = tmp_path / "variant"
    derive(source_sdf=sdf, plan_json=plan, output_dir=output)
    repaired = json.loads((output / "frame.gltf").read_text())
    assert [item["pbrMetallicRoughness"]["baseColorFactor"] for item in repaired["materials"]] == [
        [0.98, 0.96, 0.92, 1.0], [0.70, 0.72, 0.74, 1.0]
    ]
    assert repaired["meshes"] == gltf["meshes"]
    assert json.loads((source / "frame.gltf").read_text()) == gltf
    assert json.loads((output / "visual_variant_manifest.json").read_text())["changed_files"] == ["frame.gltf"]
    with pytest.raises(ValueError, match="cannot repeat"):
        duplicated = json.loads(plan.read_text())
        duplicated["edits"].append(edit("outer", [0.2, 0.2, 0.2]))
        plan.write_text(json.dumps(duplicated))
        derive(source_sdf=sdf, plan_json=plan, output_dir=tmp_path / "duplicate")
