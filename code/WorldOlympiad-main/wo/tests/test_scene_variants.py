import hashlib
import json
from pathlib import Path

import pytest

from wo.scene import _variant_sdf, resolve_scenecode
from wo.spec import TaskSpec


def test_visual_variant_rejects_uncertified_digest_bound_staging_source(tmp_path):
    source = tmp_path / "inventory" / "plate.sdf"
    source.parent.mkdir()
    source.write_text("sdf")
    variant = tmp_path / "variant"
    variant.mkdir()
    staged = variant / "plate.sdf"
    staged.write_text(source.read_text())
    manifest = variant / "visual_variant_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "scenecode_visual_variant.v1",
        "physics_geometry_unchanged": True,
        "source_sdf": str(staged),
        "source_file_sha256": {"plate.sdf": hashlib.sha256(source.read_bytes()).hexdigest()},
        "output_file_sha256": {"plate.sdf": hashlib.sha256(staged.read_bytes()).hexdigest()},
    }))
    spec = TaskSpec(case_id="x", family="pick_place", instruction="x", objects=(), steps=())
    spec = spec.__class__(**{**spec.__dict__, "visual_variants": ({"object_id": "plate", "manifest": str(manifest)},)})
    with pytest.raises(ValueError, match="file evidence changed"):
        _variant_sdf(spec, "plate", source)


def test_visual_variant_is_bound_and_hash_checked(tmp_path):
    source = tmp_path / "source.sdf"
    source.write_text("sdf")
    variant = tmp_path / "variant"
    variant.mkdir()
    output = variant / "source.sdf"
    output.write_text("visual-only sdf")
    manifest = variant / "visual_variant_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "scenecode_visual_variant.v1",
        "physics_geometry_unchanged": True,
        "source_sdf": str(source),
        "output_file_sha256": {"source.sdf": hashlib.sha256(output.read_bytes()).hexdigest()},
    }))
    spec = TaskSpec(case_id="x", family="pick_place", instruction="x", objects=(), steps=())
    # Bypass full task validation: this helper only consumes the declared variant list.
    spec = spec.__class__(**{**spec.__dict__, "visual_variants": ({"object_id": "plate", "manifest": str(manifest)},)})
    with pytest.raises(ValueError, match="file evidence changed|changed the physical SDF"):
        _variant_sdf(spec, "plate", source)


def _certified_variant(tmp_path, *, collision_changed=False, graph_changed=False,
                       verifier_flag=True):
    source_dir = tmp_path / "certified"
    variant_dir = tmp_path / "variant"
    source_dir.mkdir()
    variant_dir.mkdir()
    sdf = source_dir / "plate.sdf"
    sdf.write_text("""<sdf version='1.7'><model name='plate'><link name='base'>
      <visual><geometry><mesh><uri>visual.gltf</uri></mesh></geometry></visual>
      <collision><geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision>
    </link></model></sdf>""")
    (source_dir / "visual.gltf").write_text(json.dumps({
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}],
    }))
    (source_dir / "collision.obj").write_text("v 0 0 0\n")
    for path in source_dir.iterdir():
        (variant_dir / path.name).write_bytes(path.read_bytes())
    variant_gltf = json.loads((variant_dir / "visual.gltf").read_text())
    variant_gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 1, 1]
    if graph_changed:
        variant_gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = 1
    (variant_dir / "visual.gltf").write_text(json.dumps(variant_gltf))
    if collision_changed:
        (variant_dir / "collision.obj").write_text("v 9 9 9\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    files = lambda d: {p.name: sha(p) for p in d.iterdir() if p.is_file()}
    manifest = variant_dir / "visual_variant_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "scenecode_visual_variant.v1",
        "physics_geometry_unchanged": True,
        "gltf_geometry_and_other_materials_unchanged": verifier_flag,
        "source_sdf": str(sdf),
        "source_file_sha256": files(source_dir),
        "output_file_sha256": files(variant_dir),
        "changed_files": ["visual.gltf"],
    }))
    spec = TaskSpec(case_id="x", family="pick_place", instruction="x", objects=(), steps=())
    spec = spec.__class__(**{**spec.__dict__, "visual_variants": (
        {"object_id": "plate", "manifest": str(manifest)},)})
    return spec, sdf


def _public_spec_payload(manifest):
    return {
        "schema_version": "wo_task_spec.v1", "case_id": "public", "family": "pick_place",
        "instruction": "Move the plate onto the table.",
        "objects": [
            {"object_id": "plate", "category": "plate", "role": "manipuland"},
            {"object_id": "table", "category": "table", "role": "support"},
        ],
        "steps": [{"primitive": "transport", "subject": "plate", "target": "table",
                    "goal": {"kind": "on_region", "subject": "plate", "target": "table"}}],
        "visual_variants": [{"object_id": "plate", "manifest": str(manifest)}],
    }


def test_public_load_then_resolve_enforces_variant_guard(tmp_path):
    spec, source = _certified_variant(tmp_path, collision_changed=True, verifier_flag=False)
    house = tmp_path / "house.json"
    house.write_text(json.dumps({"rooms": {"room": {"objects": {
        "plate": {"sdf_path": str(source), "transform": {"translation": [0, 0, 0.1], "rotation_wxyz": [1, 0, 0, 0]}},
        "table": {"sdf_path": str(source), "transform": {"translation": [0, 0, 0], "rotation_wxyz": [1, 0, 0, 0]}},
    }}}}))
    (tmp_path / "task_inventory_certificate.json").write_text(json.dumps({
        "house_state": str(house), "matches": [
            {"task_object_id": oid, "scene_object_id": oid, "status": "matched", "sdf_path": str(source)}
            for oid in ("plate", "table")
        ]
    }))
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_public_spec_payload(spec.visual_variants[0]["manifest"])))
    loaded = TaskSpec.load(task_path)
    with pytest.raises(ValueError, match="non-material|file evidence|collision"):
        resolve_scenecode(loaded, tmp_path)


def test_public_load_then_resolve_accepts_certified_material_revision(tmp_path):
    spec, source = _certified_variant(tmp_path)
    house = tmp_path / "house.json"
    house.write_text(json.dumps({"rooms": {"room": {"objects": {
        "plate": {"sdf_path": str(source), "transform": {"translation": [0, 0, 0.1], "rotation_wxyz": [1, 0, 0, 0]}},
        "table": {"sdf_path": str(source), "transform": {"translation": [0, 0, 0], "rotation_wxyz": [1, 0, 0, 0]}},
    }}}}))
    (tmp_path / "task_inventory_certificate.json").write_text(json.dumps({
        "house_state": str(house), "matches": [
            {"task_object_id": oid, "scene_object_id": oid, "status": "matched", "sdf_path": str(source)}
            for oid in ("plate", "table")
        ]
    }))
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_public_spec_payload(spec.visual_variants[0]["manifest"])))
    assert resolve_scenecode(TaskSpec.load(task_path), tmp_path).objects["plate"].sdf_path.is_file()


def test_public_variant_rejects_changed_collision_resource(tmp_path):
    spec, source = _certified_variant(tmp_path, collision_changed=True)
    with pytest.raises(ValueError, match="file evidence changed|non-material|collision"):
        _variant_sdf(spec, "plate", source)


def test_public_variant_rejects_changed_gltf_graph(tmp_path):
    spec, source = _certified_variant(tmp_path, graph_changed=True)
    with pytest.raises(ValueError, match="geometry|material"):
        _variant_sdf(spec, "plate", source)


def test_public_variant_accepts_material_only_revision(tmp_path):
    spec, source = _certified_variant(tmp_path)
    assert _variant_sdf(spec, "plate", source).is_file()


def test_public_variant_ignores_known_nondotted_verdict_metadata(tmp_path):
    spec, source = _certified_variant(tmp_path)
    manifest = Path(spec.visual_variants[0]["manifest"])
    (manifest.parent / "visual_revision_verdict.json").write_text(
        json.dumps({"schema_version": "worldolympiad_visual_revision_verdict.v1", "passed": True})
    )
    house = tmp_path / "house.json"
    house.write_text(json.dumps({"rooms": {"room": {"objects": {
        "plate": {"sdf_path": str(source), "transform": {"translation": [0, 0, 0.1], "rotation_wxyz": [1, 0, 0, 0]}},
        "table": {"sdf_path": str(source), "transform": {"translation": [0, 0, 0], "rotation_wxyz": [1, 0, 0, 0]}},
    }}}}))
    (tmp_path / "task_inventory_certificate.json").write_text(json.dumps({
        "house_state": str(house), "matches": [
            {"task_object_id": oid, "scene_object_id": oid, "status": "matched", "sdf_path": str(source)}
            for oid in ("plate", "table")
        ]
    }))
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_public_spec_payload(str(manifest))))
    assert resolve_scenecode(TaskSpec.load(task_path), tmp_path).objects["plate"].sdf_path.is_file()


def test_public_variant_cannot_opt_out_of_resource_verification(tmp_path):
    spec, source = _certified_variant(tmp_path, verifier_flag=False)
    with pytest.raises(ValueError, match="file evidence changed"):
        _variant_sdf(spec, "plate", source)


def test_public_variant_rejects_changed_glb_binary_resource(tmp_path):
    import struct
    from autoresearch.robotics_visual_revision_verifier import verify_visual_revision

    source_dir, variant_dir = tmp_path / "certified", tmp_path / "variant"
    source_dir.mkdir(); variant_dir.mkdir()
    sdf = source_dir / "plate.sdf"
    sdf.write_text("""<sdf version='1.7'><model name='plate'><link name='base'>
      <visual><geometry><mesh><uri>plate.glb</uri></mesh></geometry></visual>
      <collision><geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision>
    </link></model></sdf>""")
    scene = {"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
             "materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1]}}]}
    def write_glb(path, payload, binary):
        raw = json.dumps(payload, separators=(",", ":")).encode()
        raw += b" " * ((4 - len(raw) % 4) % 4)
        blob = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(raw) + 8 + len(binary))
        blob += struct.pack("<II", len(raw), 0x4E4F534A) + raw
        blob += struct.pack("<II", len(binary), 0x004E4942) + binary
        path.write_bytes(blob)
    write_glb(source_dir / "plate.glb", scene, b"POSITION\\x00")
    (source_dir / "collision.obj").write_text("v 0 0 0\\n")
    for p in source_dir.iterdir():
        (variant_dir / p.name).write_bytes(p.read_bytes())
    changed = json.loads(json.dumps(scene))
    changed["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 1, 1]
    write_glb(variant_dir / "plate.glb", changed, b"POSITION\\x01")
    manifest = {
        "schema_version": "scenecode_visual_variant.v1",
        "physics_geometry_unchanged": True,
        "gltf_geometry_and_other_materials_unchanged": True,
        "source_sdf": str(sdf),
        "changed_files": ["plate.glb"],
    }
    from autoresearch.robotics_visual_revision_verifier import _files
    manifest["source_file_sha256"] = _files(source_dir)
    manifest["output_file_sha256"] = _files(variant_dir)
    manifest_path = variant_dir / "visual_variant_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binary geometry/resource"):
        verify_visual_revision(manifest_path=manifest_path, output=variant_dir / "verdict.json")


def test_public_variant_accepts_material_only_glb(tmp_path):
    import struct
    from autoresearch.robotics_visual_revision_verifier import verify_visual_revision

    source_dir = tmp_path / "certified"
    variant_dir = tmp_path / "variant"
    source_dir.mkdir(); variant_dir.mkdir()
    sdf = source_dir / "plate.sdf"
    sdf.write_text("""<sdf version='1.7'><model name='plate'><link name='base'>
      <visual><geometry><mesh><uri>plate.glb</uri></mesh></geometry></visual>
      <collision><geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision>
    </link></model></sdf>""")
    scene = {"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
             "materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1]}}]}
    def write_glb(path, payload):
        raw = json.dumps(payload, separators=(",", ":")).encode()
        raw += b" " * ((4 - len(raw) % 4) % 4)
        blob = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(raw))
        blob += struct.pack("<II", len(raw), 0x4E4F534A) + raw
        path.write_bytes(blob)
    write_glb(source_dir / "plate.glb", scene)
    (source_dir / "collision.obj").write_text("v 0 0 0\n")
    for p in source_dir.iterdir():
        (variant_dir / p.name).write_bytes(p.read_bytes())
    changed = json.loads(json.dumps(scene))
    changed["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 1, 1]
    write_glb(variant_dir / "plate.glb", changed)
    sha = lambda p: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
    files = lambda d: {p.name: sha(p) for p in d.iterdir() if p.is_file()}
    manifest = variant_dir / "visual_variant_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version":"scenecode_visual_variant.v1", "physics_geometry_unchanged":True,
        "gltf_geometry_and_other_materials_unchanged":True, "source_sdf":str(sdf),
        "source_file_sha256":files(source_dir), "output_file_sha256":files(variant_dir),
        "changed_files":["plate.glb"]}))
    assert verify_visual_revision(manifest_path=manifest, output=variant_dir / ".visual_revision_verdict.json",
                                  expected_source_sdf=sdf)["passed"]
