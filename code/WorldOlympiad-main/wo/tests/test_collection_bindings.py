"""Collection references must account for every physical instance and preserve containment."""
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from wo.scene import resolve_scenecode, SceneObject
from wo.spec import TaskSpec
from wo.expert import embodiment as e


def collection_spec():
    return TaskSpec.from_dict({
        "schema_version": "wo_task_spec.v1", "case_id": "collection",
        "family": "transfer", "instruction": "Move all three beads to the bowl.",
        "objects": [
            {"object_id": f"bead_{i}", "category": "bead", "role": "manipuland",
             "inventory_object_id": "beads", "scene_member_id": scene_id,
             "initial_container": "cup"}
            for i, scene_id in enumerate(("b_0", "b_0_instance_1", "b_0_instance_2"))
        ] + [
            {"object_id": "cup", "category": "cup", "role": "destination"},
            {"object_id": "bowl", "category": "bowl", "role": "destination"},
            {"object_id": "table", "category": "table", "role": "support"},
        ],
        "steps": [
            {"primitive": "transport", "subject": f"bead_{i}", "target": "bowl",
             "goal": {"kind": "inside", "subject": f"bead_{i}", "target": "bowl"}}
            for i in range(3)
        ],
    })


def scene_fixture(tmp_path):
    asset = tmp_path / "asset.sdf"
    asset.write_text("<sdf/>")
    ids = ("b_0", "b_0_instance_1", "b_0_instance_2", "cup", "bowl", "table")
    objects = {
        oid: {"sdf_path": str(asset), "transform": {
            "translation": [i * .01, 0, .8], "rotation_wxyz": [1, 0, 0, 0]}}
        for i, oid in enumerate(ids)
    }
    objects["table"]["transform"]["translation"] = [0, 0, 0]
    house = tmp_path / "house.json"
    house.write_text(json.dumps({"rooms": {"room": {"objects": objects}}}))
    matches = [
        {"task_object_id": oid, "scene_object_id": scene_id, "status": "matched",
         "sdf_path": str(asset), "sdf_sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
        for oid, scene_id in (("beads", "b_0"), ("cup", "cup"),
                              ("bowl", "bowl"), ("table", "table"))
    ]
    (tmp_path / "task_inventory_certificate.json").write_text(
        json.dumps({"house_state": str(house), "matches": matches}))
    return house


def test_all_three_members_resolve_with_distinct_identity(tmp_path):
    scene_fixture(tmp_path)
    scene = resolve_scenecode(collection_spec(), tmp_path)
    bindings = scene.source["instance_bindings"]
    assert set(bindings) == {"bead_0", "bead_1", "bead_2"}
    assert len({b["scene_object_id"] for b in bindings.values()}) == 3
    assert scene.objects["bead_1"].position != scene.objects["bead_0"].position
    assert scene.objects["bead_2"].initial_container == "cup"


def test_missing_member_fails_instead_of_certifying_partial_transfer(tmp_path):
    house = scene_fixture(tmp_path)
    data = json.loads(house.read_text())
    del data["rooms"]["room"]["objects"]["b_0_instance_2"]
    house.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no declared member"):
        resolve_scenecode(collection_spec(), tmp_path)


def test_duplicate_member_binding_is_rejected(tmp_path):
    scene_fixture(tmp_path)
    spec = collection_spec()
    spec = replace(spec, objects=tuple(
        replace(o, scene_member_id="b_0") if o.object_id == "bead_1" else o
        for o in spec.objects))
    with pytest.raises(ValueError, match="same scene member"):
        resolve_scenecode(spec, tmp_path)


def test_member_with_different_asset_is_rejected(tmp_path):
    house = scene_fixture(tmp_path)
    data = json.loads(house.read_text())
    data["rooms"]["room"]["objects"]["b_0_instance_1"]["sdf_path"] = "other.sdf"
    house.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different asset"):
        resolve_scenecode(collection_spec(), tmp_path)


def test_composite_source_members_bind_without_parent_pose_composition(tmp_path):
    house = scene_fixture(tmp_path)
    data = json.loads(house.read_text())
    objects = data['rooms']['room']['objects']
    members = [dict(objects.pop(oid), asset_id='bead_asset')
               for oid in ('b_0', 'b_0_instance_1', 'b_0_instance_2')]
    objects['pile'] = {'metadata': {'composite_type': 'pile', 'member_assets': members}}
    house.write_text(json.dumps(data))
    cert_path = tmp_path / 'task_inventory_certificate.json'
    cert = json.loads(cert_path.read_text())
    cert['matches'][0]['scene_object_id'] = 'pile::member_0::bead_asset'
    cert_path.write_text(json.dumps(cert))
    spec = collection_spec()
    spec = replace(spec, objects=tuple(
        replace(o, scene_member_id=f'pile::member_{i}::bead_asset') if i < 3 else o
        for i, o in enumerate(spec.objects)))
    scene = resolve_scenecode(spec, tmp_path)
    assert len(scene.source['instance_bindings']) == 3
    np.testing.assert_allclose(scene.objects['bead_2'].position, [.02, 0, .8])


def test_unrelated_same_asset_occurrence_is_not_collection_member(tmp_path):
    house = scene_fixture(tmp_path)
    data = json.loads(house.read_text())
    objects = data['rooms']['room']['objects']
    objects['unrelated'] = objects.pop('b_0_instance_2')
    house.write_text(json.dumps(data))
    spec = collection_spec()
    spec = replace(spec, objects=tuple(
        replace(o, scene_member_id='unrelated') if o.object_id == 'bead_2' else o
        for o in spec.objects))
    with pytest.raises(ValueError, match='certified source occurrence'):
        resolve_scenecode(spec, tmp_path)


def test_initial_container_cycle_is_rejected():
    data = collection_spec().to_dict()
    data["objects"][3]["initial_container"] = "bead_0"
    with pytest.raises(ValueError, match="cycle"):
        TaskSpec.from_dict(data)


def test_canonical_layout_preserves_all_member_offsets(monkeypatch, tmp_path):
    from wo.tests.test_embodiment_layout import Scene
    positions = {"cup": (0, 0, .75), "bead_0": (.01, .01, .79),
                 "bead_1": (-.01, 0, .78), "bead_2": (0, -.01, .8),
                 "bowl": (.3, 0, .75)}
    objects = {oid: SceneObject(
        oid, "manipuland" if oid.startswith("bead") else "destination",
        tmp_path / "a.sdf", pos, (0, 0, 0, 1),
        initial_container="cup" if oid.startswith("bead") else None)
        for oid, pos in positions.items()}
    scene = Scene(SimpleNamespace(object_id="table"), objects, {})
    class Probe:
        support_id = "table"
        def __init__(self, *args, **kwargs): pass
        def aabb(self, oid):
            if oid == "table":
                return np.array([-1, -1, 0]), np.array([1, 1, .75])
            pos = np.array(positions[oid])
            half = np.array([.06, .06, .05]) if oid in {"cup", "bowl"} else np.array([.015]*3)
            return pos - half, pos + half
        def close(self): pass
    monkeypatch.setattr(e, "World", Probe)
    monkeypatch.setattr(e, "prepare_bodies", lambda *args: {})
    result, info = e.droid_canonical_layout(scene, tmp_path)
    for oid in ("bead_0", "bead_1", "bead_2"):
        before = np.array(positions[oid]) - np.array(positions["cup"])
        after = np.array(result.objects[oid].position) - np.array(result.objects["cup"].position)
        np.testing.assert_allclose(after, before)
        assert info["moves"][oid]["containment_group"] == "cup"

def test_outside_source_contents_cannot_be_certified(monkeypatch):
    from wo import predicates
    from wo.expert.run import initial_containment_evidence
    def evaluate(world, predicate):
        return predicate.subject != "bead_2", {"supported_by": "cup"}
    monkeypatch.setattr(predicates, "evaluate", evaluate)
    with pytest.raises(RuntimeError, match="bead_2"):
        initial_containment_evidence(object(), collection_spec())
