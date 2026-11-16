"""Source contents retain layout and must be physically inside at start."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from wo.spec import TaskSpec, TaskObject
from wo.scene import SceneObject, canonicalize_support_aliases
from wo.expert import embodiment as e
from wo.expert.run import initial_containment_evidence
from wo.tests.test_embodiment_layout import Scene


def spec():
    return TaskSpec.from_dict({"schema_version": "wo_task_spec.v1", "case_id": "contents",
        "family": "transfer", "instruction": "Transfer the bead.",
        "objects": [
            {"object_id": "bead", "category": "bead", "role": "manipuland", "initial_container": "cup"},
            {"object_id": "cup", "category": "cup", "role": "destination"},
            {"object_id": "bowl", "category": "bowl", "role": "destination"},
            {"object_id": "table", "category": "table", "role": "support"}],
        "steps": [{"primitive": "transport", "subject": "bead", "target": "bowl",
                   "goal": {"kind": "inside", "subject": "bead", "target": "bowl"}}]})


def test_spec_roundtrip_and_invalid_containers():
    s = spec()
    assert TaskSpec.from_dict(s.to_dict()).object("bead").initial_container == "cup"
    for parent in ("missing", "bead"):
        with pytest.raises(ValueError, match="invalid initial container"):
            replace(s, objects=tuple(replace(o, initial_container=parent) if o.object_id == "bead" else o for o in s.objects)).validate()
    with pytest.raises(ValueError, match="cycle"):
        replace(s, objects=tuple(replace(o, initial_container="bead") if o.object_id == "cup" else o for o in s.objects)).validate()


def test_canonical_layout_preserves_contents_xyz(monkeypatch, tmp_path):
    positions = {"cup": (0, 0, .65), "bead_0": (.01, .01, .69),
                 "bead_1": (-.01, 0, .68), "bead_2": (0, -.01, .7), "bowl": (.3, 0, .75)}
    objects = {oid: SceneObject(oid, "destination", tmp_path / "a.sdf", pos, (0, 0, 0, 1),
                   initial_container="cup" if oid.startswith("bead") else None) for oid, pos in positions.items()}
    scene = Scene(SimpleNamespace(object_id="table"), objects, {})
    class Probe:
        support_id = "table"
        def __init__(self, *args, **kwargs): pass
        def aabb(self, oid):
            if oid == "table": return np.array([-1, -1, 0]), np.array([1, 1, .75])
            half = np.array([.06, .06, .05]) if oid in {"cup", "bowl"} else np.array([.015]*3)
            return np.array(positions[oid]) - half, np.array(positions[oid]) + half
        def close(self): pass
    monkeypatch.setattr(e, "World", Probe)
    monkeypatch.setattr(e, "prepare_bodies", lambda *args: {})
    result, info = e.droid_canonical_layout(scene, tmp_path)
    for oid in ("bead_0", "bead_1", "bead_2"):
        np.testing.assert_allclose(np.array(result.objects[oid].position) - result.objects["cup"].position,
                                   np.array(positions[oid]) - positions["cup"])
        assert info["moves"][oid]["containment_group"] == "cup"
    assert result.objects["cup"].position[2] > positions["cup"][2]


def test_initial_containment_is_required_and_recorded(monkeypatch):
    from wo import predicates
    monkeypatch.setattr(predicates, "evaluate", lambda world, p: (True, {"support": p.target}))
    assert initial_containment_evidence(object(), spec()) == [
        {"subject": "bead", "container": "cup", "passed": True, "evidence": {"support": "cup"}}]
    monkeypatch.setattr(predicates, "evaluate", lambda world, p: (False, {}))
    with pytest.raises(RuntimeError, match="bead"):
        initial_containment_evidence(object(), spec())


def test_support_alias_remaps_initial_relation(tmp_path):
    s = spec()
    s = replace(s, objects=tuple(replace(o, initial_container="cup") if o.object_id == "bead" else o for o in s.objects))
    (tmp_path / "task_inventory_certificate.json").write_text(json.dumps({"matches": [
        {"task_object_id": "table", "scene_object_id": "table0", "status": "matched"},
        {"task_object_id": "cup", "scene_object_id": "table0", "status": "matched"}]}))
    result = canonicalize_support_aliases(s, tmp_path)
    assert result.object("bead").initial_container == "table"
