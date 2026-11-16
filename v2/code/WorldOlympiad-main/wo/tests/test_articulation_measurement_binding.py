import json
from types import SimpleNamespace
import numpy as np
import pytest
from wo.scene import resolved_articulations
from wo.eval import cae_episodes as C

CERT = {"articulations":{"door":{"host":"cabinet","joint_name":"hinge","child_link":"panel"}}}


def test_resolved_certificate_fills_unspecified_joint_names_without_mutation():
    declared = [{"component":"door","host":"cabinet","joint_name":None}]
    before = json.dumps(declared)
    rows = resolved_articulations(CERT, declared)
    assert rows[0]["joint_name"] == "hinge" and rows[0]["child_link"] == "panel"
    assert json.dumps(declared) == before


@pytest.mark.parametrize("field,value", [("host","other"),("joint_name","wrong"),("child_link","wrong")])
def test_conflicting_graph_binding_is_not_silently_repaired(field,value):
    with pytest.raises(ValueError,match="disagrees"):
        resolved_articulations(CERT,[{"component":"door",field:value}])


def test_legacy_graph_recovers_bound_reference_metadata(tmp_path):
    (tmp_path / "certificate.json").write_text(json.dumps(CERT))
    assert C.episode_articulations({"articulation_components":[]},tmp_path)[0]["component"] == "door"
    with pytest.raises(ValueError,match="lacks"):
        resolved_articulations({},[{"component":"door","host":"cabinet"}])


def test_missing_required_subject_cannot_reduce_cae_to_tcp_only(tmp_path,monkeypatch):
    ref = tmp_path / "ref";ref.mkdir()
    (ref / "certificate.json").write_text('{}')
    (tmp_path / "episode.json").write_text(json.dumps({"reference":str(ref),"task_success":False}))
    (tmp_path / "task_graph.json").write_text(json.dumps({"success_program":[{"subject":"object"}]}))
    monkeypatch.setattr(C,"verify_binding",lambda *a:{"status":"verified"})
    monkeypatch.setattr(C,"entity_tracks",lambda *a:{"end_effector":SimpleNamespace(positions=np.zeros((4,3)))})
    monkeypatch.setattr(C,"score",lambda *a,**k:pytest.fail("partial CAE must not be scored"))
    with pytest.raises(ValueError,match="object"):
        C.episode_cae(tmp_path)


def test_complete_recorded_tracks_including_static_subject_are_accepted():
    tracks = {k:SimpleNamespace(positions=np.zeros((4,3))) for k in ["end_effector","object"]}
    C.require_entity_tracks(tracks,["end_effector","object"],"native")


def test_declared_metadata_is_kept_for_legacy_certificates():
    old = [{"component":"door","host":"cabinet","joint_name":"hinge","child_link":"panel"}]
    assert resolved_articulations({},old) == old


def test_cae_selects_current_reference_urdf_not_a_stale_sorted_match(tmp_path):
    directory = tmp_path / "physics_sdf/cabinet_s1.000"
    directory.mkdir(parents=True)
    old = directory / "a_old.urdf";old.write_text("old")
    current = directory / "z_current.urdf";current.write_text("current")
    source = tmp_path / "z_current.sdf"
    (directory / ".ok").write_text(str(source))
    (directory / ".path").write_text(str(current))
    (tmp_path / "scene_inputs.json").write_text(json.dumps({"support":{"object_id":"table"},
        "objects":{"cabinet":{"sdf_path":str(source),"scale":1.0}}}))
    assert C.articulation_urdf(tmp_path,"cabinet") == current
    # A stale marker cannot redirect the reader back to the old artifact.
    (directory / ".ok").write_text("different source")
    (directory / ".path").write_text(str(old))
    assert C.articulation_urdf(tmp_path,"cabinet") == current


def test_ambiguous_legacy_urdf_does_not_silently_select_first(tmp_path):
    directory = tmp_path / "physics_sdf/cabinet_s1.000"
    directory.mkdir(parents=True)
    for name in ["old.urdf","current.urdf"]:(directory / name).write_text("asset")
    with pytest.raises(ValueError,match="ambiguous"):
        C.articulation_urdf(tmp_path,"cabinet")


def test_fk_rejects_an_unrecorded_joint_instead_of_using_zero(tmp_path,monkeypatch):
    urdf = tmp_path / "asset.urdf"
    monkeypatch.setitem(C._ART_FK,str(urdf),SimpleNamespace(joint={"hinge":0},link_index={"panel":0}))
    with pytest.raises(ValueError,match="binding"):
        C.moving_part_position(urdf,{},"wrong_joint","panel",.8)
