from types import SimpleNamespace
import pytest
from autoresearch.cosmos_scene_state import live_object_records, dynamic_state_checkpoint


def binding(body, semantic, dynamic=True):
    return SimpleNamespace(body_name=body, trajectory_object_name=semantic, dynamic=dynamic)


def test_all_independent_contents_use_live_states_with_semantic_aliases():
    reference = {name: {"position": [9, 9, 9], "role": "manipuland"}
                 for name in ("source_container", "cube_renamed_a", "cube_renamed_b")}
    bindings = [binding("cup", "source_container"), binding("body_a", "cube_renamed_a"),
                binding("body_b", "cube_renamed_b")]
    states = {body: {"position": [idx, 0, .5]} for idx, body in enumerate(("cup", "body_a", "body_b"))}
    records = live_object_records(reference, bindings, states)
    assert records["cube_renamed_a"]["position"] == [1, 0, .5]
    assert records["cube_renamed_b"]["position"] == [2, 0, .5]
    assert reference["cube_renamed_a"]["position"] == [9, 9, 9]
    checkpoint = dynamic_state_checkpoint(bindings, states)
    assert checkpoint["scene:cube_renamed_b:position"] == [2, 0, .5]


def test_missing_dynamic_feedback_cannot_silently_reuse_the_reference_frame():
    with pytest.raises(ValueError, match="no live native state"):
        live_object_records({"new_object": {"position": [0, 0, 0]}},
                            [binding("native_object", "new_object")], {})
