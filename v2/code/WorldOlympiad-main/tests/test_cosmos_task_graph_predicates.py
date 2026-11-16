from copy import deepcopy
from pathlib import Path
import pytest
from autoresearch.cosmos_task_graph_predicates import NativeTaskGraphPredicates, protocol


def body(name, identity, *, static=False):
    geometry = {"kind": "box", "size": [.1, .1, .1], "pose": {}}
    return {"name": name, "static": static,
        "pose": {"xyz": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
        "metadata": {"task_object_id": identity, "trajectory_object_name": identity},
        "links": [{"pose": {}, "visuals": [deepcopy(geometry)], "collisions": [deepcopy(geometry)]}]}


def state(x=0, z=0, *, velocity=0):
    return {"position": [x, 0, z], "orientation_xyzw": [0, 0, 0, 1],
            "linear_velocity": [velocity, 0, 0], "angular_velocity": [0, 0, 0]}


def test_protocol_budget_is_independent_of_reference_timeline():
    assert protocol()["episode_length_s"] == 60
    assert protocol()["control_hz"] == 15
    assert "reference" not in {k for k in protocol() if k != "policy"}


def test_all_contents_must_enter_target_and_remain_stable_with_arbitrary_ids():
    graph = {"success_program": [{"predicate": "inside", "subject": "new_contents", "object": "new_bin"}]}
    plan = {"bodies": [body("a", "new_contents"), body("b", "new_contents"), body("tray", "new_bin", static=True)]}
    verifier = NativeTaskGraphPredicates(graph, plan, Path("."))
    frame = {"time_s": 0., "objects": {"new_contents": state()}, "tray": state(),
             "robot_cup_contact_links": [], "object_container_contact": False}
    # Independent instances have distinct trajectory identities, sharing one semantic group.
    plan["bodies"][0]["metadata"]["trajectory_object_name"] = "piece_a"
    plan["bodies"][1]["metadata"]["trajectory_object_name"] = "piece_b"
    frame["objects"] = {"piece_a": state(), "piece_b": state(x=.2)}
    assert verifier.update(frame) is False
    frame["time_s"] = .6
    assert verifier.update(frame) is False
    frame["objects"]["piece_b"] = state()
    frame["time_s"] = .7
    assert verifier.update(frame) is False
    frame["time_s"] = 1.3
    assert verifier.update(frame) is True


def test_static_contact_is_not_a_measured_impact():
    graph = {"success_program": [{"predicate": "contact", "subject": "new_tool", "object": "new_target", "impact_required": True}]}
    plan = {"bodies": [body("cup", "new_tool"), body("tray", "new_target", static=True)]}
    verifier = NativeTaskGraphPredicates(graph, plan, Path("."))
    frame = {"time_s": 0., "cup": state(), "tray": state(),
             "objects": {}, "object_container_contact": True, "robot_cup_contact_links": ["finger"]}
    assert verifier.update(frame) is False
    frame["time_s"] = 1.
    assert verifier.update(frame) is False
    assert verifier.last_evidence["impact_goals"] == []


def test_successful_pour_measures_contents_outside_the_lifted_tilted_source():
    graph = {"success_program": [{"predicate": "inside", "subject": "pieces", "object": "bin",
                                   "pour_required": True, "source_container": "source"}]}
    plan = {"bodies": [body("cup", "source"), body("piece", "pieces"), body("tray", "bin", static=True)]}
    verifier = NativeTaskGraphPredicates(graph, plan, Path("."))
    frame = {"time_s": 0., "cup": state(), "tray": state(x=1), "objects": {"pieces": state()},
             "robot_cup_contact_links": ["left", "right"], "object_container_contact": False}
    verifier.update(frame)
    frame["cup"] = state(x=1, z=1)
    frame["cup"]["orientation_xyzw"] = [0, 2**-.5, 0, 2**-.5]
    frame["objects"]["pieces"] = state(x=1)
    frame["time_s"] = .2
    assert verifier.update(frame) is False
    frame["time_s"] = .8
    assert verifier.update(frame) is True


def test_actual_moving_contact_after_grasp_is_measured_once():
    graph = {"success_program": [{"predicate": "contact", "subject": "tool", "object": "target", "impact_required": True}]}
    plan = {"bodies": [body("cup", "tool"), body("tray", "target", static=True)]}
    verifier = NativeTaskGraphPredicates(graph, plan, Path("."))
    frame = {"time_s": 0., "cup": state(velocity=.2), "tray": state(), "objects": {},
             "object_container_contact": False, "robot_cup_contact_links": ["finger"]}
    verifier.update(frame)
    frame["time_s"] = .1; frame["object_container_contact"] = True
    assert verifier.update(frame) is False
    frame["time_s"] = .7
    assert verifier.update(frame) is True


def test_support_region_accepts_off_center_placement_and_checks_declared_upright():
    graph = {"success_program": [{"predicate": "on_region", "subject": "renamed_item", "object": "broad_surface", "terminal_upright": True}]}
    surface = body("tray", "broad_surface", static=True)
    surface["links"][0]["visuals"][0]["size"] = [2., 2., .1]
    surface["links"][0]["collisions"][0]["size"] = [2., 2., .1]
    verifier = NativeTaskGraphPredicates(graph, {"bodies": [body("cup", "renamed_item"), surface]}, Path("."))
    frame = {"time_s": 0., "cup": state(x=.35, z=.1), "tray": state(), "objects": {},
             "object_container_contact": True, "robot_cup_contact_links": []}
    frame["cup"]["orientation_xyzw"] = [2**-.5, 0, 0, 2**-.5]
    assert verifier.update(frame) is False
    frame["time_s"] = 1.; frame["cup"]["orientation_xyzw"] = [0, 0, 0, 1]
    assert verifier.update(frame) is False
    frame["time_s"] = 1.6
    assert verifier.update(frame) is True


def test_terminal_upright_uses_declared_local_semantic_axis():
    graph = {"success_program": [{"predicate": "on_region", "subject": "item",
                                   "object": "tray", "terminal_upright": True,
                                   "terminal_upright_axis_local": [0, 1, 0],
                                   "terminal_upright_axis_world": [0, 0, 1]}]}
    verifier = NativeTaskGraphPredicates(graph, {"bodies": [
        body("cup", "item"), body("tray", "tray", static=True)]}, Path("."))
    frame = {"time_s": 0., "cup": state(z=.1), "tray": state(),
             "objects": {}, "robot_cup_contact_links": []}
    # Identity leaves the measured base-to-cap Y axis horizontal: unsolved.
    assert verifier.update(frame) is False
    frame["time_s"] = 1.0
    frame["cup"]["orientation_xyzw"] = [2**-.5, 0, 0, 2**-.5]
    assert verifier.update(frame) is False
    frame["time_s"] = 1.6
    assert verifier.update(frame) is True


def test_a_place_operation_cannot_succeed_while_still_held_inside_target():
    graph = {"controller_program": [{"operator": "place", "subjects": ["item"], "target": "box"}],
             "success_program": [{"predicate": "inside", "subject": "item", "object": "box"}]}
    verifier = NativeTaskGraphPredicates(graph, {"bodies": [body("cup", "item"), body("tray", "box", static=True)]}, Path("."))
    frame = {"time_s": 0., "cup": state(), "tray": state(), "objects": {},
             "object_container_contact": True, "robot_cup_contact_links": ["left", "right"]}
    verifier.update(frame)
    frame["time_s"] = 1.
    assert verifier.update(frame) is False
    frame["robot_cup_contact_links"] = []; frame["time_s"] = 1.1
    verifier.update(frame)
    frame["time_s"] = 1.7
    assert verifier.update(frame) is True
