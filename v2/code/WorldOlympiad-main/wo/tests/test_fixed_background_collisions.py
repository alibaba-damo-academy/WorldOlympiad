from types import SimpleNamespace

import pytest

from wo.expert.kin import COLLISION_MARGIN_M
from wo.models.lingbot_va_client import initial_background_collision_audit


class ContactWorld:
    def __init__(self, contacts):
        self.points = contacts
        self.refreshed = False
        self.queries = []

    def refresh(self):
        self.refreshed = True

    def contacts(self, actor, oid):
        assert self.refreshed
        self.queries.append((actor, oid))
        return [SimpleNamespace(separation=p) for p in self.points[oid]]


def test_legacy_empty_selection_does_not_change_measurement_path():
    world = ContactWorld({})
    assert initial_background_collision_audit({}, world) is None
    assert not world.refreshed and not world.queries


def test_actual_crowded_base_penetration_rejects_input_even_without_impulse():
    world = ContactWorld({"counter": [-.06861606240272522, -.05646586790680885]})
    audit = initial_background_collision_audit({"fixed_background_ids": ["counter"]}, world)
    assert audit["status"] == "input_mismatch"
    assert audit["violations"] == ["counter"]
    assert audit["penetration_margin_m"] == COLLISION_MARGIN_M
    assert audit["robot_contacts"]["counter"]["minimum_separation_m"] == pytest.approx(-.06861606240272522)
    assert world.queries == [("robot", "counter")]


def test_legal_support_floor_contact_and_positive_clearance_are_not_penetration():
    world = ContactWorld({"floor": [-.000401521276216954, 0., .001], "fixture": [.003]})
    audit = initial_background_collision_audit({"fixed_background_ids": ["floor", "fixture"]}, world)
    assert audit["status"] == "ready" and audit["violations"] == []
    assert audit["robot_contacts"]["floor"]["count"] == 3


def test_deep_penetration_has_no_name_role_or_floor_height_exemption():
    world = ContactWorld({"floor": [-2 * COLLISION_MARGIN_M], "counter": []})
    audit = initial_background_collision_audit({"fixed_background_ids": ["floor", "counter"]}, world)
    assert audit["violations"] == ["floor"]
    assert audit["robot_contacts"]["counter"]["minimum_separation_m"] is None
