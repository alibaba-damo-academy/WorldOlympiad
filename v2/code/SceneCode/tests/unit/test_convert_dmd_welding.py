"""Unit tests for convert_dmd_welding script."""

import copy
import unittest

from scripts.convert_dmd_welding import (
    build_object_registry,
    convert_dmd,
    convert_free_to_welded,
    convert_welded_to_free,
    extract_free_body_pose,
    extract_weld_pose,
    is_frame_weld,
    should_be_welded,
)

# Reusable pose fixture.
SAMPLE_POSE = {
    "translation": [1.0, 2.0, 0.5],
    "rotation": {"angle_axis_d": [0.0, 0.0, 1.0, 0.0]},
}

PARENT_POSE = {
    "translation": [3.0, 4.0, 0.0],
    "rotation": {"angle_axis_d": [0.0, 0.0, 1.0, 0.0]},
}

RELATIVE_POSE = {
    "translation": [0.5, 0.0, 0.3],
    "rotation": {"angle_axis_d": [0.0, 0.0, 1.0, 0.0]},
}


def _make_registry_entry(
    object_type, is_composite_member=False, parent_model_name=None
):
    """Create a minimal object registry entry."""
    return {
        "object_type": object_type,
        "room_id": "room",
        "object_id": "obj",
        "is_composite_member": is_composite_member,
        "parent_model_name": parent_model_name,
    }


def _make_add_model(name, free_pose=None):
    """Create an add_model directive."""
    model = {"add_model": {"name": name, "file": f"{name}.sdf"}}
    if free_pose is not None:
        model["add_model"]["default_free_body_pose"] = {"base_link": free_pose}
    return model


def _make_world_weld(model_name, pose, link_name="base_link"):
    """Create a world weld directive."""
    return {
        "add_weld": {
            "parent": "world",
            "child": f"{model_name}::{link_name}",
            "X_PC": pose,
        }
    }


def _make_frame_weld(frame_name, model_name, pose, link_name="base_link"):
    """Create a weld to a named frame (e.g., room_bedroom_frame)."""
    return {
        "add_weld": {
            "parent": frame_name,
            "child": f"{model_name}::{link_name}",
            "X_PC": pose,
        }
    }


def _make_add_model_with_base_frame(name, free_pose, base_frame, link_name="base_link"):
    """Create an add_model directive with base_frame in the free body pose."""
    return {
        "add_model": {
            "name": name,
            "file": f"{name}.sdf",
            "default_free_body_pose": {
                link_name: {**free_pose, "base_frame": base_frame},
            },
        }
    }


def _make_non_world_weld(
    parent_name,
    child_name,
    pose,
    parent_link="base_link",
    child_link="base_link",
):
    """Create a non-world (composite) weld directive."""
    return {
        "add_weld": {
            "parent": f"{parent_name}::{parent_link}",
            "child": f"{child_name}::{child_link}",
            "X_PC": pose,
        }
    }


class TestShouldBeWelded(unittest.TestCase):
    """Tests for should_be_welded function."""

    def test_room_geometry_always_welded(self):
        """Room geometry returns True regardless of mode."""
        for mode in ("nothing", "furniture", "all"):
            result = should_be_welded("room_geometry_0", {}, mode)
            assert result is True

    def test_frame_always_welded(self):
        """Frame models return True regardless of mode."""
        for mode in ("nothing", "furniture", "all"):
            result = should_be_welded("kitchen_frame", {}, mode)
            assert result is True

    def test_wall_mounted_always_welded(self):
        """Wall-mounted objects return True in all modes."""
        registry = {"shelf": _make_registry_entry("wall_mounted")}
        for mode in ("nothing", "furniture", "all"):
            assert should_be_welded("shelf", registry, mode) is True

    def test_ceiling_mounted_always_welded(self):
        """Ceiling-mounted objects return True in all modes."""
        registry = {"light": _make_registry_entry("ceiling_mounted")}
        for mode in ("nothing", "furniture", "all"):
            assert should_be_welded("light", registry, mode) is True

    def test_manipuland_free_in_nothing_mode(self):
        """Manipulands are free in 'nothing' mode."""
        registry = {"cup": _make_registry_entry("manipuland")}
        assert should_be_welded("cup", registry, "nothing") is False

    def test_manipuland_free_in_furniture_mode(self):
        """Manipulands are free in 'furniture' mode."""
        registry = {"cup": _make_registry_entry("manipuland")}
        assert should_be_welded("cup", registry, "furniture") is False

    def test_manipuland_welded_in_all_mode(self):
        """Manipulands are welded in 'all' mode (the bug fix)."""
        registry = {"cup": _make_registry_entry("manipuland")}
        assert should_be_welded("cup", registry, "all") is True

    def test_furniture_free_in_nothing_mode(self):
        """Furniture is free in 'nothing' mode."""
        registry = {"table": _make_registry_entry("furniture")}
        assert should_be_welded("table", registry, "nothing") is False

    def test_furniture_welded_in_furniture_mode(self):
        """Furniture is welded in 'furniture' mode."""
        registry = {"table": _make_registry_entry("furniture")}
        assert should_be_welded("table", registry, "furniture") is True

    def test_furniture_welded_in_all_mode(self):
        """Furniture is welded in 'all' mode."""
        registry = {"table": _make_registry_entry("furniture")}
        assert should_be_welded("table", registry, "all") is True

    def test_unknown_type_raises_value_error(self):
        """Unknown object type raises ValueError."""
        registry = {"thing": _make_registry_entry("unknown_type")}
        with self.assertRaises(ValueError, msg="Unknown object type"):
            should_be_welded("thing", registry, "furniture")

    def test_model_not_in_registry_raises_value_error(self):
        """Model not in registry raises ValueError."""
        with self.assertRaises(ValueError, msg="not found"):
            should_be_welded("missing_model", {}, "furniture")


class TestExtractWeldPose(unittest.TestCase):
    """Tests for extract_weld_pose function."""

    def test_extracts_x_pc(self):
        """Extracts X_PC from weld directive."""
        weld = _make_world_weld("model", SAMPLE_POSE)
        result = extract_weld_pose(weld)
        assert result == SAMPLE_POSE

    def test_raises_on_non_weld(self):
        """Raises ValueError for non-weld directives."""
        with self.assertRaises(ValueError):
            extract_weld_pose({"add_model": {}})

    def test_raises_on_missing_x_pc(self):
        """Raises ValueError when X_PC is missing."""
        with self.assertRaises(ValueError):
            extract_weld_pose({"add_weld": {"parent": "world"}})


class TestExtractFreeBodyPose(unittest.TestCase):
    """Tests for extract_free_body_pose function."""

    def test_extracts_pose_for_link(self):
        """Extracts pose for the specified link."""
        model = _make_add_model("obj", free_pose=SAMPLE_POSE)
        result = extract_free_body_pose(model, link_name="base_link")
        assert result == SAMPLE_POSE

    def test_returns_none_for_missing_link(self):
        """Returns None when link is not in free body pose."""
        model = _make_add_model("obj", free_pose=SAMPLE_POSE)
        result = extract_free_body_pose(model, link_name="other_link")
        assert result is None

    def test_returns_none_when_no_free_pose(self):
        """Returns None when model has no default_free_body_pose."""
        model = _make_add_model("obj")
        result = extract_free_body_pose(model)
        assert result is None


class TestConvertFreeToWelded(unittest.TestCase):
    """Tests for convert_free_to_welded function."""

    def test_creates_weld_and_strips_free_pose(self):
        """Converts free body to welded model and weld directive."""
        model = _make_add_model("table", free_pose=SAMPLE_POSE)
        new_model, new_weld = convert_free_to_welded(model)

        assert "default_free_body_pose" not in new_model["add_model"]
        assert new_weld["add_weld"]["parent"] == "world"
        assert new_weld["add_weld"]["child"] == "table::base_link"
        assert new_weld["add_weld"]["X_PC"] == SAMPLE_POSE

    def test_does_not_mutate_original(self):
        """Original directive is not modified."""
        model = _make_add_model("table", free_pose=SAMPLE_POSE)
        original = copy.deepcopy(model)
        convert_free_to_welded(model)
        assert model == original

    def test_uses_base_frame_as_weld_parent(self):
        """When free body pose has base_frame, weld parent uses that frame."""
        model = _make_add_model_with_base_frame(
            "room_table_0", SAMPLE_POSE, "room_bedroom_frame"
        )
        new_model, new_weld = convert_free_to_welded(model)

        assert new_weld["add_weld"]["parent"] == "room_bedroom_frame"
        assert new_weld["add_weld"]["child"] == "room_table_0::base_link"
        assert "base_frame" not in new_weld["add_weld"]["X_PC"]


class TestConvertWeldedToFree(unittest.TestCase):
    """Tests for convert_welded_to_free function."""

    def test_creates_free_body_pose(self):
        """Converts welded model to free body with pose."""
        model = _make_add_model("table")
        weld = _make_world_weld("table", SAMPLE_POSE)
        result = convert_welded_to_free(model, weld, link_name="base_link")

        free_pose = result["add_model"]["default_free_body_pose"]
        assert free_pose["base_link"] == SAMPLE_POSE

    def test_does_not_mutate_original(self):
        """Original directive is not modified."""
        model = _make_add_model("table")
        original = copy.deepcopy(model)
        weld = _make_world_weld("table", SAMPLE_POSE)
        convert_welded_to_free(model, weld)
        assert model == original

    def test_preserves_parent_frame_as_base_frame(self):
        """When weld parent is a room frame, free body pose has base_frame."""
        model = _make_add_model("room_table_0")
        weld = _make_frame_weld("room_bedroom_frame", "room_table_0", SAMPLE_POSE)
        result = convert_welded_to_free(model, weld, link_name="base_link")

        free_pose = result["add_model"]["default_free_body_pose"]
        assert free_pose["base_link"]["base_frame"] == "room_bedroom_frame"
        # Pose values should still be present.
        assert free_pose["base_link"]["translation"] == SAMPLE_POSE["translation"]


class TestBuildObjectRegistry(unittest.TestCase):
    """Tests for build_object_registry function."""

    def test_builds_registry_from_house_state(self):
        """Builds correct registry from house_state dict."""
        house_state = {
            "rooms": {
                "kitchen": {
                    "objects": {
                        "table_0": {"object_type": "furniture"},
                        "cup_0": {"object_type": "manipuland"},
                    }
                }
            }
        }
        registry = build_object_registry(house_state)
        assert "kitchen_table_0" in registry
        assert registry["kitchen_table_0"]["object_type"] == "furniture"
        assert "kitchen_cup_0" in registry
        assert registry["kitchen_cup_0"]["object_type"] == "manipuland"

    def test_skips_wall_objects(self):
        """Walls are excluded from registry."""
        house_state = {
            "rooms": {
                "room": {
                    "objects": {
                        "wall_0": {"object_type": "wall"},
                        "table_0": {"object_type": "furniture"},
                    }
                }
            }
        }
        registry = build_object_registry(house_state)
        assert "room_wall_0" not in registry
        assert "room_table_0" in registry

    def test_handles_composite_members(self):
        """Registers composite member assets with correct metadata."""
        house_state = {
            "rooms": {
                "room": {
                    "objects": {
                        "stack_0": {
                            "object_type": "manipuland",
                            "member_assets": [
                                {"name": "book_0"},
                                {"name": "book_1"},
                            ],
                        }
                    }
                }
            }
        }
        registry = build_object_registry(house_state)
        assert "room_stack_0" in registry
        assert registry["room_stack_0"]["is_composite_member"] is False

        assert "room_stack_0_member_0" in registry
        assert registry["room_stack_0_member_0"]["is_composite_member"] is True
        assert registry["room_stack_0_member_0"]["parent_model_name"] == "room_stack_0"

        assert "room_stack_0_member_1" in registry


class TestConvertDmd(unittest.TestCase):
    """Integration-style tests for convert_dmd using synthetic directives."""

    def test_free_furniture_welded_in_furniture_mode(self):
        """Free furniture becomes welded in 'furniture' mode."""
        registry = {"room_table_0": _make_registry_entry("furniture")}
        directives = [_make_add_model("room_table_0", free_pose=SAMPLE_POSE)]

        result = convert_dmd(directives, registry, mode="furniture")

        assert len(result) == 2
        assert "default_free_body_pose" not in result[0]["add_model"]
        assert result[1]["add_weld"]["parent"] == "world"

    def test_welded_furniture_freed_in_nothing_mode(self):
        """Welded furniture becomes free in 'nothing' mode."""
        registry = {"room_table_0": _make_registry_entry("furniture")}
        directives = [
            _make_add_model("room_table_0"),
            _make_world_weld("room_table_0", SAMPLE_POSE),
        ]

        result = convert_dmd(directives, registry, mode="nothing")

        assert len(result) == 1
        free_pose = result[0]["add_model"]["default_free_body_pose"]
        assert "base_link" in free_pose

    def test_already_welded_furniture_stays_welded_in_all_mode(self):
        """Already-welded furniture passes through in 'all' mode."""
        registry = {"room_table_0": _make_registry_entry("furniture")}
        directives = [
            _make_add_model("room_table_0"),
            _make_world_weld("room_table_0", SAMPLE_POSE),
        ]

        result = convert_dmd(directives, registry, mode="all")

        assert len(result) == 2
        assert "add_weld" in result[1]

    def test_manipuland_welded_in_all_mode(self):
        """Free manipuland becomes welded in 'all' mode (primary bug fix)."""
        registry = {"room_cup_0": _make_registry_entry("manipuland")}
        directives = [_make_add_model("room_cup_0", free_pose=SAMPLE_POSE)]

        result = convert_dmd(directives, registry, mode="all")

        assert len(result) == 2
        assert "default_free_body_pose" not in result[0]["add_model"]
        assert result[1]["add_weld"]["parent"] == "world"
        assert result[1]["add_weld"]["child"] == "room_cup_0::base_link"

    def test_manipuland_stays_free_in_furniture_mode(self):
        """Manipuland stays free in 'furniture' mode."""
        registry = {"room_cup_0": _make_registry_entry("manipuland")}
        directives = [_make_add_model("room_cup_0", free_pose=SAMPLE_POSE)]

        result = convert_dmd(directives, registry, mode="furniture")

        assert len(result) == 1
        assert "default_free_body_pose" in result[0]["add_model"]

    def test_unregistered_free_model_welded_in_all_mode(self):
        """Unregistered model with free pose is welded in 'all' mode."""
        registry = {}  # Empty registry, model not registered.
        directives = [_make_add_model("room_stack_0_s0_0", free_pose=SAMPLE_POSE)]

        result = convert_dmd(directives, registry, mode="all")

        assert len(result) == 2
        assert "default_free_body_pose" not in result[0]["add_model"]
        assert result[1]["add_weld"]["parent"] == "world"

    def test_unregistered_model_unchanged_in_furniture_mode(self):
        """Unregistered model passes through in 'furniture' mode."""
        registry = {}
        directives = [_make_add_model("room_stack_0_s0_0", free_pose=SAMPLE_POSE)]

        result = convert_dmd(directives, registry, mode="furniture")

        assert len(result) == 1
        assert "default_free_body_pose" in result[0]["add_model"]

    def test_unregistered_nw_weld_child_welded_to_world_in_all_mode(self):
        """Unregistered composite child with non-world weld gets welded
        to world in 'all' mode when parent pose is known."""
        registry = {}
        parent_model = _make_add_model("room_stack_0", free_pose=PARENT_POSE)
        child_model = _make_add_model("room_stack_0_s0_0")
        nw_weld = _make_non_world_weld(
            "room_stack_0", "room_stack_0_s0_0", RELATIVE_POSE
        )

        directives = [parent_model, child_model, nw_weld]
        result = convert_dmd(directives, registry, mode="all")

        # Parent should be welded (2 directives: model + weld).
        # Child should be welded to world (2 directives: model + weld).
        assert len(result) == 4
        # Check child weld is to world.
        child_weld = result[3]
        assert child_weld["add_weld"]["parent"] == "world"
        assert "room_stack_0_s0_0" in child_weld["add_weld"]["child"]

    def test_room_geometry_passes_through(self):
        """Room geometry directives pass through unchanged."""
        registry = {}
        directives = [_make_add_model("room_geometry_0")]

        result = convert_dmd(directives, registry, mode="all")

        assert len(result) == 1
        assert result[0]["add_model"]["name"] == "room_geometry_0"


class TestConvertDmdWithRoomFrames(unittest.TestCase):
    """Tests for convert_dmd with room-frame-relative poses."""

    def test_frame_welded_furniture_freed_in_nothing_mode(self):
        """Furniture welded to room frame becomes free with base_frame."""
        registry = {"room_table_0": _make_registry_entry("furniture")}
        directives = [
            _make_add_model("room_table_0"),
            _make_frame_weld("room_bedroom_frame", "room_table_0", SAMPLE_POSE),
        ]

        result = convert_dmd(directives, registry, mode="nothing")

        assert len(result) == 1
        free_pose = result[0]["add_model"]["default_free_body_pose"]
        assert "base_link" in free_pose
        assert free_pose["base_link"]["base_frame"] == "room_bedroom_frame"

    def test_free_furniture_with_base_frame_welded_in_furniture_mode(self):
        """Free furniture with base_frame welded to room frame."""
        registry = {"room_table_0": _make_registry_entry("furniture")}
        directives = [
            _make_add_model_with_base_frame(
                "room_table_0", SAMPLE_POSE, "room_bedroom_frame"
            ),
        ]

        result = convert_dmd(directives, registry, mode="furniture")

        assert len(result) == 2
        assert "default_free_body_pose" not in result[0]["add_model"]
        assert result[1]["add_weld"]["parent"] == "room_bedroom_frame"
        assert "base_frame" not in result[1]["add_weld"]["X_PC"]

    def test_frame_welded_wall_mounted_stays_welded(self):
        """Wall-mounted object welded to room frame stays welded."""
        registry = {"room_shelf_0": _make_registry_entry("wall_mounted")}
        directives = [
            _make_add_model("room_shelf_0"),
            _make_frame_weld("room_bedroom_frame", "room_shelf_0", SAMPLE_POSE),
        ]

        result = convert_dmd(directives, registry, mode="nothing")

        assert len(result) == 2
        assert result[1]["add_weld"]["parent"] == "room_bedroom_frame"

    def test_free_manipuland_with_base_frame_stays_free_in_furniture_mode(
        self,
    ):
        """Manipuland with base_frame stays free, base_frame preserved."""
        registry = {"room_cup_0": _make_registry_entry("manipuland")}
        directives = [
            _make_add_model_with_base_frame(
                "room_cup_0", SAMPLE_POSE, "room_bedroom_frame"
            ),
        ]

        result = convert_dmd(directives, registry, mode="furniture")

        assert len(result) == 1
        free_pose = result[0]["add_model"]["default_free_body_pose"]
        assert free_pose["base_link"]["base_frame"] == "room_bedroom_frame"

    def test_free_manipuland_with_base_frame_welded_in_all_mode(self):
        """Manipuland with base_frame welded to room frame in all mode."""
        registry = {"room_cup_0": _make_registry_entry("manipuland")}
        directives = [
            _make_add_model_with_base_frame(
                "room_cup_0", SAMPLE_POSE, "room_bedroom_frame"
            ),
        ]

        result = convert_dmd(directives, registry, mode="all")

        assert len(result) == 2
        assert "default_free_body_pose" not in result[0]["add_model"]
        assert result[1]["add_weld"]["parent"] == "room_bedroom_frame"
        assert "base_frame" not in result[1]["add_weld"]["X_PC"]

    def test_round_trip_frame_weld_to_free_and_back(self):
        """Weld to room frame -> free -> weld preserves room frame."""
        registry = {"room_table_0": _make_registry_entry("furniture")}

        # Start with frame weld.
        directives_welded = [
            _make_add_model("room_table_0"),
            _make_frame_weld("room_bedroom_frame", "room_table_0", SAMPLE_POSE),
        ]

        # Convert to free (nothing mode).
        freed = convert_dmd(directives_welded, registry, mode="nothing")
        assert len(freed) == 1

        # Convert back to welded (furniture mode).
        rewelded = convert_dmd(freed, registry, mode="furniture")
        assert len(rewelded) == 2
        assert rewelded[1]["add_weld"]["parent"] == "room_bedroom_frame"


class TestIsFrameWeld(unittest.TestCase):
    """Tests for is_frame_weld function."""

    def test_world_weld_is_frame_weld(self):
        """World weld is a frame weld."""
        weld = _make_world_weld("model", SAMPLE_POSE)
        assert is_frame_weld(weld) is True

    def test_room_frame_weld_is_frame_weld(self):
        """Room frame weld is a frame weld."""
        weld = _make_frame_weld("room_bedroom_frame", "model", SAMPLE_POSE)
        assert is_frame_weld(weld) is True

    def test_object_weld_is_not_frame_weld(self):
        """Object-to-object weld is not a frame weld."""
        weld = _make_non_world_weld("parent", "child", SAMPLE_POSE)
        assert is_frame_weld(weld) is False


if __name__ == "__main__":
    unittest.main()
