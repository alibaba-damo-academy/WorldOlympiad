"""Tests for geometry caching functionality."""

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock

from scenecode.agent_utils.house import HouseLayout, OpeningType, RoomSpec
from scenecode.floor_plan_agents.tools.geometry_cache import (
    GeometryCache,
    floor_cache_key,
    wall_cache_key,
    window_cache_key,
)
from scenecode.floor_plan_agents.tools.wall_geometry import WallOpening
from scenecode.utils.material import Material


class TestCacheKeyFunctions(unittest.TestCase):
    """Tests for cache key generation functions."""

    def test_floor_cache_key_deterministic(self) -> None:
        """Same inputs produce same key."""
        key1 = floor_cache_key(
            width=5.0,
            depth=4.0,
            thickness=0.1,
            material=Material.from_path(Path("materials/Wood094_1K-JPG")),
        )
        key2 = floor_cache_key(
            width=5.0,
            depth=4.0,
            thickness=0.1,
            material=Material.from_path(Path("materials/Wood094_1K-JPG")),
        )
        assert key1 == key2

    def test_floor_cache_key_different_dimensions(self) -> None:
        """Different dimensions produce different keys."""
        key1 = floor_cache_key(width=5.0, depth=4.0, thickness=0.1, material=None)
        key2 = floor_cache_key(width=6.0, depth=4.0, thickness=0.1, material=None)
        assert key1 != key2

    def test_floor_cache_key_different_material(self) -> None:
        """Different materials produce different keys."""
        key1 = floor_cache_key(
            width=5.0,
            depth=4.0,
            thickness=0.1,
            material=Material.from_path(Path("materials/Wood094_1K-JPG")),
        )
        key2 = floor_cache_key(
            width=5.0,
            depth=4.0,
            thickness=0.1,
            material=Material.from_path(Path("materials/Tile001_1K-JPG")),
        )
        assert key1 != key2

    def test_floor_cache_key_none_material(self) -> None:
        """None material produces valid key."""
        key = floor_cache_key(width=5.0, depth=4.0, thickness=0.1, material=None)
        assert len(key) == 16  # SHA-256 truncated to 16 chars.

    def test_wall_cache_key_deterministic(self) -> None:
        """Same inputs produce same key."""
        key1 = wall_cache_key(
            width=5.0,
            height=2.8,
            thickness=0.1,
            material=Material.from_path(Path("materials/Plaster001_1K-JPG")),
        )
        key2 = wall_cache_key(
            width=5.0,
            height=2.8,
            thickness=0.1,
            material=Material.from_path(Path("materials/Plaster001_1K-JPG")),
        )
        assert key1 == key2

    def test_wall_cache_key_different_dimensions(self) -> None:
        """Different dimensions produce different keys."""
        key1 = wall_cache_key(width=5.0, height=2.8, thickness=0.1, material=None)
        key2 = wall_cache_key(width=6.0, height=2.8, thickness=0.1, material=None)
        assert key1 != key2

    def test_wall_cache_key_with_openings(self) -> None:
        """Openings affect cache key."""
        openings = [
            {
                "position_along_wall": 1.0,
                "width": 0.9,
                "height": 2.0,
                "sill_height": 0.0,
            }
        ]
        key1 = wall_cache_key(
            width=5.0, height=2.8, thickness=0.1, material=None, openings=None
        )
        key2 = wall_cache_key(
            width=5.0, height=2.8, thickness=0.1, material=None, openings=openings
        )
        assert key1 != key2

    def test_window_cache_key_deterministic(self) -> None:
        """Same inputs produce same key."""
        key1 = window_cache_key(width=1.2, height=1.5, depth=0.1, is_horizontal=True)
        key2 = window_cache_key(width=1.2, height=1.5, depth=0.1, is_horizontal=True)
        assert key1 == key2

    def test_window_cache_key_different_orientation(self) -> None:
        """Different wall orientation produces different key."""
        key1 = window_cache_key(width=1.2, height=1.5, depth=0.1, is_horizontal=True)
        key2 = window_cache_key(width=1.2, height=1.5, depth=0.1, is_horizontal=False)
        assert key1 != key2


class TestWallOpening(unittest.TestCase):
    """Tests for WallOpening dataclass serialization."""

    def test_to_dict_serializes_opening_type_as_string(self) -> None:
        """OpeningType enum is serialized as string value, not raw enum."""
        opening = WallOpening(
            position_along_wall=1.0,
            width=0.9,
            height=2.1,
            sill_height=0.0,
            opening_type=OpeningType.DOOR,
        )
        result = opening.to_dict()
        assert result["opening_type"] == "door"
        assert isinstance(result["opening_type"], str)

    def test_to_dict_is_json_serializable(self) -> None:
        """WallOpening.to_dict() produces JSON-serializable output."""
        opening = WallOpening(
            position_along_wall=1.5,
            width=1.2,
            height=1.5,
            sill_height=0.9,
            opening_type=OpeningType.WINDOW,
        )
        result = opening.to_dict()
        # Should not raise TypeError: Object of type OpeningType is not JSON serializable.
        json_str = json.dumps(result)
        assert "window" in json_str

    def test_wall_cache_key_with_wall_opening_to_dict(self) -> None:
        """wall_cache_key works with WallOpening.to_dict() output."""
        opening = WallOpening(
            position_along_wall=1.0,
            width=0.9,
            height=2.0,
            sill_height=0.0,
            opening_type=OpeningType.DOOR,
        )
        # Should not raise TypeError.
        key = wall_cache_key(
            width=5.0,
            height=2.8,
            thickness=0.1,
            material=None,
            openings=[opening.to_dict()],
        )
        assert len(key) == 16


class TestHouseLayoutInvalidation(unittest.TestCase):
    """Tests for HouseLayout geometry invalidation methods."""

    def test_invalidate_room_geometry_returns_true_when_cached(self) -> None:
        """Invalidating a cached room returns True."""
        layout = HouseLayout(
            wall_height=2.8,
            room_specs=[RoomSpec(room_id="living_room", length=5.0, width=4.0)],
            house_dir=None,
        )
        layout.room_geometries["living_room"] = MagicMock()

        result = layout.invalidate_room_geometry("living_room")

        assert result is True
        assert "living_room" not in layout.room_geometries

    def test_invalidate_room_geometry_returns_false_when_not_cached(self) -> None:
        """Invalidating a non-cached room returns False."""
        layout = HouseLayout(
            wall_height=2.8,
            room_specs=[RoomSpec(room_id="living_room", length=5.0, width=4.0)],
            house_dir=None,
        )

        result = layout.invalidate_room_geometry("living_room")

        assert result is False

    def test_invalidate_room_geometry_only_affects_target(self) -> None:
        """Invalidating one room doesn't affect others."""
        layout = HouseLayout(
            wall_height=2.8,
            room_specs=[
                RoomSpec(room_id="living_room", length=5.0, width=4.0),
                RoomSpec(room_id="bedroom", length=4.0, width=3.5),
            ],
            house_dir=None,
        )
        layout.room_geometries["living_room"] = MagicMock()
        layout.room_geometries["bedroom"] = MagicMock()

        layout.invalidate_room_geometry("living_room")

        assert "living_room" not in layout.room_geometries
        assert "bedroom" in layout.room_geometries

    def test_invalidate_all_room_geometries_clears_all(self) -> None:
        """Invalidating all rooms clears the entire cache."""
        layout = HouseLayout(
            wall_height=2.8,
            room_specs=[
                RoomSpec(room_id="living_room", length=5.0, width=4.0),
                RoomSpec(room_id="bedroom", length=4.0, width=3.5),
                RoomSpec(room_id="kitchen", length=3.0, width=3.0),
            ],
            house_dir=None,
        )
        layout.room_geometries["living_room"] = MagicMock()
        layout.room_geometries["bedroom"] = MagicMock()
        layout.room_geometries["kitchen"] = MagicMock()

        count = layout.invalidate_all_room_geometries()

        assert count == 3
        assert len(layout.room_geometries) == 0

    def test_invalidate_all_room_geometries_returns_zero_when_empty(self) -> None:
        """Invalidating all rooms when cache is empty returns 0."""
        layout = HouseLayout(
            wall_height=2.8,
            room_specs=[],
            house_dir=None,
        )

        count = layout.invalidate_all_room_geometries()

        assert count == 0


class TestGeometryCache(unittest.TestCase):
    """Tests for GeometryCache class."""

    def test_cache_directory_structure_created(self) -> None:
        """Cache creates walls/floors/windows subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GeometryCache(cache_dir=Path(tmpdir))

            assert cache.walls_dir.exists()
            assert cache.floors_dir.exists()
            assert cache.windows_dir.exists()

    def test_get_or_create_wall_calls_create_on_miss(self) -> None:
        """First call to get_or_create_wall invokes create function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GeometryCache(cache_dir=Path(tmpdir))
            output_dir = Path(tmpdir) / "output"
            create_called = []

            def create_fn(output_path: Path) -> None:
                create_called.append(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("mock gltf")

            cache.get_or_create_wall(
                cache_key="test_key_123",
                output_dir=output_dir,
                create_fn=create_fn,
            )

            assert len(create_called) == 1
            assert cache.get_stats()["misses"] == 1
            assert cache.get_stats()["hits"] == 0

    def test_get_or_create_wall_uses_cache_on_hit(self) -> None:
        """Second call with same key uses cache, doesn't call create."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GeometryCache(cache_dir=Path(tmpdir))
            create_called = []

            def create_fn(output_path: Path) -> None:
                create_called.append(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("mock gltf")

            # First call - creates.
            output_dir1 = Path(tmpdir) / "output1"
            cache.get_or_create_wall(
                cache_key="shared_key",
                output_dir=output_dir1,
                create_fn=create_fn,
            )

            # Second call with same key - should use cache.
            output_dir2 = Path(tmpdir) / "output2"
            cache.get_or_create_wall(
                cache_key="shared_key",
                output_dir=output_dir2,
                create_fn=create_fn,
            )

            assert len(create_called) == 1  # Only called once.
            assert cache.get_stats()["misses"] == 1
            assert cache.get_stats()["hits"] == 1

    def test_get_or_create_floor_works(self) -> None:
        """get_or_create_floor follows same pattern as walls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GeometryCache(cache_dir=Path(tmpdir))
            output_dir = Path(tmpdir) / "output"
            create_called = []

            def create_fn(output_path: Path) -> None:
                create_called.append(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("mock floor gltf")

            result = cache.get_or_create_floor(
                cache_key="floor_key",
                output_dir=output_dir,
                create_fn=create_fn,
            )

            assert result.name == "floor.gltf"
            assert len(create_called) == 1

    def test_get_or_create_window_works(self) -> None:
        """get_or_create_window follows same pattern as walls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GeometryCache(cache_dir=Path(tmpdir))
            output_dir = Path(tmpdir) / "output"
            create_called = []

            def create_fn(output_path: Path) -> None:
                create_called.append(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("mock window gltf")

            result = cache.get_or_create_window(
                cache_key="window_key",
                output_dir=output_dir,
                create_fn=create_fn,
            )

            assert result.name == "window.gltf"
            assert len(create_called) == 1

    def test_different_keys_create_different_entries(self) -> None:
        """Different cache keys result in separate cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GeometryCache(cache_dir=Path(tmpdir))
            create_count = [0]

            def create_fn(output_path: Path) -> None:
                create_count[0] += 1
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(f"gltf {create_count[0]}")

            cache.get_or_create_wall(
                cache_key="key_a",
                output_dir=Path(tmpdir) / "out1",
                create_fn=create_fn,
            )
            cache.get_or_create_wall(
                cache_key="key_b",
                output_dir=Path(tmpdir) / "out2",
                create_fn=create_fn,
            )

            assert create_count[0] == 2
            assert cache.get_stats()["misses"] == 2
            assert cache.get_stats()["hits"] == 0

    def test_cache_stats_hit_rate(self) -> None:
        """Cache statistics correctly compute hit rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = GeometryCache(cache_dir=Path(tmpdir))

            def create_fn(output_path: Path) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("gltf")

            # 1 miss, then 3 hits.
            for i in range(4):
                cache.get_or_create_wall(
                    cache_key="same_key",
                    output_dir=Path(tmpdir) / f"out{i}",
                    create_fn=create_fn,
                )

            stats = cache.get_stats()
            assert stats["hits"] == 3
            assert stats["misses"] == 1
            assert stats["total"] == 4
            assert stats["hit_rate"] == 0.75


if __name__ == "__main__":
    unittest.main()
