"""Unit tests for WallSurface"""

import unittest

import numpy as np

from pydrake.math import RigidTransform, RotationMatrix

from scenecode.agent_utils.house import Opening, OpeningType, Wall, WallDirection
from scenecode.agent_utils.room import UniqueID
from scenecode.wall_agents.tools.wall_surface import WallSurface, _create_wall_surface


def _create_test_wall_surface(
    wall_direction: WallDirection,
    start_point: tuple[float, float],
    wall_vec: tuple[float, float],
    wall_length: float,
    ceiling_height: float = 3.0,
) -> WallSurface:
    """Create a WallSurface with explicit geometry for testing.

    Args:
        wall_direction: Cardinal direction the wall faces.
        start_point: (x, y) world coordinates of wall start.
        wall_vec: Normalized (dx, dy) vector along wall (+X in wall frame).
        wall_length: Length of wall in meters.
        ceiling_height: Height of wall in meters.

    Returns:
        WallSurface configured for testing.
    """
    inward_normal = wall_direction.get_inward_normal()
    outward_normal = (-inward_normal[0], -inward_normal[1])  # Flip to outward.

    # Build rotation matrix: X=along wall, Y=outward, Z=up.
    # Using outward (not inward) ensures right-handed coordinate system.
    col_x = np.array([wall_vec[0], wall_vec[1], 0.0])
    col_y = np.array([outward_normal[0], outward_normal[1], 0.0])
    col_z = np.array([0.0, 0.0, 1.0])

    rotation_matrix = np.column_stack([col_x, col_y, col_z])
    rotation = RotationMatrix(rotation_matrix)

    transform = RigidTransform(
        R=rotation,
        p=np.array([start_point[0], start_point[1], 0.0]),
    )

    return WallSurface(
        surface_id=UniqueID(f"test_{wall_direction.value}"),
        wall_id=f"test_room_{wall_direction.value}",
        wall_direction=wall_direction,
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[wall_length, 0.0, ceiling_height],
        transform=transform,
        excluded_regions=[],
    )


class TestToWorldPoseNorthWall(unittest.TestCase):
    """Test to_world_pose() for NORTH wall (faces -Y, at high Y)."""

    def _create_north_wall(self) -> WallSurface:
        """NORTH wall from (0, 5) to (4, 5), 4m long.

        Wall frame:
            +X = (1, 0) along wall (west to east)
            +Y = (0, -1) into room
            +Z = up
        """
        return _create_test_wall_surface(
            wall_direction=WallDirection.NORTH,
            start_point=(0.0, 5.0),
            wall_vec=(1.0, 0.0),
            wall_length=4.0,
        )

    def test_origin_position(self) -> None:
        """Object at wall origin should be at wall start point."""
        north_wall = self._create_north_wall()
        pose = north_wall.to_world_pose(position_x=0.0, position_z=0.0)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [0.0, 5.0, 0.0], atol=1e-10)

    def test_offset_along_wall(self) -> None:
        """Object offset along wall (+X) should move in +X world."""
        north_wall = self._create_north_wall()
        pose = north_wall.to_world_pose(position_x=2.0, position_z=0.0)
        translation = pose.translation()

        # +2m along wall = +2m in world X.
        np.testing.assert_allclose(translation, [2.0, 5.0, 0.0], atol=1e-10)

    def test_offset_height(self) -> None:
        """Object offset in height (+Z) should move in +Z world."""
        north_wall = self._create_north_wall()
        pose = north_wall.to_world_pose(position_x=0.0, position_z=1.5)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [0.0, 5.0, 1.5], atol=1e-10)

    def test_combined_offset(self) -> None:
        """Object with both offsets should move in both directions."""
        north_wall = self._create_north_wall()
        pose = north_wall.to_world_pose(position_x=2.0, position_z=1.5)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [2.0, 5.0, 1.5], atol=1e-10)


class TestToWorldPoseSouthWall(unittest.TestCase):
    """Test to_world_pose() for SOUTH wall (faces +Y, at low Y)."""

    def _create_south_wall(self) -> WallSurface:
        """SOUTH wall from (4, 0) to (0, 0), 4m long.

        Wall is at Y=0, faces +Y (into room which is at higher Y).
        +X along wall goes from (4,0) toward (0,0), so wall_vec = (-1, 0).

        Wall frame:
            +X = (-1, 0) along wall (east to west)
            +Y = (0, 1) into room
            +Z = up
        """
        return _create_test_wall_surface(
            wall_direction=WallDirection.SOUTH,
            start_point=(4.0, 0.0),
            wall_vec=(-1.0, 0.0),
            wall_length=4.0,
        )

    def test_origin_position(self) -> None:
        """Object at wall origin should be at wall start point."""
        south_wall = self._create_south_wall()
        pose = south_wall.to_world_pose(position_x=0.0, position_z=0.0)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [4.0, 0.0, 0.0], atol=1e-10)

    def test_offset_along_wall(self) -> None:
        """Object offset along wall (+X) should move in -X world."""
        south_wall = self._create_south_wall()
        pose = south_wall.to_world_pose(position_x=2.0, position_z=0.0)
        translation = pose.translation()

        # +2m along wall = -2m in world X (wall goes east to west).
        np.testing.assert_allclose(translation, [2.0, 0.0, 0.0], atol=1e-10)

    def test_offset_height(self) -> None:
        """Object offset in height (+Z) should move in +Z world."""
        south_wall = self._create_south_wall()
        pose = south_wall.to_world_pose(position_x=0.0, position_z=1.5)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [4.0, 0.0, 1.5], atol=1e-10)


class TestToWorldPoseEastWall(unittest.TestCase):
    """Test to_world_pose() for EAST wall (faces -X, at high X)."""

    def _create_east_wall(self) -> WallSurface:
        """EAST wall from (5, 0) to (5, 4), 4m long.

        Wall is at X=5, faces -X (into room which is at lower X).
        +X along wall goes from (5,0) toward (5,4), so wall_vec = (0, 1).

        Wall frame:
            +X = (0, 1) along wall (south to north)
            +Y = (-1, 0) into room
            +Z = up
        """
        return _create_test_wall_surface(
            wall_direction=WallDirection.EAST,
            start_point=(5.0, 0.0),
            wall_vec=(0.0, 1.0),
            wall_length=4.0,
        )

    def test_origin_position(self) -> None:
        """Object at wall origin should be at wall start point."""
        east_wall = self._create_east_wall()
        pose = east_wall.to_world_pose(position_x=0.0, position_z=0.0)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [5.0, 0.0, 0.0], atol=1e-10)

    def test_offset_along_wall(self) -> None:
        """Object offset along wall (+X) should move in +Y world."""
        east_wall = self._create_east_wall()
        pose = east_wall.to_world_pose(position_x=2.0, position_z=0.0)
        translation = pose.translation()

        # +2m along wall = +2m in world Y (wall goes south to north).
        np.testing.assert_allclose(translation, [5.0, 2.0, 0.0], atol=1e-10)

    def test_offset_height(self) -> None:
        """Object offset in height (+Z) should move in +Z world."""
        east_wall = self._create_east_wall()
        pose = east_wall.to_world_pose(position_x=0.0, position_z=1.5)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [5.0, 0.0, 1.5], atol=1e-10)


class TestToWorldPoseWestWall(unittest.TestCase):
    """Test to_world_pose() for WEST wall (faces +X, at low X)."""

    def _create_west_wall(self) -> WallSurface:
        """WEST wall from (0, 4) to (0, 0), 4m long.

        Wall is at X=0, faces +X (into room which is at higher X).
        +X along wall goes from (0,4) toward (0,0), so wall_vec = (0, -1).

        Wall frame:
            +X = (0, -1) along wall (north to south)
            +Y = (1, 0) into room
            +Z = up
        """
        return _create_test_wall_surface(
            wall_direction=WallDirection.WEST,
            start_point=(0.0, 4.0),
            wall_vec=(0.0, -1.0),
            wall_length=4.0,
        )

    def test_origin_position(self) -> None:
        """Object at wall origin should be at wall start point."""
        west_wall = self._create_west_wall()
        pose = west_wall.to_world_pose(position_x=0.0, position_z=0.0)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [0.0, 4.0, 0.0], atol=1e-10)

    def test_offset_along_wall(self) -> None:
        """Object offset along wall (+X) should move in -Y world."""
        west_wall = self._create_west_wall()
        pose = west_wall.to_world_pose(position_x=2.0, position_z=0.0)
        translation = pose.translation()

        # +2m along wall = -2m in world Y (wall goes north to south).
        np.testing.assert_allclose(translation, [0.0, 2.0, 0.0], atol=1e-10)

    def test_offset_height(self) -> None:
        """Object offset in height (+Z) should move in +Z world."""
        west_wall = self._create_west_wall()
        pose = west_wall.to_world_pose(position_x=0.0, position_z=1.5)
        translation = pose.translation()

        np.testing.assert_allclose(translation, [0.0, 4.0, 1.5], atol=1e-10)


class TestToWorldPoseRotation(unittest.TestCase):
    """Test to_world_pose() rotation handling."""

    def _create_north_wall(self) -> WallSurface:
        """Simple NORTH wall for rotation tests."""
        return _create_test_wall_surface(
            wall_direction=WallDirection.NORTH,
            start_point=(0.0, 5.0),
            wall_vec=(1.0, 0.0),
            wall_length=4.0,
        )

    def test_zero_rotation(self) -> None:
        """Zero user rotation includes 180° base rotation for wall mounting.

        Wall objects need 180° base rotation so their front (+Y after
        canonicalization) faces into the room instead of outward.
        """
        north_wall = self._create_north_wall()
        pose = north_wall.to_world_pose(
            position_x=0.0, position_z=0.0, rotation_deg=0.0
        )
        rotation = pose.rotation().matrix()

        # Wall frame for NORTH: X=(1,0,0), Y=(0,1,0) outward, Z=(0,0,1).
        # With 180° base rotation: R_z(180°) = [[-1,0,0],[0,-1,0],[0,0,1]].
        # Since north wall frame is identity, world rotation equals R_z(180°).
        expected = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(rotation, expected, atol=1e-10)

    def test_90_degree_rotation(self) -> None:
        """90 degree CCW rotation tilts object in wall plane.

        SE(2) rotation on wall means rotation about Y axis (wall normal),
        not Z axis. Combined with 180° base flip about Z:
        R_total = R_y(-90°) @ R_z(180°)

        This makes:
        - Object +Y → -Y (front faces into room)
        - Object +Z → -X (12 o'clock moves to viewer's left = CCW)
        """
        north_wall = self._create_north_wall()
        pose = north_wall.to_world_pose(
            position_x=0.0, position_z=0.0, rotation_deg=90.0
        )
        rotation = pose.rotation().matrix()

        # R_y(-90°) @ R_z(180°) for north wall (identity transform).
        expected = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(rotation, expected, atol=1e-10)

    def test_rotation_preserves_position(self) -> None:
        """Rotation should not affect translation."""
        north_wall = self._create_north_wall()
        pose_no_rot = north_wall.to_world_pose(
            position_x=2.0, position_z=1.5, rotation_deg=0.0
        )
        pose_with_rot = north_wall.to_world_pose(
            position_x=2.0, position_z=1.5, rotation_deg=45.0
        )

        np.testing.assert_allclose(
            pose_no_rot.translation(), pose_with_rot.translation(), atol=1e-10
        )


class TestExcludedRegionCoordinates(unittest.TestCase):
    """Test that excluded regions (doors/windows) are correctly positioned.

    Opening.position_along_wall is the LEFT EDGE of the opening measured from
    wall.start_point. Wall surface X coordinate may be measured from either
    end depending on wall direction. This tests the conversion from LEFT EDGE
    to CENTER and the coordinate transformation for different wall directions.
    """

    def test_north_wall_opening_position(self) -> None:
        """NORTH wall: origin at start (min X), no transform needed.

        position_along_wall=2.0 is the LEFT EDGE of a 1.0m wide window.
        Opening center from start = 2.0 + 0.5 = 2.5m.
        For NORTH wall (origin at start): x_center = 2.5.
        """
        wall = Wall(
            wall_id="test_north",
            room_id="test_room",
            direction=WallDirection.NORTH,
            start_point=(0.0, 5.0),  # West end.
            end_point=(4.0, 5.0),  # East end.
            length=4.0,
            openings=[
                Opening(
                    opening_id="window_1",
                    opening_type=OpeningType.WINDOW,
                    position_along_wall=2.0,  # LEFT EDGE 2m from start.
                    width=1.0,
                    height=1.0,
                    sill_height=1.0,
                )
            ],
        )

        surface = _create_wall_surface(wall, ceiling_height=3.0)

        # Opening center = 2.0 + 0.5 = 2.5 (no transform for NORTH).
        assert len(surface.excluded_regions) == 1
        x_min, z_min, x_max, z_max = surface.excluded_regions[0]
        x_center = (x_min + x_max) / 2
        assert abs(x_center - 2.5) < 0.01, f"Expected x_center=2.5, got {x_center}"

    def test_south_wall_opening_position(self) -> None:
        """SOUTH wall: origin at end (max X), position must be transformed.

        Wall from (0,0) to (4,0): start.x=0, end.x=4.
        Since start.x < end.x, origin is at end.
        position_along_wall=1.0 is the LEFT EDGE of a 1.0m wide window.
        Opening center from start = 1.0 + 0.5 = 1.5m.
        For SOUTH wall (origin at end): x_center = 4.0 - 1.5 = 2.5.
        """
        wall = Wall(
            wall_id="test_south",
            room_id="test_room",
            direction=WallDirection.SOUTH,
            start_point=(0.0, 0.0),  # West end.
            end_point=(4.0, 0.0),  # East end.
            length=4.0,
            openings=[
                Opening(
                    opening_id="window_1",
                    opening_type=OpeningType.WINDOW,
                    position_along_wall=1.0,  # LEFT EDGE 1m from start (west).
                    width=1.0,
                    height=1.0,
                    sill_height=1.0,
                )
            ],
        )

        surface = _create_wall_surface(wall, ceiling_height=3.0)

        # Origin at end (east), X axis points west (-X).
        # Opening center from start = 1.0 + 0.5 = 1.5.
        # x_center = 4.0 - 1.5 = 2.5.
        assert len(surface.excluded_regions) == 1
        x_min, z_min, x_max, z_max = surface.excluded_regions[0]
        x_center = (x_min + x_max) / 2
        assert abs(x_center - 2.5) < 0.01, f"Expected x_center=2.5, got {x_center}"

    def test_east_wall_opening_position(self) -> None:
        """EAST wall: origin at end (max Y), position must be transformed.

        Wall from (4,0) to (4,4): start.y=0, end.y=4.
        Since start.y < end.y, origin is at end.
        position_along_wall=1.0 is the LEFT EDGE of a 1.0m wide window.
        Opening center from start = 1.0 + 0.5 = 1.5m.
        For EAST wall (origin at end): x_center = 4.0 - 1.5 = 2.5.
        """
        wall = Wall(
            wall_id="test_east",
            room_id="test_room",
            direction=WallDirection.EAST,
            start_point=(4.0, 0.0),  # South end.
            end_point=(4.0, 4.0),  # North end.
            length=4.0,
            openings=[
                Opening(
                    opening_id="window_1",
                    opening_type=OpeningType.WINDOW,
                    position_along_wall=1.0,  # LEFT EDGE 1m from start (south).
                    width=1.0,
                    height=1.0,
                    sill_height=1.0,
                )
            ],
        )

        surface = _create_wall_surface(wall, ceiling_height=3.0)

        # Origin at end (north), X axis points south (-Y).
        # Opening center from start = 1.0 + 0.5 = 1.5.
        # x_center = 4.0 - 1.5 = 2.5.
        assert len(surface.excluded_regions) == 1
        x_min, z_min, x_max, z_max = surface.excluded_regions[0]
        x_center = (x_min + x_max) / 2
        assert abs(x_center - 2.5) < 0.01, f"Expected x_center=2.5, got {x_center}"

    def test_west_wall_opening_position(self) -> None:
        """WEST wall: origin at start (min Y), no transform needed.

        position_along_wall=2.0 is the LEFT EDGE of a 1.0m wide window.
        Opening center from start = 2.0 + 0.5 = 2.5m.
        For WEST wall (origin at start): x_center = 2.5.
        """
        wall = Wall(
            wall_id="test_west",
            room_id="test_room",
            direction=WallDirection.WEST,
            start_point=(0.0, 0.0),  # South end.
            end_point=(0.0, 4.0),  # North end.
            length=4.0,
            openings=[
                Opening(
                    opening_id="window_1",
                    opening_type=OpeningType.WINDOW,
                    position_along_wall=2.0,  # LEFT EDGE 2m from start.
                    width=1.0,
                    height=1.0,
                    sill_height=1.0,
                )
            ],
        )

        surface = _create_wall_surface(wall, ceiling_height=3.0)

        # Opening center = 2.0 + 0.5 = 2.5 (no transform for WEST).
        assert len(surface.excluded_regions) == 1
        x_min, z_min, x_max, z_max = surface.excluded_regions[0]
        x_center = (x_min + x_max) / 2
        assert abs(x_center - 2.5) < 0.01, f"Expected x_center=2.5, got {x_center}"


if __name__ == "__main__":
    unittest.main()
