import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

from tests.integration.common import has_usd_export_env
from tests.integration.usd_diagnostics import collect_scene_diagnostics

HAS_USD_EXPORT_ENV = has_usd_export_env()

if HAS_USD_EXPORT_ENV:
    from pydrake.all import RigidTransform

    from scenecode.agent_utils.house import (
        HouseLayout,
        HouseScene,
        PlacedRoom,
        RoomGeometry,
        RoomSpec,
    )
    from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID


@unittest.skipUnless(
    HAS_USD_EXPORT_ENV,
    "Requires .mujoco_venv with SCENECODE_RUN_USD_EXPORT_TESTS=1",
)
class TestUsdExportPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.test_data_dir = self.repo_root / "tests" / "test_data"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.fail(
                f"Command failed: {' '.join(args)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result

    def _write_scene_fixture(self, scene_dir: Path) -> None:
        room_dir = scene_dir / "room_main"
        combined_dir = scene_dir / "combined_house"
        room_dir.mkdir(parents=True, exist_ok=True)
        combined_dir.mkdir(parents=True, exist_ok=True)

        room_sdf = room_dir / "room_geometry.sdf"
        cube_obj = room_dir / "cube.obj"
        cabinet_sdf = room_dir / "articulated_cabinet.sdf"
        shutil.copy(self.test_data_dir / "simple_room_geometry.sdf", room_sdf)
        shutil.copy(self.test_data_dir / "usd_export_pipeline" / "cube.obj", cube_obj)
        shutil.copy(
            self.test_data_dir / "usd_export_pipeline" / "articulated_cabinet.sdf",
            cabinet_sdf,
        )

        room_geometry = RoomGeometry(
            sdf_tree=ET.parse(room_sdf),
            sdf_path=room_sdf,
            width=10.0,
            length=10.0,
            wall_height=3.0,
        )
        room = RoomScene(
            room_geometry=room_geometry,
            scene_dir=room_dir,
            room_id="main",
            room_type="room",
        )
        room.text_description = "Minimal room with articulated cabinet"
        room.add_object(
            SceneObject(
                object_id=UniqueID("cabinet_0"),
                object_type=ObjectType.FURNITURE,
                name="cabinet",
                description="Minimal articulated cabinet",
                transform=RigidTransform(p=[0.0, 0.0, 0.0]),
                sdf_path=cabinet_sdf,
            )
        )

        layout = HouseLayout(
            wall_height=3.0,
            house_prompt="Minimal USD export test scene",
            room_specs=[
                RoomSpec(
                    room_id="main",
                    room_type="room",
                    prompt="test room",
                    position=(0.0, 0.0),
                    width=10.0,
                    length=10.0,
                )
            ],
            room_geometries={"main": room_geometry},
            house_dir=scene_dir,
            placed_rooms=[
                PlacedRoom(
                    room_id="main",
                    position=(0.0, 0.0),
                    width=10.0,
                    depth=10.0,
                    walls=[],
                )
            ],
            placement_valid=True,
            connectivity_valid=True,
        )
        house = HouseScene(layout=layout, rooms={"main": room})

        (combined_dir / "house_state.json").write_text(
            json.dumps(house.to_state_dict(), indent=2)
        )
        (combined_dir / "house.dmd.yaml").write_text(
            "\n".join(
                [
                    "directives:",
                    "- add_frame:",
                    "    name: house_frame",
                    "    X_PF:",
                    "      base_frame: world",
                    "      translation: [0, 0, 0]",
                    "- add_frame:",
                    "    name: room_main_frame",
                    "    X_PF:",
                    "      base_frame: house_frame",
                    "      translation: [0, 0, 0]",
                    "- add_model:",
                    "    name: room_geometry_main",
                    "    file: package://scene/room_main/room_geometry.sdf",
                    "- add_weld:",
                    "    parent: room_main_frame",
                    "    child: room_geometry_main::room_geometry_body_link",
                    "- add_model:",
                    "    name: main_cabinet_0",
                    "    file: package://scene/room_main/articulated_cabinet.sdf",
                    "    default_free_body_pose:",
                    "      base_link:",
                    "        translation: [0, 0, 0]",
                    "        rotation: !AngleAxis",
                    "          angle_deg: 0",
                    "          axis: [0, 0, 1]",
                    "        base_frame: room_main_frame",
                    "",
                ]
            )
        )

    def _write_box_obj(
        self,
        path: Path,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        x0, y0, z0 = min_corner
        x1, y1, z1 = max_corner
        path.write_text(
            "\n".join(
                [
                    "o Box",
                    f"v {x0} {y0} {z0}",
                    f"v {x1} {y0} {z0}",
                    f"v {x1} {y1} {z0}",
                    f"v {x0} {y1} {z0}",
                    f"v {x0} {y0} {z1}",
                    f"v {x1} {y0} {z1}",
                    f"v {x1} {y1} {z1}",
                    f"v {x0} {y1} {z1}",
                    "f 1 2 3",
                    "f 1 3 4",
                    "f 5 8 7",
                    "f 5 7 6",
                    "f 1 5 6",
                    "f 1 6 2",
                    "f 2 6 7",
                    "f 2 7 3",
                    "f 3 7 8",
                    "f 3 8 4",
                    "f 5 1 4",
                    "f 5 4 8",
                    "",
                ]
            )
        )

    def _write_flat_obj(
        self,
        path: Path,
        corners: list[tuple[float, float, float]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "o Flat",
                    *(f"v {x} {y} {z}" for x, y, z in corners),
                    "f 1 2 3",
                    "f 1 3 4",
                    "",
                ]
            )
        )

    def _write_collision_basename_fixture(self, fixture_dir: Path) -> Path:
        fixture_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            self.test_data_dir / "usd_export_pipeline" / "cube.obj",
            fixture_dir / "cube.obj",
        )
        self._write_box_obj(
            fixture_dir / "base_collision" / "convex_piece_000.obj",
            (-0.5, -0.25, 0.0),
            (0.5, 0.25, 1.0),
        )
        self._write_box_obj(
            fixture_dir / "door_collision" / "convex_piece_000.obj",
            (-0.02, 0.22, 0.0),
            (0.02, 0.27, 1.0),
        )
        sdf_path = fixture_dir / "collision_basename_cabinet.sdf"
        sdf_path.write_text(
            """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="collision_basename_cabinet">
    <link name="base_link">
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <visual name="base_visual">
        <pose>0 0 0.5 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>cube.obj</uri>
            <scale>1.0 0.5 1.0</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name="base_collision_0">
        <geometry>
          <mesh>
            <uri>base_collision/convex_piece_000.obj</uri>
          </mesh>
        </geometry>
      </collision>
    </link>
    <link name="door_link">
      <pose>0.5 0 0.5 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <visual name="door_visual">
        <pose>0 0.26 0 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>cube.obj</uri>
            <scale>0.02 0.5 1.0</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name="door_collision_0">
        <geometry>
          <mesh>
            <uri>door_collision/convex_piece_000.obj</uri>
          </mesh>
        </geometry>
      </collision>
    </link>
    <joint name="door_hinge" type="revolute">
      <parent>base_link</parent>
      <child>door_link</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0</lower>
          <upper>1.5708</upper>
        </limit>
        <dynamics>
          <damping>1.0</damping>
        </dynamics>
      </axis>
    </joint>
  </model>
</sdf>
"""
        )
        return sdf_path

    def _write_degenerate_collision_fixture(self, fixture_dir: Path) -> Path:
        fixture_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            self.test_data_dir / "usd_export_pipeline" / "cube.obj",
            fixture_dir / "cube.obj",
        )
        self._write_box_obj(
            fixture_dir / "good_collision" / "solid_piece.obj",
            (-0.5, -0.5, 0.0),
            (0.5, 0.5, 1.0),
        )
        self._write_flat_obj(
            fixture_dir / "bad_collision" / "flat_piece.obj",
            [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0),
            ],
        )

        sdf_path = fixture_dir / "degenerate_collision_fixture.sdf"
        sdf_path.write_text(
            """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="degenerate_collision_fixture">
    <link name="base_link">
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <visual name="base_visual">
        <pose>0 0 0.5 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>cube.obj</uri>
          </mesh>
        </geometry>
      </visual>
      <collision name="good_collision_0">
        <geometry>
          <mesh>
            <uri>good_collision/solid_piece.obj</uri>
          </mesh>
        </geometry>
      </collision>
      <collision name="bad_collision_0">
        <geometry>
          <mesh>
            <uri>bad_collision/flat_piece.obj</uri>
          </mesh>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
        )
        return sdf_path

    def _write_two_room_geometry_dmd_fixture(self, scene_dir: Path) -> None:
        room_geometry_dir = scene_dir / "room_geometry"
        combined_dir = scene_dir / "combined_house"
        room_geometry_dir.mkdir(parents=True, exist_ok=True)
        combined_dir.mkdir(parents=True, exist_ok=True)

        alpha_wall = (
            scene_dir / "floor_plans" / "alpha" / "walls" / "north_wall" / "wall.obj"
        )
        beta_wall = (
            scene_dir / "floor_plans" / "beta" / "walls" / "north_wall" / "wall.obj"
        )
        self._write_box_obj(alpha_wall, (0.0, 0.0, 0.0), (1.0, 0.1, 1.0))
        self._write_box_obj(beta_wall, (0.0, 0.0, 0.0), (2.0, 0.1, 1.0))

        alpha_sdf = room_geometry_dir / "room_geometry_alpha.sdf"
        beta_sdf = room_geometry_dir / "room_geometry_beta.sdf"
        for sdf_path, rel_uri in [
            (alpha_sdf, "../floor_plans/alpha/walls/north_wall/wall.obj"),
            (beta_sdf, "../floor_plans/beta/walls/north_wall/wall.obj"),
        ]:
            sdf_path.write_text(
                f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{sdf_path.stem}">
    <link name="room_geometry_body_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <visual name="north_wall_visual">
        <geometry>
          <mesh>
            <uri>{rel_uri}</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""
            )

        (combined_dir / "house.dmd.yaml").write_text(
            "\n".join(
                [
                    "directives:",
                    "- add_frame:",
                    "    name: house_frame",
                    "    X_PF:",
                    "      base_frame: world",
                    "      translation: [0, 0, 0]",
                    "- add_frame:",
                    "    name: room_alpha_frame",
                    "    X_PF:",
                    "      base_frame: house_frame",
                    "      translation: [0, 0, 0]",
                    "- add_frame:",
                    "    name: room_beta_frame",
                    "    X_PF:",
                    "      base_frame: house_frame",
                    "      translation: [5, 0, 0]",
                    "- add_model:",
                    "    name: room_geometry_alpha",
                    "    file: package://scene/room_geometry/room_geometry_alpha.sdf",
                    "- add_weld:",
                    "    parent: room_alpha_frame",
                    "    child: room_geometry_alpha::room_geometry_body_link",
                    "- add_model:",
                    "    name: room_geometry_beta",
                    "    file: package://scene/room_geometry/room_geometry_beta.sdf",
                    "- add_weld:",
                    "    parent: room_beta_frame",
                    "    child: room_geometry_beta::room_geometry_body_link",
                    "",
                ]
            )
        )

    def test_full_scene_export_then_fix(self) -> None:
        scene_dir = self.temp_path / "scene_fixture"
        self._write_scene_fixture(scene_dir)

        raw_export_dir = self.temp_path / "raw_export"
        self._run(
            "scripts/export_scene_to_mujoco.py",
            str(scene_dir),
            "-o",
            str(raw_export_dir),
            "--skip-validation",
            "--usd",
            "--skip-isaac-sim-fix",
        )

        raw_usd_dir = raw_export_dir / "usd"
        raw_diagnostics = collect_scene_diagnostics(raw_usd_dir)
        self.assertGreater(raw_diagnostics.invalid_no_rb, 0)
        self.assertGreater(raw_diagnostics.ancestor_desc, 0)
        self.assertGreater(raw_diagnostics.neg_prismatic, 0)
        self.assertGreater(raw_diagnostics.collision_total, 0)
        self.assertEqual(raw_diagnostics.visual_collision, 0)

        fixed_usd_dir = self.temp_path / "fixed_usd"
        shutil.copytree(raw_usd_dir, fixed_usd_dir)
        self._run("scripts/fix_usd_isaac_sim.py", str(fixed_usd_dir))

        fixed_diagnostics = collect_scene_diagnostics(fixed_usd_dir)
        self.assertEqual(fixed_diagnostics.invalid_no_rb, 0)
        self.assertEqual(fixed_diagnostics.ancestor_desc, 0)
        self.assertEqual(fixed_diagnostics.neg_prismatic, raw_diagnostics.neg_prismatic)
        self.assertEqual(
            fixed_diagnostics.collision_total, raw_diagnostics.collision_total
        )
        self.assertEqual(
            fixed_diagnostics.collision_active, raw_diagnostics.collision_active
        )
        self.assertEqual(fixed_diagnostics.visual_collision, 0)

    def test_full_scene_export_from_dmd_only_then_fix(self) -> None:
        scene_dir = self.temp_path / "scene_fixture_dmd_only"
        self._write_scene_fixture(scene_dir)
        (scene_dir / "combined_house" / "house_state.json").unlink()

        raw_export_dir = self.temp_path / "raw_export_dmd_only"
        self._run(
            "scripts/export_scene_to_mujoco.py",
            str(scene_dir),
            "-o",
            str(raw_export_dir),
            "--skip-validation",
            "--usd",
            "--skip-isaac-sim-fix",
        )

        raw_usd_dir = raw_export_dir / "usd"
        raw_diagnostics = collect_scene_diagnostics(raw_usd_dir)
        self.assertGreater(raw_diagnostics.invalid_no_rb, 0)
        self.assertGreater(raw_diagnostics.ancestor_desc, 0)
        self.assertGreater(raw_diagnostics.neg_prismatic, 0)

        fixed_usd_dir = self.temp_path / "fixed_usd_dmd_only"
        shutil.copytree(raw_usd_dir, fixed_usd_dir)
        self._run("scripts/fix_usd_isaac_sim.py", str(fixed_usd_dir))

        fixed_diagnostics = collect_scene_diagnostics(fixed_usd_dir)
        self.assertEqual(fixed_diagnostics.invalid_no_rb, 0)
        self.assertEqual(fixed_diagnostics.ancestor_desc, 0)
        self.assertEqual(fixed_diagnostics.neg_prismatic, raw_diagnostics.neg_prismatic)
        self.assertEqual(
            fixed_diagnostics.collision_total, raw_diagnostics.collision_total
        )
        self.assertEqual(
            fixed_diagnostics.collision_active, raw_diagnostics.collision_active
        )
        self.assertEqual(fixed_diagnostics.visual_collision, 0)

    def test_collision_mesh_basenames_do_not_alias_across_articulated_links(
        self,
    ) -> None:
        fixture_dir = self.temp_path / "collision_basename_fixture"
        sdf_path = self._write_collision_basename_fixture(fixture_dir)
        export_dir = self.temp_path / "collision_basename_export"

        self._run(
            "scripts/export_scene_to_mujoco.py",
            "--sdf",
            str(sdf_path),
            "-o",
            str(export_dir),
            "--skip-validation",
        )

        scene_xml = export_dir / "scene.xml"
        root = ET.parse(scene_xml).getroot()

        mesh_files = {}
        for mesh in root.findall(".//asset/mesh"):
            name = mesh.get("name", "")
            if "base_collision_0" in name or "door_collision_0" in name:
                mesh_files[name] = mesh.get("file")

        self.assertEqual(len(mesh_files), 2)
        exported_files = set(mesh_files.values())
        self.assertEqual(len(exported_files), 2)

        base_export = (
            export_dir
            / "meshes"
            / next(
                file_name
                for name, file_name in mesh_files.items()
                if "base_collision_0" in name
            )
        )
        door_export = (
            export_dir
            / "meshes"
            / next(
                file_name
                for name, file_name in mesh_files.items()
                if "door_collision_0" in name
            )
        )

        self.assertEqual(
            hashlib.sha256(base_export.read_bytes()).hexdigest(),
            hashlib.sha256(
                (fixture_dir / "base_collision" / "convex_piece_000.obj").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            hashlib.sha256(door_export.read_bytes()).hexdigest(),
            hashlib.sha256(
                (fixture_dir / "door_collision" / "convex_piece_000.obj").read_bytes()
            ).hexdigest(),
        )

    def test_dmd_only_export_keeps_room_geometry_meshes_distinct_across_rooms(
        self,
    ) -> None:
        scene_dir = self.temp_path / "two_room_dmd_scene"
        self._write_two_room_geometry_dmd_fixture(scene_dir)
        export_dir = self.temp_path / "two_room_dmd_export"

        self._run(
            "scripts/export_scene_to_mujoco.py",
            str(scene_dir),
            "-o",
            str(export_dir),
            "--skip-validation",
        )

        scene_xml = export_dir / "scene.xml"
        root = ET.parse(scene_xml).getroot()

        mesh_files = {}
        for mesh in root.findall(".//asset/mesh"):
            name = mesh.get("name", "")
            if "room_geometry_alpha" in name or "room_geometry_beta" in name:
                mesh_files[name] = mesh.get("file")

        self.assertEqual(len(mesh_files), 2)
        exported_files = set(mesh_files.values())
        self.assertEqual(len(exported_files), 2)
        self.assertTrue(any("alpha_" in file_name for file_name in exported_files))
        self.assertTrue(any("beta_" in file_name for file_name in exported_files))

        alpha_export = (
            export_dir
            / "meshes"
            / next(
                file_name
                for name, file_name in mesh_files.items()
                if "room_geometry_alpha" in name
            )
        )
        beta_export = (
            export_dir
            / "meshes"
            / next(
                file_name
                for name, file_name in mesh_files.items()
                if "room_geometry_beta" in name
            )
        )

        self.assertEqual(
            hashlib.sha256(alpha_export.read_bytes()).hexdigest(),
            hashlib.sha256(
                (
                    scene_dir
                    / "floor_plans"
                    / "alpha"
                    / "walls"
                    / "north_wall"
                    / "wall.obj"
                ).read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            hashlib.sha256(beta_export.read_bytes()).hexdigest(),
            hashlib.sha256(
                (
                    scene_dir
                    / "floor_plans"
                    / "beta"
                    / "walls"
                    / "north_wall"
                    / "wall.obj"
                ).read_bytes()
            ).hexdigest(),
        )

    def test_standalone_export_drops_only_degenerate_collision_piece(self) -> None:
        fixture_dir = self.temp_path / "degenerate_collision_fixture"
        sdf_path = self._write_degenerate_collision_fixture(fixture_dir)
        export_dir = self.temp_path / "degenerate_collision_export"

        self._run(
            "scripts/export_scene_to_mujoco.py",
            "--sdf",
            str(sdf_path),
            "-o",
            str(export_dir),
            "--skip-validation",
        )

        scene_xml = export_dir / "scene.xml"
        root = ET.parse(scene_xml).getroot()

        geom_names = {
            geom.get("name", "") for geom in root.findall(".//worldbody//geom")
        }
        mesh_names = {mesh.get("name", "") for mesh in root.findall(".//asset/mesh")}

        self.assertTrue(
            any("good_collision_0_collision" in geom_name for geom_name in geom_names)
        )
        self.assertFalse(
            any("bad_collision_0_collision" in geom_name for geom_name in geom_names)
        )
        self.assertTrue(
            any(
                "good_collision_0_collision_mesh" in mesh_name
                for mesh_name in mesh_names
            )
        )
        self.assertFalse(
            any(
                "bad_collision_0_collision_mesh" in mesh_name
                for mesh_name in mesh_names
            )
        )


if __name__ == "__main__":
    unittest.main()
