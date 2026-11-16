import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import xml.etree.ElementTree as ET
import types

from autoresearch.robotics_robot_asset import (
    PICKNIK_COMMIT,
    ROBOT_ASSET_DIR,
    _checked_picknik_source,
    _checked_usd_source,
    _portable_manifest_sha256,
    _asset_member,
    prepare_robot_asset,
    validate_robot_asset,
)


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _asset_fixture(base: Path) -> tuple[dict, Path, Path, dict[str, Path]]:
    root = base / "outputs" / "robot_assets" / ROBOT_ASSET_DIR
    pybullet = base / "venv" / "lib" / "python3.11" / "site-packages" / "pybullet_data"
    picknik = base / "runtime" / "ros2_robotiq_gripper" / "robotiq_description"
    panda_mesh = pybullet / "franka_panda" / "meshes" / "collision" / "link0.obj"
    robotiq_source = picknik / "meshes" / "visual" / "2f_85" / "left_finger.dae"
    collision_mesh = root / "collision_meshes" / "left_finger.obj"
    visual_mesh = root / "visual_meshes" / "left_finger.obj"
    material = root / "visual_meshes" / "left_finger.mtl"
    panda_urdf = pybullet / "franka_panda" / "panda.urdf"
    macro = picknik / "urdf" / "robotiq_2f_85_macro.urdf.xacro"
    usd = base / "runtime" / "RoboLab" / "franka_robotiq_2f_85_flattened.usd"
    generated_urdf = root / "franka_robotiq_2f_85.urdf"

    panda_hash = _write(panda_urdf, b"panda urdf")
    panda_mesh_hash = _write(panda_mesh, b"panda mesh")
    source_hash = _write(robotiq_source, b"robotiq source mesh")
    macro_hash = _write(macro, b"robotiq macro")
    usd_hash = _write(usd, b"registered usd")
    collision_hash = _write(collision_mesh, b"generated collision mesh")
    visual_hash = _write(visual_mesh, b"generated visual mesh")
    material_hash = _write(material, b"generated material")

    robot = ET.Element("robot", {"name": "fixture"})
    link = ET.SubElement(robot, "link", {"name": "panda_link0"})
    for kind, mesh in (("visual", panda_mesh), ("collision", collision_mesh), ("visual", visual_mesh)):
        node = ET.SubElement(link, kind)
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": str(mesh)})
    ET.ElementTree(robot).write(generated_urdf, encoding="utf-8", xml_declaration=True)
    urdf_hash = hashlib.sha256(generated_urdf.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "robolab_droid_portable_urdf.v2",
        "robot_embodiment": "robolab_droid",
        "generated_urdf": str(generated_urdf),
        "generated_urdf_sha256": urdf_hash,
        "panda_urdf": str(panda_urdf),
        "panda_urdf_sha256": panda_hash,
        "picknik_repo": str(picknik.parents[1]),
        "picknik_commit": "2c047340aeb2440f7a60e429264221aab9658707",
        "picknik_macro": str(macro),
        "picknik_macro_sha256": macro_hash,
        "robolab_commit": "f72d39445ab699c9cfd9bf551e8d21b6075293b9",
        "robolab_usd": str(usd),
        "robolab_usd_sha256": "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41",
        "mount": {"parent": "panda_link8", "child": "base_link", "xyz_m": [0, 0, 0.018174022]},
        "gripper": {"open_driver_rad": 0.0, "closed_driver_rad": 0.7853981633974483},
        "mesh_sources": [
            {"path": str(panda_mesh), "sha256": panda_mesh_hash},
            {"path": str(robotiq_source), "sha256": source_hash},
        ],
        "generated_collision_meshes": [
            {"path": str(collision_mesh), "sha256": collision_hash},
        ],
        "visual_conversion": {
            "authority": "official SceneCode Blender runtime",
            "blender_executable": str(base / "runtime" / "blender-4.4.3" / "blender"),
            "converter_script": str(base / "scripts" / "convert_collada_to_obj_blender.py"),
            "converter_script_sha256": "1e2847e732c14aa24ecd32abdecec31441d6fcdcb623d7d422b5de983d1b257e",
            "coordinate_contract": "metres, source Z-up preserved via OBJ -Y-forward/Z-up",
            "records": [
                {
                    "source": str(robotiq_source),
                    "source_sha256": source_hash,
                    "destination": str(visual_mesh),
                    "destination_sha256": visual_hash,
                    "material": str(material),
                    "material_sha256": material_hash,
                    "mesh_object_count": 1,
                    "world_coordinate_bounds": {"minimum_m": [0, 0, 0], "maximum_m": [1, 1, 1]},
                }
            ],
        },
    }
    manifest_path = root / "robot_asset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))
    return manifest, root, generated_urdf, {
        "collision": collision_mesh,
        "panda_mesh": panda_mesh,
        "output_root": root.parents[1],
    }


class PortableRobotAssetTests(unittest.TestCase):
    def test_pinned_picknik_checkout_can_be_resolved_from_sibling_branch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "WOV2"
            workspace = root / "branches" / "active"
            local = workspace / "code/dependencies/ros2_robotiq_gripper_2c047340"
            sibling = (
                root / "branches" / "source"
                / "code/dependencies/ros2_robotiq_gripper_2c047340"
            )
            local.mkdir(parents=True)
            sibling.mkdir(parents=True)

            def revision_for(command, **_kwargs):
                path = Path(command[2])
                revision = PICKNIK_COMMIT if path == sibling else "wrong-revision"
                return types.SimpleNamespace(stdout=revision + "\n")

            with patch(
                "autoresearch.robotics_robot_asset.subprocess.run",
                side_effect=revision_for,
            ):
                self.assertEqual(_checked_picknik_source(workspace), sibling.resolve())

    def test_picknik_resolution_rejects_unpinned_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "WOV2"
            workspace = root / "branches" / "active"
            local = workspace / "code/dependencies/ros2_robotiq_gripper_2c047340"
            local.mkdir(parents=True)
            with patch(
                "autoresearch.robotics_robot_asset.subprocess.run",
                return_value=types.SimpleNamespace(stdout="wrong-revision\n"),
            ), self.assertRaisesRegex(FileNotFoundError, PICKNIK_COMMIT):
                _checked_picknik_source(workspace)

    def test_shared_runtime_usd_is_used_when_checkout_source_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "WOV2"
            workspace = root / "branches" / "branch-a"
            usd = root / "runtime" / "RoboLab" / "franka_robotiq_2f_85_flattened.usd"
            digest = _write(usd, b"pinned shared usd")

            with patch("autoresearch.robotics_robot_asset.USD_SHA256", digest):
                self.assertEqual(_checked_usd_source(workspace), usd)

    def test_shared_runtime_usd_digest_is_still_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "WOV2"
            workspace = root / "branches" / "branch-a"
            usd = root / "runtime" / "RoboLab" / "franka_robotiq_2f_85_flattened.usd"
            _write(usd, b"not the pinned usd")

            with patch(
                "autoresearch.robotics_robot_asset.USD_SHA256", "0" * 64
            ), self.assertRaisesRegex(ValueError, "digest changed"):
                _checked_usd_source(workspace)

    def test_portable_fingerprint_ignores_installation_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            first = _asset_fixture(tmp_path / "user_a" / "WOV2")
            second = _asset_fixture(tmp_path / "user_b" / "another_workspace")

            self.assertEqual(
                _portable_manifest_sha256(first[0], first[1], first[2]),
                _portable_manifest_sha256(second[0], second[1], second[2]),
            )

    def test_official_asset_cache_path_has_same_registered_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, root, urdf, _ = _asset_fixture(Path(temporary) / "workspace")
            baseline = _portable_manifest_sha256(manifest, root, urdf)
            manifest["robolab_usd"] = (
                "/workspace/cache/official_assets/robolab/"
                "f72d39445ab699c9cfd9bf551e8d21b6075293b9/"
                "franka_robotiq_2f_85_flattened.usd"
            )
            self.assertEqual(
                _portable_manifest_sha256(manifest, root, urdf), baseline
            )

    def test_official_asset_cache_source_identity_rejects_unregistered_revision(self):
        manifest_path = (
            "/workspace/cache/official_assets/robolab/other-revision/"
            "franka_robotiq_2f_85_flattened.usd"
        )
        from autoresearch.robotics_robot_asset import _source_identity

        with self.assertRaisesRegex(ValueError, "unrecognized DROID source path"):
            _source_identity(manifest_path)


    def test_portable_fingerprint_detects_generated_mesh_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, root, urdf, files = _asset_fixture(Path(temporary) / "user")
            files["collision"].write_bytes(b"different mesh")

            with self.assertRaisesRegex(ValueError, "mesh digest changed"):
                _portable_manifest_sha256(manifest, root, urdf)


    def test_portable_fingerprint_detects_robot_semantic_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, root, urdf, _ = _asset_fixture(Path(temporary) / "user")
            baseline = _portable_manifest_sha256(manifest, root, urdf)
            manifest["mount"]["xyz_m"][2] += 0.001

            self.assertNotEqual(
                _portable_manifest_sha256(manifest, root, urdf), baseline
            )

    def test_legacy_manifest_and_mesh_paths_rebind_after_workspace_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = _asset_fixture(Path(temporary) / "user_a" / "WOV2")
            second = _asset_fixture(Path(temporary) / "user_b" / "different_root")
            old_manifest, _, old_urdf, _ = first
            new_manifest, new_root, new_urdf, new_files = second

            # Simulate copying only the registered bundle while external
            # pybullet_data has been installed under the second user's root.
            copied_urdf = new_root / old_urdf.name
            copied_urdf.write_bytes(old_urdf.read_bytes())
            manifest_path = new_root / "robot_asset_manifest.json"
            manifest_path.write_text(json.dumps(old_manifest))
            self.assertEqual(
                _asset_member(old_manifest["generated_urdf"], new_root),
                copied_urdf.resolve(),
            )

            fake_pybullet = types.SimpleNamespace(
                getDataPath=lambda: str(
                    new_files["panda_mesh"].parents[3]
                )
            )
            expected = _portable_manifest_sha256(new_manifest, new_root, new_urdf)
            with patch.dict("sys.modules", {"pybullet_data": fake_pybullet}), patch(
                "autoresearch.robotics_robot_asset.ROBOT_ASSET_PORTABLE_SHA256",
                expected,
            ):
                found_manifest, runtime_urdf = validate_robot_asset(
                    new_files["output_root"]
                )

            self.assertEqual(found_manifest, manifest_path.resolve())
            self.assertTrue(runtime_urdf.is_file())
            self.assertNotEqual(runtime_urdf, copied_urdf)
            runtime_root = ET.parse(runtime_urdf).getroot()
            filenames = [node.attrib["filename"] for node in runtime_root.iter("mesh")]
            self.assertIn(str(new_files["panda_mesh"].resolve()), filenames)

    def test_restore_recovers_invalid_local_bundle_from_verified_shared_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            workspace = base / "WOV2" / "branches" / "isolated"
            workspace.mkdir(parents=True)
            source = base / "WOV2" / "runtime" / "robot_assets" / ROBOT_ASSET_DIR
            source.mkdir(parents=True)
            (source / "robot_asset_manifest.json").write_text("registered")
            (source / "asset.bin").write_bytes(b"verified asset")
            output_root = workspace / "outputs"
            destination = output_root / "robot_assets" / ROBOT_ASSET_DIR
            destination.mkdir(parents=True)
            (destination / "robot_asset_manifest.json").write_text("local")
            (destination / "asset.bin").write_bytes(b"invalid asset")

            def validate(root):
                bundle = Path(root) / "robot_assets" / ROBOT_ASSET_DIR
                if (bundle / "asset.bin").read_bytes() == b"invalid asset":
                    raise ValueError("portable asset fingerprint changed")
                return bundle / "robot_asset_manifest.json", bundle / "asset.bin"

            from contextlib import nullcontext
            with patch(
                "autoresearch.robotics_robot_asset.validate_robot_asset",
                side_effect=validate,
            ), patch(
                "autoresearch.robotics_robot_asset.exclusive_case_asset_edit",
                return_value=nullcontext(),
            ):
                manifest, urdf = prepare_robot_asset(workspace, output_root, restore=True)

            self.assertEqual(manifest, destination / "robot_asset_manifest.json")
            self.assertEqual(urdf.read_bytes(), b"verified asset")
            preserved = list(destination.parent.glob(f"{ROBOT_ASSET_DIR}.invalid-*"))
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0] / "asset.bin").read_bytes(), b"invalid asset")


if __name__ == "__main__":
    unittest.main()
