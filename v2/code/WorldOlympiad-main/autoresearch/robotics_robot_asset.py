"""Validate and restore the registered DROID embodiment from pinned local sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET

from .io import sha256_file
from .robotics_resource_lock import exclusive_case_asset_edit

ROBOT_ASSET_DIR = "robolab_droid_franka_robotiq_2f85_f72d_2c047340_20260823_v3"
# Historical raw-file pins are retained for provenance; they include absolute paths.
ROBOT_ASSET_MANIFEST_SHA256 = "803d9420c23a71d489af724e4dcbae65291c66b073eb5da8900837577e72789a"
ROBOT_URDF_SHA256 = "0894862f40c135e1774e5d0b9620c9fa245a9056bf16e1d6d1ee46f0d849c128"
USD_SHA256 = "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41"
BUILDER_SHA256 = "d46198abb1f937ddae474b86f5980219b6d780f937c08eeea1b907a42f48d225"
CONVERTER_SHA256 = "1e2847e732c14aa24ecd32abdecec31441d6fcdcb623d7d422b5de983d1b257e"
PICKNIK_COMMIT = "2c047340aeb2440f7a60e429264221aab9658707"
ROBOLAB_COMMIT = "f72d39445ab699c9cfd9bf551e8d21b6075293b9"
ROBOT_ASSET_PORTABLE_SHA256 = "26f3c83fafcd8c49e5040472fea89df5ff37ffdfe8ba4e0cdd3da9ca6b7df01e"


def _checked(path: Path, expected: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"DROID resource missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"DROID resource digest changed: {path}")
    return path


def _path_parts(value: str) -> tuple[str, ...]:
    """Parse manifest paths consistently, including paths written on Windows."""
    return PurePosixPath(value.replace("\\", "/")).parts


def _source_identity(value: str) -> str:
    """Return a stable source identity without its machine-specific prefix."""
    parts = _path_parts(value)
    for anchor in ("pybullet_data", "robotiq_description", "RoboLab"):
        if anchor in parts:
            index = parts.index(anchor)
            return "/".join(parts[index:])
    # SceneCode may materialize the pinned USD in its content-addressed
    # official-assets cache rather than from the RoboLab checkout. Normalize
    # that known layout to the checkout identity, but only for the registered
    # source revision; the USD digest is independently checked by the asset
    # validator.
    if "official_assets" in parts:
        index = parts.index("official_assets")
        if (
            len(parts) >= index + 4
            and parts[index + 1] == "robolab"
            and parts[index + 2] == ROBOLAB_COMMIT
        ):
            return f"RoboLab/{parts[index + 3]}"
    if "ros2_robotiq_gripper" in parts:
        return "picknik_repo"
    raise ValueError(f"unrecognized DROID source path in manifest: {value}")


def _asset_member(value: str, root: Path) -> Path:
    """Resolve generated-file paths from either v2 absolute or relative manifests."""
    root = root.resolve()
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            pass
        parts = _path_parts(value)
        for index in range(len(parts) - 1):
            if parts[index] == "robot_assets" and parts[index + 1] == ROBOT_ASSET_DIR:
                candidate = root.joinpath(*parts[index + 2 :])
                break
        else:
            raise ValueError(f"DROID generated path is outside its asset bundle: {value}")
    else:
        candidate = root / candidate

    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"DROID generated path escapes its asset bundle: {value}") from exc
    return resolved


def _asset_relative(value: str, root: Path) -> str:
    return _asset_member(value, root).relative_to(root.resolve()).as_posix()


def _pybullet_data_roots(root: Path) -> list[Path]:
    roots = []
    try:
        import pybullet_data

        roots.append(Path(pybullet_data.getDataPath()).resolve())
    except ImportError:
        pass

    # The stage process may use a lightweight environment without pybullet_data;
    # inspect the workspace's runtime environments without assuming Python X.Y.
    workspace = root.resolve().parents[2]
    for pattern in (
        "envs/*/lib/python*/site-packages/pybullet_data",
        "venvs/*/lib/python*/site-packages/pybullet_data",
        "venv/lib/python*/site-packages/pybullet_data",
        ".venv/lib/python*/site-packages/pybullet_data",
        "runtime/python-*/lib/python*/site-packages/pybullet_data",
    ):
        roots.extend(path.resolve() for path in workspace.glob(pattern))
    unique = []
    for candidate in roots:
        if candidate not in unique and (candidate / "franka_panda/panda.urdf").is_file():
            unique.append(candidate)
    return unique


def _resolve_mesh_path(
    value: str,
    urdf_path: Path,
    root: Path,
    expected_sha256: str | None = None,
) -> Path:
    """Rebind legacy absolute references to this install's asset/runtime roots."""
    parts = _path_parts(value)
    if "robot_assets" in parts:
        for index in range(len(parts) - 1):
            if parts[index] == "robot_assets" and parts[index + 1] == ROBOT_ASSET_DIR:
                return _asset_member(value, root)
    if "pybullet_data" in parts:
        index = parts.index("pybullet_data")
        suffix = parts[index + 1 :]
        candidates = [base.joinpath(*suffix).resolve() for base in _pybullet_data_roots(root)]
        original = Path(value)
        if original.is_file():
            candidates.append(original.resolve())
        for candidate in candidates:
            if candidate.is_file() and (
                expected_sha256 is None or sha256_file(candidate) == expected_sha256
            ):
                return candidate
        raise FileNotFoundError(
            f"no installed pybullet_data mesh matches the registered DROID input: {value}"
        )
    path = Path(value)
    if not path.is_absolute():
        return (urdf_path.parent / path).resolve()
    try:
        path.resolve().relative_to(root.resolve())
        return path.resolve()
    except ValueError as exc:
        raise ValueError(f"unrecognized external DROID mesh path: {value}") from exc


def _portable_manifest_sha256(manifest: dict, root: Path, urdf_path: Path) -> str:
    """Fingerprint asset bytes and robot semantics, excluding installation paths."""
    source_hashes: dict[str, str] = {}
    for item in manifest.get("mesh_sources", []):
        identity = _source_identity(item["path"])
        digest = item["sha256"]
        if identity in source_hashes and source_hashes[identity] != digest:
            raise ValueError(f"conflicting DROID source hashes for {identity}")
        source_hashes[identity] = digest

    generated_hashes: dict[str, str] = {}
    collision_meshes = []
    for item in manifest.get("generated_collision_meshes", []):
        relative = _asset_relative(item["path"], root)
        collision_meshes.append({"path": relative, "sha256": item["sha256"]})
        generated_hashes[relative] = item["sha256"]

    visual_records = []
    for item in manifest.get("visual_conversion", {}).get("records", []):
        source_id = _source_identity(item["source"])
        destination = _asset_relative(item["destination"], root)
        material = _asset_relative(item["material"], root) if item.get("material") else None
        generated_hashes[destination] = item["destination_sha256"]
        if material:
            generated_hashes[material] = item["material_sha256"]
        visual_records.append(
            {
                "source": source_id,
                "source_sha256": item["source_sha256"],
                "destination": destination,
                "destination_sha256": item["destination_sha256"],
                "material": material,
                "material_sha256": item.get("material_sha256"),
                "mesh_object_count": item["mesh_object_count"],
                "world_coordinate_bounds": item["world_coordinate_bounds"],
            }
        )

    robot = ET.parse(urdf_path).getroot()
    for node in robot.iter("mesh"):
        value = node.attrib.get("filename", "")
        try:
            relative = _asset_relative(value, root)
        except ValueError:
            relative = None
        expected = (
            generated_hashes.get(relative)
            if relative is not None
            else source_hashes.get(_source_identity(value))
        )
        if expected is None:
            raise ValueError(f"URDF mesh has no portable asset record: {value}")
        path = _resolve_mesh_path(value, urdf_path, root, expected)
        if not path.is_file():
            raise FileNotFoundError(f"DROID URDF mesh is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"DROID URDF mesh digest changed: {path}")
        node.set("filename", f"sha256:{actual}")

    canonical_urdf = ET.canonicalize(
        ET.tostring(robot, encoding="unicode"), strip_text=True
    )
    payload = {
        "lock_schema": "worldolympiad.droid-asset-content.v1",
        "builder_sha256": BUILDER_SHA256,
        "converter_sha256": CONVERTER_SHA256,
        "robot": {
            "schema_version": manifest["schema_version"],
            "robot_embodiment": manifest["robot_embodiment"],
            "canonical_urdf_sha256": hashlib.sha256(canonical_urdf.encode("utf-8")).hexdigest(),
            "panda_urdf_sha256": manifest["panda_urdf_sha256"],
            "picknik_commit": manifest["picknik_commit"],
            "picknik_macro": _source_identity(manifest["picknik_macro"]),
            "picknik_macro_sha256": manifest["picknik_macro_sha256"],
            "robolab_commit": manifest["robolab_commit"],
            "robolab_usd": _source_identity(manifest["robolab_usd"]),
            "robolab_usd_sha256": manifest["robolab_usd_sha256"],
            "mount": manifest["mount"],
            "gripper": manifest["gripper"],
            "mesh_sources": sorted(
                ({"path": path, "sha256": digest} for path, digest in source_hashes.items()),
                key=lambda row: (row["path"], row["sha256"]),
            ),
            "generated_collision_meshes": sorted(
                collision_meshes, key=lambda row: (row["path"], row["sha256"])
            ),
            "visual_conversion": {
                "authority": manifest["visual_conversion"]["authority"],
                "converter_script_sha256": manifest["visual_conversion"][
                    "converter_script_sha256"
                ],
                "coordinate_contract": manifest["visual_conversion"][
                    "coordinate_contract"
                ],
                "records": sorted(visual_records, key=lambda row: row["source"]),
            },
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_robot_asset(output_root: Path) -> tuple[Path, Path]:
    """Validate the portable DROID content lock and generated dependencies."""
    root = output_root.resolve() / "robot_assets" / ROBOT_ASSET_DIR
    manifest_path = root / "robot_asset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DROID resource missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "robolab_droid_portable_urdf.v2":
        raise ValueError("unsupported DROID portable robot asset")
    if manifest.get("picknik_commit") != PICKNIK_COMMIT:
        raise ValueError("DROID PickNik source revision is not the registered revision")
    if manifest.get("robolab_commit") != ROBOLAB_COMMIT:
        raise ValueError("DROID RoboLab source revision is not the registered revision")
    if manifest.get("robolab_usd_sha256") != USD_SHA256:
        raise ValueError("DROID RoboLab USD is not the registered asset")
    if manifest.get("visual_conversion", {}).get("converter_script_sha256") != CONVERTER_SHA256:
        raise ValueError("DROID COLLADA converter is not the registered version")

    urdf = _asset_member(manifest["generated_urdf"], root)
    if urdf != (root / "franka_robotiq_2f_85.urdf").resolve():
        raise ValueError("DROID manifest names a different asset directory")
    source_hashes = {
        _source_identity(item["path"]): item["sha256"]
        for item in manifest.get("mesh_sources", [])
    }
    recorded: dict[Path, str] = {}
    for item in manifest.get("generated_collision_meshes", []):
        path = _asset_member(item["path"], root)
        recorded[path] = item["sha256"]
        _checked(path, item["sha256"])
    for item in manifest["visual_conversion"]["records"]:
        for name in ("destination", "material"):
            if item.get(name):
                path = _asset_member(item[name], root)
                expected = item[name + "_sha256"]
                recorded[path] = expected
                _checked(path, expected)
    robot = ET.parse(urdf).getroot()
    visual_nodes = {id(node) for node in robot.findall("./link/visual/geometry/mesh")}
    colour_names = {node.get("name") for node in robot.iter("material") if node.find("color") is not None}
    colour_override_nodes = {
        id(node)
        for visual in robot.findall("./link/visual")
        if (material := visual.find("material")) is not None
        and (material.find("color") is not None or material.get("name") in colour_names)
        for node in visual.findall("./geometry/mesh")
    }
    relocated_mesh_paths = False
    resolved_mesh_paths = []
    for node in robot.iter("mesh"):
        original = node.attrib["filename"]
        try:
            relative = _asset_relative(original, root)
        except ValueError:
            relative = None
        expected = recorded.get((root / relative).resolve()) if relative is not None else None
        if expected is None:
            expected = source_hashes.get(_source_identity(original))
        if expected is None:
            raise ValueError(f"URDF mesh has no portable dependency record: {original}")
        path = _resolve_mesh_path(original, urdf, root, expected)
        resolved_mesh_paths.append(path)
        original_path = Path(original)
        if not original_path.is_absolute():
            original_path = urdf.parent / original_path
        if original_path.resolve() != path:
            node.set("filename", str(path))
            relocated_mesh_paths = True
        if recorded.get(path) != expected:
            _checked(path, expected)
            recorded[path] = expected
        # Panda's OBJ material sidecars are implicit URDF dependencies. Check
        # these too: an intact URDF/OBJ must not conceal a missing texture.
        if id(node) in visual_nodes and path.suffix.lower() == ".obj":
            for line in path.read_text(errors="strict").splitlines():
                words = shlex.split(line, comments=True)
                if not words or words[0] != "mtllib":
                    continue
                for name in words[1:]:
                    material = path.parent / name
                    if not material.is_file():
                        # The pinned stock Panda base uses a collision OBJ as
                        # its visual and supplies panda_white in the URDF. Its
                        # unused mtllib was never shipped by pybullet_data.
                        if id(node) in colour_override_nodes:
                            continue
                        raise FileNotFoundError(f"DROID material missing: {material}")
                    for material_line in material.read_text().splitlines():
                        fields = shlex.split(material_line, comments=True)
                        if fields and (fields[0].lower().startswith("map_") or fields[0].lower() in {"bump", "disp", "decal"}):
                            texture = material.parent / fields[-1]
                            if not texture.is_file():
                                raise FileNotFoundError(f"DROID texture missing: {texture}")
    actual_portable_hash = _portable_manifest_sha256(manifest, root, urdf)
    if actual_portable_hash != ROBOT_ASSET_PORTABLE_SHA256:
        raise ValueError(
            "DROID portable asset fingerprint changed: "
            f"expected {ROBOT_ASSET_PORTABLE_SHA256}, got {actual_portable_hash}"
        )
    if not relocated_mesh_paths:
        return manifest_path, urdf

    path_signature = hashlib.sha256(
        "\n".join(sorted(str(path) for path in resolved_mesh_paths)).encode("utf-8")
    ).hexdigest()[:16]
    runtime_urdf = (
        output_root.resolve()
        / ".robot_asset_runtime"
        / ROBOT_ASSET_DIR
        / path_signature
        / urdf.name
    )
    runtime_urdf.parent.mkdir(parents=True, exist_ok=True)
    temporary = runtime_urdf.with_suffix(".tmp")
    ET.ElementTree(robot).write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(runtime_urdf)
    return manifest_path, runtime_urdf


def _checked_picknik_source(workspace: Path) -> Path:
    """Find the pinned PickNik checkout, including another local WOV2 branch.

    Isolated branch checkouts sometimes omit dependencies while a sibling checkout
    has the exact immutable source. Accept only the pinned Git revision; never
    fall back to an unversioned runtime copy for robot-asset generation.
    """
    expected = PICKNIK_COMMIT
    candidates = [workspace / "code/dependencies/ros2_robotiq_gripper_2c047340"]
    branches = workspace.parent
    if branches.is_dir():
        candidates.extend(
            branch / "code/dependencies/ros2_robotiq_gripper_2c047340"
            for branch in sorted(branches.iterdir())
            if branch.is_dir() and branch != workspace
        )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            revision = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if revision == expected:
            return candidate.resolve()
    raise FileNotFoundError(
        f"pinned PickNik checkout {expected} missing from {workspace} and sibling WOV2 branches"
    )


def _checked_usd_source(workspace: Path) -> Path:
    """Find the pinned RoboLab USD in the checkout or a shared WOV2 runtime."""
    filename = "franka_robotiq_2f_85_flattened.usd"
    candidates = [
        workspace / "code/RoboLab-isaac51/assets/robots" / filename,
        *(parent / "runtime/RoboLab" / filename for parent in workspace.parents),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return _checked(candidate, USD_SHA256)
    # Keep the primary expected location in the error for actionable diagnostics.
    raise FileNotFoundError(f"DROID resource missing: {candidates[0]}")


def _verified_shared_robot_asset(workspace: Path) -> Path | None:
    """Locate a shared immutable DROID bundle, accepting it only after full validation."""
    for parent in workspace.parents:
        candidate = parent / "runtime/robot_assets" / ROBOT_ASSET_DIR
        if not candidate.is_dir():
            continue
        try:
            validate_robot_asset(candidate.parent.parent)
        except (FileNotFoundError, ValueError, ET.ParseError):
            continue
        return candidate
    return None


def _install_shared_robot_asset(source: Path, destination: Path) -> None:
    """Install a previously validated shared bundle without discarding bad evidence."""
    if destination.exists():
        digest_input = destination / "robot_asset_manifest.json"
        suffix = sha256_file(digest_input)[:12] if digest_input.is_file() else "unmanifested"
        backup = destination.with_name(f"{destination.name}.invalid-{suffix}")
        if backup.exists():
            raise FileExistsError(f"refusing to overwrite preserved DROID evidence: {backup}")
        destination.rename(backup)
    shutil.copytree(source, destination)


def prepare_robot_asset(workspace: Path, output_root: Path, *, restore: bool = False) -> tuple[Path, Path]:
    """Restore a validated bundle; retain invalid generated evidence during recovery."""
    workspace, output_root = workspace.resolve(), output_root.resolve()
    with exclusive_case_asset_edit(output_root, "registered_robot_asset", ROBOT_ASSET_DIR):
        root = output_root / "robot_assets" / ROBOT_ASSET_DIR
        if root.exists():
            try:
                return validate_robot_asset(output_root)
            except (FileNotFoundError, ValueError, ET.ParseError):
                if not restore:
                    raise
                shared = _verified_shared_robot_asset(workspace)
                if shared is None:
                    raise
                _install_shared_robot_asset(shared, root)
                return validate_robot_asset(output_root)
        if not restore:
            return validate_robot_asset(output_root)
        shared = _verified_shared_robot_asset(workspace)
        if shared is not None:
            _install_shared_robot_asset(shared, root)
            return validate_robot_asset(output_root)
        bridge = workspace / "code/scenecode_pybullet_minimal/scripts"
        builder = _checked(bridge / "build_robolab_droid_urdf.py", BUILDER_SHA256)
        _checked(bridge / "convert_collada_to_obj_blender.py", CONVERTER_SHA256)
        usd = workspace / "cache/official_assets/robolab/f72d39445ab699c9cfd9bf551e8d21b6075293b9/franka_robotiq_2f_85_flattened.usd"
        if not usd.exists():
            source = _checked_usd_source(workspace)
            usd.parent.mkdir(parents=True, exist_ok=True)
            temporary = usd.with_suffix(".restore.tmp")
            shutil.copy2(source, temporary)
            _checked(temporary, USD_SHA256)
            temporary.replace(usd)
        _checked(usd, USD_SHA256)
        python = workspace / "envs/autoresearch/bin/python"
        completed = subprocess.run(
            [
                str(python),
                "-c",
                "import pybullet_data; print(pybullet_data.getDataPath())",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        panda_urdf = Path(completed.stdout.strip()) / "franka_panda" / "panda.urdf"
        command = [
            str(python), str(builder),
            "--panda-urdf", str(panda_urdf),
            "--picknik-repo", str(_checked_picknik_source(workspace)),
            "--robolab-usd", str(usd),
            "--blender-executable", str(workspace / "runtime/blender-4.4.3-linux-x64/blender"),
            "--output-dir", str(root),
        ]
        subprocess.run(command, check=True, timeout=300)
        return validate_robot_asset(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--case-id")
    args = parser.parse_args()
    root = args.output_root or args.workspace / "outputs"
    if bool(args.batch_id) != bool(args.case_id):
        parser.error("batch-id and case-id must be supplied together")
    if args.batch_id:
        from .robotics_scenecode_case_generator import _load_case
        task, _, _ = _load_case(output_root=root, batch_id=args.batch_id, case_id=args.case_id)
        if "cosmos3_nano_droid" not in task.requested_wams:
            print(json.dumps({"status": "not_required", "reason": "selected task does not use DROID"}))
            return
    manifest, urdf = prepare_robot_asset(args.workspace, root, restore=args.restore)
    print(json.dumps({"status": "verified", "manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "portable_asset_sha256": ROBOT_ASSET_PORTABLE_SHA256, "urdf": str(urdf), "urdf_sha256": sha256_file(urdf)}))


if __name__ == "__main__":
    main()
