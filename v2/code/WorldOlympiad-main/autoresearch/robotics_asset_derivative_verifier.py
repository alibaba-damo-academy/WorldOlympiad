"""Independent structural admission for a graspable container derivative."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numbers(element: ET.Element, path: str, count: int) -> list[float]:
    import math

    value = element.findtext(path)
    if value is None:
        raise ValueError(f"asset derivative lacks {path}")
    numbers = [float(token) for token in value.split()]
    if len(numbers) != count or not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"asset derivative has invalid {path}")
    return numbers


def verify_derivative(asset_root: Path, output: Path) -> dict:
    asset_root, output = asset_root.resolve(), output.resolve()
    manifest_path = asset_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source_sdf"]).resolve()
    candidate = Path(manifest["candidate_sdf"]).resolve()
    candidate.relative_to(asset_root)
    if (manifest.get("schema_version") != "worldolympiad_asset_derivative.v1"
            or not source.is_file() or not candidate.is_file()
            or manifest.get("source_sha256") != _sha(source)
            or manifest.get("candidate_sha256") != _sha(candidate)):
        raise ValueError("asset derivative does not bind exact source and candidate SDFs")
    for old in source.parent.iterdir():
        if old.is_file() and old != source:
            copied = asset_root / old.name
            if not copied.is_file() or _sha(copied) != _sha(old):
                raise ValueError("asset derivative changed a source mesh")
    old_root = ET.parse(source).getroot()
    new_root = ET.parse(candidate).getroot()
    links = new_root.findall("./model/link")
    if len(links) != 1:
        raise ValueError("grasp derivative must retain one rigid source link")
    link = links[0]
    collision = [item for item in link.findall("collision")
                 if item.get("name") == "grasp_tab_collision"]
    visual = [item for item in link.findall("visual")
              if item.get("name") == "grasp_tab_visual"]
    if len(collision) != 1 or len(visual) != 1:
        raise ValueError("grasp derivative needs one matching collision and visual tab")
    pose = _numbers(collision[0], "pose", 6)
    size = _numbers(collision[0], "geometry/box/size", 3)
    if (_numbers(visual[0], "pose", 6) != pose
            or _numbers(visual[0], "geometry/box/size", 3) != size
            or any(abs(value) > 1e-8 for value in pose[3:])
            or not all(0.005 <= value <= limit for value, limit in
                       zip(size, (0.10, 0.08, 0.05)))):
        raise ValueError("grasp tab collision and appearance differ or exceed envelope")
    declared_pose = [float(value) for value in manifest["tab_pose_xyz_m"]]
    declared_size = [float(value) for value in manifest["tab_size_xyz_m"]]
    grasp_site = [float(value) for value in manifest["grasp_site_local_xy_m"]]
    if (pose[:3] != declared_pose or size != declared_size
            or abs(grasp_site[0] - pose[0]) > size[0] / 2
            or abs(grasp_site[1] - pose[1]) > size[1] / 2):
        raise ValueError("grasp site is not bound to the candidate tab")
    link.remove(collision[0])
    link.remove(visual[0])
    canonical = lambda root: ET.canonicalize(
        ET.tostring(root, encoding="unicode"), strip_text=True)
    if canonical(old_root) != canonical(new_root):
        raise ValueError("asset derivative changed original scene geometry")
    result = {
        "schema_version": "worldolympiad_asset_derivative_verdict.v1",
        "passed": True,
        "source_sha256": _sha(source),
        "candidate_sha256": _sha(candidate),
        "candidate_manifest_sha256": _sha(manifest_path),
        "grasp_site_local_xy_m": grasp_site,
        "visual_review": "required_in_formal_reference_context",
        "source_geometry_preserved": True,
        "collision_visual_tab_coherent": True,
    }
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("published derivative verdict changed")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(verify_derivative(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
