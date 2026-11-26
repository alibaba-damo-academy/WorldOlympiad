"""Read scaled per-link glTF visual bounds from an SDF without changing its assets."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ..paths import WORKSPACE

def _load_visual_meshes(sdf: Path):
    runtime = Path(os.environ.get(
        "SCENECODE_RUNTIME_ROOT", WORKSPACE / "runtime/scenecode"
    ))
    sys.path.insert(0, str(runtime))
    import bpy  # Load Blender's mathutils before the existing geometry reader.
    from scenecode.agent_utils.support_surface_extraction import _load_sdf_visual_link_meshes
    return _load_sdf_visual_link_meshes(sdf)


def measure_asset(sdf: Path) -> dict:
    sdf = sdf.resolve(strict=True)
    if sdf.suffix.lower() != ".sdf":
        raise ValueError("expected a scaled source .sdf file")
    digest = hashlib.sha256(sdf.read_bytes()).hexdigest()
    for visual in ET.fromstring(sdf.read_bytes()).findall(".//visual"):
        uri = visual.findtext("geometry/mesh/uri")
        if not uri or Path(uri).suffix.lower() not in {".gltf", ".glb"}:
            raise ValueError("this reader supports glTF-backed SDF visuals only; no partial measurement")
    with contextlib.redirect_stdout(sys.stderr):
        meshes = _load_visual_meshes(sdf)
    links = {}
    for _, name, mesh in meshes:
        lo, hi = [[float(x) for x in row] for row in mesh.bounds]
        if any(not math.isfinite(x) for x in lo + hi) or any(a > b for a, b in zip(lo, hi)):
            raise ValueError(f"invalid visual bounds for {name!r}")
        if name in links:
            old = links[name]
            lo = [min(a, b) for a, b in zip(old["min_m"], lo)]
            hi = [max(a, b) for a, b in zip(old["max_m"], hi)]
        links[name] = {"name": name, "min_m": lo, "max_m": hi,
                       "size_m": [b - a for a, b in zip(lo, hi)]}
    if not links:
        raise ValueError("no readable SDF visual meshes; measurement is unavailable")
    if hashlib.sha256(sdf.read_bytes()).hexdigest() != digest:
        raise RuntimeError("SDF changed during measurement; inspect the completed asset")
    return {"schema_version": "wo_asset_geometry.v1", "sdf": str(sdf), "sdf_sha256": digest,
            "measurement": "visual meshes; model frame at default joint poses; declared scales applied once",
            "links": list(links.values()), "grasp_collision_execution_tested": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure_asset(args.sdf), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
