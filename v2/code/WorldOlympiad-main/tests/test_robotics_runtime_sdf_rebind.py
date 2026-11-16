from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from autoresearch.robotics_runtime_sdf_rebind import bind_runtime_physics_sdfs
from autoresearch.robotics_views import sha256_file


def test_deleted_cache_mesh_is_rebound_to_hashed_bundle_asset(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    source = bundle / "source"
    merged = bundle / "merged"
    (source / "collision").mkdir(parents=True)
    merged.mkdir()
    mesh = source / "collision" / "shape.obj"
    mesh.write_text("v 0 0 0\n")
    source_sdf = source / "object.sdf"
    source_sdf.write_text("<sdf><model><link name='body'><collision><geometry><mesh>"
                          "<uri>collision/shape.obj</uri></mesh></geometry>"
                          "</collision></link></model></sdf>")
    merged_sdf = merged / "object.sdf"
    merged_sdf.write_text("<sdf><model><link name='body'><collision><geometry><mesh>"
                          "<uri>/deleted/cache/shape.obj</uri></mesh></geometry>"
                          "</collision></link></model></sdf>")
    manifest = bundle / "robotics_reference_manifest.json"
    manifest.write_text("{}")
    prepared = {
        role: {"source_sdf": "source/object.sdf", "physics_sdf": "merged/object.sdf"}
        for role in ("table", "cup", "tray")
    }
    paths = bind_runtime_physics_sdfs(
        manifest, prepared, tmp_path / "runtime",
        {"source/collision/shape.obj": sha256_file(mesh)},
    )
    for path in paths.values():
        assert ET.parse(path).findtext(".//mesh/uri") == str(mesh)
    receipt = json.loads((tmp_path / "runtime/asset_binding_receipt.json").read_text())
    assert receipt["reference_manifest_sha256"] == sha256_file(manifest)
    assert len(receipt["assets"]) == 3
    mesh.write_text("changed")
    with pytest.raises(ValueError, match="digest changed"):
        bind_runtime_physics_sdfs(
            manifest, prepared, tmp_path / "runtime",
            {"source/collision/shape.obj": receipt["assets"][0]["mesh_bindings"][0]["sha256"]},
        )
