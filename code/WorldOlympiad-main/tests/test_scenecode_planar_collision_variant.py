import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.derive_scenecode_planar_collision_variant import (
    derive_planar_collision_variant,
)


def _source(root: Path, *, with_collision: bool = False) -> Path:
    asset = root / "source"
    asset.mkdir()
    (asset / "mat.gltf").write_text("{}", encoding="utf-8")
    collision = (
        "<collision name='old'><geometry><box><size>1 1 1</size></box></geometry></collision>"
        if with_collision
        else ""
    )
    sdf = asset / "mat.sdf"
    sdf.write_text(
        "<sdf version='1.10'><model name='mat'><static>true</static>"
        "<link name='body'><visual name='visual'><geometry><mesh>"
        "<uri>mat.gltf</uri></mesh></geometry></visual>"
        f"{collision}</link></model></sdf>",
        encoding="utf-8",
    )
    return sdf


def test_planar_variant_adds_only_bounds_matched_collision(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "variant"
    result = derive_planar_collision_variant(
        source,
        output,
        variant_id="teal_mat_planar_collision_v1",
        size_xyz_m=(0.18, 0.18, 0.002),
    )
    collision = ET.parse(result).getroot().find("model/link/collision")
    assert collision is not None
    assert collision.findtext("geometry/box/size") == "0.18 0.18 0.002"
    assert collision.findtext("pose") == "0 0 0.001 0 0 0"
    manifest = json.loads((output / "physical_variant_manifest.json").read_text())
    assert manifest["schema_version"] == "scenecode_physical_variant.v2"
    assert manifest["changed_files"] == ["mat.sdf"]
    assert manifest["visual_geometry_unchanged"] is True


def test_planar_variant_rejects_existing_collision(tmp_path: Path) -> None:
    source = _source(tmp_path, with_collision=True)
    with pytest.raises(ValueError, match="collisionless source"):
        derive_planar_collision_variant(
            source,
            tmp_path / "variant",
            variant_id="duplicate_collision",
            size_xyz_m=(0.18, 0.18, 0.002),
        )
