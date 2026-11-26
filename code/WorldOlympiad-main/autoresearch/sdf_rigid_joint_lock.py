"""Derive a rigid (joint-locked) representation of an articulated SDF asset.

A task that never manipulates an internal articulation may still need a rigid
body for native grasp/physics support.  Rather than editing the generated asset
or claiming an articulation the task does not use, this operator produces a
*derived* SDF whose declared joints are locked to a fixed pose while every
link's visual, collision and inertial content is preserved byte-for-byte as
XML.  The derivation is deterministic and carries source provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

SCHEMA_VERSION = "sdf_rigid_joint_lock.v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _link_fingerprint(model: ET.Element) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {}
    for link in model.findall("link"):
        name = link.get("name")
        if name in fingerprint:
            raise ValueError(f"duplicate link name {name!r}")
        fingerprint[name] = {
            "visual": len(link.findall("visual")),
            "collision": len(link.findall("collision")),
            "inertial": len(link.findall("inertial")),
            "visual_sha256": hashlib.sha256(
                b"".join(ET.tostring(item) for item in link.findall("visual"))
            ).hexdigest(),
            "collision_sha256": hashlib.sha256(
                b"".join(ET.tostring(item) for item in link.findall("collision"))
            ).hexdigest(),
            "inertial_sha256": hashlib.sha256(
                b"".join(ET.tostring(item) for item in link.findall("inertial"))
            ).hexdigest(),
        }
    return fingerprint


def _limit_value(limit: ET.Element, key: str) -> str:
    """Read a limit bound from either the attribute or the child-element form."""

    value = limit.get(key)
    if value is None:
        child = limit.find(key)
        value = child.text if child is not None else None
    return "0" if value is None else str(value).strip()


def _absolutise_uris(element: ET.Element, base: Path) -> None:
    for uri in element.iter("uri"):
        if uri.text is None:
            continue
        value = uri.text.strip()
        if value.startswith("/") or "://" in value:
            continue
        candidate = (base / value).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"SDF mesh uri does not resolve: {value}")
        uri.text = str(candidate)


def rigidify_sdf_joints(
    source: Path,
    output: Path,
    *,
    joints: list[str],
    provenance: dict[str, Any],
    absolutise_meshes: bool = True,
) -> dict[str, Any]:
    """Lock the named joints to ``fixed`` and write a derived SDF.

    The locked pose is the joint's declared zero, which must lie inside the
    original limits, so the derived body keeps the asset's rest configuration.
    """

    source = Path(source)
    output = Path(output)
    if not joints:
        raise ValueError("at least one joint must be declared")
    if len(joints) != len(set(joints)):
        raise ValueError("joint names must be unique")
    tree = ET.parse(source)
    root = tree.getroot()
    models = root.findall("model")
    if len(models) != 1:
        raise ValueError("rigid joint lock requires exactly one SDF model")
    model = models[0]
    if absolutise_meshes:
        # Resolve mesh references before fingerprinting: the derived file only
        # changes where the same geometry is loaded from, never the geometry.
        _absolutise_uris(model, source.parent)
    before = _link_fingerprint(model)

    locked: list[dict[str, Any]] = []
    for name in joints:
        joint = next(
            (item for item in model.findall("joint") if item.get("name") == name), None
        )
        if joint is None:
            raise ValueError(f"joint {name!r} is not declared by {source}")
        original_type = joint.get("type")
        if original_type == "fixed":
            raise ValueError(f"joint {name!r} is already fixed")
        axis = joint.find("axis")
        limits = None
        if axis is not None:
            limit = axis.find("limit")
            if limit is not None:
                lower = float(_limit_value(limit, "lower"))
                upper = float(_limit_value(limit, "upper"))
                limits = [lower, upper]
                if not lower <= 0.0 <= upper:
                    raise ValueError(
                        f"joint {name!r} zero pose is outside its declared limits"
                    )
            joint.remove(axis)
        joint.set("type", "fixed")
        locked.append(
            {
                "joint": name,
                "original_type": original_type,
                "original_limits": limits,
                "locked_value": 0.0,
                "axis_preserved": False,
                "parent": (joint.findtext("parent") or "").strip(),
                "child": (joint.findtext("child") or "").strip(),
            }
        )

    if absolutise_meshes:
        _absolutise_uris(model, source.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    reparsed = ET.parse(output).getroot()
    re_model = reparsed.find("model")
    after = _link_fingerprint(re_model)
    remaining = [
        {"joint": item.get("name"), "type": item.get("type")}
        for item in re_model.findall("joint")
        if item.get("type") != "fixed"
    ]
    checks = {
        "links_unchanged": set(before) == set(after),
        "visuals_unchanged": all(
            before[name]["visual_sha256"] == after[name]["visual_sha256"]
            and before[name]["visual"] == after[name]["visual"]
            for name in before
        ),
        "collisions_unchanged": all(
            before[name]["collision_sha256"] == after[name]["collision_sha256"]
            and before[name]["collision"] == after[name]["collision"]
            for name in before
        ),
        "inertials_unchanged": all(
            before[name]["inertial_sha256"] == after[name]["inertial_sha256"]
            and before[name]["inertial"] == after[name]["inertial"]
            for name in before
        ),
        "declared_joints_are_fixed": not remaining,
    }
    if not all(checks.values()):
        raise ValueError(f"derived SDF failed its structural invariants: {checks}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_sdf": str(source),
        "source_sdf_sha256": sha256_file(source),
        "derived_sdf": str(output),
        "derived_sdf_sha256": sha256_file(output),
        "model_name": model.get("name"),
        "locked_joints": locked,
        "remaining_non_fixed_joints": remaining,
        "link_count": len(before),
        "links": before,
        "mesh_uris_absolutised": bool(absolutise_meshes),
        "checks": checks,
        "passed": True,
        "provenance": provenance,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint", action="append", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    report = rigidify_sdf_joints(
        args.source,
        args.output,
        joints=args.joint,
        provenance=provenance,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("passed", "derived_sdf", "derived_sdf_sha256")}, indent=2))


if __name__ == "__main__":
    main()
