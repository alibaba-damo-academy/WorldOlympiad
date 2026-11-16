#!/usr/bin/env python3
"""Run one reusable multi-object DROID operator graph from a typed manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenecode_pybullet_bridge import (
    DroidGraphObject,
    DroidOperatorGraphInputs,
    DroidPickTerminalStep,
    run_droid_operator_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-manifest", type=Path, required=True)
    parser.add_argument("--table-sdf", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.graph_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "robotics_droid_operator_graph_input.v1":
        raise ValueError("unsupported DROID operator-graph input manifest")
    objects = tuple(
        DroidGraphObject(
            object_id=item["object_id"],
            sdf_path=Path(item["sdf_path"]),
            center_xy=tuple(item["center_xy"]),
            role=item["role"],
            orientation_xyzw=tuple(
                item.get("orientation_xyzw", (0.0, 0.0, 0.0, 1.0))
            ),
            lateral_friction=float(item.get("lateral_friction", 1.2)),
            collision_enabled=bool(item.get("collision_enabled", True)),
            fixed=bool(item.get("fixed", False)),
        )
        for item in payload.get("objects") or []
    )
    steps = tuple(
        DroidPickTerminalStep(
            subject_id=item["subject_id"],
            target_id=item["target_id"],
            terminal_operator=item["terminal_operator"],
            success_predicate=item["success_predicate"],
            grasp_base_offset_z_m=float(item["grasp_base_offset_z_m"]),
            hover_base_offset_z_m=float(item["hover_base_offset_z_m"]),
            lift_base_offset_z_m=float(item["lift_base_offset_z_m"]),
            place_target_offset_z_m=float(
                item.get("place_target_offset_z_m", 0.0)
            ),
            place_target_offset_xy_m=tuple(
                item.get("place_target_offset_xy_m", (0.0, 0.0))
            ),
            grasp_yaw_deg=float(item.get("grasp_yaw_deg", 90.0)),
            terminal_alignment_axis_local=(
                tuple(item["terminal_alignment_axis_local"])
                if item.get("terminal_alignment_axis_local") is not None
                else None
            ),
            terminal_alignment_axis_world=(
                tuple(item["terminal_alignment_axis_world"])
                if item.get("terminal_alignment_axis_world") is not None
                else None
            ),
        )
        for item in payload.get("steps") or []
    )
    result = run_droid_operator_graph(
        DroidOperatorGraphInputs(
            table_sdf=args.table_sdf,
            panda_urdf=args.panda_urdf,
            output_dir=args.output_dir,
            objects=objects,
            steps=steps,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
