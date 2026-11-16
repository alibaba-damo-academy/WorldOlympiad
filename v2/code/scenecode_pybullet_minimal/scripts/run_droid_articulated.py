#!/usr/bin/env python3
"""Run one SceneCode-certified articulated task in PyBullet."""

from __future__ import annotations

# Pin the numerical backends to a single thread before numpy/scipy are imported.
# The contact planner solves small bounded least-squares problems; multi-threaded
# BLAS reductions make those solutions depend on thread scheduling, and the
# contact dynamics of a shallow handle amplify a 1e-12 solve difference into a
# different grasp outcome.  A pinned backend keeps one reference reproducible.
import os as _os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_variable, "1")

import argparse
import json
from pathlib import Path

from scenecode_pybullet_bridge import (
    ArticulationBinding,
    ArticulatedSceneInputs,
    run_droid_articulated,
)


_SUCCESS_PREDICATE_ALIASES = {
    "joint_at_open_limit": "joint_at_open_limit",
    "joint_at_closed_limit": "joint_at_closed_limit",
    "drawer_open": "joint_at_open_limit",
    "door_open": "joint_at_open_limit",
    "pressed": "joint_at_open_limit",
    "closed": "joint_at_closed_limit",
    "drawer_closed": "joint_at_closed_limit",
    "door_closed": "joint_at_closed_limit",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument(
        "--table-sdf",
        type=Path,
        default=None,
        help="optional official support SDF; when absent, use fixed ground plane",
    )
    parser.add_argument("--host-sdf", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-center-xy", type=float, nargs=2, default=(0.0, 0.01))
    parser.add_argument(
        "--robot-base-xy",
        type=float,
        nargs=2,
        default=None,
        help="explicit robot base world XY; omitted uses the registered offset from the host",
    )
    parser.add_argument("--operator", choices=("pull", "press", "close"), default="pull")
    parser.add_argument(
        "--reference-joint-actuation",
        action="store_true",
        help="opt-in reference-video actuator mode; not a DROID contact-success certificate",
    )
    parser.add_argument(
        "--gripper-geometry-scale",
        type=float,
        default=1.0,
        help="PyBullet-only Robotiq geometry scale; Panda arm dimensions are unchanged",
    )
    parser.add_argument(
        "--handle-collision-scale",
        type=float,
        default=1.0,
        help="PyBullet-only scale for handle-bearing collision links",
    )
    parser.add_argument("--contact-force-scale", type=float, default=1.0)
    parser.add_argument(
        "--grasp-standoff",
        type=float,
        default=0.010,
        help="plan the grasp this far in front of the certified contact target",
    )
    parser.add_argument(
        "--actuate-steps",
        type=int,
        default=None,
        help="actuation action count for a full travel sweep (slower pull)",
    )
    parser.add_argument(
        "--grasp-lead",
        type=float,
        default=0.050,
        help="maximum angular lead of the commanded grasp over the measured joint",
    )
    parser.add_argument("--open-fraction-required", type=float, default=0.90, help="Explicit reference joint-travel acceptance fraction; recorded with measured evidence")
    parser.add_argument("--arm-position-gain", type=float, default=0.1)
    parser.add_argument("--arm-force-scale", type=float, default=1.0)
    parser.add_argument(
        "--arm-velocity-feedforward",
        type=float,
        default=2.0,
        help="feed-forward gain applied to the planned joint velocity during contact actuation",
    )
    parser.add_argument(
        "--actuation-hold-frames",
        type=int,
        default=2,
        help="hold the reachable grasp for the final actuation actions",
    )
    parser.add_argument(
        "--release-settle-velocity",
        type=float,
        default=0.01,
        help="hold the closed grasp until the articulation is below this speed before releasing",
    )
    parser.add_argument("--actuation-standoff", type=float, default=None)
    parser.add_argument("--contact-pull-force", type=float, default=0.0)
    parser.add_argument("--asset-origin", choices=("scenecode", "deterministic_fixture"), default="scenecode")
    parser.add_argument("--robot-base-z", type=float, default=0.0)
    parser.add_argument("--robot-base-yaw", type=float, default=None)
    parser.add_argument("--preserve-host-orientation", action="store_true")
    parser.add_argument(
        "--success-predicate",
        help="task predicate or its canonical joint-limit evaluator",
        default="joint_at_open_limit",
    )
    args = parser.parse_args()
    try:
        success_predicate = _SUCCESS_PREDICATE_ALIASES[args.success_predicate]
    except KeyError as exc:
        parser.error(
            "unsupported articulated success predicate; expected a known task "
            "predicate or joint_at_open_limit/joint_at_closed_limit"
        )
    binding = ArticulationBinding.from_json(args.binding)
    result = run_droid_articulated(
        ArticulatedSceneInputs(
            table_sdf=args.table_sdf,
            host_sdf=args.host_sdf,
            panda_urdf=args.panda_urdf,
            output_dir=args.output_dir,
            binding=binding,
            host_center_xy=tuple(args.host_center_xy),
            robot_base_xy=tuple(args.robot_base_xy) if args.robot_base_xy is not None else None,
            operator=args.operator,
            success_predicate=success_predicate,
            open_fraction_required=args.open_fraction_required,
            reference_joint_actuation=args.reference_joint_actuation,
            gripper_geometry_scale=args.gripper_geometry_scale,
            handle_collision_scale=args.handle_collision_scale,
            contact_force_scale=args.contact_force_scale,
            grasp_standoff_m=args.grasp_standoff,
            actuation_standoff_m=args.actuation_standoff,
            actuate_steps=args.actuate_steps,
            grasp_lead_rad=args.grasp_lead,
            arm_position_gain=args.arm_position_gain,
            arm_force_scale=args.arm_force_scale,
            arm_velocity_feedforward=args.arm_velocity_feedforward,
            actuation_hold_frames=args.actuation_hold_frames,
            release_settle_velocity_rad_s=args.release_settle_velocity,
            contact_pull_force=args.contact_pull_force,
            asset_origin=args.asset_origin,
            robot_base_z=args.robot_base_z,
            robot_base_yaw=args.robot_base_yaw,
            preserve_host_orientation=args.preserve_host_orientation,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
