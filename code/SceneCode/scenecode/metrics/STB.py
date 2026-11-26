"""Compute STB (static equilibrium rate) for SceneCode output scenes."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CODE_SCENE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_SCENE_ROOT))

from pydrake.all import DiagramBuilder, RigidTransform, Simulator

from scenecode.agent_utils.house import HouseScene
from scenecode.agent_utils.physical_feasibility import (
    _create_drake_plant_for_ik,
    _effective_to_scene_transform,
    _floating_positions_to_transform,
)
from scenecode.agent_utils.room import ObjectType, RoomScene, UniqueID


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUTS_DIR = Path(
    "outputs"
)
EVALUATED_TYPES = {
    ObjectType.FURNITURE,
    ObjectType.MANIPULAND,
    ObjectType.WALL_MOUNTED,
    ObjectType.CEILING_MOUNTED,
}


@dataclass
class ObjectSTBDetail:
    object_id: str
    name: str
    object_type: str
    support_type: str
    stable: bool
    displacement_m: float
    rotation_delta_rad: float
    welded: bool


@dataclass
class SceneSTBResult:
    scene_id: str
    num_evaluated: int
    num_positive: int
    rate: float
    status: str
    error: str = ""
    objects: list[ObjectSTBDetail] = field(default_factory=list)
    skipped_objects: list[dict[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0


def _load_subset_order(outputs_dir: Path) -> list[str] | None:
    subset_path = outputs_dir.parent / "subset_30.csv"
    if not subset_path.exists():
        return None

    scene_ids: list[str] = []
    with open(subset_path, newline="") as f:
        reader = csv.DictReader(f)
        if "ID" not in (reader.fieldnames or []):
            return None
        for row in reader:
            raw_id = row.get("ID", "").strip()
            if raw_id:
                scene_ids.append(f"scene_{int(raw_id):03d}")
    return scene_ids


def discover_scene_dirs(outputs_dir: Path) -> list[Path]:
    scene_dirs = [
        p
        for p in outputs_dir.glob("scene_*")
        if p.is_dir() and (p / "combined_house" / "house_state.json").exists()
    ]
    subset_order = _load_subset_order(outputs_dir)
    if subset_order is None:
        return sorted(scene_dirs, key=lambda p: p.name)

    by_name = {p.name: p for p in scene_dirs}
    ordered = [by_name[name] for name in subset_order if name in by_name]
    extras = sorted(
        [p for p in scene_dirs if p.name not in set(subset_order)], key=lambda p: p.name
    )
    missing = [name for name in subset_order if name not in by_name]
    if missing:
        LOGGER.warning("subset_30.csv lists missing scenes: %s", ", ".join(missing))
    return ordered + extras


def load_house_scene(scene_dir: Path) -> HouseScene:
    state_path = scene_dir / "combined_house" / "house_state.json"
    with open(state_path) as f:
        state = json.load(f)
    return HouseScene.from_state_dict(state, house_dir=scene_dir)


def load_support_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path) as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "labels" in payload and isinstance(
        payload["labels"], dict
    ):
        payload = payload["labels"]
    if not isinstance(payload, dict):
        raise ValueError("support labels JSON must be an object_id -> support_type map")
    return {str(k): str(v) for k, v in payload.items()}


def infer_support_type(obj: Any) -> str:
    if obj.object_type == ObjectType.WALL_MOUNTED:
        return "wall"
    if obj.object_type == ObjectType.CEILING_MOUNTED:
        return "ceiling"
    if obj.object_type == ObjectType.MANIPULAND and obj.placement_info is not None:
        return "object-supported"
    if obj.object_type == ObjectType.FURNITURE:
        return "ground"
    return "unknown"


def _evaluated_objects(scene: RoomScene) -> dict[str, Any]:
    evaluated: dict[str, Any] = {}
    for obj_id, obj in scene.objects.items():
        if obj.object_type not in EVALUATED_TYPES:
            continue
        if obj.sdf_path is None or not obj.sdf_path.exists():
            continue
        evaluated[str(obj_id)] = obj
    return evaluated


def _skipped_objects(scene: RoomScene) -> list[dict[str, str]]:
    skipped: list[dict[str, str]] = []
    for obj_id, obj in scene.objects.items():
        if obj.object_type not in EVALUATED_TYPES:
            continue
        if obj.sdf_path is None:
            skipped.append(
                {"object_id": str(obj_id), "name": obj.name, "reason": "missing_sdf_path"}
            )
        elif not obj.sdf_path.exists():
            skipped.append(
                {
                    "object_id": str(obj_id),
                    "name": obj.name,
                    "reason": f"sdf_not_found:{obj.sdf_path}",
                }
            )
    return skipped


def rotation_delta_rad(a: RigidTransform, b: RigidTransform) -> float:
    delta = b.rotation().multiply(a.rotation().inverse())
    return float(abs(delta.ToAngleAxis().angle()))


def _final_transform_for_object(
    obj: Any,
    plant: Any,
    plant_context: Any,
    object_indices: dict[UniqueID, tuple[Any, Any]],
) -> tuple[RigidTransform, bool]:
    indices = object_indices.get(obj.object_id)
    if indices is None:
        return obj.transform, True

    model_idx, body_idx = indices
    body = plant.get_body(body_idx)
    if not body.is_floating():
        return obj.transform, True

    positions = plant.GetPositions(plant_context, model_idx)
    if len(positions) < 7:
        return obj.transform, True

    effective_transform = _floating_positions_to_transform(positions)
    return _effective_to_scene_transform(effective_transform, obj.internal_model_pose), False


def compute_scene_static_equilibrium(
    scene: RoomScene,
    support_labels: dict[str, str],
    simulation_time_s: float = 5.0,
    time_step_s: float = 0.001,
    displacement_threshold_m: float = 0.01,
    rotation_threshold_rad: float = 0.1,
) -> tuple[list[ObjectSTBDetail], list[dict[str, str]]]:
    evaluated = _evaluated_objects(scene)
    skipped = _skipped_objects(scene)
    if not evaluated:
        return [], skipped

    support_by_id = {
        object_id: support_labels.get(object_id, infer_support_type(obj))
        for object_id, obj in evaluated.items()
    }
    free_object_ids = [
        UniqueID(object_id)
        for object_id, support_type in support_by_id.items()
        if support_type in {"ground", "object-supported"}
    ]

    initial_transforms = {
        object_id: obj.transform for object_id, obj in evaluated.items()
    }

    if free_object_ids:
        builder = DiagramBuilder()
        plant, _, object_indices, _ = _create_drake_plant_for_ik(
            scene=scene,
            builder=builder,
            weld_furniture=True,
            time_step=time_step_s,
            free_objects=free_object_ids,
        )
        diagram = builder.Build()
        simulator = Simulator(diagram)
        simulator.AdvanceTo(simulation_time_s)
        root_context = simulator.get_context()
        plant_context = plant.GetMyContextFromRoot(root_context)
    else:
        plant = None
        plant_context = None
        object_indices = {}

    details: list[ObjectSTBDetail] = []
    for object_id, obj in evaluated.items():
        initial = initial_transforms[object_id]
        if plant is None or plant_context is None:
            final = initial
            welded = True
        else:
            final, welded = _final_transform_for_object(
                obj=obj,
                plant=plant,
                plant_context=plant_context,
                object_indices=object_indices,
            )

        displacement_m = float(
            np.linalg.norm(final.translation() - initial.translation())
        )
        rotation_moved = rotation_delta_rad(initial, final)
        stable = (
            displacement_m < displacement_threshold_m
            and rotation_moved < rotation_threshold_rad
        )
        details.append(
            ObjectSTBDetail(
                object_id=object_id,
                name=obj.name,
                object_type=obj.object_type.value,
                support_type=support_by_id[object_id],
                stable=stable,
                displacement_m=displacement_m,
                rotation_delta_rad=rotation_moved,
                welded=welded,
            )
        )

    return details, skipped


def evaluate_scene(
    scene_dir: Path,
    support_labels: dict[str, str],
    simulation_time_s: float,
    time_step_s: float,
    displacement_threshold_m: float,
    rotation_threshold_rad: float,
) -> SceneSTBResult:
    start_time = time.time()
    scene_id = scene_dir.name
    try:
        house = load_house_scene(scene_dir)
        all_details: list[ObjectSTBDetail] = []
        all_skipped: list[dict[str, str]] = []

        for room_id, room in house.rooms.items():
            room_details, skipped = compute_scene_static_equilibrium(
                scene=room,
                support_labels=support_labels,
                simulation_time_s=simulation_time_s,
                time_step_s=time_step_s,
                displacement_threshold_m=displacement_threshold_m,
                rotation_threshold_rad=rotation_threshold_rad,
            )
            for detail in room_details:
                detail.object_id = f"{room_id}:{detail.object_id}"
            all_details.extend(room_details)
            for item in skipped:
                item = dict(item)
                item["room_id"] = room_id
                all_skipped.append(item)

        num_evaluated = len(all_details)
        num_positive = sum(1 for item in all_details if item.stable)
        rate = num_positive / num_evaluated if num_evaluated else 0.0
        return SceneSTBResult(
            scene_id=scene_id,
            num_evaluated=num_evaluated,
            num_positive=num_positive,
            rate=rate,
            status="ok",
            objects=all_details,
            skipped_objects=all_skipped,
            elapsed_s=time.time() - start_time,
        )
    except Exception as exc:
        LOGGER.exception("Failed to evaluate STB for %s", scene_id)
        return SceneSTBResult(
            scene_id=scene_id,
            num_evaluated=0,
            num_positive=0,
            rate=0.0,
            status="error",
            error=str(exc),
            elapsed_s=time.time() - start_time,
        )


def _summary(results: list[SceneSTBResult]) -> dict[str, Any]:
    ok_results = [r for r in results if r.status == "ok"]
    total_evaluated = sum(r.num_evaluated for r in ok_results)
    total_positive = sum(r.num_positive for r in ok_results)
    return {
        "num_scenes": len(results),
        "num_ok_scenes": len(ok_results),
        "num_error_scenes": len(results) - len(ok_results),
        "total_evaluated_objects": total_evaluated,
        "total_positive_objects": total_positive,
        "macro_average": (
            sum(r.rate for r in ok_results) / len(ok_results) if ok_results else 0.0
        ),
        "overall_rate": total_positive / total_evaluated if total_evaluated else 0.0,
        "failed_scenes": [r.scene_id for r in results if r.status != "ok"],
    }


def _write_outputs(results: list[SceneSTBResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "STB_results.csv"
    json_path = output_dir / "STB_results.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "num_evaluated",
                "num_positive",
                "rate",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scene_id": result.scene_id,
                    "num_evaluated": result.num_evaluated,
                    "num_positive": result.num_positive,
                    "rate": result.rate,
                    "status": result.status,
                    "error": result.error,
                }
            )

    payload = {
        "metric": "STB",
        "definition": (
            "fraction of evaluated objects that remain within displacement and "
            "rotation thresholds after gravity simulation"
        ),
        "summary": _summary(results),
        "scenes": [asdict(r) for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute STB for SceneCode outputs.")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--support-labels", type=Path, default=None)
    parser.add_argument("--simulation-time-s", type=float, default=5.0)
    parser.add_argument("--time-step-s", type=float, default=0.001)
    parser.add_argument("--displacement-threshold-m", type=float, default=0.01)
    parser.add_argument("--rotation-threshold-rad", type=float, default=0.1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    support_labels = load_support_labels(args.support_labels)
    scene_dirs = discover_scene_dirs(args.outputs)
    output_dir = args.output_dir or (args.outputs / "metrics")
    results = [
        evaluate_scene(
            scene_dir=p,
            support_labels=support_labels,
            simulation_time_s=args.simulation_time_s,
            time_step_s=args.time_step_s,
            displacement_threshold_m=args.displacement_threshold_m,
            rotation_threshold_rad=args.rotation_threshold_rad,
        )
        for p in scene_dirs
    ]
    _write_outputs(results, output_dir)

    summary = _summary(results)
    print("STB summary")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir / 'STB_results.csv'}")
    print(f"Wrote {output_dir / 'STB_results.json'}")
    return 0 if summary["num_error_scenes"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
