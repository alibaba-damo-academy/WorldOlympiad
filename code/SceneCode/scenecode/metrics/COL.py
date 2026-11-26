"""Compute COL (collision rate) for SceneCode output scenes.

COL is the fraction of evaluated scene objects that are involved in at least one
collision with penetration depth exceeding 1mm.
"""

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


CODE_SCENE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_SCENE_ROOT))

from pydrake.all import DiagramBuilder, QueryObject

from scenecode.agent_utils.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenecode.agent_utils.house import HouseScene
from scenecode.agent_utils.physics_validation import (
    _compute_floor_penetration_depth,
    _get_object_info_from_geometry_id,
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
class CollisionDetail:
    object_a_id: str
    object_a_name: str
    object_b_id: str
    object_b_name: str
    penetration_m: float


@dataclass
class SceneCOLResult:
    scene_id: str
    num_evaluated: int
    num_positive: int
    rate: float
    status: str
    error: str = ""
    warning: str = ""
    colliding_object_ids: list[str] = field(default_factory=list)
    collision_pairs: list[CollisionDetail] = field(default_factory=list)
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
    """Discover scene directories containing final combined house states."""
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


def compute_scene_collision_rate(
    scene: RoomScene,
    penetration_threshold_m: float = 0.001,
    exclude_room_geometry: bool = False,
) -> tuple[set[str], list[CollisionDetail], int, list[dict[str, str]]]:
    """Compute colliding object IDs and collision details for one room."""
    evaluated = _evaluated_objects(scene)
    skipped = _skipped_objects(scene)
    if not evaluated:
        return set(), [], 0, skipped

    builder = DiagramBuilder()
    _, scene_graph = create_drake_plant_and_scene_graph_from_scene(
        scene=scene,
        builder=builder,
        exclude_room_geometry=exclude_room_geometry,
        weld_furniture=False,
        free_mounted_objects_for_collision=True,
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.0)
    colliding_ids: set[str] = set()
    collision_details: list[CollisionDetail] = []
    seen_pairs: set[tuple[str, str]] = set()

    for pair in pairs:
        if pair.distance >= -penetration_threshold_m:
            continue

        object_a_info = _get_object_info_from_geometry_id(pair.id_A, scene, query_object)
        object_b_info = _get_object_info_from_geometry_id(pair.id_B, scene, query_object)
        object_a_id = object_a_info["id"]
        object_b_id = object_b_info["id"]
        if object_a_id == object_b_id:
            continue

        penetration_m = abs(float(pair.distance))
        if object_a_info["name"] == "floor" or object_b_info["name"] == "floor":
            floor_penetration = _compute_floor_penetration_depth(
                scene=scene,
                object_a_info=object_a_info,
                object_b_info=object_b_info,
            )
            if floor_penetration is not None:
                penetration_m = floor_penetration
            if penetration_m <= penetration_threshold_m:
                continue

        if object_a_id not in evaluated and object_b_id not in evaluated:
            continue

        pair_key = tuple(sorted((object_a_id, object_b_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        if object_a_id in evaluated:
            colliding_ids.add(object_a_id)
        if object_b_id in evaluated:
            colliding_ids.add(object_b_id)

        collision_details.append(
            CollisionDetail(
                object_a_id=object_a_id,
                object_a_name=object_a_info["name"],
                object_b_id=object_b_id,
                object_b_name=object_b_info["name"],
                penetration_m=penetration_m,
            )
        )

    collision_details.sort(key=lambda c: c.penetration_m, reverse=True)
    return colliding_ids, collision_details, len(evaluated), skipped


def evaluate_scene(scene_dir: Path, penetration_threshold_m: float) -> SceneCOLResult:
    start_time = time.time()
    scene_id = scene_dir.name
    try:
        house = load_house_scene(scene_dir)
        all_colliding_ids: set[str] = set()
        all_pairs: list[CollisionDetail] = []
        all_skipped: list[dict[str, str]] = []
        num_evaluated = 0
        warnings: list[str] = []

        for room_id, room in house.rooms.items():
            try:
                colliding_ids, pairs, room_evaluated, skipped = (
                    compute_scene_collision_rate(
                        room, penetration_threshold_m=penetration_threshold_m
                    )
                )
            except RuntimeError as exc:
                warnings.append(
                    f"{room_id}: full-scene query failed; retried without "
                    f"room geometry ({exc})"
                )
                colliding_ids, pairs, room_evaluated, skipped = (
                    compute_scene_collision_rate(
                        room,
                        penetration_threshold_m=penetration_threshold_m,
                        exclude_room_geometry=True,
                    )
                )
            num_evaluated += room_evaluated
            all_colliding_ids.update(f"{room_id}:{obj_id}" for obj_id in colliding_ids)
            all_pairs.extend(pairs)
            for item in skipped:
                item = dict(item)
                item["room_id"] = room_id
                all_skipped.append(item)

        num_positive = len(all_colliding_ids)
        rate = num_positive / num_evaluated if num_evaluated else 0.0
        return SceneCOLResult(
            scene_id=scene_id,
            num_evaluated=num_evaluated,
            num_positive=num_positive,
            rate=rate,
            status="ok",
            warning=" | ".join(warnings),
            colliding_object_ids=sorted(all_colliding_ids),
            collision_pairs=all_pairs,
            skipped_objects=all_skipped,
            elapsed_s=time.time() - start_time,
        )
    except Exception as exc:
        LOGGER.exception("Failed to evaluate COL for %s", scene_id)
        return SceneCOLResult(
            scene_id=scene_id,
            num_evaluated=0,
            num_positive=0,
            rate=0.0,
            status="error",
            error=str(exc),
            elapsed_s=time.time() - start_time,
        )


def _summary(results: list[SceneCOLResult]) -> dict[str, Any]:
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


def _write_outputs(
    results: list[SceneCOLResult], output_dir: Path, threshold_m: float
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "COL_results.csv"
    json_path = output_dir / "COL_results.json"

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
                "warning",
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
                    "warning": result.warning,
                }
            )

    payload = {
        "metric": "COL",
        "definition": (
            "fraction of evaluated objects involved in at least one collision "
            "with penetration depth exceeding threshold_m"
        ),
        "threshold_m": threshold_m,
        "summary": _summary(results),
        "scenes": [asdict(r) for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute COL for SceneCode outputs.")
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold-m", type=float, default=0.001)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    scene_dirs = discover_scene_dirs(args.outputs)
    output_dir = args.output_dir or (args.outputs / "metrics")
    results = [evaluate_scene(p, args.threshold_m) for p in scene_dirs]
    _write_outputs(results, output_dir, threshold_m=args.threshold_m)

    summary = _summary(results)
    print("COL summary")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir / 'COL_results.csv'}")
    print(f"Wrote {output_dir / 'COL_results.json'}")
    return 0 if summary["num_error_scenes"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
