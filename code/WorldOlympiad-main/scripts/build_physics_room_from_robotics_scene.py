#!/usr/bin/env python3
"""Build the physics reference world on an already certified SceneCode room.

The physics suite originally generated its own SceneCode room. That generator is
slow and rate limited, while the physics tasks depend only on a support surface
and three declared manipulands. This builder therefore reuses a completed
SceneCode robotics room as the static background, binds its table as the
support, and adds the red sphere, blue cube and yellow tall block as procedural
single-piece bodies with the declared sizes.
"""
import argparse, json, sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--house-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--table-token", default="table")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    repository = Path(args.repository).resolve()
    house = Path(args.house_state).resolve()
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repository))
    from autoresearch.models import WorldIR
    from autoresearch.physical_apparatus import solid
    from autoresearch.scenecode_importer import import_house_state

    world = import_house_state(house, scene_id="physical_reference_generated")
    labels = {
        body.name: " ".join(
            [body.name, body.semantic_category, str(body.metadata.get("description", ""))]
        ).lower()
        for body in world.bodies
    }
    tables = [name for name, label in labels.items() if args.table_token in label]
    if len(tables) != 1:
        raise SystemExit(f"expected exactly one support surface, found {tables}")
    table = tables[0]

    manipulands = [
        solid("ball", color=(0.8, 0.08, 0.045, 1), radius=0.06),
        solid("cube", color=(0.06, 0.24, 0.7, 1), size=(0.12, 0.12, 0.12)),
        solid("tower", color=(0.95, 0.65, 0.03, 1), size=(0.10, 0.10, 0.24)),
    ]
    base = next(body.pose for body in world.bodies if body.name == table)
    for index, body in enumerate(manipulands):
        body.pose = type(body.pose)(
            xyz=(base.xyz[0], base.xyz[1], base.xyz[2] + 1.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        world.bodies.append(body)

    world_path = destination / "world_ir.json"
    WorldIR.model_validate(world.model_dump()).write_json(world_path)
    bindings = {
        "table": {"body": table, "grounding": "wooden table"},
        "ball": {"body": "ball", "grounding": "red sphere"},
        "cube": {"body": "cube", "grounding": "blue cube"},
        "tower": {"body": "tower", "grounding": "yellow tall block"},
    }
    (destination / "bindings.json").write_text(json.dumps(bindings, indent=2) + "\n")
    print(json.dumps({
        "house_state": str(house),
        "bodies": len(world.bodies),
        "static_bodies": sum(body.static for body in world.bodies),
        "table": table,
        "world_ir": str(world_path),
        "bindings": str(destination / "bindings.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
