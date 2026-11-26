"""Command-line entry point for the minimal SDF bridge."""

from __future__ import annotations

import argparse
import json

from .sdf import prepare_physics_sdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_sdf")
    parser.add_argument("destination_sdf")
    args = parser.parse_args()
    artifact = prepare_physics_sdf(args.source_sdf, args.destination_sdf)
    print(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

