"""Task-bound, digest-verified assets for native task executors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .io import sha256_file as sha


@dataclass(frozen=True)
class TaskAssets:
    """Only assets declared by one selected generation and its inventory."""

    sdf_by_object_id: dict[str, str]
    robot_urdf: str
    generation_sha256: str
    inventory_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_task_assets(*, workspace: str | Path, generation: str | Path) -> TaskAssets:
    """Resolve assets from one immutable generation; never search by task kind."""
    workspace = Path(workspace).resolve()
    generation = Path(generation).resolve()
    outputs = workspace / "outputs"
    generation.relative_to(outputs / "robotics_scenecode_generations")
    generation_payload = _read(generation)
    inventory_path = generation.parent / "task_inventory_certificate.json"
    inventory = _read(inventory_path)
    if (
        generation_payload.get("passed") is not True
        or inventory.get("passed") is not True
        or generation_payload.get("task_inventory_certificate_sha256") != sha(inventory_path)
    ):
        raise ValueError("selected generation inventory is absent or stale")

    sdf_by_object_id: dict[str, str] = {}
    for row in inventory.get("matches", []):
        object_id = str(row["task_object_id"])
        if object_id in sdf_by_object_id:
            raise ValueError("selected inventory contains duplicate task object ids")
        path = Path(row["sdf_path"]).resolve()
        path.relative_to(generation.parent)
        if not path.is_file() or sha(path) != row["sdf_sha256"]:
            raise ValueError("selected inventory asset changed: " + object_id)
        sdf_by_object_id[object_id] = str(path)
    if not sdf_by_object_id:
        raise ValueError("selected task has no resolvable object assets")

    robots: list[Path] = []
    for manifest_path in sorted((outputs / "robot_assets").glob("*/robot_asset_manifest.json")):
        manifest = _read(manifest_path)
        urdf = Path(manifest.get("generated_urdf", "")).resolve()
        if (
            manifest.get("robot_embodiment") == "robolab_droid"
            and urdf.is_file()
            and sha(urdf) == manifest.get("generated_urdf_sha256")
        ):
            robots.append(urdf)
    if len(robots) != 1:
        raise ValueError("registered DROID robot asset must resolve uniquely")
    return TaskAssets(
        sdf_by_object_id=sdf_by_object_id,
        robot_urdf=str(robots[0]),
        generation_sha256=sha(generation),
        inventory_sha256=sha(inventory_path),
    )
