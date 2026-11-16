"""USD structural diagnostics used by the USD export integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneDiagnostics:
    scene: str
    nested: int
    joints: int
    invalid_no_rb: int
    ancestor_desc: int
    neg_prismatic: int
    collision_active: int
    collision_total: int
    visual_collision: int
    visual_total: int


def _is_active(prim: Any) -> bool:
    current = prim
    while current and current.IsValid():
        if not current.IsActive():
            return False
        current = current.GetParent()
    return True


def _find_stage_path(usd_dir: Path) -> Path:
    if usd_dir.is_file():
        return usd_dir

    top_level = sorted(usd_dir.glob("scene_*.usda"))
    if top_level:
        return top_level[0]

    generic_top_level = sorted(
        path for path in usd_dir.glob("*.usda") if path.name != "scene_for_usd.xml"
    )
    if generic_top_level:
        return generic_top_level[0]

    physics_path = usd_dir / "Payload" / "Physics.usda"
    if physics_path.exists():
        return physics_path

    raise FileNotFoundError(f"No scene_*.usda or Payload/Physics.usda in {usd_dir}")


def collect_scene_diagnostics(usd_dir: Path) -> SceneDiagnostics:
    """Collect structural diagnostics for a single scene USD directory."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(_find_stage_path(usd_dir)))
    if not stage:
        raise RuntimeError(f"Could not open USD stage for {usd_dir}")

    rigid_paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    rigid_set = set(rigid_paths)

    nested = 0
    for path in rigid_paths:
        ancestor = path.GetParentPath()
        while ancestor and ancestor != ancestor.GetParentPath():
            if ancestor in rigid_set:
                nested += 1
            ancestor = ancestor.GetParentPath()

    joints = 0
    invalid_no_rb = 0
    ancestor_desc = 0
    neg_prismatic = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue

        joints += 1
        joint = UsdPhysics.Joint(prim)
        body_targets = []
        for rel in [joint.GetBody0Rel(), joint.GetBody1Rel()]:
            if not rel:
                continue
            targets = rel.GetTargets()
            if targets:
                body_targets.append(targets[0])

        for target in body_targets:
            target_prim = stage.GetPrimAtPath(target)
            if (
                target_prim
                and target_prim.IsValid()
                and not target_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                invalid_no_rb += 1

        if len(body_targets) == 2:
            path_a, path_b = body_targets
            if str(path_a).startswith(str(path_b) + "/") or str(path_b).startswith(
                str(path_a) + "/"
            ):
                ancestor_desc += 1

        if prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic = UsdPhysics.PrismaticJoint(prim)
            lower = prismatic.GetLowerLimitAttr().Get()
            if lower is not None and lower < 0:
                neg_prismatic += 1

    collision_active = 0
    collision_total = 0
    visual_collision = 0
    visual_total = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        name = prim.GetName().lower()
        if "collision" in name:
            collision_total += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI) and _is_active(prim):
                collision_active += 1
        if "visual" in name:
            visual_total += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI) and _is_active(prim):
                visual_collision += 1

    return SceneDiagnostics(
        scene=usd_dir.name,
        nested=nested,
        joints=joints,
        invalid_no_rb=invalid_no_rb,
        ancestor_desc=ancestor_desc,
        neg_prismatic=neg_prismatic,
        collision_active=collision_active,
        collision_total=collision_total,
        visual_collision=visual_collision,
        visual_total=visual_total,
    )
