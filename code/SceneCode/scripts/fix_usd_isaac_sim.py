"""Fix USD physics for Isaac Sim compatibility.

The mujoco-usd-converter (v0.1.0a3) generates PhysicsFixedJoint prims that
connect objects to the root Xform, but the root has no PhysicsRigidBodyAPI.
PhysX requires valid physics bodies on both sides of a joint, so the
constraint solver pulls everything to (0,0,0).

This script post-processes Physics.usda files to fix three object categories:

1. **Static objects** (walls, desks, beds): Remove all physics body APIs and
   joints, leaving only collision geometry. Isaac Sim treats these as static
   colliders.

2. **Dynamic objects** (mugs, books): Flatten nested rigid bodies by moving
   MassAPI from base_link to wrapper, removing inner RigidBodyAPI, and
   deleting the internal FixedJoint.

3. **Articulated objects** (wardrobes with doors, fridges): Promote invalid
   base-body Xforms to real rigid bodies, reparent articulated links as
   siblings when needed, preserve authored collision geometry by default, and
   recreate self-collision filters (mirroring MuJoCo's ``<contact><exclude>``
   pairs). Optionally, articulated collision can be regenerated from visual
   meshes using Isaac-compatible mesh approximations.

Usage:
    # Fix single scene USD directory.
    python scripts/fix_usd_isaac_sim.py /path/to/scene/mujoco/usd

    # Fix all scenes recursively with parallel workers.
    python scripts/fix_usd_isaac_sim.py /path/to/SceneAgent_Cleaned \\
        --recursive --workers 16
"""

import argparse
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

console_logger = logging.getLogger(__name__)


def remove_rigid_body_api(prim: Usd.Prim) -> bool:
    """Remove PhysicsRigidBodyAPI from a prim if present."""
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        return True
    return False


def remove_mass_api(prim: Usd.Prim) -> None:
    """Remove PhysicsMassAPI and all mass properties from a prim."""
    if not prim.HasAPI(UsdPhysics.MassAPI):
        return
    prim.RemoveAPI(UsdPhysics.MassAPI)
    for prop_name in [
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ]:
        prop = prim.GetProperty(prop_name)
        if prop:
            prim.RemoveProperty(prop_name)


def copy_mass_to_prim(source: Usd.Prim, target: Usd.Prim) -> None:
    """Copy PhysicsMassAPI and its properties from source to target prim."""
    if not source.HasAPI(UsdPhysics.MassAPI):
        return
    UsdPhysics.MassAPI.Apply(target)

    mass_props = [
        ("physics:mass", "float"),
        ("physics:centerOfMass", "point3f"),
        ("physics:diagonalInertia", "float3"),
        ("physics:principalAxes", "quatf"),
    ]
    for prop_name, _ in mass_props:
        src_attr = source.GetAttribute(prop_name)
        if src_attr and src_attr.HasValue():
            tgt_attr = target.GetAttribute(prop_name)
            if not tgt_attr:
                # Create with same type as source.
                tgt_attr = target.CreateAttribute(prop_name, src_attr.GetTypeName())
            tgt_attr.Set(src_attr.Get())


def find_fixed_joints_with_body0(
    root_prim: Usd.Prim, body0_path: Sdf.Path
) -> list[Sdf.Path]:
    """Find all PhysicsFixedJoint descendants whose body0 targets body0_path."""
    joint_paths = []
    for descendant in Usd.PrimRange(root_prim):
        if descendant.GetTypeName() == "PhysicsFixedJoint":
            body0_rel = descendant.GetRelationship("physics:body0")
            if body0_rel:
                targets = body0_rel.GetTargets()
                if targets and targets[0] == body0_path:
                    joint_paths.append(descendant.GetPath())
    return joint_paths


def delete_prims(stage: Usd.Stage, paths: list[Sdf.Path]) -> int:
    """Delete prims at the given paths. Returns count of deleted prims."""
    count = 0
    for path in paths:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
            count += 1
    return count


def _reparent_rigid_body_children(
    stage: Usd.Stage,
    source_parent_prim: Usd.Prim,
    new_parent_prim: Usd.Prim,
    all_layers: list[Sdf.Layer],
) -> int:
    """Move rigid body children from one parent to another across all layers."""
    source_path = source_parent_prim.GetPath()
    new_parent_path = new_parent_prim.GetPath()

    prims_to_move: list[Sdf.Path] = []
    for child in source_parent_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            prims_to_move.append(child.GetPath())

    if not prims_to_move:
        return 0

    path_mapping: dict[str, str] = {}
    for old_path in prims_to_move:
        new_path = new_parent_path.AppendChild(old_path.name)
        path_mapping[str(old_path)] = str(new_path)

    for layer in all_layers:
        edit = Sdf.BatchNamespaceEdit()
        has_edits = False
        for old_path in prims_to_move:
            if layer.GetPrimAtPath(old_path):
                new_path = new_parent_path.AppendChild(old_path.name)
                edit.Add(old_path, new_path)
                has_edits = True
        if has_edits:
            if not layer.Apply(edit):
                console_logger.warning(
                    f"Failed to reparent in layer {layer.identifier}"
                )

    for descendant in Usd.PrimRange(new_parent_prim):
        for rel in descendant.GetRelationships():
            targets = rel.GetTargets()
            new_targets = []
            changed = False
            for target in targets:
                target_str = str(target)
                for old_str, new_str in path_mapping.items():
                    if target_str == old_str or target_str.startswith(old_str + "/"):
                        target_str = new_str + target_str[len(old_str) :]
                        changed = True
                        break
                new_targets.append(Sdf.Path(target_str))
            if changed:
                rel.SetTargets(new_targets)

    console_logger.debug(
        f"  {new_parent_path.name}: reparented {len(prims_to_move)} rigid "
        f"children from {source_path.name}"
    )
    return len(prims_to_move)


def _has_articulated_joint_descendants(wrapper_prim: Usd.Prim) -> bool:
    """Return True if the wrapper contains a non-fixed physics joint."""
    for descendant in Usd.PrimRange(wrapper_prim):
        if (
            descendant.IsA(UsdPhysics.Joint)
            and descendant.GetTypeName() != "PhysicsFixedJoint"
        ):
            return True
    return False


def _collect_invalid_direct_joint_targets(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> list[Usd.Prim]:
    """Collect invalid direct child body0 targets that should be base rigid bodies."""
    wrapper_path = wrapper_prim.GetPath()
    targets: dict[str, Usd.Prim] = {}

    for descendant in Usd.PrimRange(wrapper_prim):
        if not descendant.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(descendant)
        rel = joint.GetBody0Rel()
        if not rel:
            continue
        for target in rel.GetTargets():
            if target.GetParentPath() != wrapper_path:
                continue
            target_prim = stage.GetPrimAtPath(target)
            if (
                target_prim
                and target_prim.IsValid()
                and not target_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                targets[str(target)] = target_prim

    return list(targets.values())


def _collect_articulation_base_prims(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> list[Usd.Prim]:
    """Collect direct-child rigid bodies that anchor the articulation."""
    wrapper_path = wrapper_prim.GetPath()
    bases: dict[str, Usd.Prim] = {}

    for descendant in Usd.PrimRange(wrapper_prim):
        if not descendant.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(descendant)
        for rel in [joint.GetBody0Rel(), joint.GetBody1Rel()]:
            if not rel:
                continue
            for target in rel.GetTargets():
                path = target
                while path and path != wrapper_path:
                    prim = stage.GetPrimAtPath(path)
                    if (
                        prim
                        and prim.IsValid()
                        and prim.GetPath().GetParentPath() == wrapper_path
                        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    ):
                        bases[str(path)] = prim
                        break
                    path = path.GetParentPath()

    if not bases:
        for child in wrapper_prim.GetChildren():
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                bases[str(child.GetPath())] = child

    return list(bases.values())


def _promote_base_rigid_body(
    wrapper_prim: Usd.Prim,
    base_prim: Usd.Prim,
) -> None:
    """Move the base rigid-body authoring from the wrapper onto the base prim."""
    if not base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
    if wrapper_prim.HasAPI(UsdPhysics.MassAPI):
        copy_mass_to_prim(source=wrapper_prim, target=base_prim)
        remove_mass_api(wrapper_prim)
    remove_rigid_body_api(wrapper_prim)


def _find_internal_fixed_joints(
    wrapper_prim: Usd.Prim,
    wrapper_path: Sdf.Path,
    base_paths: set[Sdf.Path],
) -> list[Sdf.Path]:
    """Find internal fixed joints that connect the wrapper to a promoted base."""
    joint_paths: list[Sdf.Path] = []
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetTypeName() != "PhysicsFixedJoint":
            continue
        targets: set[Sdf.Path] = set()
        for rel_name in ["physics:body0", "physics:body1"]:
            rel = descendant.GetRelationship(rel_name)
            if rel:
                targets.update(rel.GetTargets())
        if wrapper_path in targets and base_paths.intersection(targets):
            joint_paths.append(descendant.GetPath())
    return joint_paths


def _clear_wrapper_target_from_fixed_joint(
    stage: Usd.Stage,
    joint_path: Sdf.Path,
    wrapper_path: Sdf.Path,
) -> None:
    """Clear whichever side of a fixed joint still targets the wrapper."""
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim:
        return
    for rel_name in ["physics:body0", "physics:body1"]:
        rel = joint_prim.GetRelationship(rel_name)
        if not rel:
            continue
        targets = rel.GetTargets()
        if wrapper_path in targets:
            remaining = [target for target in targets if target != wrapper_path]
            if remaining:
                rel.SetTargets(remaining)
            else:
                rel.ClearTargets(True)


def _find_nearest_rigid_body_ancestor(
    stage: Usd.Stage,
    target_path: Sdf.Path,
    stop_path: Sdf.Path,
) -> Sdf.Path | None:
    """Return the nearest rigid-body ancestor within the wrapper subtree."""
    path = target_path
    while str(path).startswith(str(stop_path)):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return path
        if path == stop_path:
            break
        path = path.GetParentPath()
    return None


def _flatten_nested_articulated_links(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    all_layers: list[Sdf.Layer],
) -> int:
    """Reparent nested articulated rigid bodies so links become siblings."""
    total_moved = 0

    while True:
        moved_this_pass = 0
        wrapper_prim = stage.GetPrimAtPath(wrapper_prim.GetPath())
        for child in list(wrapper_prim.GetChildren()):
            if not child.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            moved_this_pass += _reparent_rigid_body_children(
                stage=stage,
                source_parent_prim=child,
                new_parent_prim=wrapper_prim,
                all_layers=all_layers,
            )
        if not moved_this_pass:
            break
        total_moved += moved_this_pass

    return total_moved


def _retag_articulated_collision_meshes(
    wrapper_prim: Usd.Prim,
    approximation: str,
) -> dict[str, int]:
    """Replace articulated collision meshes with visual-mesh collision."""
    deactivated = 0
    visual_tagged = 0

    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetTypeName() != "Mesh":
            continue

        name_lower = descendant.GetName().lower()
        if "collision" in name_lower:
            if descendant.IsActive():
                descendant.SetActive(False)
                deactivated += 1
            continue

        if "visual" not in name_lower:
            continue

        if not descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(descendant)
        if not descendant.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(descendant)

        mesh_collision = UsdPhysics.MeshCollisionAPI(descendant)
        mesh_collision.GetApproximationAttr().Set(approximation)
        visual_tagged += 1

    return {
        "collision_deactivated": deactivated,
        "visual_tagged": visual_tagged,
    }


def _find_composed_scene_path(physics_usda_path: Path) -> Path | None:
    """Find the top-level composed scene stage for a USD export directory."""
    usd_dir = physics_usda_path.parent.parent
    scene_files = sorted(usd_dir.glob("scene_*.usda"))
    if scene_files:
        return scene_files[0]

    generic_scene_files = sorted(
        path
        for path in usd_dir.glob("*.usda")
        if path.parent == usd_dir and path.name != "scene_for_usd.xml"
    )
    if generic_scene_files:
        return generic_scene_files[0]

    return None


def _collect_articulated_wrapper_paths_from_joints(
    scene_stage: Usd.Stage,
) -> set[Sdf.Path]:
    """Find top-level geometry wrappers that contain non-fixed joints."""
    root_prim = scene_stage.GetDefaultPrim()
    if not root_prim:
        return set()

    geometry_path = root_prim.GetPath().AppendChild("Geometry")
    wrapper_paths: set[Sdf.Path] = set()

    for prim in scene_stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        if prim.GetTypeName() == "PhysicsFixedJoint":
            continue

        path = prim.GetPath()
        while path and path != geometry_path:
            if path.GetParentPath() == geometry_path:
                wrapper_paths.add(path)
                break
            path = path.GetParentPath()

    return wrapper_paths


def _apply_articulated_collision_mode(
    physics_usda_path: Path,
    collision_mode: str,
) -> dict[str, int]:
    """Apply articulated collision regeneration on the composed scene stage."""
    if collision_mode == "preserve":
        return {"wrappers": 0, "collision_deactivated": 0, "visual_tagged": 0}

    if collision_mode == "convex-hull":
        approximation = str(UsdPhysics.Tokens.convexHull)
    elif collision_mode == "convex-decomposition":
        approximation = str(UsdPhysics.Tokens.convexDecomposition)
    else:
        raise ValueError(f"Unsupported articulated collision mode: {collision_mode}")

    scene_path = _find_composed_scene_path(physics_usda_path)
    if scene_path is None:
        console_logger.warning(
            f"Could not find composed scene root for {physics_usda_path}; "
            "skipping articulated collision regeneration"
        )
        return {"wrappers": 0, "collision_deactivated": 0, "visual_tagged": 0}

    scene_stage = Usd.Stage.Open(str(scene_path))
    if not scene_stage:
        raise RuntimeError(f"Could not open composed scene stage: {scene_path}")

    totals = {"wrappers": 0, "collision_deactivated": 0, "visual_tagged": 0}
    for wrapper_path in sorted(
        _collect_articulated_wrapper_paths_from_joints(scene_stage),
        key=str,
    ):
        wrapper_prim = scene_stage.GetPrimAtPath(wrapper_path)
        if not wrapper_prim or not wrapper_prim.IsValid():
            continue
        counts = _retag_articulated_collision_meshes(wrapper_prim, approximation)
        totals["wrappers"] += 1
        totals["collision_deactivated"] += counts["collision_deactivated"]
        totals["visual_tagged"] += counts["visual_tagged"]

    scene_stage.GetRootLayer().Save()
    return totals


def _add_self_collision_filter(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> None:
    """Add self-collision filtering for all rigid bodies in an articulated object.

    The MuJoCo source has ``<contact><exclude>`` pairs that prevent adjacent
    articulated links from colliding (e.g. wardrobe body vs. its doors).
    The mujoco_usd_converter does not convert these (``Tf.Warn("excludes
    are not supported")``), so we recreate them using a PhysicsCollisionGroup
    that includes all rigid bodies within the object and filters against
    itself.

    Without this, PhysX detects collisions between overlapping bodies at
    hinge points, which prevents joints from moving interactively.
    """
    # Collect all rigid body prims under the wrapper.
    rigid_bodies = []
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(descendant.GetPath())

    if len(rigid_bodies) < 2:
        return  # No self-collision possible with fewer than 2 bodies.

    # Create a PhysicsCollisionGroup under the wrapper.
    group_path = wrapper_prim.GetPath().AppendChild("selfCollisionFilter")
    group = UsdPhysics.CollisionGroup.Define(stage, group_path)

    # Add all rigid bodies to the group via CollectionAPI.
    collection = group.GetCollidersCollectionAPI()
    includes_rel = collection.CreateIncludesRel()
    for body_path in rigid_bodies:
        includes_rel.AddTarget(body_path)

    # Filter the group against itself → disables collision between members.
    filtered_rel = group.GetFilteredGroupsRel()
    filtered_rel.AddTarget(group_path)

    console_logger.debug(
        f"  {wrapper_prim.GetPath().name}: self-collision filter for "
        f"{len(rigid_bodies)} bodies"
    )


def _has_nested_rigid_bodies(wrapper_prim: Usd.Prim) -> bool:
    """Check if any child rigid body has a child that is also a rigid body."""
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            for grandchild in child.GetChildren():
                if grandchild.HasAPI(UsdPhysics.RigidBodyAPI):
                    return True
    return False


def _has_dynamic_mass_or_rigid_body_authoring(wrapper_prim: Usd.Prim) -> bool:
    """Return True when the wrapper already looks like a movable rigid object."""
    if wrapper_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return True
    if wrapper_prim.HasAPI(UsdPhysics.MassAPI):
        return True
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(UsdPhysics.MassAPI):
            return True
    return False


def classify_object(
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> str:
    """Classify an object as 'static', 'dynamic', or 'articulated'.

    Classification logic:
    1. Check ArticulationRootAPI first — articulated objects may not have
       FixedJoints to root (e.g. when furniture uses freejoints in MuJoCo).
    2. Check for nested rigid bodies — this catches partially-fixed objects
       from prior runs where ArticulationRootAPI was already removed but
       bodies were not yet reparented as siblings.
    3. Check if wrapper has a FixedJoint descendant with body0 targeting root.
    4. If welded and no ArticulationRootAPI -> 'static'.
    5. If not welded and no ArticulationRootAPI -> 'dynamic'.
    """
    if _has_articulated_joint_descendants(wrapper_prim):
        return "articulated"
    if wrapper_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return "articulated"
    welded_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    if welded_joints:
        return "static"
    if _has_dynamic_mass_or_rigid_body_authoring(wrapper_prim):
        return "dynamic"
    return "static"


def fix_static_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> None:
    """Fix a static object by removing all physics body APIs and joints.

    Leaves only PhysicsCollisionAPI on collision geometry, making the object
    a static collider in Isaac Sim.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Remove RigidBodyAPI from wrapper.
    remove_rigid_body_api(wrapper_prim)

    # Remove RigidBodyAPI + MassAPI from all descendants.
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetPath() == wrapper_path:
            continue
        remove_rigid_body_api(descendant)
        remove_mass_api(descendant)

    # Delete FixedJoint from wrapper to root.
    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    delete_prims(stage, root_joints)

    # Delete FixedJoint from base_link/body_link to wrapper.
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_dynamic_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> None:
    """Fix a dynamic object by flattening to a single rigid body.

    Moves MassAPI from base_link to wrapper and removes the inner
    RigidBodyAPI and FixedJoint.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Find the immediate child (base_link) that has MassAPI.
    base_link = None
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.MassAPI):
            base_link = child
            break

    if base_link is None:
        console_logger.debug(
            f"Dynamic object {wrapper_path} has no child with MassAPI, "
            "skipping mass copy."
        )
    else:
        # Copy mass properties from base_link to wrapper.
        copy_mass_to_prim(source=base_link, target=wrapper_prim)
        # Remove MassAPI + RigidBodyAPI from base_link.
        remove_mass_api(base_link)
        remove_rigid_body_api(base_link)

    # Delete FixedJoint inside base_link (base_link→wrapper).
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_articulated_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
    all_layers: list[Sdf.Layer],
) -> None:
    """Repair articulated objects without replacing authored collision."""
    wrapper_path = wrapper_prim.GetPath()

    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    is_welded = len(root_joints) > 0

    promoted_bases = _collect_invalid_direct_joint_targets(stage, wrapper_prim)
    for base_prim in promoted_bases:
        _promote_base_rigid_body(wrapper_prim, base_prim)

    moved_links = _flatten_nested_articulated_links(
        stage=stage,
        wrapper_prim=wrapper_prim,
        all_layers=all_layers,
    )

    base_paths = {
        prim.GetPath() for prim in _collect_articulation_base_prims(stage, wrapper_prim)
    }
    wrapper_to_base_joints = _find_internal_fixed_joints(
        wrapper_prim=wrapper_prim,
        wrapper_path=wrapper_path,
        base_paths=base_paths,
    )
    if is_welded:
        for joint_path in wrapper_to_base_joints:
            _clear_wrapper_target_from_fixed_joint(
                stage=stage,
                joint_path=joint_path,
                wrapper_path=wrapper_path,
            )
        console_logger.debug(f"  {wrapper_path.name}: fixed-base articulation")
    else:
        delete_prims(stage, wrapper_to_base_joints)
        console_logger.debug(f"  {wrapper_path.name}: free articulation")

    delete_prims(stage, root_joints)
    if base_paths:
        remove_rigid_body_api(wrapper_prim)

    for descendant in Usd.PrimRange(wrapper_prim):
        if not descendant.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(descendant)
        for rel in [joint.GetBody0Rel(), joint.GetBody1Rel()]:
            if not rel:
                continue
            targets = rel.GetTargets()
            if not targets:
                continue
            target_path = targets[0]
            target_prim = stage.GetPrimAtPath(target_path)
            if (
                target_prim
                and target_prim.IsValid()
                and target_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                continue
            replacement = _find_nearest_rigid_body_ancestor(
                stage=stage,
                target_path=target_path,
                stop_path=wrapper_path,
            )
            if replacement and replacement != target_path:
                rel.SetTargets([replacement])

    _add_self_collision_filter(stage, wrapper_prim)
    if promoted_bases or moved_links:
        console_logger.debug(
            f"  {wrapper_path.name}: promoted {len(promoted_bases)} base bodies, "
            f"reparented {moved_links} articulated links"
        )


def fix_physics_layer(
    physics_usda_path: Path,
    articulated_collision_mode: str = "preserve",
) -> dict[str, int]:
    """Fix physics in a Physics.usda file for Isaac Sim compatibility.

    Opens the composed stage and fixes all objects. For articulated
    objects, reparenting is applied across ALL sublayers (Physics,
    Geometry, Materials) so mesh data and materials move with the prims.

    Args:
        physics_usda_path: Path to the Physics.usda file.

    Returns:
        Dict with counts of objects fixed per category.
    """
    stage = Usd.Stage.Open(str(physics_usda_path))
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError(f"No default prim in {physics_usda_path}")

    root_path = root_prim.GetPath()

    # Find the Geometry scope.
    geometry_path = root_path.AppendChild("Geometry")
    geometry_prim = stage.GetPrimAtPath(geometry_path)
    if not geometry_prim:
        raise RuntimeError(f"No Geometry scope found at {geometry_path}")

    # Collect ALL sublayers in the Payload directory for reparenting.
    # The Payload dir contains Physics.usda, Geometry.usda, Materials.usda.
    payload_dir = physics_usda_path.parent
    all_layers: list[Sdf.Layer] = []
    for usda_file in sorted(payload_dir.glob("*.usda")):
        layer = Sdf.Layer.FindOrOpen(str(usda_file))
        if layer:
            all_layers.append(layer)

    counts: dict[str, int] = {"static": 0, "dynamic": 0, "articulated": 0}

    for wrapper_prim in geometry_prim.GetChildren():
        category = classify_object(
            wrapper_prim=wrapper_prim,
            root_path=root_path,
        )
        counts[category] += 1

        if category == "static":
            fix_static_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
            )
        elif category == "dynamic":
            fix_dynamic_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
            )
        elif category == "articulated":
            fix_articulated_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
                all_layers=all_layers,
            )

    # Save ALL modified layers (Physics + Geometry + Materials).
    stage.GetRootLayer().Save()
    for layer in all_layers:
        if layer.dirty:
            layer.Save()

    collision_counts = _apply_articulated_collision_mode(
        physics_usda_path=physics_usda_path,
        collision_mode=articulated_collision_mode,
    )

    console_logger.info(
        f"Fixed {physics_usda_path}: "
        f"{counts['static']} static, "
        f"{counts['dynamic']} dynamic, "
        f"{counts['articulated']} articulated, "
        f"{collision_counts['wrappers']} articulated wrapper(s) collision-regenerated"
    )
    return counts


def _fix_single_scene(
    usd_dir: Path,
    articulated_collision_mode: str = "preserve",
) -> tuple[Path, dict[str, int] | str]:
    """Fix a single scene's Physics.usda. Returns (path, counts_or_error)."""
    physics_path = usd_dir / "Payload" / "Physics.usda"
    if not physics_path.exists():
        return usd_dir, "no Physics.usda found"
    try:
        counts = fix_physics_layer(
            physics_path,
            articulated_collision_mode=articulated_collision_mode,
        )
        return usd_dir, counts
    except Exception as e:
        return usd_dir, f"error: {e}"


def find_usd_dirs(base_path: Path, recursive: bool) -> list[Path]:
    """Find USD directories (containing Payload/Physics.usda)."""
    if not recursive:
        # Single scene: base_path should be the usd directory itself.
        if (base_path / "Payload" / "Physics.usda").exists():
            return [base_path]
        return []

    # Recursive: find all Physics.usda files.
    usd_dirs = []
    for physics_file in base_path.rglob("Payload/Physics.usda"):
        usd_dirs.append(physics_file.parent.parent)
    return sorted(usd_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix USD physics for Isaac Sim compatibility"
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a single USD directory (containing Payload/), "
            "or a parent directory when using --recursive"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for all USD scenes under the given path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for recursive mode (default: 1)",
    )
    parser.add_argument(
        "--articulated-collision-mode",
        choices=["preserve", "convex-hull", "convex-decomposition"],
        default="preserve",
        help=(
            "How to handle collision on articulated objects only. "
            "'preserve' keeps authored collision meshes, while the other "
            "modes deactivate authored *collision* meshes and regenerate "
            "collision from *visual* meshes."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    usd_dirs = find_usd_dirs(base_path=args.path, recursive=args.recursive)
    if not usd_dirs:
        console_logger.error(f"No USD scenes found at {args.path}")
        return

    console_logger.info(f"Found {len(usd_dirs)} USD scene(s) to fix")

    total_counts: dict[str, int] = {
        "static": 0,
        "dynamic": 0,
        "articulated": 0,
    }
    errors = 0

    if args.workers > 1 and len(usd_dirs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _fix_single_scene,
                    d,
                    args.articulated_collision_mode,
                ): d
                for d in usd_dirs
            }
            for future in as_completed(futures):
                path, result = future.result()
                if isinstance(result, str):
                    console_logger.warning(f"{path}: {result}")
                    errors += 1
                else:
                    for k, v in result.items():
                        total_counts[k] += v
    else:
        for usd_dir in usd_dirs:
            path, result = _fix_single_scene(
                usd_dir,
                args.articulated_collision_mode,
            )
            if isinstance(result, str):
                console_logger.warning(f"{path}: {result}")
                errors += 1
            else:
                for k, v in result.items():
                    total_counts[k] += v

    console_logger.info(
        f"Done. Fixed {len(usd_dirs) - errors}/{len(usd_dirs)} scenes: "
        f"{total_counts['static']} static, "
        f"{total_counts['dynamic']} dynamic, "
        f"{total_counts['articulated']} articulated objects total"
    )
    if errors:
        console_logger.warning(f"{errors} scene(s) had errors")


if __name__ == "__main__":
    main()
