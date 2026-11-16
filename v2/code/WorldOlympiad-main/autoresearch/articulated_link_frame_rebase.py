"""Bind an imported SDF moving-link frame to a measured reference frame.

SceneCode may express a link origin at a model-frame offset while the PyBullet
reference records the executable child frame at the joint anchor.  Relabeling
the native link frame is safe only if every visual, collision and COM keeps its
world pose.  This derivative never changes source SDF or mesh bytes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .math3d import compose
from .models import BodyIR, Pose


def _relative(parent: Pose, world: Pose) -> Pose:
    parent_rotation = Rotation.from_quat(parent.quat_xyzw)
    child_rotation = Rotation.from_quat(world.quat_xyzw)
    position = parent_rotation.inv().apply(
        np.asarray(world.xyz, dtype=float) - np.asarray(parent.xyz, dtype=float)
    )
    return Pose(xyz=tuple(position),
                quat_xyzw=tuple((parent_rotation.inv() * child_rotation).as_quat()))


def rebase_measured_child_frame(
    body: BodyIR, *, child_link: str, measured_world: Pose,
    maximum_translation_m: float = 1.0,
) -> dict:
    """Mutate one runtime WorldIR link while proving its geometry is invariant."""

    matches = [link for link in body.links if link.name == child_link]
    if len(matches) != 1:
        raise ValueError("moving link is missing or ambiguous")
    link = matches[0]
    old = link.pose
    new = _relative(body.pose, measured_world)
    delta = _relative(new, old)
    displacement = float(np.linalg.norm(delta.xyz))
    angular = float(Rotation.from_quat(delta.quat_xyzw).magnitude())
    if displacement > maximum_translation_m or angular > 1e-4:
        raise ValueError("measured child frame requires an unsafe rotation or translation")
    before = [compose(body.pose, compose(old, geometry.pose))
              for geometry in [*link.visuals, *link.collisions]]
    for geometry in [*link.visuals, *link.collisions]:
        geometry.pose = compose(delta, geometry.pose)
    if link.inertial is not None:
        link.inertial.com = tuple(
            np.asarray(delta.xyz) + np.asarray(link.inertial.com)
        )
    link.pose = new
    after = [compose(body.pose, compose(new, geometry.pose))
             for geometry in [*link.visuals, *link.collisions]]
    maximum_geometry_error = max(
        (float(np.linalg.norm(np.asarray(left.xyz) - np.asarray(right.xyz)))
         for left, right in zip(before, after)), default=0.0,
    )
    maximum_orientation_error = max(
        (float((Rotation.from_quat(left.quat_xyzw).inv()
                * Rotation.from_quat(right.quat_xyzw)).magnitude())
         for left, right in zip(before, after)), default=0.0,
    )
    measured_error = float(np.linalg.norm(
        np.asarray(compose(body.pose, link.pose).xyz) - np.asarray(measured_world.xyz)
    ))
    if max(maximum_geometry_error, maximum_orientation_error, measured_error) > 1e-5:
        raise ValueError("moving-link frame rebase changed source geometry")
    return {
        "schema_version": "worldir_articulated_link_frame_rebase.v1",
        "child_link": child_link,
        "old_link_pose": old.model_dump(mode="json"),
        "new_link_pose": new.model_dump(mode="json"),
        "frame_displacement_m": displacement,
        "frame_rotation_rad": angular,
        "visual_count": len(link.visuals),
        "collision_count": len(link.collisions),
        "maximum_world_geometry_position_error_m": maximum_geometry_error,
        "maximum_world_geometry_orientation_error_rad": maximum_orientation_error,
        "measured_child_position_error_m": measured_error,
        "source_asset_geometry_unchanged": True,
        "passed": True,
    }
