"""Resolve a SceneCode generation into simulator inputs.

The SceneCode layout is kept as authored (world frame = SceneCode house frame).
The robot base is placed on the support edge that best reaches the task objects.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .spec import TaskSpec
from .paths import WORKSPACE
from autoresearch.robotics_visual_revision_verifier import verify_visual_revision

ROBOT_URDF = (
    WORKSPACE
    / "outputs/robot_assets/robolab_droid_franka_robotiq_2f85_f72d_2c047340_20260823_v3"
    / "franka_robotiq_2f_85.urdf"
)
# base sits this far inside the support edge; DROID mounts the arm at the table edge
BASE_EDGE_INSET_M = 0.08
REACH_M = 0.80


@dataclass
class SceneObject:
    object_id: str
    role: str
    sdf_path: Path
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    fixed: bool = False
    scale: float = 1.0
    bbox_local: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    initial_container: str | None = None


@dataclass
class SceneInputs:
    case_id: str
    support: SceneObject
    objects: dict[str, SceneObject]
    robot_urdf: Path
    robot_base_position: tuple[float, float, float]
    robot_base_yaw: float
    source: dict[str, Any] = field(default_factory=dict)
    # component_id -> {"host": object_id, "joint_name": str | None, "joint_type": str}
    articulations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def obj(o: SceneObject) -> dict[str, Any]:
            return {
                "object_id": o.object_id,
                "role": o.role,
                "sdf_path": str(o.sdf_path),
                "position": list(o.position),
                "orientation_xyzw": list(o.orientation_xyzw),
                "fixed": o.fixed,
                "scale": o.scale,
                "initial_container": o.initial_container,
            }

        return {
            "case_id": self.case_id,
            "support": obj(self.support),
            "objects": {k: obj(v) for k, v in self.objects.items()},
            "robot_urdf": str(self.robot_urdf),
            "robot_base_position": list(self.robot_base_position),
            "robot_base_yaw": self.robot_base_yaw,
            "source": self.source,
            "articulations": self.articulations,
        }


def fixed_background_ids(spec: TaskSpec, objects, *, required_visible=(), hosts=()) -> set[str]:
    declared = {o.object_id for o in spec.objects if o.role == 'distractor'
                and o.geometry_constraints.get('fixed') is True}
    if not declared:
        return set()
    involved = set(required_visible) | set(hosts)
    involved.update(a.host_object_id for a in spec.articulations)
    involved.update(o.initial_container for o in spec.objects if o.initial_container)
    for step in spec.steps:
        involved.update(ref for ref in (step.subject, step.target, step.tool,
                        step.goal.subject, step.goal.target) if ref is not None)
    values = objects.values() if isinstance(objects, dict) else objects
    def get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key)
    return {get(o, 'object_id') for o in values if get(o, 'fixed') is True
            and get(o, 'role') == 'distractor' and get(o, 'object_id') in declared - involved}


def _house_objects(house_state: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for room in payload.get("rooms", {}).values():
        from .scene_objects import physical_objects
        members = physical_objects(room.get("objects") or {})
        for record in members.values():
            # Composite transforms are model WORLD poses; the exporter uses
            # T_world_model @ T_model_base. Never compose with the parent.
            internal = record.get('internal_model_pose')
            if record.get('metadata', {}).get('composite_parent_id') and internal:
                transform = record['transform']
                rotation = _quat_mat(_xyzw(transform['rotation_wxyz']))
                position = np.asarray(transform['translation']) + rotation @ np.asarray(internal['translation'])
                q = _mat_quat(rotation @ _quat_mat(_xyzw(internal['rotation_wxyz'])))
                record['transform'] = {'translation': position.tolist(),
                                       'rotation_wxyz': [q[3], q[0], q[1], q[2]]}
        overlap = out.keys() & members.keys()
        if overlap:
            raise ValueError(f'Duplicate scene occurrence IDs across rooms: {sorted(overlap)}')
        out.update(members)
    return out


def _xyzw(wxyz: list[float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        x = (r[2, 1] - r[1, 2]) / (4 * w)
        y = (r[0, 2] - r[2, 0]) / (4 * w)
        z = (r[1, 0] - r[0, 1]) / (4 * w)
    else:  # 180 deg rotation; fall back to the generic routine
        from scipy.spatial.transform import Rotation

        x, y, z, w = Rotation.from_matrix(r).as_quat()
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def _rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def robot_base_candidates(
    support_lo: np.ndarray, support_hi: np.ndarray, targets: list[tuple[np.ndarray, np.ndarray]],
    *, outside_only: bool = False,
) -> list[tuple[float, tuple[float, float, float], float]]:
    """Candidate (cost, base, yaw) on every support edge (world frame = support frame).

    Positive inset puts the base on the support; negative inset puts it on a
    pedestal just outside the edge (rendered as the trajectory's robot_mount).
    Reach is measured to the nearest point of each target's xy footprint.
    """

    centre = (support_lo[:2] + support_hi[:2]) / 2
    half = (support_hi[:2] - support_lo[:2]) / 2
    top = float(support_hi[2])
    t_centres = [((lo[:2] + hi[:2]) / 2) for lo, hi in targets]
    centroid = np.mean(t_centres, axis=0)

    def reach(base: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        nearest = np.clip(base, lo[:2], hi[:2])
        return float(np.linalg.norm(nearest - base))

    out = []
    for axis in (0, 1):
        for sign in (-1.0, 1.0):
            normal = np.zeros(2)
            normal[axis] = sign
            tangent = np.array([-normal[1], normal[0]])
            edge_mid = centre + normal * half[axis]
            half_len = half[1 - axis]
            along0 = float((centroid - edge_mid) @ tangent)
            for d_along in (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3):
                along = float(np.clip(along0 + d_along, -half_len + 0.1, half_len - 0.1))
                insets = (-0.15, -0.25, -0.35, -0.45) if outside_only else \
                    (BASE_EDGE_INSET_M, 0.0, -0.08, -0.16)
                for inset in insets:
                    base = edge_mid + tangent * along - normal * inset
                    dists = [reach(base, lo, hi) for lo, hi in targets]
                    far = max(dists)
                    cost = far + 0.1 * float(np.mean(dists)) + 0.05 * abs(d_along) \
                        + (1.0 if far > REACH_M else 0.0)
                    out.append((cost, (float(base[0]), float(base[1]), top + 0.002),
                                math.atan2(-normal[1], -normal[0])))
    out.sort(key=lambda c: c[0])
    return out


def canonicalize_support_aliases(spec: TaskSpec, generation_dir: str | Path) -> TaskSpec:
    """Collapse inventory aliases of the support, never instantiate a second table.

    Identity uses SceneCode object IDs, not shared asset paths: two real objects
    may intentionally use the same asset. Manipuland aliases are invalid.
    """
    from dataclasses import replace
    cert = json.loads((Path(generation_dir) / "task_inventory_certificate.json").read_text())
    matches = {m['task_object_id']: m['scene_object_id'] for m in cert['matches']
               if m.get('status') == 'matched'}
    support = spec.support.object_id
    identity = matches.get(support)
    aliases = {o.object_id for o in spec.objects if o.object_id != support
               and identity is not None and matches.get(o.object_id) == identity}
    for obj in spec.objects:
        if obj.object_id in aliases and obj.role != 'destination':
            raise ValueError(f"non-destination {obj.object_id} aliases the support")
    def name(value):
        return support if value in aliases else value
    return replace(spec, objects=tuple(replace(o, initial_container=name(o.initial_container))
                                      for o in spec.objects if o.object_id not in aliases),
                   steps=tuple(replace(st, subject=name(st.subject), target=name(st.target),
                                  tool=name(st.tool), goal=replace(st.goal,
                                  subject=name(st.goal.subject), target=name(st.goal.target)))
                          for st in spec.steps))


def _repair_reciprocal_bindings(spec: TaskSpec, matches: dict[str, dict[str, Any]]):
    """Repair one certified, unambiguous lid/container inversion.

    The inventory matcher can confuse complementary parts when their descriptions
    share vocabulary.  This repair is deliberately conservative: both task
    declarations and both scene names must identify the same lid/container pair,
    and there must be exactly one possible pair.  In particular, a random object
    matched to a scene whose name contains ``lid`` must never be swapped.
    """
    def tokens(value):
        return {x for x in str(value or '').lower().replace('-', '_').split('_') if x}

    def kind(words):
        if 'lid' in words:
            return 'lid'
        if 'container' in words or 'box' in words:
            return 'container'
        return None

    task_ids = {o.object_id for o in spec.objects}
    # An explicit inventory/member declaration is an authoritative occurrence
    # binding.  Do not rewrite either its task object or the inventory key it
    # names: swapping the latter makes the declared member fail certification
    # even when the reciprocal repair would otherwise look semantically safe.
    protected = set()
    for declaration in spec.objects:
        if declaration.inventory_object_id and declaration.scene_member_id:
            protected.add(declaration.object_id)
            protected.add(declaration.inventory_object_id)
    candidates = []
    for a_id, a in matches.items():
        if a_id not in task_ids:
            continue
        for b_id, b in matches.items():
            if b_id <= a_id or b_id not in task_ids:
                continue
            if a.get('scene_object_id') == b.get('scene_object_id'):
                continue
            da, db = spec.object(a_id), spec.object(b_id)
            ca, cb = tokens(da.category), tokens(db.category)
            sa, sb = tokens(a.get('scene_name')), tokens(b.get('scene_name'))
            # Require complementary semantic kinds in both namespaces.
            if {kind(ca), kind(cb)} != {'lid', 'container'} or \
               {kind(sa), kind(sb)} != {'lid', 'container'}:
                continue
            # A shared product/domain token makes this a pair, rather than an
            # accidental match to an unrelated lid or box elsewhere in a room.
            # The product/domain vocabulary must agree exactly on both sides.
            # A loose intersection is unsafe: e.g. ``one_lid`` and two
            # ``box`` records can manufacture a reciprocal pair from stale
            # inventory loop records.  Keep only a closed semantic pair whose
            # non-generic domain tokens are identical.
            generic = {'lid', 'container', 'box'}
            task_domain = (ca | cb) - generic
            scene_domain = (sa | sb) - generic
            if not task_domain or task_domain != scene_domain:
                continue
            if kind(ca) == kind(sa):
                continue
            candidates.append((a_id, b_id))

    # Multiple possible occurrences are ambiguous; do not create duplicate
    # source bindings or perform a sequence of pairwise swaps.
    if len(candidates) != 1:
        return []
    a_id, b_id = candidates[0]
    if a_id in protected or b_id in protected:
        return []
    matches[a_id], matches[b_id] = matches[b_id], matches[a_id]
    return [{'objects': [a_id, b_id],
             'reason': 'reciprocal_lid_container_semantic_binding'}]


def _variant_sdf(spec: TaskSpec, object_id: str, source_sdf: Path) -> Path:
    """Return a validated visual-only derivative for one bound scene object."""
    entries = [v for v in spec.visual_variants if v.get("object_id") == object_id]
    if not entries:
        return source_sdf
    if len(entries) != 1:
        raise ValueError(f"multiple visual variants for {object_id!r}")
    entry = entries[0]
    manifest_path = Path(entry.get("manifest", "")).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"visual variant manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "scenecode_visual_variant.v1" or not manifest.get("physics_geometry_unchanged"):
        raise ValueError("visual variant is not a physics-preserving SceneCode manifest")
    declared_source = Path(manifest.get("source_sdf", "")).resolve()
    # Derivative bundlers may copy the certified physics SDF into a staging
    # directory. Bind that copy by its recorded digest, not by its pathname;
    # the inventory path remains the authoritative source occurrence. Keep the
    # historical path check for manifests lacking source-file evidence.
    if declared_source != source_sdf.resolve():
        source_hashes = manifest.get("source_file_sha256", {})
        expected = source_hashes.get(source_sdf.name)
        actual = hashlib.sha256(source_sdf.read_bytes()).hexdigest() if source_sdf.is_file() else None
        if not expected or expected != actual:
            raise ValueError(f"visual variant source does not match {object_id!r}")
    output_sdf = manifest_path.parent / source_sdf.name
    if not output_sdf.is_file():
        raise FileNotFoundError(f"visual variant output missing: {output_sdf}")
    hashes = manifest.get("output_file_sha256", {})
    for name, digest in hashes.items():
        path = manifest_path.parent / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"visual variant evidence changed: {name}")
    # The SDF is the physics source, not a visual payload.  For certified
    # Always verify at the native consumption boundary.  The manifest flag is
    # evidence that must be checked, never an opt-in bypass: otherwise a copied
    # derivative can opt out and alter collision or node resources.
    verify_visual_revision(
        manifest_path=manifest_path,
        output=manifest_path.parent / ".visual_revision_verdict.json",
        source_sdf_override=source_sdf,
        expected_source_sdf=source_sdf,
        write_output=False,
    )
    # This compact resolver still binds the selected physical SDF by bytes;
    # the independent verifier above additionally binds its resource graph.
    if not source_sdf.is_file() or output_sdf.read_bytes() != source_sdf.read_bytes():
        raise ValueError("visual variant changed the physical SDF")
    return output_sdf


def _resolve_bound_resource(raw: str | Path, generation: Path) -> Path:
    """Resolve a certificate resource without depending on the worker cwd."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    generation_candidate = generation / path
    if generation_candidate.exists():
        return generation_candidate
    project_candidate = WORKSPACE / path
    if project_candidate.exists():
        return project_candidate
    return generation_candidate


def resolve_scenecode(spec: TaskSpec, generation_dir: str | Path) -> SceneInputs:
    spec = canonicalize_support_aliases(spec, generation_dir)
    generation = Path(generation_dir)
    certificate = json.loads(
        (generation / "task_inventory_certificate.json").read_text(encoding="utf-8")
    )
    def bound_path(raw: str | Path) -> Path:
        """Resolve certificate-relative resources at the generation boundary.

        SceneCode certificates commonly store paths relative to the published
        generation, while registered stages run from the agent worktree.  Do
        not let the caller's current directory change which asset is bound.
        Absolute paths remain supported for legacy certificates.
        """
        return _resolve_bound_resource(raw, generation)

    house = _house_objects(bound_path(certificate["house_state"]))
    matches = {m["task_object_id"]: m for m in certificate["matches"] if m.get("status") == "matched"}

    # Correct only a certified reciprocal pair, never by coordinates or case id.
    binding_corrections = _repair_reciprocal_bindings(spec, matches)

    instance_bindings = {}
    used_occurrences = {}

    def scene_object(task_id: str, role: str) -> SceneObject:
        declaration = spec.object(task_id)
        match = matches.get(declaration.inventory_object_id or task_id)
        if match is None:
            raise ValueError(f"SceneCode inventory has no match for task object {task_id!r}")
        root_id = match["scene_object_id"]
        record_id = declaration.scene_member_id or root_id
        if record_id not in house:
            raise ValueError(f"SceneCode has no declared member {record_id!r}")
        record = house[record_id]
        if declaration.scene_member_id:
            root = house[root_id]
            metadata = record.get('metadata') or {}
            root_metadata = root.get('metadata') or {}
            same_composite_source = (
                metadata.get('composite_parent_id') is not None
                and metadata.get('composite_parent_id') == root_metadata.get('composite_parent_id')
                and metadata.get('source_asset_id') == root_metadata.get('source_asset_id'))
            if record_id != root_id and not (
                    record_id.startswith(root_id + '_instance_') or same_composite_source):
                raise ValueError('declared member is not a certified source occurrence')
            if record.get('sdf_path') != root.get('sdf_path'):
                raise ValueError('declared collection member has a different asset')
            instance_bindings[task_id] = {
                'inventory_object_id': declaration.inventory_object_id,
                'scene_object_id': record_id, 'sdf_path': str(bound_path(match['sdf_path'])),
                'sdf_sha256': match.get('sdf_sha256'),
            }
        if record_id in used_occurrences:
            raise ValueError('two task objects bind the same scene member')
        used_occurrences[record_id] = task_id
        transform = record["transform"]
        return SceneObject(
            object_id=task_id,
            role=role,
            sdf_path=_variant_sdf(spec, task_id, bound_path(match["sdf_path"])),
            position=tuple(float(v) for v in transform["translation"]),
            orientation_xyzw=_xyzw(transform["rotation_wxyz"]),
            fixed=role == "support" or declaration.geometry_constraints.get("fixed", False),
            initial_container=spec.object(task_id).initial_container,
            bbox_local=(tuple(record["bbox_min"]), tuple(record["bbox_max"]))
            if "bbox_min" in record and "bbox_max" in record else None,
        )

    support_spec = spec.support
    if support_spec is None:
        raise ValueError("task has no support object")
    support = scene_object(support_spec.object_id, "support")
    objects = {
        o.object_id: scene_object(o.object_id, o.role)
        for o in spec.objects
        if o.role != "support"
    }
    # world frame := support frame (the renderer welds the support at the origin)
    rs = _quat_mat(support.orientation_xyzw)
    ps = np.asarray(support.position)
    for obj in objects.values():
        r = rs.T @ _quat_mat(obj.orientation_xyzw)
        obj.position = tuple((rs.T @ (np.asarray(obj.position) - ps)).tolist())
        obj.orientation_xyzw = _mat_quat(r)
    support.position = (0.0, 0.0, 0.0)
    support.orientation_xyzw = (0.0, 0.0, 0.0, 1.0)
    base, yaw = (0.0, -0.5, 0.0), math.pi / 2  # placeholder until place_robot() measures AABBs
    articulations = {}
    for comp in spec.articulations:
        binding = (WORKSPACE / "outputs/robotics_articulation_bindings" / generation.parent.name
                   / generation.name / f"{comp.component_id}.json")
        joint_name = None
        if binding.is_file():
            joint_name = json.loads(binding.read_text()).get("joint_name")
        articulations[comp.component_id] = {
            "host": comp.host_object_id, "joint_name": joint_name, "joint_type": comp.joint_type,
        }
    return SceneInputs(
        case_id=spec.case_id,
        support=support,
        objects=objects,
        robot_urdf=ROBOT_URDF,
        robot_base_position=base,
        robot_base_yaw=yaw,
        articulations=articulations,
        source={
            "generation_dir": str(generation),
            "house_state": certificate["house_state"],
            "house_state_sha256": certificate.get("house_state_sha256"),
            "instance_bindings": instance_bindings,
            "binding_corrections": binding_corrections,
            "visual_variants": [dict(v) for v in spec.visual_variants],
        },
    )


def resolved_articulations(certificate: dict, declared: list[dict] = ()) -> list[dict]:
    """Use the expert's measured joint/link binding, retaining complete legacy metadata."""
    bindings = {a["component"]: dict(a) for a in declared}
    for component, measured in (certificate.get("articulations") or {}).items():
        prior = bindings.get(component, {})
        for key in ("host", "joint_name", "child_link"):
            if prior.get(key) and measured.get(key) and prior[key] != measured[key]:
                raise ValueError(f"articulation {component} {key} disagrees with reference certificate")
        bindings[component] = {**prior, **measured, "component": component}
    for component, binding in bindings.items():
        if not all(binding.get(k) for k in ("host", "joint_name", "child_link")):
            raise ValueError(f"articulation {component} lacks a resolved host/joint/child binding")
    return list(bindings.values())
