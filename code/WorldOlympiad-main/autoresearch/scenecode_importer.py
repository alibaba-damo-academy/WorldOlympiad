"""Import SceneCode state/SDF artifacts into the simulator-neutral WorldIR."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .io import sha256_file
from .materials import MATERIALS
from .math3d import compose, pose_from_scenecode, pose_from_sdf
from .models import (
    BodyIR,
    ContactMaterial,
    Geometry,
    Inertial,
    JointIR,
    LinkIR,
    Pose,
    RoomIR,
    WorldIR,
)


def _floats(text: str | None, count: int, default: float = 0.0) -> tuple[float, ...]:
    values = [float(value) for value in (text or "").split()]
    values += [default] * (count - len(values))
    return tuple(values[:count])


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return cleaned or "unnamed"


def _model_frame_poses(model: ET.Element) -> dict[str, Pose]:
    """Resolve SDF link/frame ``relative_to`` into the model frame.

    Generated SceneCode assets commonly put a fixed handle at an identity pose
    relative to a moving door. Reading that identity as a model-frame pose
    detaches the handle in native USD while the source SDF remains valid.
    """
    elements = {}
    for element in [*model.findall("link"), *model.findall("frame")]:
        name = element.get("name")
        if not name or name in elements:
            raise ValueError("SDF link/frame names must be unique and nonempty")
        elements[name] = element
    resolved = {"__model__": Pose()}
    visiting: set[str] = set()

    def resolve(raw_name: str) -> Pose:
        name = raw_name.removeprefix(f"{model.get('name')}::")
        if name in resolved:
            return resolved[name]
        if name in visiting or name not in elements:
            raise ValueError(f"cyclic or unresolved SDF link/frame: {raw_name}")
        visiting.add(name)
        element = elements[name]
        pose = element.find("pose")
        default_parent = (element.get("attached_to", "__model__")
                          if element.tag == "frame" else "__model__")
        parent = pose.get("relative_to", default_parent) if pose is not None else default_parent
        result = compose(resolve(parent), pose_from_sdf(
            pose.text if pose is not None else None))
        visiting.remove(name)
        resolved[name] = result
        return result

    for name in elements:
        resolve(name)
    return resolved


def _geometry(element: ET.Element, base_dir: Path, name: str) -> Geometry:
    geometry = element.find("geometry")
    if geometry is None:
        raise ValueError(f"{name}: missing <geometry>")
    pose = pose_from_sdf(element.findtext("pose"))
    diffuse_text = element.findtext("material/diffuse")
    diffuse = _floats(diffuse_text, 4, 1.0)
    if (mesh := geometry.find("mesh")) is not None:
        uri = (mesh.findtext("uri") or "").strip()
        if not uri:
            raise ValueError(f"{name}: mesh URI is empty")
        if uri.startswith("file://"):
            uri = uri[7:]
        path = Path(uri)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name}: mesh not found: {path}")
        if path.suffix.lower() == ".gltf" and diffuse_text is None:
            materials = json.loads(path.read_text()).get("materials") or []
            if len(materials) == 1:
                factor = (materials[0].get("pbrMetallicRoughness") or {}).get(
                    "baseColorFactor")
                if factor is not None:
                    if (not isinstance(factor, list) or len(factor) != 4
                            or any(not isinstance(x, (int, float))
                                   or not math.isfinite(x) or not 0 <= x <= 1
                                   for x in factor)):
                        raise ValueError(f"{name}: invalid glTF base color")
                    diffuse = tuple(float(x) for x in factor)
        return Geometry(
            name=name,
            kind="mesh",
            pose=pose,
            path=str(path),
            scale=_floats(mesh.findtext("scale"), 3, 1.0),
            rgba=diffuse,
        )
    if (box := geometry.find("box")) is not None:
        return Geometry(name=name, kind="box", pose=pose, size=_floats(box.findtext("size"), 3), rgba=diffuse)
    if (sphere := geometry.find("sphere")) is not None:
        return Geometry(name=name, kind="sphere", pose=pose, radius=float(sphere.findtext("radius") or 0), rgba=diffuse)
    if (cylinder := geometry.find("cylinder")) is not None:
        return Geometry(
            name=name,
            kind="cylinder",
            pose=pose,
            radius=float(cylinder.findtext("radius") or 0),
            length=float(cylinder.findtext("length") or 0),
            rgba=diffuse,
        )
    raise ValueError(f"{name}: unsupported SDF geometry")


def _inertial(link: ET.Element) -> Inertial | None:
    element = link.find("inertial")
    if element is None:
        return None
    mass = float(element.findtext("mass") or 0)
    if mass <= 0:
        return None
    pose = pose_from_sdf(element.findtext("pose"))
    inertia = element.find("inertia")
    if inertia is None:
        return None
    values = tuple(
        float(inertia.findtext(key) or 0)
        for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    )
    if min(values[:3]) <= 0:
        return None
    return Inertial(mass=mass, com=pose.xyz, inertia=values)


def import_sdf_body(
    path: str | Path,
    *,
    body_name: str | None = None,
    semantic_category: str = "generic",
    pose: Pose | None = None,
    internal_model_pose: Pose | None = None,
    static: bool = False,
    target: bool = False,
    support: str | None = None,
    material: ContactMaterial | None = None,
    metadata: dict[str, Any] | None = None,
) -> BodyIR:
    """Parse one SDF model while rejecting missing or silently dropped geometry."""

    source = Path(path).resolve()
    model = ET.parse(source).getroot().find("model")
    if model is None:
        raise ValueError(f"SDF has no model: {source}")
    name = _safe_name(body_name or model.get("name") or source.stem)
    frame_poses = _model_frame_poses(model)
    links: list[LinkIR] = []
    for link_index, link in enumerate(model.findall("link")):
        link_name = _safe_name(link.get("name") or f"link_{link_index}")
        visuals = [
            _geometry(item, source.parent, f"{link_name}_visual_{index}")
            for index, item in enumerate(link.findall("visual"))
        ]
        collisions = [
            _geometry(item, source.parent, f"{link_name}_collision_{index}")
            for index, item in enumerate(link.findall("collision"))
        ]
        if visuals and not collisions:
            raise ValueError(f"{source}: link {link_name!r} has visual geometry but no collision")
        links.append(
            LinkIR(
                name=link_name,
                pose=frame_poses[link.get("name")],
                inertial=_inertial(link),
                visuals=visuals,
                collisions=collisions,
            )
        )
    if not links:
        raise ValueError(f"SDF model has no links: {source}")

    joints: list[JointIR] = []
    link_pose_by_name = {link.name: link.pose for link in links}
    implicit_joint_pose_names: list[str] = []
    for index, joint in enumerate(model.findall("joint")):
        joint_type = (joint.get("type") or "fixed").lower()
        if joint_type not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError(f"unsupported joint type {joint_type!r} in {source}")
        lower = joint.findtext("axis/limit/lower")
        upper = joint.findtext("axis/limit/upper")
        joint_name = _safe_name(joint.get("name") or f"joint_{index}")
        child_name = _safe_name(joint.findtext("child") or "")
        joint_pose_text = joint.findtext("pose")
        if joint_pose_text is None and child_name in link_pose_by_name:
            # In SDF an omitted joint pose leaves the joint at the child-link
            # frame. URDF instead locates the child link through the joint
            # origin. Transfer the imported child pose explicitly; otherwise
            # articulated doors are compiled at the model origin and can end
            # up buried in the floor.
            joint_pose = link_pose_by_name[child_name]
            implicit_joint_pose_names.append(joint_name)
        else:
            joint_pose = pose_from_sdf(joint_pose_text)
        joints.append(
            JointIR(
                name=joint_name,
                type=joint_type,
                parent=_safe_name(joint.findtext("parent") or ""),
                child=child_name,
                pose=joint_pose,
                axis=_floats(joint.findtext("axis/xyz"), 3),
                lower=float(lower) if lower is not None else None,
                upper=float(upper) if upper is not None else None,
                damping=float(joint.findtext("axis/dynamics/damping") or 0.1),
            )
        )

    body_metadata = dict(metadata or {})
    if implicit_joint_pose_names:
        body_metadata["simready_sdf_implicit_joint_pose_names"] = (
            implicit_joint_pose_names
        )
    return BodyIR(
        name=name,
        semantic_category=semantic_category,
        pose=pose or Pose(),
        internal_model_pose=internal_model_pose or Pose(),
        links=links,
        joints=joints,
        material=material or MATERIALS.get(semantic_category, ContactMaterial()),
        static=static,
        target=target,
        support=support,
        source_path=str(source),
        source_sha256=sha256_file(source),
        metadata=body_metadata,
    )


def _resolve_scene_path(value: str, room_dir: Path, scene_root: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [room_dir / path, scene_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"SceneCode asset path cannot be resolved: {value}")


def _gltf_dependencies(path: Path) -> list[Path]:
    if path.suffix.lower() != ".gltf":
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot audit GLTF dependencies for {path}: {exc}") from exc
    dependencies: list[Path] = []
    for item in [*raw.get("buffers", []), *raw.get("images", [])]:
        uri = item.get("uri")
        if not uri or str(uri).startswith("data:"):
            continue
        candidate = (path.parent / str(uri)).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"GLTF dependency is missing: {candidate}")
        dependencies.append(candidate)
    return dependencies


def import_house_state(path: str | Path, *, scene_id: str | None = None) -> WorldIR:
    """Import SceneCode's canonical ``combined_house/house_state.json`` registry."""

    state_path = Path(path).resolve()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    scene_root = state_path.parent.parent
    bodies: list[BodyIR] = []
    rooms: list[RoomIR] = []
    prompt_parts: list[str] = []
    structural_object_mappings: list[dict[str, str]] = []
    composite_object_mappings: list[dict[str, Any]] = []
    manipuland_asset_registries: list[dict[str, str]] = []
    for room_id, room_data in raw.get("rooms", {}).items():
        room_dir = scene_root / f"room_{room_id}"
        manipuland_registry = (
            room_dir / "generated_assets" / "manipuland" / "asset_registry.json"
        )
        if manipuland_registry.is_file():
            manipuland_asset_registries.append(
                {
                    "room_id": str(room_id),
                    "path": str(manipuland_registry.resolve()),
                    "sha256": sha256_file(manipuland_registry),
                }
            )
        room_body_names: list[str] = []
        room_body: BodyIR | None = None
        room_sdf: Path | None = None
        if room_data.get("text_description"):
            prompt_parts.append(str(room_data["text_description"]))
        room_geometry = room_data.get("room_geometry") or {}
        if room_geometry and not room_geometry.get("sdf_path"):
            raise ValueError(
                f"SceneCode room {room_id} declares room_geometry without sdf_path"
            )
        if room_geometry.get("sdf_path"):
            room_sdf = _resolve_scene_path(room_geometry["sdf_path"], room_dir, scene_root)
            room_body = import_sdf_body(
                room_sdf,
                body_name=f"{room_id}_room_geometry",
                semantic_category="room",
                static=True,
                material=MATERIALS["room"],
                metadata={"room_id": room_id, "object_type": "room_geometry"},
            )
            bodies.append(room_body)
            room_body_names.append(room_body.name)
        declared_structural_walls = {
            str(item.get("object_id")): item
            for item in room_geometry.get("walls", [])
            if isinstance(item, dict) and item.get("object_id")
        }
        for object_id, item in room_data.get("objects", {}).items():
            sdf_value = item.get("sdf_path")
            if not sdf_value:
                # SceneCode intentionally duplicates immutable structural wall
                # registry entries under ``objects`` even though their visual
                # and collision geometry is carried by the single room SDF.
                # Accept only an exact duplicate with an audited room artifact;
                # every other declared object still fails closed below.
                structural_wall = declared_structural_walls.get(str(object_id))
                if (
                    str(item.get("object_type", "")).lower() == "wall"
                    and item.get("immutable") is True
                    and room_sdf is not None
                    and structural_wall == item
                ):
                    structural_object_mappings.append(
                        {
                            "room_id": str(room_id),
                            "object_id": str(object_id),
                            "represented_by_body": str(room_body.name),
                            "source_sdf": str(room_sdf),
                        }
                    )
                    continue
                # SceneCode's manipuland designer can add a logical stack
                # object whose geometry is exactly the already-declared member
                # assets. It intentionally has no SDF of its own. Accept only
                # a fully auditable duplicate mapping; malformed or orphaned
                # composites continue to fail closed.
                metadata = item.get("metadata") or {}
                members = metadata.get("member_assets") or []
                room_objects = room_data.get("objects", {})
                member_mappings: list[dict[str, str]] = []
                composite_ok = (
                    str(item.get("object_type", "")).lower() == "manipuland"
                    and metadata.get("composite_type") == "stack"
                    and isinstance(members, list)
                    and bool(members)
                )
                if composite_ok:
                    expanded_members: list[BodyIR] = []
                    registry_path = (
                        room_dir
                        / "generated_assets"
                        / "manipuland"
                        / "asset_registry.json"
                    )
                    registry = (
                        json.loads(registry_path.read_text(encoding="utf-8"))
                        if registry_path.is_file()
                        else {}
                    )
                    for member_index, member in enumerate(members):
                        member_id = str(member.get("asset_id", ""))
                        member_sdf = member.get("sdf_path")
                        source_item = room_objects.get(member_id)
                        source_sdf = source_item.get("sdf_path") if source_item else None
                        if not member_id or not member_sdf:
                            composite_ok = False
                            break
                        resolved_member = _resolve_scene_path(
                            str(member_sdf), room_dir, scene_root
                        )
                        if not resolved_member.is_file():
                            composite_ok = False
                            break
                        if source_sdf:
                            resolved_source = _resolve_scene_path(
                                str(source_sdf), room_dir, scene_root
                            )
                            if resolved_member != resolved_source:
                                composite_ok = False
                                break
                        else:
                            registry_item = registry.get(member_id)
                            if not isinstance(registry_item, dict):
                                composite_ok = False
                                break
                            registry_sdf = registry_item.get("sdf_path")
                            if not registry_sdf or _resolve_scene_path(
                                str(registry_sdf), room_dir, scene_root
                            ) != resolved_member:
                                composite_ok = False
                                break
                            registry_metadata = registry_item.get("metadata") or {}
                            expanded_members.append(
                                import_sdf_body(
                                    resolved_member,
                                    body_name=(
                                        f"{room_id}_{object_id}_member_{member_index}"
                                    ),
                                    semantic_category=str(
                                        registry_item.get("name", "cube_block")
                                    ).lower(),
                                    pose=pose_from_scenecode(member.get("transform")),
                                    internal_model_pose=pose_from_scenecode(
                                        member.get("internal_model_pose")
                                    ),
                                    static=False,
                                    material=MATERIALS["wood_block"],
                                    metadata={
                                        "room_id": room_id,
                                        "object_id": (
                                            f"{object_id}::member_{member_index}"
                                        ),
                                        "object_type": "manipuland",
                                        "scene_description": registry_item.get(
                                            "description", ""
                                        ),
                                        "bbox_min": registry_item.get("bbox_min"),
                                        "bbox_max": registry_item.get("bbox_max"),
                                        "scenecode_scale_factor": registry_item.get(
                                            "scale_factor", 1.0
                                        ),
                                        "composite_parent": str(object_id),
                                        "composite_member_index": member_index,
                                        **registry_metadata,
                                    },
                                )
                            )
                        member_mappings.append(
                            {
                                "object_id": (
                                    member_id
                                    if source_sdf
                                    else f"{object_id}::member_{member_index}"
                                ),
                                "source_sdf": str(resolved_member),
                            }
                        )
                if composite_ok:
                    bodies.extend(expanded_members)
                    room_body_names.extend(body.name for body in expanded_members)
                    composite_object_mappings.append(
                        {
                            "room_id": str(room_id),
                            "object_id": str(object_id),
                            "composite_type": "stack",
                            "represented_by_members": member_mappings,
                        }
                    )
                    continue
                raise ValueError(
                    f"SceneCode object {room_id}/{object_id} has no sdf_path; "
                    "benchmark import forbids silently dropping declared objects"
                )
            object_type = str(item.get("object_type", "generic"))
            category = str(item.get("metadata", {}).get("category", item.get("name", object_type))).lower()
            scene_pose = pose_from_scenecode(item.get("transform"))
            body = import_sdf_body(
                _resolve_scene_path(sdf_value, room_dir, scene_root),
                body_name=f"{room_id}_{object_id}",
                semantic_category=category,
                pose=scene_pose,
                internal_model_pose=pose_from_scenecode(item.get("internal_model_pose")),
                static=object_type != "manipuland",
                metadata={
                    "room_id": room_id,
                    "object_id": object_id,
                    "object_type": object_type,
                    "scene_description": item.get("description", ""),
                    "bbox_min": item.get("bbox_min"),
                    "bbox_max": item.get("bbox_max"),
                    # SceneCode may resize generated assets during its visual
                    # placement/refinement pass and records that cumulative
                    # transform at the object-registry level.  Preserve it so
                    # the SimReady event compiler can deterministically recover
                    # the dimensions of the generated asset before placement.
                    "scenecode_scale_factor": item.get("scale_factor", 1.0),
                    "support_surfaces": item.get("support_surfaces", []),
                    "scenecode_metadata": item.get("metadata", {}),
                },
            )
            bodies.append(body)
            room_body_names.append(body.name)
        rooms.append(RoomIR(name=room_id, body_names=room_body_names))
    if not bodies:
        raise ValueError(f"no importable SDF bodies found in {state_path}")
    source_paths: set[Path] = {state_path}
    for body in bodies:
        if body.source_path:
            source_paths.add(Path(body.source_path).resolve())
        for link in body.links:
            for geometry in [*link.visuals, *link.collisions]:
                if geometry.kind == "mesh" and geometry.path:
                    mesh_path = Path(geometry.path).resolve()
                    source_paths.add(mesh_path)
                    source_paths.update(_gltf_dependencies(mesh_path))
    return WorldIR(
        scene_id=scene_id or state_path.parent.parent.name,
        prompt=" ".join(prompt_parts),
        provenance={
            "source_type": "scenecode_house_state",
            "house_state_path": str(state_path),
            "house_state_sha256": sha256_file(state_path),
            "source_file_sha256": {
                str(source): sha256_file(source) for source in sorted(source_paths)
            },
            "structural_object_mappings": structural_object_mappings,
            "composite_object_mappings": composite_object_mappings,
            # SceneCode can generate several candidates while repairing a
            # placement.  They are not silently added to the physical world,
            # but the event compiler may audit this registry and select a
            # better task-fit candidate with explicit provenance.
            "manipuland_asset_registries": manipuland_asset_registries,
        },
        rooms=rooms,
        bodies=bodies,
    )
