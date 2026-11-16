#!/usr/bin/env python3
"""Convert DMD file welding configs.

This script converts Drake Model Directive (DMD) files between different welding
configs. It supports three modes:

- nothing: Only wall/ceiling-mounted objects welded (furniture FREE, composites FREE)
- furniture: Furniture welded, composites FREE
- all: Everything welded (furniture + manipulands)

The script requires house_state.json metadata to determine object types. It will
fail with an error if the metadata is not found or if a model is not found in
the metadata (no fallback heuristics).

Example usage:
    python scripts/convert_dmd_welding.py combined_house/house.dmd.yaml -m furniture
    python scripts/convert_dmd_welding.py house.dmd.yaml -m nothing -o house_free.dmd.yaml
"""

import argparse
import copy
import json
import logging
import math

from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
console_logger = logging.getLogger(__name__)

# Object types that are always welded (regardless of mode).
ALWAYS_WELDED_TYPES = {"wall_mounted", "ceiling_mounted"}

# Object types that are free in all modes.
ALWAYS_FREE_TYPES = {"manipuland"}

# Asset sources that are always welded (regardless of mode).
# Thin coverings (rugs, carpets, tablecloths) have no collision geometry,
# so they must remain welded to avoid unrealistic physics behavior.
ALWAYS_WELDED_ASSET_SOURCES = {"thin_covering"}


def _angle_axis_to_matrix(angle_deg: float, axis: list[float]) -> np.ndarray:
    """Convert angle-axis rotation to a 3x3 rotation matrix."""
    angle_rad = math.radians(angle_deg)
    ax = np.array(axis, dtype=float)
    norm = np.linalg.norm(ax)
    if norm < 1e-12:
        return np.eye(3)
    ax = ax / norm
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def _matrix_to_angle_axis(
    r: np.ndarray,
) -> tuple[float, list[float]]:
    """Convert a 3x3 rotation matrix to angle-axis (degrees, unit axis)."""
    trace_val = float(np.trace(r))
    cos_angle = max(-1.0, min(1.0, (trace_val - 1.0) / 2.0))
    angle_rad = math.acos(cos_angle)
    if abs(angle_rad) < 1e-10:
        return 0.0, [0.0, 0.0, 1.0]
    sin_angle = math.sin(angle_rad)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (
        2.0 * sin_angle
    )
    return math.degrees(angle_rad), axis.tolist()


def _extract_translation_and_rotation(
    pose: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract translation vector and rotation matrix from a DMD pose dict."""
    t = np.array(pose.get("translation", [0, 0, 0]), dtype=float)
    rot_data = pose.get("rotation")
    if rot_data and "!AngleAxis" in rot_data:
        aa = rot_data["!AngleAxis"]
        r = _angle_axis_to_matrix(aa["angle_deg"], aa["axis"])
    else:
        r = np.eye(3)
    return t, r


def _compose_poses(parent_pose: dict, child_pose: dict) -> dict:
    """Compose two DMD poses: T_world_child = T_world_parent * T_parent_child."""
    pt, pr = _extract_translation_and_rotation(parent_pose)
    ct, cr = _extract_translation_and_rotation(child_pose)
    world_t = pt + pr @ ct
    world_r = pr @ cr
    angle_deg, axis = _matrix_to_angle_axis(world_r)
    return {
        "translation": [float(world_t[0]), float(world_t[1]), float(world_t[2])],
        "rotation": {
            "!AngleAxis": {
                "angle_deg": angle_deg,
                "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
            }
        },
    }


def load_house_state(state_path: Path) -> dict:
    """Load and parse house_state.json.

    Args:
        state_path: Path to house_state.json.

    Returns:
        Parsed house state dictionary.

    Raises:
        FileNotFoundError: If house_state.json does not exist.
    """
    if not state_path.exists():
        raise FileNotFoundError(
            f"house_state.json not found at {state_path}. "
            "This script requires scenecode metadata to determine object types."
        )
    with open(state_path) as f:
        return json.load(f)


def build_object_registry(house_state: dict) -> dict[str, dict]:
    """Build a registry mapping model names to object metadata.

    Model names in DMD files follow the pattern: {room_id}_{object_id}
    For example: hallway_console_table_0, bedroom_2_bed_0

    Args:
        house_state: Parsed house_state.json dictionary.

    Returns:
        Dictionary mapping model names to object metadata including object_type,
        room_id, object_id, and whether it's a composite member.
    """
    registry = {}

    rooms = house_state.get("rooms", {})
    for room_id, room_data in rooms.items():
        objects = room_data.get("objects", {})
        for object_id, obj in objects.items():
            # Skip wall objects (they're not in DMD, they're in room_geometry SDF).
            if obj.get("object_type") == "wall":
                continue

            # Main object model name.
            model_name = f"{room_id}_{object_id}"
            metadata = obj.get("metadata", {})
            registry[model_name] = {
                "object_type": obj.get("object_type"),
                "asset_source": metadata.get("asset_source"),
                "room_id": room_id,
                "object_id": object_id,
                "is_composite_member": False,
                "parent_model_name": None,
            }

            # If object has member_assets (composite), register those too.
            member_assets = obj.get("member_assets", [])
            for i, member in enumerate(member_assets):
                member_model_name = f"{room_id}_{object_id}_member_{i}"
                registry[member_model_name] = {
                    "object_type": obj.get("object_type"),
                    "asset_source": metadata.get("asset_source"),
                    "room_id": room_id,
                    "object_id": object_id,
                    "is_composite_member": True,
                    "parent_model_name": model_name,
                    "member_index": i,
                }

    return registry


def parse_dmd_yaml(dmd_path: Path) -> list[dict]:
    """Parse DMD YAML file into a list of directives.

    Args:
        dmd_path: Path to the DMD YAML file.

    Returns:
        List of directive dictionaries.

    Raises:
        FileNotFoundError: If the DMD file does not exist.
    """
    if not dmd_path.exists():
        raise FileNotFoundError(f"DMD file not found: {dmd_path}")

    with open(dmd_path) as f:
        content = f.read()

    # Parse YAML with custom tag handling for !AngleAxis.
    def angle_axis_constructor(loader, node):
        return {"!AngleAxis": loader.construct_mapping(node)}

    yaml.SafeLoader.add_constructor("!AngleAxis", angle_axis_constructor)
    data = yaml.safe_load(content)

    return data.get("directives", [])


def write_dmd_yaml(directives: list[dict], output_path: Path) -> None:
    """Write directives to DMD YAML file.

    Args:
        directives: List of directive dictionaries.
        output_path: Path to write the output file.
    """

    # Custom representer for AngleAxis.
    def angle_axis_representer(dumper, data):
        if "!AngleAxis" in data:
            return dumper.represent_mapping(
                "!AngleAxis", data["!AngleAxis"], flow_style=False
            )
        return dumper.represent_dict(data)

    yaml.SafeDumper.add_representer(dict, angle_axis_representer)

    output = {"directives": directives}

    with open(output_path, "w") as f:
        yaml.safe_dump(
            output,
            f,
            default_flow_style=None,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        )


def should_be_welded(
    model_name: str, object_registry: dict[str, dict], mode: str
) -> bool:
    """Determine if a model should be welded based on mode and object type.

    Args:
        model_name: The model name from the DMD file.
        object_registry: Registry mapping model names to object metadata.
        mode: Welding mode ('nothing', 'furniture', or 'all').

    Returns:
        True if the model should be welded to world.

    Raises:
        ValueError: If model_name is not found in registry.
    """
    # Room geometry and frames are handled separately (always welded to frame).
    if model_name.startswith("room_geometry_") or model_name.endswith("_frame"):
        return True

    if model_name not in object_registry:
        raise ValueError(
            f"Model '{model_name}' not found in house_state.json metadata. "
            "This indicates a mismatch between the DMD file and metadata."
        )

    obj_info = object_registry[model_name]
    obj_type = obj_info["object_type"]
    asset_source = obj_info.get("asset_source")

    # Always-welded types (wall_mounted, ceiling_mounted).
    if obj_type in ALWAYS_WELDED_TYPES:
        return True

    # Always-welded asset sources (thin coverings have no collision geometry).
    if asset_source in ALWAYS_WELDED_ASSET_SOURCES:
        return True

    # Manipulands are free except in 'all' mode where everything is welded.
    if obj_type in ALWAYS_FREE_TYPES:
        if mode == "all":
            return True
        return False

    # Furniture.
    if obj_type == "furniture":
        if mode == "nothing":
            return False
        # modes 'furniture' and 'all' weld furniture.
        return True

    # Unknown type - fail fast.
    raise ValueError(f"Unknown object type '{obj_type}' for model '{model_name}'.")


def extract_weld_pose(weld_directive: dict) -> dict:
    """Extract pose from an add_weld directive.

    Args:
        weld_directive: The add_weld directive dictionary.

    Returns:
        Pose dictionary with translation and rotation.

    Raises:
        ValueError: If the weld directive has unexpected structure.
    """
    if "add_weld" not in weld_directive:
        raise ValueError("Expected add_weld directive")

    x_pc = weld_directive["add_weld"].get("X_PC")
    if not x_pc:
        raise ValueError("add_weld missing X_PC pose")

    return x_pc


def _resolve_link_name(add_model_directive: dict, link_name: str | None) -> str | None:
    """Resolve the link name from a free body pose.

    If link_name is provided, uses it directly. Otherwise discovers the
    first (and typically only) link from default_free_body_pose.
    """
    if link_name is not None:
        return link_name
    model_data = add_model_directive.get("add_model", {})
    free_body_pose = model_data.get("default_free_body_pose", {})
    if not free_body_pose:
        return None
    return next(iter(free_body_pose))


def extract_free_body_pose(add_model_directive: dict, link_name: str | None = None):
    """Extract pose from a free body's default_free_body_pose.

    Strips the ``base_frame`` key so the returned dict contains only pose
    data (translation/rotation) suitable for use as ``X_PC``.

    Args:
        add_model_directive: The add_model directive dictionary.
        link_name: The link name to extract pose for. If None, uses the
            first link found.

    Returns:
        Pose dictionary with translation and rotation, or None if not found.
    """
    resolved = _resolve_link_name(add_model_directive, link_name)
    if resolved is None:
        return None
    model_data = add_model_directive.get("add_model", {})
    free_body_pose = model_data.get("default_free_body_pose", {})
    pose = free_body_pose.get(resolved)
    if pose is None:
        return None
    # Strip base_frame so it doesn't leak into X_PC.
    pose = {k: v for k, v in pose.items() if k != "base_frame"}
    return pose


def extract_base_frame(
    add_model_directive: dict, link_name: str | None = None
) -> str | None:
    """Extract base_frame from a free body's default_free_body_pose.

    Args:
        add_model_directive: The add_model directive dictionary.
        link_name: The link name to look up. If None, uses the first link.

    Returns:
        The base_frame string, or None if not present.
    """
    resolved = _resolve_link_name(add_model_directive, link_name)
    if resolved is None:
        return None
    model_data = add_model_directive.get("add_model", {})
    free_body_pose = model_data.get("default_free_body_pose", {})
    pose = free_body_pose.get(resolved)
    if pose is None:
        return None
    return pose.get("base_frame")


def convert_welded_to_free(
    add_model: dict, weld: dict, link_name: str = "base_link"
) -> dict:
    """Convert a welded model to a free body.

    Preserves the weld's parent frame as ``base_frame`` in the free body pose
    when the parent is not ``"world"``.

    Args:
        add_model: The add_model directive.
        weld: The corresponding add_weld directive.
        link_name: The link name for the free body pose.

    Returns:
        New add_model directive with default_free_body_pose.
    """
    new_model = copy.deepcopy(add_model)
    pose = extract_weld_pose(weld)
    parent_frame = get_weld_parent_frame(weld)

    free_pose = dict(pose)
    if parent_frame != "world":
        free_pose["base_frame"] = parent_frame

    new_model["add_model"]["default_free_body_pose"] = {link_name: free_pose}

    return new_model


def convert_free_to_welded(
    add_model: dict, link_name: str | None = None
) -> tuple[dict, dict]:
    """Convert a free body to a welded model.

    Uses the ``base_frame`` from the free body pose as the weld parent
    (falls back to ``"world"`` if absent).

    Args:
        add_model: The add_model directive with default_free_body_pose.
        link_name: The link name to extract pose from. If None,
            auto-discovers from the free body pose.

    Returns:
        Tuple of (new add_model without pose, new add_weld directive).

    Raises:
        ValueError: If no free body pose found.
    """
    resolved_link = _resolve_link_name(add_model, link_name)
    base_frame = extract_base_frame(add_model, resolved_link)
    pose = extract_free_body_pose(add_model, resolved_link)
    if pose is None:
        raise ValueError(
            f"No default_free_body_pose for {resolved_link} in model "
            f"{add_model.get('add_model', {}).get('name', 'unknown')}"
        )

    parent = base_frame if base_frame else "world"

    new_model = copy.deepcopy(add_model)
    del new_model["add_model"]["default_free_body_pose"]

    model_name = add_model["add_model"]["name"]
    weld = {
        "add_weld": {
            "parent": parent,
            "child": f"{model_name}::{resolved_link}",
            "X_PC": pose,
        }
    }

    return new_model, weld


def is_frame_weld(weld: dict) -> bool:
    """Check if a weld is to a frame (world or named room frame).

    Frame welds have parents like "world" or "room_bedroom_frame".
    Object welds have parents like "model_name::link_name".
    The distinguisher is whether the parent contains "::".
    """
    parent = weld.get("add_weld", {}).get("parent", "")
    return "::" not in parent


def get_weld_parent_frame(weld: dict) -> str:
    """Extract the parent frame name from a frame weld."""
    return weld.get("add_weld", {}).get("parent", "world")


def get_weld_model_name(weld: dict) -> str | None:
    """Extract model name from a weld's child field."""
    child = weld.get("add_weld", {}).get("child", "")
    if "::" in child:
        return child.split("::")[0]
    return None


def get_weld_child_link_name(weld: dict) -> str:
    """Extract link name from a weld's child field."""
    child = weld.get("add_weld", {}).get("child", "")
    if "::" in child:
        return child.split("::")[1]
    return "base_link"


def _should_free_composites(mode: str) -> bool:
    """Return True if composite members should be free bodies in this mode."""
    return mode in ("nothing", "furniture")


def _get_parent_model_name(weld: dict) -> str | None:
    """Extract parent model name from a non-world weld's parent field.

    Parent field looks like 'model_name::link_name'.
    """
    parent = weld.get("add_weld", {}).get("parent", "")
    if "::" in parent:
        return parent.split("::")[0]
    return None


def _build_non_world_weld_index(
    directives: list[dict],
) -> dict[str, dict]:
    """Build index of non-world welds keyed by child model name."""
    index: dict[str, dict] = {}
    for directive in directives:
        if "add_weld" in directive and not is_frame_weld(directive):
            child_name = get_weld_model_name(directive)
            if child_name:
                index[child_name] = directive
    return index


def convert_dmd(
    directives: list[dict], object_registry: dict[str, dict], mode: str
) -> list[dict]:
    """Convert DMD directives to the specified welding mode.

    Args:
        directives: List of DMD directive dictionaries.
        object_registry: Registry mapping model names to object metadata.
        mode: Target welding mode ('nothing', 'furniture', or 'all').

    Returns:
        New list of directives with converted welding.
    """
    result = []
    free_composites = _should_free_composites(mode)

    # Build index of non-world welds (composite member welds) by child name.
    non_world_welds = _build_non_world_weld_index(directives)

    # Set of child model names that have non-world welds (composite members).
    composite_children: set[str] = set(non_world_welds.keys())

    # Build index of world welds by model name for easy lookup.
    weld_by_model: dict[str, dict] = {}
    for directive in directives:
        if "add_weld" in directive and is_frame_weld(directive):
            model_name = get_weld_model_name(directive)
            if model_name:
                weld_by_model[model_name] = directive

    # Track which models we've seen to detect duplicates.
    processed_models: set[str] = set()

    # Track world poses for models (needed for composing composite poses).
    model_world_poses: dict[str, dict] = {}

    # Track parent frame for each model (for round-tripping frame welds).
    model_parent_frames: dict[str, str] = {}

    i = 0
    while i < len(directives):
        directive = directives[i]

        if "add_model" in directive:
            model_name = directive["add_model"]["name"]

            # Skip room geometry and frames (pass through unchanged).
            if model_name.startswith("room_geometry_"):
                result.append(directive)
                i += 1
                continue

            # Handle composite child models not in registry.
            if model_name not in object_registry:
                if free_composites and model_name in composite_children:
                    # This model is a composite member that should be free.
                    # Check if next directive is a non-world weld for it.
                    has_nw_weld = (
                        i + 1 < len(directives)
                        and "add_weld" in directives[i + 1]
                        and not is_frame_weld(directives[i + 1])
                        and get_weld_model_name(directives[i + 1]) == model_name
                    )
                    if has_nw_weld:
                        weld = directives[i + 1]
                        link = get_weld_child_link_name(weld)
                        parent_name = _get_parent_model_name(weld)
                        parent_pose = model_world_poses.get(parent_name)
                        if parent_pose is not None:
                            rel_pose = extract_weld_pose(weld)
                            world_pose = _compose_poses(parent_pose, rel_pose)
                            # Inherit base_frame from parent so Drake
                            # interprets the composed pose in the correct
                            # coordinate frame (e.g. room frame).
                            parent_frame = model_parent_frames.get(parent_name)
                            if parent_frame and parent_frame != "world":
                                world_pose["base_frame"] = parent_frame
                            new_model = copy.deepcopy(directive)
                            new_model["add_model"]["default_free_body_pose"] = {
                                link: world_pose
                            }
                            result.append(new_model)
                            model_world_poses[model_name] = {
                                k: v for k, v in world_pose.items() if k != "base_frame"
                            }
                            if parent_frame:
                                model_parent_frames[model_name] = parent_frame
                            i += 2  # Skip the weld.
                            continue
                        else:
                            console_logger.warning(
                                f"Could not find parent pose for "
                                f"'{parent_name}' (child: '{model_name}'). "
                                f"Keeping weld unchanged."
                            )

                # In "all" mode, weld unregistered models too.
                if mode == "all":
                    has_free_pose = "default_free_body_pose" in directive["add_model"]
                    has_next_world_weld = (
                        i + 1 < len(directives)
                        and "add_weld" in directives[i + 1]
                        and is_frame_weld(directives[i + 1])
                        and get_weld_model_name(directives[i + 1]) == model_name
                    )
                    has_next_nw_weld = (
                        i + 1 < len(directives)
                        and "add_weld" in directives[i + 1]
                        and not is_frame_weld(directives[i + 1])
                        and get_weld_model_name(directives[i + 1]) == model_name
                    )

                    if has_next_world_weld:
                        # Already welded to world, keep as-is.
                        result.append(directive)
                        weld_pose = extract_weld_pose(directives[i + 1])
                        model_world_poses[model_name] = weld_pose
                        result.append(directives[i + 1])
                        i += 2
                        continue
                    elif has_free_pose:
                        # Convert free to welded.
                        new_model, new_weld = convert_free_to_welded(directive)
                        result.append(new_model)
                        result.append(new_weld)
                        model_world_poses[model_name] = extract_free_body_pose(
                            directive
                        )
                        i += 1
                        continue
                    elif has_next_nw_weld:
                        # Composite child with non-world weld. Compose
                        # pose with parent to create world weld.
                        weld = directives[i + 1]
                        link = get_weld_child_link_name(weld)
                        parent_name = _get_parent_model_name(weld)
                        parent_pose = model_world_poses.get(parent_name)
                        if parent_pose is not None:
                            rel_pose = extract_weld_pose(weld)
                            world_pose = _compose_poses(parent_pose, rel_pose)
                            new_model = copy.deepcopy(directive)
                            if "default_free_body_pose" in (new_model["add_model"]):
                                del new_model["add_model"]["default_free_body_pose"]
                            new_weld = {
                                "add_weld": {
                                    "parent": "world",
                                    "child": f"{model_name}::{link}",
                                    "X_PC": world_pose,
                                }
                            }
                            result.append(new_model)
                            result.append(new_weld)
                            model_world_poses[model_name] = world_pose
                            i += 2
                            continue
                        else:
                            console_logger.warning(
                                f"Could not find parent pose for "
                                f"'{parent_name}' "
                                f"(child: '{model_name}'). "
                                f"Keeping weld unchanged."
                            )

                # Pass through unchanged (not in registry, not a
                # composite child we need to modify).
                result.append(directive)
                # Track pose if it has one (composite base objects).
                free_pose = extract_free_body_pose(directive)
                if free_pose:
                    model_world_poses[model_name] = free_pose
                i += 1
                continue

            processed_models.add(model_name)
            target_welded = should_be_welded(model_name, object_registry, mode)
            has_free_pose = "default_free_body_pose" in directive["add_model"]

            # Check if next directive is a weld for this model.
            has_weld = (
                i + 1 < len(directives)
                and "add_weld" in directives[i + 1]
                and is_frame_weld(directives[i + 1])
                and get_weld_model_name(directives[i + 1]) == model_name
            )

            if target_welded:
                if has_weld:
                    # Already welded, keep as-is.
                    result.append(directive)
                    result.append(directives[i + 1])
                    weld_pose = extract_weld_pose(directives[i + 1])
                    model_world_poses[model_name] = weld_pose
                    parent_frame = get_weld_parent_frame(directives[i + 1])
                    model_parent_frames[model_name] = parent_frame
                    i += 2
                elif has_free_pose:
                    # Convert free to welded.
                    base_frame = extract_base_frame(directive)
                    new_model, new_weld = convert_free_to_welded(directive)
                    result.append(new_model)
                    result.append(new_weld)
                    model_world_poses[model_name] = extract_free_body_pose(directive)
                    if base_frame:
                        model_parent_frames[model_name] = base_frame
                    i += 1
                else:
                    raise ValueError(
                        f"Model '{model_name}' should be welded but has no "
                        "pose information (no weld and no "
                        "default_free_body_pose)."
                    )
            else:
                # Should be free.
                if has_free_pose:
                    # Already free, keep as-is.
                    result.append(directive)
                    model_world_poses[model_name] = extract_free_body_pose(directive)
                    base_frame = extract_base_frame(directive)
                    if base_frame:
                        model_parent_frames[model_name] = base_frame
                    i += 1
                elif has_weld:
                    # Convert welded to free.
                    link = get_weld_child_link_name(directives[i + 1])
                    new_model = convert_welded_to_free(
                        directive, directives[i + 1], link_name=link
                    )
                    result.append(new_model)
                    weld_pose = extract_weld_pose(directives[i + 1])
                    model_world_poses[model_name] = weld_pose
                    parent_frame = get_weld_parent_frame(directives[i + 1])
                    model_parent_frames[model_name] = parent_frame
                    i += 2  # Skip the weld directive.
                else:
                    raise ValueError(
                        f"Model '{model_name}' should be free but has no "
                        "pose information (no weld and no "
                        "default_free_body_pose)."
                    )

        elif "add_weld" in directive:
            # Welds to world are handled with their add_model above.
            if is_frame_weld(directive):
                model_name = get_weld_model_name(directive)
                # Skip if already processed with add_model.
                if model_name and model_name in processed_models:
                    i += 1
                    continue
            else:
                # Non-world weld (composite member).
                child_name = get_weld_model_name(directive)
                if free_composites and child_name in composite_children:
                    # Already handled when processing the child's add_model
                    # (converted to free body pose). But if the add_model
                    # wasn't adjacent, we might reach here. Skip it.
                    if child_name in model_world_poses:
                        i += 1
                        continue
            result.append(directive)
            i += 1

        else:
            # Pass through other directives (add_frame, etc.).
            result.append(directive)
            i += 1

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert DMD file welding configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Path to input DMD YAML file")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["nothing", "furniture", "all"],
        default="nothing",
        help="Welding mode (default: nothing)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: INPUT with _<mode> suffix)",
    )
    parser.add_argument(
        "--scene-state",
        type=Path,
        default=None,
        help="Path to house_state.json (auto-detected if in same dir)",
    )

    args = parser.parse_args()

    # Determine house_state.json path.
    if args.scene_state:
        state_path = args.scene_state
    else:
        # Auto-detect: look in same directory as input.
        state_path = args.input.parent / "house_state.json"

    # Determine output path.
    if args.output:
        output_path = args.output
    else:
        stem = args.input.stem
        if stem.endswith(f"_{args.mode}"):
            # Already has suffix, don't add another.
            output_path = args.input.parent / f"{stem}.dmd.yaml"
        else:
            # Remove existing mode suffix if present.
            for mode_suffix in ["_nothing", "_furniture", "_all"]:
                if stem.endswith(mode_suffix):
                    stem = stem[: -len(mode_suffix)]
                    break
            output_path = args.input.parent / f"{stem}_{args.mode}.dmd.yaml"

    console_logger.info(f"Loading house state from {state_path}")
    house_state = load_house_state(state_path)

    console_logger.info("Building object registry from metadata")
    object_registry = build_object_registry(house_state)
    console_logger.info(f"Found {len(object_registry)} objects in registry")

    console_logger.info(f"Parsing DMD file {args.input}")
    directives = parse_dmd_yaml(args.input)
    console_logger.info(f"Parsed {len(directives)} directives")

    console_logger.info(f"Converting to mode '{args.mode}'")
    converted = convert_dmd(directives, object_registry, args.mode)
    console_logger.info(f"Converted to {len(converted)} directives")

    console_logger.info(f"Writing output to {output_path}")
    write_dmd_yaml(converted, output_path)

    console_logger.info("Done!")


if __name__ == "__main__":
    main()
