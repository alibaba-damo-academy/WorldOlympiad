"""Preserve authored geometry through image-conditioned Code_Object generation."""
import json
import math
import os
import shutil
import time
import hashlib
from pathlib import Path

SCENE_MAX_CHARS = 1600
DESCRIPTION_MAX_CHARS = 400
CONTEXT_MAX_CHARS = 1600
PROMPT_MAX_CHARS = 40000
PROMPT_MAX_BYTES = 80000


def check_text_limit(text, *, field, max_chars):
    if not isinstance(text, str):
        raise ValueError(f"{field} must be text")
    if len(text) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters; simplify the authored input before retrying")
    return text


def _asset_owner(task, asset_name):
    if not isinstance(asset_name, str) or not asset_name.strip():
        raise ValueError("typed asset requirements need a canonical asset name or explicit alias")
    aliases = {}
    manifest_name = os.environ.get("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST")
    if manifest_name:
        manifest = Path(manifest_name)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != os.environ.get("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256"):
            raise ValueError("object reference manifest changed before geometry projection")
        payload = json.loads(manifest.read_text())
        if payload.get("case_id") != task.get("case_id"):
            raise ValueError("object reference manifest belongs to a different task")
        for item in payload.get("items", []):
            aliases.setdefault(item["object_id"], []).extend(item.get("aliases", []))
    matches = []
    for obj in task.get("objects", []):
        names = [obj["object_id"], *obj.get("aliases", []), *aliases.get(obj["object_id"], [])]
        if asset_name.casefold() in [name.casefold() for name in names]:
            matches.append(obj)
    if len(matches) != 1:
        raise ValueError("canonical asset identity is absent or ambiguous; dimensions are not an identity selector")
    return matches[0]


def _compatible_geometry(task, owner_id):
    objects = {obj["object_id"]: obj for obj in task.get("objects", [])}
    peers = set()
    for predicate in task.get("success_predicates", []):
        subject, target = predicate.get("subject"), predicate.get("object")
        if owner_id == subject and target in objects:
            peers.add(target)
        elif owner_id == target and subject in objects:
            peers.add(subject)
    keys = {"overall_dimensions_m", "clear_inner_opening_m", "cavity"}
    return [{key: value for key, value in objects[peer].get("geometry_constraints", {}).items() if key in keys}
            for peer in sorted(peers)]


def object_requirements(description="", dimensions=None, *, asset_name=None):
    """Project only the identified asset; the complete typed ENV remains unchanged."""
    authored_task = os.environ.get("SCENECODE_AUTHORED_TASK_REQUIREMENTS", "")
    geometry = {}
    parts = []
    if authored_task:
        task = json.loads(authored_task)
        owner = _asset_owner(task, asset_name)
        description = owner["visual_description"]
        geometry = owner.get("geometry_constraints", {})
        expected = geometry.get("overall_dimensions_m")
        if dimensions is not None and expected is not None and list(dimensions) != list(expected):
            raise ValueError("asset dimensions differ from its identified geometry contract")
        dimensions = expected if expected is not None else dimensions
        parts.append("Selected asset: " + owner["object_id"])
        components = []
        for component in task.get("articulation_components", []):
            if component.get("host_object_id") != owner["object_id"]:
                continue
            if component.get("visual_description"):
                check_text_limit(component["visual_description"], field="component visual_description", max_chars=DESCRIPTION_MAX_CHARS)
            components.append({key: component[key] for key in
                               ("component_id", "visual_description", "joint_type", "semantic_kind", "geometry_constraints")
                               if key in component})
        compatibility = _compatible_geometry(task, owner["object_id"])
    else:
        components, compatibility = [], []
    check_text_limit(description, field="visual_description", max_chars=DESCRIPTION_MAX_CHARS)
    if description:
        parts.append(description)
    if geometry:
        parts.append("geometry_constraints=" + json.dumps(geometry, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if components:
        parts.append("components=" + json.dumps(components, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if compatibility:
        parts.append("compatible_geometry=" + json.dumps(compatibility, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if dimensions is not None:
        dims = [float(x) for x in dimensions]
        if len(dims) != 3 or not all(math.isfinite(x) and x > 0 for x in dims):
            raise ValueError("object dimensions must be three finite positive metre values")
        if not geometry:
            parts.append("overall_dimensions_m=" + json.dumps(dims))
        factor = max(dims)
        parts.append(f"One model unit is {factor:g} m; divide metre lengths by {factor:g}.")
    return check_text_limit("\n".join(parts), field="own asset context", max_chars=CONTEXT_MAX_CHARS)


def prepare_request_context(output_dir, context):
    """Do not resume code made without the current authored requirements."""
    check_text_limit(context, field="own asset context", max_chars=CONTEXT_MAX_CHARS)
    if not context:
        return
    out = Path(output_dir)
    path = out / "request_context.txt"
    previous = path.read_text() if path.exists() else ""
    if (out / "ObjectPlan.json").exists() and previous != context:
        history = out / ".context_history" / str(time.time_ns())
        history.mkdir(parents=True)
        for item in list(out.iterdir()):
            if item.name != ".context_history":
                shutil.move(str(item), str(history / item.name))
    path.write_text(context)


def measure_messages(messages, *, tools=None, tool_choice=None):
    """Measure the canonical text envelope, excluding image data and image URLs."""
    images = []
    def text_only(value):
        if isinstance(value, dict):
            if value.get("type") in {"image_url", "input_image", "image_file"}:
                image = value.get("image_url", {})
                url = image.get("url", "") if isinstance(image, dict) else image
                images.append({"type": value["type"], "reference_characters": len(url) if isinstance(url, str) else 0})
                return {"type": value["type"]}
            return {key: text_only(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [text_only(item) for item in value]
        return value
    envelope = {"messages": text_only(messages)}
    if tools is not None:
        envelope["tools"] = text_only(tools)
    if tool_choice is not None:
        envelope["tool_choice"] = text_only(tool_choice)
    text = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return {"characters": len(text), "utf8_bytes": len(text.encode("utf-8")), "images": images,
            "measurement": "canonical text-only JSON envelope; image data excluded", "token_count": None}


def contextual_messages(messages, context, *, tools=None, tool_choice=None):
    check_text_limit(context, field="own asset context", max_chars=CONTEXT_MAX_CHARS)
    result = [*messages, {"role": "user", "content": context}] if context else messages
    size = measure_messages(result, tools=tools, tool_choice=tool_choice)
    if size["characters"] > PROMPT_MAX_CHARS or size["utf8_bytes"] > PROMPT_MAX_BYTES:
        raise ValueError(f"assembled prompt exceeds hard text budget ({size['characters']} characters, {size['utf8_bytes']} UTF-8 bytes); simplify before retrying")
    return result
