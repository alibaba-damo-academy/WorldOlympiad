"""Opt-in reuse of an interrupted Code_Object asset, with exact input/part matching."""
import json
import logging
import os
from pathlib import Path
import shutil

logger = logging.getLogger("CodeObjectResume")


def _source(text_input):
    value = os.getenv("SCENECODE_CODE_OBJECT_RECOVERY_DIR")
    if not value or not text_input:
        return None
    root = Path(value)
    if (root / "input_text.txt").read_text().strip() != text_input.strip():
        return None
    return root


def seed_plan(output_dir, text_input):
    root = _source(text_input)
    dest = Path(output_dir) / "ObjectPlan.json"
    if root is None or dest.exists():
        return
    plan = json.loads((root / "ObjectPlan.json").read_text())
    if not plan.get("parts"):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "ObjectPlan.json", dest)
    logger.info("Restored interrupted Code_Object plan from %s; plan checking remains enabled", root)


def seed_part(part, text_input, code_path):
    root = _source(text_input)
    dest = Path(code_path)
    if root is None or dest.exists():
        return
    plan = json.loads((root / "ObjectPlan.json").read_text())
    matching = [p for p in plan["parts"] if p == part.to_dict()]
    if len(matching) != 1:
        return
    candidates = list((root / "code").glob(f"parts_*/{part.name}.py"))
    if len(candidates) != 1:
        return
    source = candidates[0]
    trace_path = source.with_name(f"{part.name}_trace.json")
    if not trace_path.is_file():
        return
    trace = json.loads(trace_path.read_text())
    if not trace.get("actions") or trace["actions"][-1].get("result") != "success" or trace["actions"][-1].get("action_type") != "execute_code":
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    shutil.copy2(trace_path, dest.with_name(trace_path.name))
    logger.info("Reused completed Code_Object part: %s from %s", part.name, source)
