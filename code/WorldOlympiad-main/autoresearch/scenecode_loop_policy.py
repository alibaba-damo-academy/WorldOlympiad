"""Bounded SceneCode robotics generation and deterministic Blender API repair."""
from __future__ import annotations

import ast
import os
import re

POLICY_VERSION = "robotics_scenecode_loop.v2"

COMPACT_PLAN_RULES = """
WorldOlympiad robotics reference asset policy:
Preserve the exact requested shape, color, dimensions, openings, contact surfaces,
collision topology, and independently moving components. Use the fewest parts
that represent those requirements. Do not invent hidden electronics, wiring,
screws, mounting hardware, logos or decorative subassemblies unless explicitly
requested or needed for the requested physical interaction. A plain solid block
is one part. A hollow vessel must remain hollow and its handle/openings must
remain physically usable. Do not simplify articulated joints or functional
contact geometry. Do not report missing unrequested decorative details as errors.
Represent repeated fixed parts with the same geometry and material as one
PartPlan using explicit instances. Preserve every instance's placement and
required collision shape. Keep independently moving components separate.
"""

BLENDER_API_RULES = """
Blender API requirement: WeightedNormalModifier.weight is an integer in [1, 100].
Use e.g. modifier.weight = 50, not 50.0. It is not a normalized 0..1 weight.
"""

_WEIGHT_HELPER = '''
def _wo_assign_blender_weight(value, owner):
    import bpy
    import math
    if isinstance(owner, bpy.types.WeightedNormalModifier) and isinstance(value, float):
        prop = owner.bl_rna.properties["weight"]
        if not math.isfinite(value) or not value.is_integer() or not prop.hard_min <= value <= prop.hard_max:
            raise TypeError("WeightedNormalModifier.weight requires an integer in [1, 100]; refusing lossy conversion: " + repr(value))
        value = int(value)
        print("WorldOlympiad Blender API repair: exact integer conversion for WeightedNormalModifier.weight")
    setattr(owner, "weight", value)
'''

_JOIN_HELPER = '''
def _wo_join_selected_geometry(*args, **kwargs):
    import bpy as _wog_bpy
    import json as _wog_json
    native_join = _wog_bpy.ops.object.join
    geometry_types = {"MESH", "CURVE", "SURFACE", "FONT", "META"}
    selected = list(_wog_bpy.context.selected_objects)
    geometry = [obj for obj in selected if obj.type in geometry_types]
    active = _wog_bpy.context.view_layer.objects.active
    types = [obj.type for obj in geometry]
    if len(set(types)) < 2 or active not in geometry or _wog_bpy.context.mode != "OBJECT":
        return native_join(*args, **kwargs)
    converted = []
    active_after = None
    for obj in geometry:
        _wog_bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        _wog_bpy.context.view_layer.objects.active = obj
        was_active = obj == active
        transform = obj.matrix_world.copy()
        if obj.type != "MESH":
            _wog_bpy.ops.object.convert(target="MESH")
        converted_obj = _wog_bpy.context.view_layer.objects.active
        converted_obj.matrix_world = transform
        converted.append(converted_obj)
        if was_active:
            active_after = converted_obj
    _wog_bpy.ops.object.select_all(action="DESELECT")
    for obj in converted:
        obj.select_set(True)
    _wog_bpy.context.view_layer.objects.active = active_after
    before_vertices = sum(len(obj.data.vertices) for obj in converted)
    result = native_join(*args, **kwargs)
    joined = _wog_bpy.context.view_layer.objects.active
    after_vertices = len(joined.data.vertices)
    if after_vertices != before_vertices:
        raise RuntimeError("mixed-geometry join lost vertices")
    print("GEOMETRY_JOIN_EVIDENCE:" + _wog_json.dumps({
        "schema_version": "scenecode_mixed_geometry_join.v1",
        "component_types_before": types, "component_count": len(converted),
        "vertices_before_join": before_vertices, "vertices_after_join": after_vertices,
        "world_transforms_preserved": True}))
    return result
'''

def robotics_profile_enabled() -> bool:
    return os.environ.get("SCENECODE_ROBOTICS_FAST", "0") == "1"


def bounded_attempts(configured: int, cap: int) -> int:
    value = max(1, int(configured))
    return min(value, cap) if robotics_profile_enabled() else value


def failure_fingerprint(message: str) -> str:
    message = re.sub(r"0x[0-9a-fA-F]+", "<address>", str(message))
    return re.sub(r"\s+", " ", message).strip()


def plan_error_fingerprint(issues) -> tuple:
    return tuple(sorted(
        (issue.category, failure_fingerprint(issue.description),
         tuple(sorted(issue.affected_parts or [])))
        for issue in issues if issue.severity == "error"
    ))


def normalize_blender_code(code: str) -> str:
    """Repair typed API assignments and preserve mixed renderable components.

    The actual RNA object type is checked inside Blender. RHS evaluation occurs
    before the target evaluation, as for the original assignment. The rewritten
    source is persisted, so execution and later export use the same correction.
    """
    original=code
    body,marker,footer=code.partition("\n# WORLDOLYMPIAD_PART_BOUNDS_CONTRACT")
    code=body if marker else code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return original  # The normal constructor error/fix path owns invalid Python.
    bpy_aliases={alias.asname or 'bpy' for node in ast.walk(tree) if isinstance(node,ast.Import)
                 for alias in node.names if alias.name=='bpy'}

    class WeightAssignments(ast.NodeTransformer):
        changed = False
        join_changed = False

        def visit_FunctionDef(self,node):
            if node.name=='_wo_join_selected_geometry':return node
            return self.generic_visit(node)

        def visit_Call(self,node):
            self.generic_visit(node)
            func=node.func;chain=[]
            while isinstance(func,ast.Attribute):
                chain.append(func.attr);func=func.value
            if (isinstance(func,ast.Name) and func.id in bpy_aliases
                    and list(reversed(chain))==['ops','object','join']):
                self.join_changed=True
                node.func=ast.Name(id='_wo_join_selected_geometry',ctx=ast.Load())
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) != 1:
                return node
            target = node.targets[0]
            if not isinstance(target, ast.Attribute) or target.attr != "weight":
                return node
            self.changed = True
            replacement = ast.Expr(value=ast.Call(
                func=ast.Name(id="_wo_assign_blender_weight", ctx=ast.Load()),
                args=[node.value, target.value], keywords=[],
            ))
            return ast.copy_location(replacement, node)

    transformer = WeightAssignments()
    tree = transformer.visit(tree)
    if not transformer.changed and not transformer.join_changed:
        return original
    helpers=''
    if transformer.changed and not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
               node.name == "_wo_assign_blender_weight" for node in tree.body):
        helpers+=_WEIGHT_HELPER
    if transformer.join_changed and not any(isinstance(node,ast.FunctionDef) and
            node.name=='_wo_join_selected_geometry' for node in tree.body):
        helpers+=_JOIN_HELPER
    if helpers:
        position = 0
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
            position = 1
        while position < len(tree.body) and isinstance(tree.body[position], ast.ImportFrom) and tree.body[position].module == "__future__":
            position += 1
        tree.body[position:position] = ast.parse(helpers).body
    result=ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    return result+(marker+footer if marker else '')


def asset_support_offset(asset) -> float:
    """Place the lowest geometry point on a support plane, not its origin.

    Bounds belong to raw SDF geometry. Include the declared internal model pose
    before projecting onto the object's local support normal (local +Z).
    """
    import itertools
    import math
    import numpy as np
    if asset.bbox_min is None or asset.bbox_max is None:
        raise ValueError("Support placement requires finite asset bounds")
    corners = itertools.product(*zip(asset.bbox_min, asset.bbox_max))
    heights = [float((asset.internal_model_pose @ np.asarray(corner, dtype=float))[2])
               for corner in corners]
    if not all(math.isfinite(height) for height in heights):
        raise ValueError("Support placement requires finite asset bounds")
    return -min(heights)
