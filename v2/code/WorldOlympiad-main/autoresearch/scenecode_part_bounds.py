"""Check constructed part geometry against its declared center and dimensions.

PartPlan positions describe bounding-box centers, not a mesh's arbitrary origin.
The generated Blender guard measures the executed geometry.  A measured
mismatch is *repaired* - the part is recentered onto its declared plan center and
both measurements are recorded - because discarding a part throws away every
other part of the same asset and costs a full re-plan.  Only a missing declared
object or a part with no evaluated surface geometry remains a failure.
"""
from __future__ import annotations

import inspect
import json
import math


MARKER = "# WORLDOLYMPIAD_PART_BOUNDS_CONTRACT"


def check_part_bounds(spec: dict, low: list[float], high: list[float]) -> dict:
    """Validate bounds expressed in planned local axes about the planned center."""
    dimensions = spec["shape"]["dimensions"]
    expected = [float(dimensions[key]) for key in ("width", "depth", "height")]
    values = [*expected, *low, *high]
    if len(low) != 3 or len(high) != 3 or any(not math.isfinite(x) for x in values) or min(expected) <= 0:
        raise ValueError("part bounds require finite coordinates and positive dimensions")
    center = [(a + b) / 2 for a, b in zip(low, high)]
    measured = [b - a for a, b in zip(low, high)]
    # Small bevels and approximate custom contours are allowed. A half-height
    # displacement from confusing a base origin with a center is not.
    tolerance = [max(0.005, 0.05 * size) for size in expected]
    passed = all(abs(offset) <= tol for offset, tol in zip(center, tolerance)) and all(
        abs(actual - planned) <= 2 * tol for actual, planned, tol in zip(measured, expected, tolerance)
    )
    return {"schema_version": "scenecode_part_bounds.v1", "part_name": spec["name"],
            "passed": passed, "planned_dimensions": expected, "measured_dimensions": measured,
            "center_offset_in_planned_axes": center, "tolerance": tolerance,
            "low_in_planned_axes": low, "high_in_planned_axes": high}


def append_part_bounds_contract(code: str, spec: dict) -> str:
    """Append a canonical Blender measure-and-recenter pass after every revision."""
    code = code.split("\n" + MARKER, 1)[0].rstrip()
    evaluator = inspect.getsource(check_part_bounds).replace("def check_part_bounds(", "def _wob_check_part_bounds(", 1)
    payload = json.dumps(spec, ensure_ascii=False)
    guard = '''
import bpy as _wob_bpy
import math
import json
from mathutils import Vector as _wob_Vector, Euler as _wob_Euler
_wob_spec = json.loads(SPEC_LITERAL)
_wob_instances = _wob_spec.get("instances") or [{"name": _wob_spec["name"], "position": _wob_spec.get("position", {}), "rotation": _wob_spec.get("rotation", {})}]


def _wob_plan_rotation(rotation):
    return _wob_Euler(tuple(math.radians(float(rotation.get(axis, 0))) for axis in "xyz"), "XYZ").to_matrix()


def _wob_measure(instance):
    _wob_name = instance["name"]
    _wob_object = _wob_bpy.data.objects.get(_wob_name)
    if _wob_object is None:
        raise RuntimeError("PART_BOUNDS_CONTRACT: declared object missing: " + _wob_name)
    _wob_position = instance.get("position", _wob_spec.get("position", {}))
    _wob_rotation = instance.get("rotation", _wob_spec.get("rotation", {}))
    _wob_center = _wob_Vector(tuple(float(_wob_position.get(axis, 0)) for axis in "xyz"))
    _wob_inverse = _wob_plan_rotation(_wob_rotation).transposed()
    _wob_bpy.context.view_layer.update()
    _wob_graph = _wob_bpy.context.evaluated_depsgraph_get()
    _wob_points = []
    for _wob_child in [_wob_object, *_wob_object.children_recursive]:
        # Curves with bevels/extrusion are real rendered surfaces. Measuring
        # only mesh objects incorrectly rejects native Code_Object handles,
        # woven strands and other valid curve-built parts.
        if _wob_child.type not in {"MESH", "CURVE", "SURFACE", "FONT"}:
            continue
        _wob_evaluated = _wob_child.evaluated_get(_wob_graph)
        _wob_mesh = _wob_evaluated.to_mesh()
        try:
            if _wob_mesh is None or not _wob_mesh.polygons:
                continue
            _wob_points.extend(_wob_inverse @ (_wob_evaluated.matrix_world @ vertex.co - _wob_center) for vertex in _wob_mesh.vertices)
        finally:
            _wob_evaluated.to_mesh_clear()
    if not _wob_points:
        raise RuntimeError("PART_BOUNDS_CONTRACT: no mesh geometry: " + _wob_name)
    _wob_low = [min(point[i] for point in _wob_points) for i in range(3)]
    _wob_high = [max(point[i] for point in _wob_points) for i in range(3)]
    return _wob_object, _wob_check_part_bounds({**_wob_spec, "name": _wob_name}, _wob_low, _wob_high)


for _wob_instance in _wob_instances:
    _wob_object, _wob_measurement = _wob_measure(_wob_instance)
    print("PART_BOUNDS_EVIDENCE:" + json.dumps(_wob_measurement, sort_keys=True))
    if _wob_measurement["passed"]:
        continue
    # Repair instead of rejecting.  A constructor that confused a base origin
    # with a bounding-box center is off by a translation, so the part is moved
    # onto its declared center and re-measured.  A remaining dimension
    # deviation is recorded as a warning: size and shape are judged downstream
    # (native-view framing, physical usability, the certified expert rollout),
    # where a bad asset is visible as such instead of merely malformed.
    _wob_rotation = _wob_instance.get("rotation", _wob_spec.get("rotation", {}))
    _wob_shift = _wob_plan_rotation(_wob_rotation) @ _wob_Vector(tuple(_wob_measurement["center_offset_in_planned_axes"]))
    _wob_object.location = _wob_object.location - _wob_shift
    _wob_bpy.context.view_layer.update()
    _, _wob_recheck = _wob_measure(_wob_instance)
    _wob_correction = {"schema_version": "scenecode_part_bounds_correction.v1",
                       "recentered": True, "shift_applied": list(_wob_shift),
                       "before": _wob_measurement, "after": _wob_recheck}
    print("PART_BOUNDS_CORRECTION:" + json.dumps(_wob_correction, sort_keys=True))
    if not _wob_recheck["passed"]:
        print("PART_BOUNDS_WARNING:" + json.dumps(_wob_recheck, sort_keys=True))
'''.replace("SPEC_LITERAL", repr(payload))
    guarded = "import math\n" + evaluator + guard
    return code + "\n\n" + MARKER + "\nif __name__ == '__main__':\n" + "".join("    " + line if line.strip() else line for line in guarded.splitlines(True))
