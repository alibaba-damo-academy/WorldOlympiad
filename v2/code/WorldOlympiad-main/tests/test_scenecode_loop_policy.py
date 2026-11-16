from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from autoresearch.scenecode_loop_policy import bounded_attempts, normalize_blender_code

RUNTIME = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", "/tmp/SceneCode-main-worldolympiad"))
if not (RUNTIME / "scenecode/code_object/agent_framework").is_dir():
    pytest.skip("requires the staged SceneCode runtime", allow_module_level=True)
pytest.importorskip("bpy", reason="requires the actual Blender Python runtime")
sys.path.insert(0, str(RUNTIME / "scenecode/code_object"))
from agent_framework.agents.part_constructor import PartConstructorAgent
from agent_framework.agents.planner_checker import PlannerCheckerAgent, CheckIssue, CheckResult


def test_budget_profile_preserves_explicit_nonrobotics_mode(monkeypatch):
    monkeypatch.delenv("SCENECODE_ROBOTICS_FAST", raising=False)
    assert bounded_attempts(5, 2) == 5
    monkeypatch.setenv("SCENECODE_ROBOTICS_FAST", "1")
    assert bounded_attempts(5, 2) == 2
    assert PartConstructorAgent().max_retry == 2
    assert PlannerCheckerAgent().max_retry == 3


def test_actual_blender_rejects_old_weight_and_accepts_exact_repair():
    import bpy
    mesh = bpy.data.meshes.new("wo_weight_test")
    obj = bpy.data.objects.new("wo_weight_test", mesh)
    try:
        modifier = obj.modifiers.new(name="weighted", type="WEIGHTED_NORMAL")
        with pytest.raises(TypeError):
            modifier.weight = 50.0
        code = normalize_blender_code("modifier.weight = float(50)\n")
        exec(code, {"modifier": modifier})
        assert modifier.weight == 50
        assert normalize_blender_code(code) == code
        with pytest.raises(TypeError, match="refusing lossy conversion"):
            exec(normalize_blender_code("modifier.weight = 0.5"), {"modifier": modifier})
    finally:
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)


def test_non_blender_weights_and_python_evaluation_order_are_preserved():
    events = []
    obj = SimpleNamespace(weight=0)
    def value():
        events.append("rhs")
        return 0.5
    def target():
        events.append("target")
        return obj
    exec(normalize_blender_code("target().weight = value()"), {"target": target, "value": value})
    assert events == ["rhs", "target"]
    assert obj.weight == 0.5


def test_future_import_and_idempotence():
    code = '"""example"""\nfrom __future__ import annotations\nmod.weight = 50.0\n'
    fixed = normalize_blender_code(code)
    compile(fixed, "<test>", "exec")
    assert normalize_blender_code(fixed) == fixed


@pytest.mark.parametrize('alias',['bpy','geometry_bpy'])
def test_real_blender_mixed_mesh_curve_join_preserves_curve_and_world_placement(alias):
    import bpy
    from mathutils import Vector
    code=f'''
import bpy as {alias}
{alias}.ops.object.select_all(action='DESELECT')
{alias}.ops.mesh.primitive_cube_add(size=.2, location=(0,0,0))
mesh={alias}.context.object
mesh.name='wo_mixed_join_mesh'
curve_data={alias}.data.curves.new('wo_mixed_join_curve_data',type='CURVE')
curve_data.dimensions='3D'
curve_data.bevel_depth=.02
spline=curve_data.splines.new('POLY')
spline.points.add(1)
spline.points[0].co=(0,0,0,1)
spline.points[1].co=(0,0,1,1)
curve={alias}.data.objects.new('wo_mixed_join_curve',curve_data)
{alias}.context.collection.objects.link(curve)
curve.location=(2,0,0)
{alias}.ops.object.select_all(action='DESELECT')
mesh.select_set(True)
curve.select_set(True)
{alias}.context.view_layer.objects.active=mesh
{alias}.ops.object.join()
'''
    try:
        # Blender itself drops the curve from this mixed join.
        scope={};exec(code,scope)
        assert len(scope['mesh'].data.vertices)==8
        assert bpy.data.objects.get('wo_mixed_join_curve') is not None
        for obj in list(bpy.data.objects):
            if obj.name.startswith('wo_mixed_join_'):bpy.data.objects.remove(obj,do_unlink=True)
        fixed=normalize_blender_code(code)
        assert normalize_blender_code(fixed)==fixed
        exec(fixed,{})
        joined=bpy.context.object
        assert len(joined.data.vertices)>8
        assert max((joined.matrix_world@Vector(v)).x for v in joined.bound_box)>1.9
        assert bpy.data.objects.get('wo_mixed_join_curve') is None
    finally:
        for obj in list(bpy.data.objects):
            if obj.name.startswith('wo_mixed_join_'):bpy.data.objects.remove(obj,do_unlink=True)


def test_join_normalization_preserves_the_part_bounds_footer():
    from autoresearch.scenecode_part_bounds import append_part_bounds_contract,MARKER
    spec={'name':'handle','shape':{'dimensions':{'width':.2,'depth':.1,'height':.3}},
        'position':{'x':0,'y':0,'z':0},'rotation':{'x':0,'y':0,'z':0}}
    code=append_part_bounds_contract('import bpy\nbpy.ops.object.join()',spec)
    fixed=normalize_blender_code(code)
    assert fixed.count(MARKER)==1
    assert append_part_bounds_contract(fixed,spec).count(MARKER)==1


def test_constructor_fixes_known_type_before_any_llm_retry(tmp_path):
    import bpy
    mesh = bpy.data.meshes.new("wo_constructor_test")
    mesh.from_pydata([(x,y,z) for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]], [],
                     [(0,1,3,2), (4,6,7,5), (0,4,5,1), (2,3,7,6), (0,2,6,4), (1,5,7,3)])
    obj = bpy.data.objects.new("plain_block", mesh)
    bpy.context.collection.objects.link(obj)
    try:
        modifier = obj.modifiers.new(name="weighted", type="WEIGHTED_NORMAL")
        constructor = PartConstructorAgent(llm_client=object(), blender_client=object())
        constructor._generate_code_with_llm = AsyncMock(return_value="modifier.weight = 50.0")
        constructor._generate_code_fix_from_error = AsyncMock()
        async def execute(_tool, parameters):
            exec(parameters["code"], {"modifier": modifier})
            return {"ok": True}
        constructor._call_blender = execute
        spec={'name':'plain_block','shape':{'dimensions':{'width':1.,'depth':1.,'height':1.}},
            'position':{'x':0.,'y':0.,'z':0.},'rotation':{'x':0.,'y':0.,'z':0.}}
        part = SimpleNamespace(name="plain_block", instances=[], to_dict=lambda: spec)
        result = asyncio.run(constructor.construct_part(part=part, image_path=None, output_dir=str(tmp_path), text_input="a plain block"))
        assert result["execution_result"]["success"]
        constructor._generate_code_fix_from_error.assert_not_awaited()
        assert "_wo_assign_blender_weight" in Path(result["code_path"]).read_text()
    finally:
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)


def test_constructor_stops_identical_error_after_one_repair(tmp_path, monkeypatch):
    monkeypatch.delenv("SCENECODE_ROBOTICS_FAST", raising=False)
    constructor = PartConstructorAgent(llm_client=object(), blender_client=object())
    constructor._generate_code_with_llm = AsyncMock(return_value="pass")
    constructor._generate_code_fix_from_error = AsyncMock(return_value="pass")
    constructor._execute_code = AsyncMock(return_value={"success": False, "error": "same invalid socket"})
    part = SimpleNamespace(name="part", instances=[], to_dict=lambda: {"name": "part"})
    with pytest.raises(RuntimeError, match="same invalid socket"):
        asyncio.run(constructor.construct_part(part=part, image_path=None, output_dir=str(tmp_path), text_input="part"))
    assert constructor._execute_code.await_count == 2
    assert constructor._generate_code_fix_from_error.await_count == 1


@pytest.mark.parametrize("corrections", [{}, {"body": "fix"}])
def test_checker_stops_repeated_errors_and_never_revises_after_last_check(monkeypatch, corrections):
    monkeypatch.setenv("SCENECODE_ROBOTICS_FAST", "1")
    bad = CheckResult(False, 80, [CheckIssue("error", "geometry", "same missing opening", "fix opening", ["body"])], corrections, "bad")
    checker = PlannerCheckerAgent()
    checker.run = AsyncMock(return_value=bad)
    planner = SimpleNamespace(revise_plan=AsyncMock(return_value=object()))
    _, history = asyncio.run(checker.check_and_fix(plan=object(), text_input="hollow vessel", planner_agent=planner))
    assert len(history) == 2
    assert planner.revise_plan.await_count == 1

    checker.run = AsyncMock(return_value=bad)
    planner.revise_plan.reset_mock()
    _, history = asyncio.run(checker.check_and_fix(plan=object(), text_input="vessel", planner_agent=planner, max_retry=1))
    assert len(history) == 1
    planner.revise_plan.assert_not_awaited()


def test_centered_and_internal_pose_assets_rest_on_support_plane():
    import itertools
    import numpy as np
    from pydrake.all import RigidTransform, RollPitchYaw
    from autoresearch.scenecode_loop_policy import asset_support_offset
    for inner in (RigidTransform(), RigidTransform(RollPitchYaw([0.3, 0.2, 0]), [0.01, 0.02, 0.04])):
        asset = SimpleNamespace(bbox_min=np.array([-0.2, -0.15, -0.1]),
                                bbox_max=np.array([0.2, 0.15, 0.1]),
                                internal_model_pose=inner)
        support_height = 0.8
        z_offset = asset_support_offset(asset)
        transform = RigidTransform([0.3, -0.2, support_height + z_offset])
        corners = itertools.product(*zip(asset.bbox_min, asset.bbox_max))
        actual_lowest = min(float((transform @ inner @ np.asarray(c))[2]) for c in corners)
        assert actual_lowest == pytest.approx(support_height)


def test_rescale_keeps_existing_surface_contact(tmp_path):
    import numpy as np
    sys.path.insert(0, str(RUNTIME))
    from pydrake.all import RigidTransform
    from scenecode.agent_utils.room import SceneObject, ObjectType
    from scenecode.agent_utils.rescale_helpers import rescale_object_common
    sdf = tmp_path / "box.sdf"
    sdf.write_text('<sdf version="1.7"><model name="box"><link name="body"><collision name="body_collision"><geometry><box><size>0.4 0.3 0.2</size></box></geometry></collision></link></model></sdf>')
    obj = SceneObject(object_id="box", object_type=ObjectType.MANIPULAND, name="box",
                      description="a centered box", transform=RigidTransform([0.3, 0.2, 0.9]),
                      sdf_path=sdf, bbox_min=np.array([-0.2, -0.15, -0.1]),
                      bbox_max=np.array([0.2, 0.15, 0.1]),
                      placement_info=SimpleNamespace(parent_surface_id="S_0"))
    scene = SimpleNamespace(objects={"box": obj}, get_object=lambda _: obj)
    for factor in (0.5, 2.0):
        before = obj.compute_world_bounds()
        result = rescale_object_common(scene, "box", factor, "manipuland")
        assert result.success
        after = obj.compute_world_bounds()
        assert after[0][2] == pytest.approx(0.8)
        assert after[1][2] - after[0][2] == pytest.approx((before[1][2] - before[0][2]) * factor)


@pytest.mark.parametrize("severity", ["error", "warning"])
def test_checker_uses_issue_guidance_when_corrections_are_empty(severity):
    bad = CheckResult(False, 73, [CheckIssue(severity, "internal_structure",
        "Cabinet cavity is missing", "Add side walls and a bottom panel", ["cabinet_body"])], {}, "unresolved cavity")
    good = CheckResult(True, 100, [], {}, "valid cavity")
    checker = PlannerCheckerAgent()
    checker.run = AsyncMock(side_effect=[bad, good])
    original, revised = object(), object()
    planner = SimpleNamespace(revise_plan=AsyncMock(return_value=revised))
    actual, history = asyncio.run(checker.check_and_fix(
        plan=original, text_input="kitchen counter", planner_agent=planner, max_retry=3))
    assert actual is revised
    assert history == [bad, good]
    planner.revise_plan.assert_awaited_once()
    arguments = planner.revise_plan.call_args.kwargs
    assert arguments["original_plan"] is original
    assert arguments["text_input"] == "kitchen counter"
    assert "Add side walls and a bottom panel" in arguments["correction_context"]
    assert "Cabinet cavity is missing" in arguments["correction_context"]
    assert checker.run.call_args_list[-1].args[0] is revised


@pytest.mark.parametrize("has_planner, issues", [
    (True, []),
    (True, [CheckIssue("suggestion", "style", "Use warmer wood", "", [])]),
    (False, [CheckIssue("error", "geometry", "Missing cavity", "Add cavity", [])]),
])
def test_checker_does_not_retry_without_actionable_guidance_or_planner(has_planner, issues):
    bad = CheckResult(False, 50, issues, {}, "unresolved")
    checker = PlannerCheckerAgent()
    checker.run = AsyncMock(return_value=bad)
    planner = SimpleNamespace(revise_plan=AsyncMock())
    original = object()
    actual, history = asyncio.run(checker.check_and_fix(
        plan=original, text_input="counter", planner_agent=planner if has_planner else None, max_retry=3))
    assert actual is original
    assert history == [bad]
    planner.revise_plan.assert_not_awaited()
    assert checker.run.await_count == 1
