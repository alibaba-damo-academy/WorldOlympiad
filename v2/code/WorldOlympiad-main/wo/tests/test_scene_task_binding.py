import json
import pytest

from wo.spec import TaskSpec
from wo.tasks.scene import _pinned_generation, legacy_task


def test_generated_scene_task_binds_seed_and_parameters():
    spec = TaskSpec.from_dict({
        "schema_version": "wo_task_spec.v1", "case_id": "t", "family": "pick",
        "instruction": "pick the cube", "seed": 7,
        "objects": [
            {"object_id": "cube", "category": "cube", "role": "manipuland", "description": "red cube 5 cm"},
            {"object_id": "table", "category": "table", "role": "support", "description": "table"},
        ],
        "steps": [{"primitive": "transport", "subject": "cube", "target": "table",
                    "goal": {"kind": "on_region", "subject": "cube", "target": "table"}}],
    })
    generated = legacy_task(spec)
    assert generated["case_seed"] == 7
    assert generated["case_parameters"]["layout_seed"] == 7


def test_legacy_scene_task_normalizes_readable_categories_to_safe_symbols():
    from autoresearch.robotics_contracts import TaskInstructionSpec

    spec = TaskSpec.from_dict({
        "schema_version": "wo_task_spec.v1", "case_id": "plate_case", "family": "pick_place",
        "instruction": "place apple on the serving plate",
        "objects": [
            {"object_id": "apple", "category": "apple", "role": "manipuland", "description": "small apple"},
            {"object_id": "plate", "category": "serving plate", "role": "destination", "description": "wide serving plate"},
            {"object_id": "table", "category": "table", "role": "support", "description": "table"},
        ],
        "steps": [{"primitive": "transport", "subject": "apple", "target": "plate",
                   "goal": {"kind": "on", "subject": "apple", "target": "plate"}}],
    })

    payload = legacy_task(spec)
    assert payload["objects"][1]["category"] == "serving_plate"
    assert payload["objects"][1]["visual_description"] == "wide serving plate"
    assert TaskInstructionSpec.model_validate(payload).objects[1].category == "serving_plate"


def test_certificate_project_relative_resource_ignores_worker_cwd(tmp_path, monkeypatch):
    from wo.scene import _resolve_bound_resource
    project = tmp_path / "project"
    generation = tmp_path / "published" / "case"
    resource = project / "outputs" / "asset.sdf"
    resource.parent.mkdir(parents=True)
    generation.mkdir(parents=True)
    resource.write_text("asset")
    monkeypatch.setattr("wo.scene.WORKSPACE", project)
    assert _resolve_bound_resource("outputs/asset.sdf", generation) == resource


def test_validate_source_is_fresh_admission_for_shifted_missing_and_valid_assets(tmp_path, monkeypatch):
    import trimesh
    from types import SimpleNamespace
    from wo.tasks.scene import validate_source

    mesh = tmp_path / 'box.obj'
    trimesh.creation.box(extents=[.08, .08, .05]).export(mesh)
    def make_sdf(name, collision=True, shift=0.0):
        extra = (f'<collision><pose>0 0 {shift} 0 0 0</pose><geometry><mesh><uri>{mesh.name}</uri>'
                 '</mesh></geometry></collision>') if collision else ''
        path = tmp_path / name
        path.write_text(f'<sdf><model><link><visual><geometry><mesh><uri>{mesh.name}</uri>'
                        f'</mesh></geometry></visual>{extra}</link></model></sdf>')
        return path

    generation = tmp_path / 'generation'
    generation.mkdir()
    (generation / 'task_inventory_certificate.json').write_text('{}')
    task = {
        'schema_version': 'wo_task_spec.v1', 'case_id': 'admission', 'family': 'pick_place',
        'instruction': 'move the box', 'generation': str(generation),
        'objects': [
            {'object_id': 'box', 'category': 'box', 'role': 'manipuland',
             'geometry_constraints': {'overall_dimensions_m': [.08, .08, .05]}},
            {'object_id': 'table', 'category': 'table', 'role': 'support'}],
        'steps': [{'primitive': 'transport', 'subject': 'box', 'target': 'table',
                   'goal': {'kind': 'on_region', 'subject': 'box', 'target': 'table'}}]}
    task_path = tmp_path / 'task.json'
    task_path.write_text(json.dumps(task))

    scene = lambda sdf: SimpleNamespace(
        support=SimpleNamespace(object_id='table'),
        objects={'box': SimpleNamespace(sdf_path=sdf, scale=1.0)})
    import wo.scene as scene_module
    monkeypatch.setattr(scene_module, 'resolve_scenecode',
                        lambda _spec, _generation: scene(make_sdf('valid.sdf')))
    assert validate_source(task_path) == generation.resolve()

    monkeypatch.setattr(scene_module, 'resolve_scenecode',
                        lambda _spec, _generation: scene(make_sdf('shifted.sdf', shift=.075)))
    with pytest.raises(RuntimeError, match='violates declared physical sizes'):
        validate_source(task_path)

    monkeypatch.setattr(scene_module, 'resolve_scenecode',
                        lambda _spec, _generation: scene(make_sdf('missing.sdf', collision=False)))
    with pytest.raises(RuntimeError, match='violates declared physical sizes'):
        validate_source(task_path)


def test_pinned_generation_routes_from_raw_payload_without_compilation(tmp_path, monkeypatch):
    generation = tmp_path / "certified"
    generation.mkdir()
    (generation / "task_inventory_certificate.json").write_text("{}")
    spec = TaskSpec.from_dict({
        "schema_version": "wo_task_spec.v1", "case_id": "reuse", "family": "pick",
        "instruction": "pick the cube",
        "objects": [
            {"object_id": "cube", "category": "cube", "role": "manipuland"},
            {"object_id": "table", "category": "table", "role": "support"},
        ],
        "steps": [{"primitive": "transport", "subject": "cube", "target": "table",
                    "goal": {"kind": "on_region", "subject": "cube", "target": "table"}}],
    })
    # The real resolver is the admission boundary; this test isolates routing.
    monkeypatch.setattr("wo.scene.resolve_scenecode", lambda _spec, _generation: object())
    assert _pinned_generation({"generation": str(generation)}, spec) == generation.resolve()
