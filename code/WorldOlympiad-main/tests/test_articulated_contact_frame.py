import json
from pathlib import Path
import pytest
from test_robotics_articulation import _task, _sdf
from autoresearch.robotics_articulation import resolve_articulation_binding
from autoresearch.robotics_scene_prompt_compiler import compile_task_prompt
from autoresearch.scene_prompting import compile_scene_prompt

def test_component_geometry_reaches_host_prompt():
    task=_task()
    task.articulation_components[0].geometry_constraints={
        "handle_orientation":"horizontal", "handle_mount_count":2,
        "handle_clearance_m":.09,
    }
    prompt=compile_scene_prompt(compile_task_prompt(task))
    assert "handle mounting post count: 2" in prompt
    assert "clearance between handle bar and panel: 0.09" in prompt

def test_binding_preserves_declared_contact_frame(tmp_path,monkeypatch):
    monkeypatch.setattr("autoresearch.robotics_articulation._scenecode_joint_endpoints",
                        lambda lo,hi:(hi,lo))
    task=_task()
    task.articulation_components[0].geometry_constraints={
        "contact_target_local_xyz":[0,.1,0],
        "contact_normal_local_xyz":[0,-2,0],
        "handle_axis_local_xyz":[1,0,0],
    }
    p=tmp_path/"cabinet.sdf";p.write_text(_sdf())
    result=resolve_articulation_binding(task=task,component_id="drawer",host_sdf=p)
    assert result.contact_target_local_xyz==(0,.1,0)
    assert result.contact_normal_local_xyz==(0,-1,0)
    assert result.handle_axis_local_xyz==(1,0,0)

def test_binding_rejects_partial_contact_frame(tmp_path,monkeypatch):
    monkeypatch.setattr("autoresearch.robotics_articulation._scenecode_joint_endpoints",
                        lambda lo,hi:(hi,lo))
    task=_task();task.articulation_components[0].geometry_constraints={"contact_target_local_xyz":[0,.1,0]}
    p=tmp_path/"cabinet.sdf";p.write_text(_sdf())
    with pytest.raises(ValueError,match="three complete"):
        resolve_articulation_binding(task=task,component_id="drawer",host_sdf=p)
