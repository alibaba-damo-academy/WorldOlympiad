from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.robotics_articulation import resolve_articulation_binding
from autoresearch.robotics_contracts import TaskInstructionSpec
from autoresearch.robotics_task_compiler import compile_task
from autoresearch.robotics_scene_prompt_compiler import compile_task_prompt
from autoresearch.scene_prompting import assess_scene_prompt, compile_scene_prompt


def _task() -> TaskInstructionSpec:
    return TaskInstructionSpec(
        case_id="pull_drawer",
        instruction="Pull the drawer open.",
        scene_brief="A waist-height cabinet with one closed drawer and a visible handle.",
        initial_state="The cabinet is fixed and the drawer is fully closed.",
        robot="franka_robotiq_2f85",
        objects=[
            {
                "object_id": "cabinet",
                "category": "articulated_cabinet",
                "role": "support",
                "visual_description": (
                    "waist-height matte wood cabinet with one closed sliding drawer "
                    "and a large horizontal handle"
                ),
            }
        ],
        articulation_components=[
            {
                "component_id": "drawer",
                "host_object_id": "cabinet",
                "semantic_kind": "drawer",
                "joint_type": "prismatic",
                "visual_description": "the single closed drawer with a horizontal handle",
            }
        ],
        action_program=[
            {"operation": "pull", "subjects": ["drawer"], "target": "cabinet"}
        ],
        success_predicates=[
            {"kind": "drawer_open", "subject": "drawer", "object": "cabinet"}
        ],
        requested_wams=[],
    )


def _sdf(*, second_drawer: bool = False, collision: bool = True) -> str:
    other = """
    <link name="drawer_spare_link">
      <visual name="v"><geometry><box><size>0.3 0.3 0.1</size></box></geometry></visual>
      <collision name="c"><geometry><box><size>0.3 0.3 0.1</size></box></geometry></collision>
    </link>
    <joint name="drawer_spare_slide" type="prismatic">
      <parent>base_link</parent><child>drawer_spare_link</child>
      <axis><xyz>0 1 0</xyz><limit><lower>-0.2</lower><upper>0</upper></limit></axis>
    </joint>
    """ if second_drawer else ""
    child_collision = (
        '<collision name="c"><geometry><box><size>0.4 0.3 0.12</size></box></geometry></collision>'
        if collision
        else ""
    )
    return f"""<?xml version="1.0"?>
<sdf version="1.7"><model name="cabinet">
  <link name="base_link">
    <visual name="v"><geometry><box><size>0.8 0.5 0.8</size></box></geometry></visual>
    <collision name="c"><geometry><box><size>0.8 0.5 0.8</size></box></geometry></collision>
  </link>
  <link name="drawer_link">
    <pose>0 0 0.2 0 0 0</pose>
    <visual name="v"><geometry><box><size>0.4 0.3 0.12</size></box></geometry></visual>
    {child_collision}
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent>base_link</parent><child>drawer_link</child>
    <axis><xyz>0 1 0</xyz><limit><lower>-0.3</lower><upper>0</upper></limit></axis>
  </joint>
  {other}
</model></sdf>"""


def test_component_binding_preserves_official_scenecode_joint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "autoresearch.robotics_articulation._scenecode_joint_endpoints",
        lambda lower, upper: (upper, lower),
    )
    sdf = tmp_path / "cabinet.sdf"
    sdf.write_text(_sdf(), encoding="utf-8")
    output = tmp_path / "binding.json"
    certificate = resolve_articulation_binding(
        task=_task(),
        component_id="drawer",
        host_sdf=sdf,
        output=output,
    )
    assert certificate.passed
    assert certificate.joint_name == "drawer_slide"
    assert certificate.child_link == "drawer_link"
    assert certificate.axis_xyz == (0.0, 1.0, 0.0)
    assert certificate.closed_position == 0.0
    assert certificate.open_position == -0.3
    assert certificate.travel == pytest.approx(0.3)
    assert certificate.open_delta_xyz == pytest.approx((0.0, -0.3, 0.0))
    assert output.is_file()

    plan = compile_task(_task())
    assert plan["scene_generation"]["articulated_object_ids"] == ["cabinet"]
    assert plan["articulation_components"][0]["component_id"] == "drawer"
    prompt_spec = compile_task_prompt(_task())
    assert prompt_spec.objects[0].task_role == "support_target"
    prompt = compile_scene_prompt(prompt_spec)
    assert "closed sliding drawer" in prompt
    assert assess_scene_prompt(prompt_spec, prompt).passed


def test_component_binding_rejects_ambiguous_drawer_joints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "autoresearch.robotics_articulation._scenecode_joint_endpoints",
        lambda lower, upper: (upper, lower),
    )
    sdf = tmp_path / "cabinet.sdf"
    sdf.write_text(_sdf(second_drawer=True), encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_articulation_binding(
            task=_task(), component_id="drawer", host_sdf=sdf
        )


def test_component_binding_rejects_collisionless_moving_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "autoresearch.robotics_articulation._scenecode_joint_endpoints",
        lambda lower, upper: (upper, lower),
    )
    sdf = tmp_path / "cabinet.sdf"
    sdf.write_text(_sdf(collision=False), encoding="utf-8")
    with pytest.raises(ValueError, match="visual geometry but no collision"):
        resolve_articulation_binding(
            task=_task(), component_id="drawer", host_sdf=sdf
        )
