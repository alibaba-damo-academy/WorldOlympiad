from pathlib import Path

import pytest

from wo.eval import reference_binding
from wo.models import scene_check
from wo.agent import stages


def test_scene_check_rejects_stale_reference_before_plan_or_output_change(tmp_path, monkeypatch):
    out = tmp_path / "out"; out.mkdir(); (out / "old_result").write_text("preserve")
    def reject(ref, task):
        raise ValueError("reference version mismatch")
    monkeypatch.setattr(reference_binding, "verify_task", reject)
    monkeypatch.setattr(scene_check.rt, "build_plan", lambda *a, **kw: pytest.fail("invalid reference reached scene adapter"))
    with pytest.raises(ValueError, match="reference version mismatch"):
        scene_check.run(tmp_path / "ref", out, tmp_path / "task.json")
    assert (out / "old_result").read_text() == "preserve"


def test_scene_check_valid_reference_continues_existing_plan_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(reference_binding, "verify_task", lambda r,t: calls.append((r,t)))
    def stop_at_plan(*a, **kw):
        raise RuntimeError("plan reached")
    monkeypatch.setattr(scene_check.rt, "build_plan", stop_at_plan)
    ref, task = tmp_path / "ref", tmp_path / "task.json"
    with pytest.raises(RuntimeError, match="plan reached"):
        scene_check.run(ref, tmp_path / "out", task)
    assert calls == [(ref, task)]


def test_registered_scene_check_passes_authoritative_task_path():
    args = {"run":"run", "case":"case"}
    command = stages.STAGES["scene_check"].build(args, 1)
    assert command[command.index("--task")+1] == stages.run_dir("run") / "tasks/case.json"
