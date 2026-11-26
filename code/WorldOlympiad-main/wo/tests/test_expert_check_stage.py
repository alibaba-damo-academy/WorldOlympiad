import json
import pytest
from wo.agent import stages


def test_expert_control_does_not_render_or_overwrite_formal_reference():
    stage = stages.STAGES["expert_check"]
    command = " ".join(stage.command({"run":"run","case":"case"},1))
    assert "wo.expert.run" in command
    assert "wo.render" not in command
    assert "/expert_checks/case" in command
    assert "/references/case" not in command
    assert stage.gpu_gb == 0


def test_expert_control_still_requires_real_success(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "RUNS", tmp_path)
    folder = tmp_path / "run/expert_checks/case"
    folder.mkdir(parents=True)
    cert = folder / "certificate.json"
    cert.write_text(json.dumps({"success": False, "steps": [{"stage":"verify","detail":"not inserted"}]}))
    with pytest.raises(RuntimeError, match="not inserted"):
        stages.STAGES["expert_check"].collect({"run":"run","case":"case"})
    cert.write_text(json.dumps({"success": True,"frame_count":5}))
    assert stages.STAGES["expert_check"].collect({"run":"run","case":"case"})["control_only"] is True
