import json
import sys
import pytest
from wo.expert import run
from wo.agent import stages


def test_reference_cli_requires_success_before_render():
    command = " ".join(stages.STAGES["reference"].command({"run":"run","case":"case"},1))
    before = command.split("&&")[0]
    assert "--require-success" in before
    assert "wo.render" in command


@pytest.mark.parametrize("success", [False, True])
def test_reference_exit_status_preserves_real_certificate_result(tmp_path, monkeypatch, success):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"generation": str(tmp_path)}))
    certificate = {"case_id":"case","success":success,"frame_count":1,"duration_s":1,"wall_time_s":1,"steps":[]}
    monkeypatch.setattr(run, "run_case", lambda *a, **kw: certificate)
    monkeypatch.setattr(sys, "argv", ["expert", "--task", str(task), "--out", str(tmp_path), "--require-success"])
    if success:
        run.main()
    else:
        with pytest.raises(SystemExit) as exc:
            run.main()
        assert exc.value.code == 1
