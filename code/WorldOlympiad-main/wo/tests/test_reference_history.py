import json

import pytest

from wo.expert.run import archive_previous_attempt, run_case


@pytest.mark.parametrize("success", [True, False])
def test_retry_preserves_evidence_and_removes_stale_outputs(tmp_path, success):
    out = tmp_path / "reference"
    (out / "replay").mkdir(parents=True)
    (out / "certificate.json").write_text(json.dumps({"success": success}))
    (out / "trajectory.json").write_bytes(b"old trajectory")
    (out / "replay/exterior_left.mp4").write_bytes(b"old video")
    (out / "physics_sdf").mkdir()
    (out / "physics_sdf/object.urdf").write_bytes(b"cached geometry")
    history = archive_previous_attempt(out)
    assert json.loads((history / "certificate.json").read_text())["success"] is success
    assert (history / "trajectory.json").read_bytes() == b"old trajectory"
    assert (history / "replay/exterior_left.mp4").read_bytes() == b"old video"
    assert not (out / "certificate.json").exists()
    assert not (out / "replay").exists()
    assert (out / "physics_sdf/object.urdf").read_bytes() == b"cached geometry"
    assert archive_previous_attempt(out) is None
    (out / "certificate.json").write_text('{"success":false}')
    assert archive_previous_attempt(out) != history
    assert (history / "trajectory.json").exists()


def test_failure_before_simulation_cannot_leave_old_success(tmp_path):
    out = tmp_path / "reference"
    out.mkdir()
    (out / "certificate.json").write_text('{"success":true}')
    task = tmp_path / "task.json"
    task.write_text("invalid task json")
    with pytest.raises(json.JSONDecodeError):
        run_case(task, tmp_path / "generation", out)
    assert not (out / "certificate.json").exists()
    assert (out / "task_snapshot.json").read_bytes() == task.read_bytes()
    assert len(list((out / ".history").glob("*/certificate.json"))) == 1
