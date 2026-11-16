from __future__ import annotations

from autoresearch.robotics_resource_lock import exclusive_blender_job


def test_blender_job_lock_is_scoped_under_output_root(tmp_path) -> None:
    with exclusive_blender_job(tmp_path) as lock_path:
        assert lock_path == tmp_path / ".locks" / "scenecode_blender_job.lock"
        assert lock_path.is_file()
        assert lock_path.stat().st_mode & 0o777 == 0o600
