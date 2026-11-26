import fcntl
import os
from pathlib import Path

import pytest

from wo.agent.episode_stage import _episode_output_lock, run_bound


def _release(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def test_run_bound_rejects_duplicate_active_output_before_archiving(tmp_path):
    output = tmp_path / "case" / "lingbot_va_s0"
    output.mkdir(parents=True)
    sentinel = output / "in_progress"
    sentinel.write_text("first worker owns this attempt")
    fd = _episode_output_lock(output)
    try:
        with pytest.raises(RuntimeError, match="episode output is already active"):
            # Lock contention must be caught before reference checks, directory
            # archival, or adapter launch can touch the first worker's output.
            run_bound(Path("unused-run"), Path("unused-ref"), output, ["unused"])
        assert sentinel.read_text() == "first worker owns this attempt"
    finally:
        _release(fd)

    # A completed/failed worker releases its lock so an explicit retry can run.
    retry_fd = _episode_output_lock(output)
    _release(retry_fd)


def test_episode_output_locks_are_scoped_to_distinct_outputs(tmp_path):
    first = _episode_output_lock(tmp_path / "case-a" / "model_s0")
    try:
        second = _episode_output_lock(tmp_path / "case-b" / "model_s0")
        _release(second)
    finally:
        _release(first)
