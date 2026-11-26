"""Shared Blender request lock adapter for Code_Object.

SceneCode owns the canonical implementation, including owner sidecar and
heartbeat diagnostics. Code_Object can also run standalone, so this module
falls back to the same fcntl lock path when SceneCode is not importable.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import socket

from collections.abc import Iterator
from pathlib import Path

try:
    from scenecode.agent_utils.blender.request_lock import (
        acquire_blender_request_lock as _scenecode_acquire_blender_request_lock,
    )
except Exception:  # pragma: no cover - exercised only in standalone environments.
    _scenecode_acquire_blender_request_lock = None


console_logger = logging.getLogger(__name__)

LOCK_ENV_VAR = "SCENECODE_BLENDER_GLOBAL_LOCK"
DEFAULT_LOCK_PATH = Path("/tmp/scenecode_blender_locks/blender_requests.lock")


def get_blender_request_lock_path() -> Path:
    lock_path = os.environ.get(LOCK_ENV_VAR)
    if lock_path:
        return Path(lock_path).expanduser()
    return DEFAULT_LOCK_PATH


@contextlib.contextmanager
def _fallback_acquire_blender_request_lock(purpose: str) -> Iterator[Path]:
    lock_path = get_blender_request_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        stat_result = lock_path.stat()
        console_logger.info(
            "Waiting for Blender request lock (%s): %s pid=%s hostname=%s st_dev=%s st_ino=%s",
            purpose,
            lock_path,
            os.getpid(),
            socket.gethostname(),
            stat_result.st_dev,
            stat_result.st_ino,
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            console_logger.info(
                "Acquired Blender request lock (%s): %s pid=%s hostname=%s st_dev=%s st_ino=%s",
                purpose,
                lock_path,
                os.getpid(),
                socket.gethostname(),
                stat_result.st_dev,
                stat_result.st_ino,
            )
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            console_logger.info(
                "Released Blender request lock (%s): %s pid=%s hostname=%s st_dev=%s st_ino=%s",
                purpose,
                lock_path,
                os.getpid(),
                socket.gethostname(),
                stat_result.st_dev,
                stat_result.st_ino,
            )


def acquire_blender_request_lock(purpose: str) -> contextlib.AbstractContextManager[Path]:
    if _scenecode_acquire_blender_request_lock is not None:
        return _scenecode_acquire_blender_request_lock(purpose)
    return _fallback_acquire_blender_request_lock(purpose)
