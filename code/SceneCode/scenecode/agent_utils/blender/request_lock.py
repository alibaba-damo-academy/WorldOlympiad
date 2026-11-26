import contextlib
import fcntl
import json
import logging
import os
import socket
import threading
import time
import uuid

from collections.abc import Iterator
from pathlib import Path

console_logger = logging.getLogger(__name__)

LOCK_ENV_VAR = "SCENECODE_BLENDER_GLOBAL_LOCK"
DEFAULT_LOCK_PATH = Path("/tmp/scenecode_blender_locks/blender_requests.lock")
HEARTBEAT_INTERVAL_SECONDS = 5.0


def get_blender_request_lock_path() -> Path:
    """Return the shared lock path for node-local Blender request serialization."""
    lock_path = os.environ.get(LOCK_ENV_VAR)
    if lock_path:
        return Path(lock_path).expanduser()
    return DEFAULT_LOCK_PATH


def get_blender_request_lock_owner_path(lock_path: Path | None = None) -> Path:
    """Return the sidecar owner path for a Blender request lock."""
    resolved_lock_path = lock_path if lock_path is not None else get_blender_request_lock_path()
    return resolved_lock_path.with_name(f"{resolved_lock_path.name}.owner.json")


def _lock_metadata(lock_path: Path) -> dict:
    stat_result = lock_path.stat()
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "lock_path": str(lock_path),
        "st_dev": stat_result.st_dev,
        "st_ino": stat_result.st_ino,
    }


def _write_owner_file(owner_path: Path, owner_data: dict) -> None:
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = owner_path.with_name(
        f"{owner_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temp_path.write_text(json.dumps(owner_data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, owner_path)


def _read_owner_file(owner_path: Path) -> dict | None:
    try:
        return json.loads(owner_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        console_logger.warning("Failed to read Blender request lock owner file %s: %s", owner_path, exc)
        return None


def _start_owner_heartbeat(
    *,
    owner_path: Path,
    owner_data: dict,
    stop_event: threading.Event,
) -> threading.Thread:
    def heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
            heartbeat_data = dict(owner_data)
            heartbeat_data["heartbeat_at"] = time.time()
            try:
                _write_owner_file(owner_path, heartbeat_data)
            except Exception as exc:
                console_logger.warning(
                    "Failed to update Blender request lock heartbeat %s: %s",
                    owner_path,
                    exc,
                )

    thread = threading.Thread(
        target=heartbeat,
        name="blender-request-lock-heartbeat",
        daemon=True,
    )
    thread.start()
    return thread


@contextlib.contextmanager
def acquire_blender_request_lock(purpose: str) -> Iterator[Path]:
    """Acquire the shared Blender request lock for a single heavy request."""
    lock_path = get_blender_request_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    metadata = _lock_metadata(lock_path)

    console_logger.info(
        "Waiting for Blender request lock (%s): %s pid=%s hostname=%s st_dev=%s st_ino=%s",
        purpose,
        lock_path,
        metadata["pid"],
        metadata["hostname"],
        metadata["st_dev"],
        metadata["st_ino"],
    )
    lock_acquired = False
    owner_path = get_blender_request_lock_owner_path(lock_path)
    owner_token = uuid.uuid4().hex
    owner_data = {
        **metadata,
        "purpose": purpose,
        "owner_token": owner_token,
        "acquired_at": None,
        "heartbeat_at": None,
    }
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None

    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        lock_acquired = True
        acquired_at = time.time()
        owner_data["acquired_at"] = acquired_at
        owner_data["heartbeat_at"] = acquired_at
        _write_owner_file(owner_path, owner_data)
        heartbeat_thread = _start_owner_heartbeat(
            owner_path=owner_path,
            owner_data=owner_data,
            stop_event=heartbeat_stop,
        )
        console_logger.info(
            "Acquired Blender request lock (%s): %s pid=%s hostname=%s st_dev=%s st_ino=%s owner=%s",
            purpose,
            lock_path,
            metadata["pid"],
            metadata["hostname"],
            metadata["st_dev"],
            metadata["st_ino"],
            owner_path,
        )
        yield lock_path
    finally:
        try:
            if lock_acquired:
                heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=1.0)
                current_owner = _read_owner_file(owner_path)
                if current_owner is None:
                    console_logger.warning(
                        "Blender request lock owner file disappeared before release: %s",
                        owner_path,
                    )
                elif current_owner.get("owner_token") == owner_token:
                    try:
                        owner_path.unlink()
                    except FileNotFoundError:
                        pass
                else:
                    console_logger.warning(
                        "Blender request lock owner file was overwritten before release: %s",
                        owner_path,
                    )
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                console_logger.info(
                    "Released Blender request lock (%s): %s pid=%s hostname=%s st_dev=%s st_ino=%s",
                    purpose,
                    lock_path,
                    metadata["pid"],
                    metadata["hostname"],
                    metadata["st_dev"],
                    metadata["st_ino"],
                )
        finally:
            lock_file.close()
