"""
Blender headless backend process manager.

Starts/stops a per-run Blender background process that hosts
`agent_framework/blender/headless_server.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import time

from pathlib import Path
from typing import Optional

from ..utils.gpu_diagnostics import describe_gpu_snapshot
from .request_lock import acquire_blender_request_lock


class BlenderBackendManager:
    """Manage lifecycle of a dedicated Blender headless server process."""

    def __init__(
        self,
        blender_command: str,
        host: str = "127.0.0.1",
        port: int = 9876,
        server_script: Optional[str] = None,
        startup_timeout: float = 45.0,
        stop_timeout: float = 10.0,
        poll_interval: float = 0.2,
        log_file: Optional[str] = None,
    ) -> None:
        self.blender_command = blender_command
        self.host = host
        self.port = int(port)
        self.server_script = (
            os.path.abspath(server_script)
            if server_script
            else str(Path(__file__).resolve().parent / "headless_server.py")
        )
        self.startup_timeout = float(startup_timeout)
        self.stop_timeout = float(stop_timeout)
        self.poll_interval = float(poll_interval)
        self.log_file = log_file
        self.logger = logging.getLogger("Pipeline.BlenderBackend")

        self._proc: Optional[subprocess.Popen] = None
        self._log_fp = None

    @property
    def pid(self) -> Optional[int]:
        return None if self._proc is None else self._proc.pid

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    async def start(self) -> None:
        if self.is_running():
            return

        if not self.blender_command:
            raise RuntimeError("Blender command is empty")
        if not os.path.exists(self.server_script):
            raise FileNotFoundError(f"Headless server script not found: {self.server_script}")

        stdout_target = subprocess.DEVNULL
        if self.log_file:
            log_path = os.path.abspath(self.log_file)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._log_fp = open(log_path, "a", encoding="utf-8")
            stdout_target = self._log_fp

        cmd = [
            self.blender_command,
            "--background",
            "--python",
            self.server_script,
            "--",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        with acquire_blender_request_lock("code_object.backend.start"):
            self.logger.info(
                "Launching Blender backend host=%s port=%s command=%s",
                self.host,
                self.port,
                self.blender_command,
            )
            self._proc = subprocess.Popen(
                cmd,
                stdout=stdout_target,
                stderr=subprocess.STDOUT,
            )
            self.logger.info(
                "Started Blender backend process pid=%s host=%s port=%s",
                self.pid,
                self.host,
                self.port,
            )

            try:
                await self._wait_until_ready()
            except Exception:
                self.logger.exception(
                    "Blender backend failed during startup pid=%s host=%s port=%s",
                    self.pid,
                    self.host,
                    self.port,
                )
                self.logger.info(describe_gpu_snapshot("Code_Object backend start failure"))
                raise

        self.logger.info(
            "Blender backend ready pid=%s host=%s port=%s",
            self.pid,
            self.host,
            self.port,
        )
        self.logger.info(describe_gpu_snapshot("Code_Object backend ready"))

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc is None:
                raise RuntimeError("Blender process was not created")
            exit_code = self._proc.poll()
            if exit_code is not None:
                raise RuntimeError(f"Blender process exited early with code {exit_code}")
            if self._can_connect():
                await asyncio.sleep(self.poll_interval)
                exit_code = self._proc.poll()
                if exit_code is not None:
                    raise RuntimeError(f"Blender process exited early with code {exit_code}")
                return
            await asyncio.sleep(self.poll_interval)
        await self.stop()
        raise TimeoutError(
            f"Timed out waiting for Blender backend at {self.host}:{self.port} "
            f"(timeout={self.startup_timeout}s)"
        )

    async def stop(self) -> None:
        if self._proc is None:
            self._close_log_file()
            return

        proc = self._proc
        proc_pid = proc.pid
        self.logger.info(
            "Stopping Blender backend pid=%s host=%s port=%s",
            proc_pid,
            self.host,
            self.port,
        )
        with acquire_blender_request_lock("code_object.backend.stop"):
            if proc.poll() is None:
                self._send_shutdown_command()
                deadline = time.monotonic() + self.stop_timeout
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        break
                    await asyncio.sleep(0.1)

                if proc.poll() is None:
                    proc.terminate()
                    try:
                        await asyncio.to_thread(proc.wait, 3.0)
                    except Exception:
                        pass
                if proc.poll() is None:
                    proc.kill()
                    try:
                        await asyncio.to_thread(proc.wait, 3.0)
                    except Exception:
                        pass

        return_code = proc.poll()
        self._proc = None
        self._close_log_file()
        self.logger.info(
            "Blender backend stopped pid=%s returncode=%s host=%s port=%s",
            proc_pid,
            return_code,
            self.host,
            self.port,
        )
        self.logger.info(describe_gpu_snapshot("Code_Object backend stopped"))

    def stop_sync(self) -> None:
        if self._proc is None:
            self._close_log_file()
            return

        proc = self._proc
        proc_pid = proc.pid
        self.logger.info(
            "Stopping Blender backend synchronously pid=%s host=%s port=%s",
            proc_pid,
            self.host,
            self.port,
        )
        with acquire_blender_request_lock("code_object.backend.stop"):
            if proc.poll() is None:
                self._send_shutdown_command()
                deadline = time.monotonic() + self.stop_timeout
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)

                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3.0)
                    except Exception:
                        pass
                if proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=3.0)
                    except Exception:
                        pass

        return_code = proc.poll()
        self._proc = None
        self._close_log_file()
        self.logger.info(
            "Blender backend stopped synchronously pid=%s returncode=%s host=%s port=%s",
            proc_pid,
            return_code,
            self.host,
            self.port,
        )
        self.logger.info(describe_gpu_snapshot("Code_Object backend stopped"))

    def _close_log_file(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None

    def _can_connect(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.3):
                return True
        except OSError:
            return False

    def _send_shutdown_command(self) -> None:
        payload = {"type": "shutdown_server", "params": {}}
        raw = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5) as sock:
                sock.sendall(raw)
        except OSError:
            pass
