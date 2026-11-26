from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "deployment/ziplab5090/patches/scenecode_backend_startup_timeout_cleanup.patch"
STAGER = ROOT / "deployment/ziplab5090/stage_scenecode_runtime.sh"


def test_timeout_cleanup_patch_applies_and_avoids_reentrant_lock_acquisition(tmp_path):
    source = tmp_path / "scenecode/code_object/agent_framework/blender/backend_manager.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import asyncio\n\n"
        "class Backend:\n"
        "    async def _wait_until_ready(self):\n"
        "        while True:\n"
        "            if self._can_connect():\n"
        "                await asyncio.sleep(self.poll_interval)\n"
        "                exit_code = self._proc.poll()\n"
        "                if exit_code is not None:\n"
        "                    raise RuntimeError('early exit')\n"
        "                return\n"
        "            await asyncio.sleep(self.poll_interval)\n"
        "        await self.stop()\n"
        "        raise TimeoutError(\n"
        "            f\"Timed out waiting for Blender backend at {self.host}:{self.port} \"\n"
        "            f\"(timeout={self.startup_timeout}s)\"\n"
        "        )\n"
    )

    completed = subprocess.run(
        ["patch", "-p1", "-i", str(PATCH)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    patched = source.read_text()
    compile(patched, str(source), "exec")
    method = patched.split("    async def _wait_until_ready", 1)[1]
    method = method.split("    async def stop", 1)[0]
    assert "await self.stop()" not in method
    assert "proc.terminate()" in method
    assert "proc.kill()" in method
    assert "self._close_log_file()" in method
    assert "raise TimeoutError(" in method


def test_runtime_stager_applies_and_verifies_timeout_cleanup_patch():
    text = STAGER.read_text()
    assert 'patches/scenecode_backend_startup_timeout_cleanup.patch' in text
    assert "scenecode_backend_startup_timeout_cleanup.patch)" in text
    assert "start() still owns the shared Blender request lock here" in text
    assert "scenecode_backend_startup_timeout_cleanup'" in text
