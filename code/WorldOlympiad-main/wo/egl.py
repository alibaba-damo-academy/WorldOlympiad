"""Optional host EGL vendor selection for renderer subprocesses."""
from __future__ import annotations

import os
import sys
from pathlib import Path

NVIDIA_EGL = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
EGL_VENDOR_ENV = "WOV2_EGL_VENDOR_JSON"


def configure_egl_vendor(env: dict[str, str]) -> None:
    configured = env.get(EGL_VENDOR_ENV)
    path = Path(configured).expanduser() if configured else NVIDIA_EGL
    if path.is_file() and os.access(path, os.R_OK):
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(path.resolve())
        return
    if configured:
        print(f"Warning: {EGL_VENDOR_ENV} is unavailable at {path}; using host EGL discovery.",
              file=sys.stderr)
