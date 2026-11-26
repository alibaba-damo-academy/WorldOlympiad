from __future__ import annotations

import sys

from pathlib import Path


def _load_shared_helpers():
    try:
        from scenecode.agent_utils.gpu_diagnostics import (
            collect_gpu_snapshot,
            describe_gpu_snapshot,
            format_gpu_snapshot,
        )
        return collect_gpu_snapshot, describe_gpu_snapshot, format_gpu_snapshot
    except ImportError:
        workspace_root = Path(__file__).resolve().parents[3]
        code_scene_root = workspace_root / "SceneCode"
        if code_scene_root.exists() and str(code_scene_root) not in sys.path:
            sys.path.insert(0, str(code_scene_root))

        try:
            from scenecode.agent_utils.gpu_diagnostics import (
                collect_gpu_snapshot,
                describe_gpu_snapshot,
                format_gpu_snapshot,
            )
            return collect_gpu_snapshot, describe_gpu_snapshot, format_gpu_snapshot
        except Exception as exc:
            import_error = f"Failed to import shared GPU diagnostics helper: {exc}"

            def collect_gpu_snapshot(label: str) -> dict[str, object]:
                return {
                    "label": label,
                    "pid": None,
                    "cuda_visible_devices": "<unavailable>",
                    "backend": "unavailable",
                    "devices": [],
                    "processes": [],
                    "error": import_error,
                }

            def format_gpu_snapshot(snapshot: dict[str, object]) -> str:
                return (
                    f"GPU snapshot [{snapshot['label']}] backend={snapshot['backend']} "
                    f"reason: {snapshot.get('error', 'unknown')}"
                )

            def describe_gpu_snapshot(label: str) -> str:
                return format_gpu_snapshot(collect_gpu_snapshot(label))

            return collect_gpu_snapshot, describe_gpu_snapshot, format_gpu_snapshot


collect_gpu_snapshot, describe_gpu_snapshot, format_gpu_snapshot = _load_shared_helpers()

__all__ = [
    "collect_gpu_snapshot",
    "describe_gpu_snapshot",
    "format_gpu_snapshot",
]
