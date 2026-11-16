"""Render an expert reference or native model trajectory with baseline cameras.

Relative outputs retain the historical REF/<out> layout. Absolute output paths
allow episode views to be rendered without writing into the frozen reference.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .egl import configure_egl_vendor
from .render_plan import build_plan, VIEWS
from .scene import WORKSPACE

REPO = WORKSPACE / "code/WorldOlympiad-main"
SCENECODE = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", WORKSPACE / "runtime/scenecode"))
BLENDER_PY = WORKSPACE / "envs/scenecode/bin/python3.11"
SETTINGS = REPO / "deployment/ziplab5090/scenecode_robotics_render_settings.py"


def _render_environment() -> dict[str, str]:
    env = dict(os.environ, PYTHONPATH=str(SCENECODE))
    if not env.get("WO_FFMPEG"):
        import imageio_ffmpeg
        env["WO_FFMPEG"] = imageio_ffmpeg.get_ffmpeg_exe()
    configure_egl_vendor(env)
    return env


def render(ref: Path, *, trajectory: str | Path = "trajectory.json",
           out: str | Path = "replay", views=tuple(VIEWS), gpu_id: int | None = None) -> Path:
    """Prefer NVIDIA EGL when configured, otherwise use host vendor discovery.

    gpu_id is retained for caller compatibility. EGL selects its own device.
    """
    ref = ref.resolve()
    out_path = Path(out)
    target = out_path if out_path.is_absolute() else ref / out_path
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    plan_path = target.with_name(target.name + "_plan.json")
    log_path = target.with_name(target.name + ".log")
    traj_path = Path(trajectory)
    traj_name = str(traj_path.resolve()) if traj_path.is_absolute() else str(traj_path)
    plan = build_plan(ref, views=views, traj_name=traj_name)
    plan_path.write_text(json.dumps(plan))
    env = _render_environment()

    with log_path.open("w") as log:
        subprocess.run([str(BLENDER_PY), str(REPO / "wo/blender_replay.py"), str(plan_path),
                        str(target), str(SETTINGS)], cwd=SCENECODE, env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    missing = [v for v in plan["views"]
               if not (target / f"{v}.mp4").is_file() or not (target / f"{v}.mp4").stat().st_size]
    if missing:
        raise RuntimeError(f"render produced no video for {missing}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, default=Path("trajectory.json"))
    ap.add_argument("--out", type=Path, default=Path("replay"))
    ap.add_argument("--views", nargs="+", choices=[*VIEWS, "showcase"], default=list(VIEWS))
    ap.add_argument("--gpu-id", type=int)
    a = ap.parse_args()
    print(render(a.ref, trajectory=a.trajectory, out=a.out, views=tuple(a.views), gpu_id=a.gpu_id))


if __name__ == "__main__":
    main()
