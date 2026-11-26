"""Validate shared RoboTwin scene, native cameras and predicates without policy inference."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from . import lingbot_va as rt


def audit_renderer(out: Path) -> dict:
    log = (out / 'logs/client.log').read_text(errors='replace')
    errors = [line for line in log.splitlines() if '[svulkan2]' in line and '[error]' in line]
    config = out / 'native_run/renderer_config.json'
    renderer = {'status':'failed' if errors or not config.is_file() else 'ready',
                'settings':json.loads(config.read_text()) if config.is_file() else None,
                'errors':errors}
    (out / 'renderer_audit.json').write_text(json.dumps(renderer, indent=2))
    if renderer['status'] != 'ready':
        raise RuntimeError(f"native renderer check failed; see {out / 'renderer_audit.json'}")
    return renderer


def run(ref: Path, out: Path, task: Path | None = None):
    if task is not None:
        from ..eval.reference_binding import verify_task
        verify_task(ref, task)
        # Scene checks are an admission boundary too: re-measure the bound
        # source instead of accepting a copied export-time report.
        from ..tasks.scene import validate_source
        validate_source(task)
    if out.exists():
        history = out.parent / ".scene_check_history" / out.name
        history.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out),str(history / str(time.time_ns())))
    (out / "logs").mkdir(parents=True)
    gpu = int(os.environ.get("WO_TESTED_MODEL_GPU", "0"))
    native = rt.run_scene_check(ref, out, gpu=gpu, seed=0)
    required = native.get("required_visible", [])
    framing = {"outside": native.get("missing_objects", []),
               "visible_pixels": native.get("visible_pixels", {}),
               "camera_shapes": native.get("camera_shapes", {})}
    audit = {"status": "ready" if native.get("status") == "ready" else "failed",
             "required_visible": required, "source": "native_actor_segmentation"}
    result = {"scene_ready": audit["status"] == "ready" and not framing["outside"],
              "renderer": native.get("renderer"), "input_audit": audit,
              "reference": str(ref), "required_visible": required,
              "framing": framing, "native_state": native.get("native_objects"),
              "support_representation": "reference-bound SAPIEN mesh actors"}
    (out / "scene_ready.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--task",type=Path)
    args = parser.parse_args();run(args.ref,args.out,args.task)


if __name__ == "__main__":main()
