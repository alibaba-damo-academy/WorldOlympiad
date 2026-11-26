"""Run a stock RoboTwin policy control through the worker, outside matrix rows."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from wo.models.lingbot_va_bridge import policy_check_receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lingbot_va", "fastwam", "longwam"], required=True)
    ap.add_argument("--stock-task", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("WO_TESTED_MODEL_GPU", "0")))
    ap.add_argument("--simulator-gpu", type=int,
                    default=int(os.environ.get("WO_SIMULATOR_GPU",
                                               os.environ.get("WO_TESTED_MODEL_GPU", "0"))))
    ap.add_argument("--prompt")
    a = ap.parse_args()
    if a.out.exists():
        history = a.out.parent / ".policy_check_history" / a.out.name
        history.mkdir(parents=True, exist_ok=True)
        shutil.move(str(a.out), str(history / str(time.time_ns())))
    a.out.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "wo.models." + a.model,
                    "--stock-task", a.stock_task, "--out", str(a.out),
                    "--window-s", str(a.window_s), "--seed", str(a.seed),
                    "--port", str(a.port),
                    "--gpu", str(a.gpu), "--simulator-gpu", str(a.simulator_gpu),
                    *(["--prompt", a.prompt] if a.prompt else [])], check=True)
    meta = json.loads((a.out / "native_run/capture/capture_metadata.json").read_text())
    receipt = policy_check_receipt(a.stock_task, a.model, meta)
    (a.out / "policy_check.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
