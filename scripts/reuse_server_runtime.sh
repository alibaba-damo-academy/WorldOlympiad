#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
server_root=
while (($#)); do
  case "$1" in
    --server-root) (($# >= 2)) || { echo "--server-root requires a path" >&2; exit 2; }; server_root=$2; shift 2 ;;
    -h|--help) echo 'Usage: ./scripts/reuse_server_runtime.sh --server-root EXISTING_WOV2_ROOT'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n $server_root ]] || { echo "--server-root is required" >&2; exit 2; }
server_root=$(realpath -e -- "$server_root")
[[ -d $server_root/envs && -d $server_root/runtime ]] || {
  echo "existing server environment root is incomplete: $server_root" >&2; exit 2;
}
python3 - "$ROOT" "$server_root" <<'PY'
from __future__ import annotations
import json, os, sys
from pathlib import Path
root, source = map(Path, sys.argv[1:3])
links = {
    "envs/autoresearch": "envs/autoresearch",
    "envs/worldeval": "envs/worldeval",
    "envs/scenecode": "envs/scenecode",
    "envs/robotwin": "envs/robotwin",
    "envs/lingbot": "envs/lingbot",
    "runtime/robot_assets": "runtime/robot_assets",
    "runtime/robotwin_assets": "runtime/RoboTwin/assets",
    "runtime/blender-4.4.3-linux-x64": "runtime/blender-4.4.3-linux-x64",
}
resolved = {}
for rel, source_rel in links.items():
    src = source / source_rel
    dst = root / rel
    if not src.exists():
        raise SystemExit(f"required reusable server path is missing: {src}")
    if rel.startswith("envs/") and not (src / "pyvenv.cfg").is_file():
        raise SystemExit(f"existing runtime is not a Python virtual environment: {src}")
    if rel.endswith("blender-4.4.3-linux-x64") and not (src / "blender").is_file():
        raise SystemExit(f"existing Blender runtime has no executable: {src}")
    if rel == "runtime/robotwin_assets":
        required = ("embodiments/aloha-agilex/config.yml", "objects/objaverse/list.json")
        missing = [item for item in required if not (src / item).is_file()]
        if missing:
            raise SystemExit(f"existing RoboTwin asset source is incomplete: {', '.join(missing)}")
    resolved[rel] = (dst, src.resolve())
for rel, (dst, src) in resolved.items():
    if dst.exists() or dst.is_symlink():
        if dst.resolve() != src:
            raise SystemExit(f"occupied evaluator path was left untouched: {dst}")
for rel, (dst, src) in resolved.items():
    if not (dst.exists() or dst.is_symlink()):
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst, target_is_directory=src.is_dir())
        print(f"reused {rel} -> {src}")
receipt = root / "runtime/server-runtime-reuse.json"
record = {"schema_version": 1, "server_root": str(source),
          "links": {rel: str(src) for rel, (_, src) in resolved.items()}}
if receipt.exists():
    try: existing = json.loads(receipt.read_text())
    except Exception as exc: raise SystemExit(f"runtime reuse receipt is unreadable ({type(exc).__name__}); preserved")
    if existing.get("schema_version") != 1 or existing.get("server_root") != str(source):
        raise SystemExit("runtime reuse receipt belongs to a different source root; preserved")
    old_links = existing.get("links")
    if not isinstance(old_links, dict) or any(record["links"].get(k) != v for k, v in old_links.items()):
        raise SystemExit("a recorded runtime link changed; receipt and paths were left untouched")
    merged = dict(record)
    merged["links"] = {**old_links, **record["links"]}
    if merged != existing:
        temp = receipt.with_name(f".{receipt.name}.partial.{os.getpid()}")
        if temp.exists(): raise SystemExit(f"receipt update partial exists; preserved: {temp}")
        temp.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        os.chmod(temp, 0o600)
        os.replace(temp, receipt)
else:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.chmod(receipt, 0o600)
PY
