#!/usr/bin/env python3
"""Check or explicitly download RoboTwin embodiment/object data into this workspace."""
from __future__ import annotations
import argparse
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/"runtime/robotwin_assets"
REQUIRED=("embodiments/aloha-agilex/config.yml","objects/objaverse/list.json")
ARCHIVES=("embodiments.zip","objects.zip")

def stop(message: str, code: int=2) -> None:
    raise SystemExit(f"RoboTwin assets blocked: {message}")

def check(path: Path) -> None:
    missing=[rel for rel in REQUIRED if not (path/rel).is_file()]
    if missing: stop("missing " + ", ".join(missing) + ". Run the explicit asset download command after obtaining Hugging Face access, or reuse an existing verified asset tree.")
    print(f"PASS: RoboTwin asset paths are present at {path.resolve()}")

def safe_extract(archive: Path, root: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            name=PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                stop(f"unsafe member in {archive.name}: {item.filename}")
            mode=(item.external_attr >> 16) & 0o170000
            if mode==stat.S_IFLNK: stop(f"symbolic link is not allowed in {archive.name}: {item.filename}")
            dest=root.joinpath(*name.parts)
            if not dest.resolve(strict=False).is_relative_to(root.resolve()):
                stop(f"archive member escapes destination: {item.filename}")
            if item.is_dir(): dest.mkdir(parents=True,exist_ok=True);continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with zf.open(item) as src,dest.open("wb") as out: shutil.copyfileobj(src,out,1024*1024)
            perm=(item.external_attr >> 16) & 0o777
            if perm: os.chmod(dest,perm)

def download() -> None:
    if TARGET.exists() or TARGET.is_symlink(): stop(f"destination is occupied and will not be overwritten: {TARGET}")
    python=ROOT/"envs/robotwin/bin/python"
    downloader=ROOT/"code/RoboTwin/assets/_download.py"
    if not python.is_file() or not downloader.is_file(): stop("RoboTwin runtime or pinned source downloader is missing; prepare those first")
    staging=ROOT/"runtime"/f".robotwin_assets.partial.{time.time_ns()}"
    staging.mkdir(mode=0o700)
    code=("from huggingface_hub import snapshot_download; "
          "snapshot_download(repo_id='TianxingChen/RoboTwin2.0', "
          "allow_patterns=['embodiments.zip','objects.zip'], local_dir='.', "
          "repo_type='dataset', resume_download=True)")
    env=os.environ.copy();env.setdefault("HF_HUB_VERBOSITY","warning")
    result=subprocess.run([str(python),"-c",code],cwd=staging,env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False)
    if result.returncode:
        stop(f"Hugging Face download failed with exit {result.returncode}; confirm dataset access/login and inspect the preserved partial directory: {staging}. No automatic retry was made")
    total=0
    for name in ARCHIVES:
        archive=staging/name
        if not archive.is_file(): stop(f"download did not produce {name}; partial preserved at {staging}")
        try:
            with zipfile.ZipFile(archive) as zf: total+=sum(info.file_size for info in zf.infolist())
        except Exception as exc: stop(f"{name} is not a valid ZIP ({type(exc).__name__}); partial preserved at {staging}")
    if shutil.disk_usage(staging).free < total + 512*1024*1024:
        stop(f"insufficient free disk space to extract {total} bytes; partial preserved at {staging}")
    for name in ARCHIVES: safe_extract(staging/name,staging)
    missing=[rel for rel in REQUIRED if not (staging/rel).is_file()]
    if missing: stop("downloaded archives lack " + ", ".join(missing) + f"; partial preserved at {staging}")
    if TARGET.exists() or TARGET.is_symlink(): stop(f"destination appeared during download; partial preserved at {staging}")
    os.rename(staging,TARGET)
    check(TARGET)

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--check-only",action="store_true",help="validate required data without downloading")
    group.add_argument("--download",action="store_true",help="explicitly fetch the two required data archives")
    a=p.parse_args()
    if a.download: download()
    else: check(TARGET)
    return 0
if __name__=="__main__": sys.exit(main())
