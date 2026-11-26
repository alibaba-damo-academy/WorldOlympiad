#!/usr/bin/env python3
"""Verify and materialize the immutable DROID robot asset bundle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'assets.lock.json'

def stop(message: str) -> None:
    raise SystemExit(f'robot asset setup blocked: {message}')

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def entries(base: Path) -> list[dict]:
    result=[]
    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(),key=lambda p:p.name):
            rel=child.relative_to(base).as_posix(); st=child.lstat(); mode=stat.S_IMODE(st.st_mode)
            if stat.S_ISDIR(st.st_mode): visit(child)
            elif stat.S_ISREG(st.st_mode):
                data=child.read_bytes(); result.append({'path':rel,'type':'file','mode':mode,'size':len(data),'sha256':digest(data)})
            elif stat.S_ISLNK(st.st_mode): result.append({'path':rel,'type':'symlink','mode':mode,'target':os.readlink(child)})
            else: stop(f'unsupported special asset: {child}')
    visit(base)
    return sorted(result,key=lambda item:item['path'])

def canonical(value: object) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()

def verify_bundle(record: dict, base: Path) -> None:
    if base.is_symlink() or not base.is_dir(): stop(f'asset bundle is missing or redirected: {base}')
    manifest_path=ROOT/record['manifest']
    try:
        raw=manifest_path.read_bytes(); manifest=json.loads(raw)
    except Exception as exc: stop(f'asset file manifest is unreadable ({type(exc).__name__})')
    if digest(raw)!=record['manifest_sha256']: stop('asset file-manifest SHA-256 differs from assets.lock.json')
    expected=manifest.get('entries')
    if manifest.get('name')!=record['name'] or not isinstance(expected,list): stop('asset manifest identity is invalid')
    tree=digest(canonical(expected))
    if tree!=manifest.get('tree_sha256') or tree!=record['tree_sha256']: stop('asset manifest tree digest differs from lock')
    actual=entries(base)
    if actual!=expected:
        expected_map={e['path']:e for e in expected}; actual_map={e['path']:e for e in actual}
        missing=sorted(set(expected_map)-set(actual_map)); extra=sorted(set(actual_map)-set(expected_map))
        changed=sorted(p for p in set(expected_map)&set(actual_map) if expected_map[p]!=actual_map[p])
        stop(f'asset bundle differs from manifest (missing={missing[:4]}, extra={extra[:4]}, changed={changed[:4]})')
    app_manifest=base/'robot_asset_manifest.json'
    if not app_manifest.is_file() or digest(app_manifest.read_bytes())!=record['robot_asset_manifest_sha256']:
        stop('robot_asset_manifest.json differs from the lock')
    try: app=json.loads(app_manifest.read_text(encoding='utf-8'))
    except Exception as exc: stop(f'robot asset receipt is unreadable ({type(exc).__name__})')
    if app.get('robolab_commit')!=record['robolab_commit'] or app.get('picknik_commit')!=record['picknik_commit']:
        stop('robot asset upstream source revisions differ from the lock')
    urdf=base/'franka_robotiq_2f_85.urdf'
    if not urdf.is_file() or digest(urdf.read_bytes())!=record['urdf_sha256']:
        stop('generated DROID URDF differs from the lock')

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true',help='verify source bundle and any already materialized output')
    a=p.parse_args()
    try: lock=json.loads(LOCK.read_text(encoding='utf-8'))
    except Exception as exc: stop(f'cannot read assets.lock.json ({type(exc).__name__})')
    if lock.get('schema_version')!=1 or len(lock.get('assets',[]))!=1: stop('unsupported or ambiguous asset lock')
    rec=lock['assets'][0]; source=ROOT/rec['path']; verify_bundle(rec,source)
    output=ROOT/'outputs/robot_assets'/rec['name']
    if output.exists() or output.is_symlink():
        verify_bundle(rec,output)
        print(f'PASS: reused exact robot asset at {output}')
        return 0
    if a.verify_only: stop(f'materialized robot asset is missing: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.parent/f'.{output.name}.partial.{rec["tree_sha256"][:12]}'
    if staging.exists() or staging.is_symlink(): stop(f'partial asset copy exists and was preserved: {staging}')
    shutil.copytree(source,staging,symlinks=True,copy_function=shutil.copy2)
    verify_bundle(rec,staging)
    if output.exists() or output.is_symlink(): stop(f'output appeared during copy; partial preserved at {staging}')
    os.rename(staging,output)
    print(f'PASS: materialized exact robot asset at {output}')
    return 0

if __name__=='__main__':
    sys.exit(main())
