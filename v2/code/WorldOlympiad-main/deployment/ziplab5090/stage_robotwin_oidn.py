"""Stage a pinned OIDN overlay without changing the shared RoboTwin environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tarfile

VERSION = '2.3.3'
ARCHIVE_SHA256 = '3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f'
SOURCE = 'https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz'
LOADER_SHA256 = '8e1808bf2633f266720834d6dad8213c10059c463285c37b55ac7d97e2a59345'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(workspace: Path, archive: Path):
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError('OIDN archive does not match the reviewed official release')
    environment = workspace / 'envs/robotwin'
    site = next((environment / 'lib').glob('python*/site-packages'))
    source = site / 'sapien'
    if sha(source / '_oidn_tricks.py') != LOADER_SHA256:
        raise ValueError('SAPIEN loader differs from the reviewed 3.0.0b1 runtime')
    target = workspace / 'runtime/robotwin_oidn' / VERSION
    if target.exists():
        raise FileExistsError(f'preserve existing runtime: {target}')
    partial = target.with_name(target.name + '.partial')
    partial.mkdir(parents=True)
    package = 'oidn-2.3.3.x86_64.linux'
    with tarfile.open(archive) as stream:
        stream.extractall(partial, filter='data')
    overlay = partial / 'sapien_overlay/sapien'
    overlay.mkdir(parents=True)
    for path in source.iterdir():
        if path.name not in {'_oidn_tricks.py', 'oidn_library', '__pycache__'}:
            (overlay / path.name).symlink_to(path)
    (overlay / 'oidn_library').symlink_to(Path('../..') / package / 'lib')
    (partial / 'sapien_overlay/sapien.libs').symlink_to(site / 'sapien.libs')
    (overlay / '_oidn_tricks.py').write_text(
        (source / '_oidn_tricks.py').read_text().replace('2.0.1', VERSION))
    launcher = partial / 'python'
    launcher.write_text('#!/bin/sh\nexport PYTHONPATH=' +
        shlex.quote(str(target / 'sapien_overlay')) + ':"${PYTHONPATH:-}"\n' +
        'export LD_LIBRARY_PATH=' + shlex.quote(str(target / package / 'lib')) +
        ':"${LD_LIBRARY_PATH:-}"\nexec ' + shlex.quote(str(environment / 'bin/python')) + ' "$@"\n')
    launcher.chmod(0o755)
    libraries = {str(target / path.relative_to(partial)):sha(path)
                 for path in (partial / package / 'lib').glob('libOpenImageDenoise*.so.2.3.3')}
    receipt = {'version':VERSION, 'source':SOURCE, 'archive_sha256':ARCHIVE_SHA256,
               'original_loader_sha256':LOADER_SHA256, 'launcher_sha256':sha(launcher),
               'library_sha256':libraries}
    (partial / 'runtime_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    partial.rename(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(stage(args.workspace.resolve(), args.archive.resolve()))
