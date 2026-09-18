"""Copy a verified code snapshot so independent development cannot mix a sweep.

Large datasets, checkpoints and upstream references stay at explicit shared
paths; their input hashes remain each experiment's responsibility. The uv
interpreter is shared, with pyproject/lock copied into the snapshot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone


EXCLUDE = {'__pycache__', '.git', '.venv', '.pytest_cache', '.cache', 'reports', 'references', 'checkpoints'}
TREES = ('models', 'flow_matching', 'triton_kernels', 'molmetal', 'molmetal_lam', 'utils', 'configs', 'data', 'environments')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Snapshot destination must be new; existing experiment is preserved')
    if destination.is_relative_to(source):
        raise ValueError('Use a destination outside the live source tree')
    files = []
    for name in TREES:
        tree = source / name
        if not tree.is_dir():
            continue
        for directory, dirs, names in os.walk(tree, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE and not (Path(directory)/d).is_symlink())
            files += [Path(directory)/n for n in sorted(names) if not n.endswith(('.pyc', '.pyo'))]
    files += [source/n for n in ('pyproject.toml', 'uv.lock', '.python-version') if (source/n).is_file()]
    before = {str(f.relative_to(source)): digest(f) for f in files}
    destination.mkdir(parents=True)
    for relative in before:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target, follow_symlinks=True)
    mismatches = [r for r, h in before.items()
                  if not (source/r).is_file() or digest(source/r) != h or digest(destination/r) != h]
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(source),
                'snapshot': str(destination), 'files_sha256': before, 'mismatches': mismatches,
                'verified': not mismatches, 'shared_paths': {}}
    for relative in ('.venv', 'molmetal/references', 'molmetal/checkpoints'):
        shared = source/relative
        if shared.exists():
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(shared, target_is_directory=shared.is_dir())
            metadata['shared_paths'][relative] = str(shared)
    (destination/'snapshot_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if mismatches:
        raise RuntimeError('Source changed while snapshotting; retain failed snapshot and retry with a new destination')
    print(json.dumps({'snapshot': str(destination), 'verified_files': len(before), 'shared_paths': metadata['shared_paths']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    snapshot(args.source, args.destination)
