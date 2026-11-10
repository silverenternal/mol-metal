#!/usr/bin/env python3
"""Recreate the source server's links to an existing pocket/data tree."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/recovery/external_pocket_links.json'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pocket_data', type=Path, help='Existing pocket/data directory')
    args = parser.parse_args()
    base = args.pocket_data.resolve(strict=True)
    if not base.is_dir():
        parser.error('pocket_data must be a directory')
    rows = json.loads(MANIFEST.read_text())['links']
    plan = []
    for item in rows:
        link_rel = Path(item['path'])
        target_rel = Path(item['target_relative_to_pocket_data'])
        if any(p.is_absolute() or '..' in p.parts for p in (link_rel, target_rel)):
            raise ValueError(f'Unsafe path in manifest: {item!r}')
        link, target = ROOT / link_rel, base / target_rel
        if not target.exists():
            raise FileNotFoundError(f'External data missing: {target}')
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise FileExistsError(f'Link points elsewhere: {link}')
            continue
        if link.exists():
            raise FileExistsError(f'Refusing to replace existing path: {link}')
        if any((ROOT / parent).is_symlink() for parent in link_rel.parents if str(parent) != '.'):
            raise ValueError(f'Link parent is a symlink: {link}')
        plan.append((link, target))
    for link, target in plan:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())
    print(f'Pocket data links: {len(plan)} created, {len(rows) - len(plan)} already correct.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
