#!/usr/bin/env python3
"""Write a relocated copy of a legacy YAML without changing its hyperparameters."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

OLD_ROOT = '/home/hugo/codes/mol-metal'
REPO_ROOT = Path(__file__).resolve().parents[1]


def relocate(value: object, old_root: str, new_root: str) -> tuple[object, int]:
    """Replace the old root in string values, preserving all other YAML values."""
    if isinstance(value, str):
        pattern = re.compile(r'(?<![A-Za-z0-9_./-])' + re.escape(old_root.rstrip('/')) + r'(?=/|$)')
        return pattern.subn(lambda _: new_root.rstrip('/'), value)
    if isinstance(value, list):
        out, count = [], 0
        for item in value:
            mapped, n = relocate(item, old_root, new_root)
            out.append(mapped)
            count += n
        return out, count
    if isinstance(value, dict):
        out, count = {}, 0
        for key, item in value.items():
            mapped, n = relocate(item, old_root, new_root)
            out[key] = mapped
            count += n
        return out, count
    return value, 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path, help='Original YAML; never overwritten')
    p.add_argument('destination', type=Path, help='New YAML; must not already exist')
    p.add_argument('--old-root', default=OLD_ROOT)
    p.add_argument('--new-root', type=Path, default=REPO_ROOT)
    args = p.parse_args()
    if not args.old_root.startswith('/') or args.old_root == '/':
        p.error('--old-root must be an absolute non-root directory')
    new_root = str(args.new_root.resolve())
    if any(c.isspace() for c in new_root):
        p.error('new root contains whitespace; legacy shell command templates may not quote it')
    source = args.source.read_bytes()
    obj = yaml.safe_load(source)
    if not isinstance(obj, dict):
        p.error('source must contain a YAML mapping')
    mapped, replacements = relocate(obj, args.old_root, new_root)
    rendered = yaml.safe_dump(mapped, sort_keys=False, allow_unicode=True).encode()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.destination.open('xb') as out:
            out.write(rendered)
    except FileExistsError:
        p.error('destination already exists; refusing to overwrite')
    print(f'Wrote {args.destination} with {replacements} root substitutions')
    print(f'source_sha256={hashlib.sha256(source).hexdigest()}')
    print(f'destination_sha256={hashlib.sha256(rendered).hexdigest()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
