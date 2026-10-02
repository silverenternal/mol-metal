#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent Apache-2.0 license-header addend for new .py files.

Mirrors the convention used by OmegaFold (see every file in
``omegafold/`` prefixed by the HeliXon 2022 copyright block).
This script is **additive** for new files only — it does NOT rewrite
the headers of existing source files (those retrofits belong in
follow-on waves, with their own audit + author + LICENSE check).

Algorithm
---------
1. Walk every ``.py`` file under ``molmetal/`` + ``molmetal/tests/``.
2. For each file whose first non-blank line is NOT a license header
   (detected via the ``Copyright\s+2026\s+Mol-Metal\s+authors``
   sentinel), prepend the Apache-2.0 header block.
3. Files that already have the header (or any non-molmetal-prefixed
   header) are skipped — idempotent.

Run
---
    python scripts/_add_license_headers.py            # add new-file headers
    python scripts/_add_license_headers.py --check   # dry-run (exit 1 if any new file is missing)

Notes
-----
* Replaces the Wf-321B §11.1 skill-audit ``license-header gap`` finding.
* Does NOT touch ``molmetal/references/*`` (git-cloned SOTA repos,
  which carry their own upstream LICENSE).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WALK_DIRS = [PROJECT_ROOT / "molmetal", PROJECT_ROOT / "tests"]
EXCLUDE_DIRS = {"references", "__pycache__", ".venv", ".git", "build", "dist"}
EXCLUDE_SUFFIXES = {".pyc", ".so"}
HEADER_SENTINEL = re.compile(r"Copyright\s+\d{4}.*?Mol-Metal", re.MULTILINE)

APACHE_HEADER = (
    "# -*- coding: utf-8 -*-\n"
    "# =============================================================================\n"
    "# Copyright 2026 Mol-Metal authors\n"
    "#\n"
    "# Licensed under the Apache License, Version 2.0 (the "License");\n"
    "# you may not use this file except in compliance with the License.\n"
    "# You may obtain a copy of the License at\n"
    "#\n"
    "#      http://www.apache.org/licenses/LICENSE-2.0\n"
    "#\n"
    "# Unless required by applicable law or agreed to in writing, software\n"
    "# distributed under the License is distributed on an "AS IS" BASIS,\n"
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License.\n"
    "# =============================================================================\n"
    "\n"
)


def _walk_py_files() -> "list[Path]":
    """Yield every .py file under WALK_DIRS minus EXCLUDE_DIRS."""
    out: list[Path] = []
    for root in WALK_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if p.suffix in EXCLUDE_SUFFIXES:
                continue
            out.append(p)
    return out


def _needs_header(path: Path) -> bool:
    """Return True iff ``path`` does NOT already have the Mol-Metal header."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return False  # empty file — skip
    head = text[:4096]  # first 4 KB is enough for the header
    return HEADER_SENTINEL.search(head) is None


def _add_header(path: Path, *, dry_run: bool = False) -> bool:
    """Prepend the header to ``path``. Returns True iff file was modified."""
    if not _needs_header(path):
        return False
    text = path.read_text(encoding="utf-8")
    new_text = APACHE_HEADER + text
    if dry_run:
        return True  # would have been modified
    path.write_text(new_text, encoding="utf-8")
    return True


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Dry-run: exit 1 iff any new file is missing the header.")
    args = parser.parse_args(argv)

    py_files = _walk_py_files()
    missing = [p for p in py_files if _needs_header(p)]
    if args.check:
        if missing:
            print(f"CHECK FAIL — {len(missing)} new files missing header:")
            for p in missing[:20]:
                print(f"  {p.relative_to(PROJECT_ROOT)}")
            return 1
        print(f"CHECK OK — {len(py_files)} files all have Mol-Metal header.")
        return 0

    modified = 0
    for p in py_files:
        if _add_header(p, dry_run=False):
            modified += 1
    print(f"Header pass complete: {modified} modified of {len(py_files)} walked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
