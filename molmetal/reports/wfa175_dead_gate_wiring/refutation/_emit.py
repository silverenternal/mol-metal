#!/usr/bin/env python3
"""Emit the refutation artifacts via the repo's §5.6 atomic-write primitive.

READ-ONLY everywhere except this directory.
"""

from __future__ import annotations

import importlib.util as ilu
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
HERE = Path(__file__).resolve().parent

s = ilu.spec_from_file_location(
    "aw", REPO / "molmetal" / "scripts" / "wf386a3_atomic_write.py"
)
aw = ilu.module_from_spec(s)
sys.modules["aw"] = aw
s.loader.exec_module(aw)

for stem in ("refutation.json", "refutation.md"):
    payload = HERE / f"_{stem.replace('.', '_')}_payload"
    if not payload.exists():
        payload = HERE / f"_{stem.split('.')[0]}_payload.{stem.split('.')[1]}"
    content = payload.read_text(encoding="utf-8")
    aw.atomic_write_file(str(HERE / stem), content)
    print(f"wrote {HERE / stem} ({len(content)} bytes)")
