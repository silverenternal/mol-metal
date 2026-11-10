#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A269 — the allow-list AUDIT: live hits vs stale entries.

Uses the shipped gate's OWN detection (``defect_hits``) so the audit can
never disagree with the gate it is auditing.

Prints three sections:
  1. the STALE allow-list entries (in ALLOWLIST, no longer a hit);
  2. the LIVE allow-listed hits, each with its source line;
  3. the RESIDUAL (not allow-listed) hits — the true violations.

Usage (CPU-only, read-only):
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        /tmp/wfa269k_allowlist_audit.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "molmetal" / "tests" / "test_wfa269_no_hardcoded_repo_paths.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("wfa269_gate", GATE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wfa269_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    g = load_gate()
    hits = g.defect_hits(REPO)
    bykey = {h["key"]: h for h in hits}

    live = [h for h in hits if h["key"] in g.ALLOWLIST_KEYS]
    residual = [h for h in hits if h["key"] not in g.ALLOWLIST_KEYS]
    stale = sorted(g.ALLOWLIST_KEYS - set(bykey))

    print(f"total defect hits now          : {len(hits)}")
    print(f"ALLOWLIST entries              : {len(g.ALLOWLIST)}")
    print(f"allow-listed hits still present: {len(live)}")
    print(f"RESIDUAL (un-allow-listed)      : {len(residual)}")
    print(f"STALE allow-list entries       : {len(stale)}")

    print()
    print("=== STALE ENTRIES (in ALLOWLIST, no longer a hit) ===")
    for k in stale:
        print(f"  STALE {k}")
    if not stale:
        print("  (none)")

    print()
    print("=== RESIDUAL (not allow-listed) — the true violations ===")
    for h in sorted(residual, key=lambda h: (h["file"], h["line"])):
        print(f"  {h['file']}:{h['line']}  [{h['class']}/{h['target_kind']}]  {h['abs']}")
    if not residual:
        print("  (none)")

    print()
    print("=== LIVE allow-listed hits, with source line ===")
    for h in sorted(live, key=lambda h: (h["file"], h["line"])):
        try:
            src = (REPO / h["file"]).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            src = []
        line = src[h["line"] - 1].strip() if 0 < h["line"] <= len(src) else "?"
        print(f"{h['file']}:{h['line']}  [{h['class']}/{h['target_kind']}]  {h['abs']}")
        print(f"      SRC: {line[:170]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
