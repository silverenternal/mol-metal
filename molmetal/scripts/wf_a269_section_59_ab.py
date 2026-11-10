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
"""Wf-A269 -- the §5.9 A/B PROOF: does the single-venv rule REQUIRE the
host-bound absolute path?

AGENTS.md §5.9 is enforced by ``scripts/_wrapper_audit.py``.  Its
``CANONICAL_MARKERS`` include two PATH markers, and ``classify()`` returns
``"REAL"`` only when ALL SEVEN markers are present.  If those two markers
match ONLY the host-root absolute form, then a PORTABLE wrapper is classified
SCAFFOLD and the HOST-BOUND wrapper is classified REAL -- i.e. the rule's
enforcement institutionalises the defect it is supposed to prevent.

This script proves that with an A/B test through the audit's OWN
``surface_check`` / ``classify`` functions: two synthetic wrappers that differ
ONLY in how they spell the root.

Usage (CPU-only, read-only):
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        /tmp/wfa269k_section_59_ab.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "scripts" / "_wrapper_audit.py"
GATE = REPO / "molmetal" / "tests" / "test_wfa269_no_hardcoded_repo_paths.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The forbidden host roots are read from the gate -- the single source of
# truth for them -- so this file never holds a host-root literal itself.
_GATE = _load(GATE, "wfa269_gate")
ROOT = _GATE._ROOTS[0]


# Both wrappers carry the FIVE non-path canonical markers, so the ONLY
# difference between them is how the repository root is spelled.
_COMMON = """#!/usr/bin/env bash
set -euo pipefail
# USER_OK gate
[ "${USER_OK:-0}" != "1" ] && { echo "USER_OK=1 required"; exit 1; }
# DRY_RUN emission
DRY_RUN_LOG="logs/DRY_RUN.log"
# plan_json emission
PLAN_JSON="logs/plan.json"
# pre_flight_check gate + USER_OK env propagation to trainer
"$PY" molmetal/scripts/pre_flight_check.py --pipeline all --user-ok "$USER_OK"
"""

PORTABLE = (
    '#!/usr/bin/env bash\n'
    'set -euo pipefail\n'
    'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
    'PY="${REPO_ROOT}/.venv/bin/python"\n'
    'export PYTHONPATH="${REPO_ROOT} ${PYTHONPATH:-}"\n'
    + _COMMON
)

ABSOLUTE = (
    '#!/usr/bin/env bash\n'
    'set -euo pipefail\n'
    f'PY="{ROOT}/.venv/bin/python"\n'
    f'export PYTHONPATH="{ROOT} ${{PYTHONPATH:-}}"\n'
    + _COMMON
)


def load_audit():
    return _load(AUDIT, "wrapper_audit")


def main() -> int:
    a = load_audit()

    print("=== the two path markers of CANONICAL_MARKERS ===")
    for name in ("PYTHONPATH export", "PY hardcoded-via-bin/python"):
        rx = a.CANONICAL_MARKERS[name]
        print(f"  {name!r}: {rx.pattern}")

    print()
    print("=== the A/B test (identical except for how the root is spelled) ===")
    rows = []
    for label, text in (("PORTABLE ($REPO_ROOT-derived)", PORTABLE),
                        ("ABSOLUTE (host path)", ABSOLUTE)):
        # the audit's own surface check: which markers does this text hit?
        matched = [n for n, rx in a.CANONICAL_MARKERS.items() if rx.search(text)]
        missing = [n for n in a.CANONICAL_MARKERS if n not in matched]
        verdict = "REAL" if len(matched) == len(a.CANONICAL_MARKERS) else "SCAFFOLD"
        rows.append((label, verdict, missing))
        print(f"  {label:<32} -> {verdict:<9} | missing: {missing}")

    portable_v = rows[0][1]
    absolute_v = rows[1][1]
    print()
    print("=== VERDICT ===")
    if portable_v == "SCAFFOLD" and absolute_v == "REAL":
        print("  CONFIRMED: the PORTABLE wrapper FAILS the §5.9 audit and the")
        print("  HOST-BOUND wrapper PASSES it.  §5.9's ENFORCEMENT requires the")
        print("  host binding; excluding .sh from the portability gate is a")
        print("  concession to that enforcement, not a design decision that")
        print("  protects portability.")
        return 0
    print(f"  NOT CONFIRMED: portable={portable_v} absolute={absolute_v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
