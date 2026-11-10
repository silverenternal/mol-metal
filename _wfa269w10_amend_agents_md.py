#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Add the §6.269.17 pointer to AGENTS.md (3 shared sites) + the §7 table row.

Append-style, minimal, and byte-preserving elsewhere (AGENTS.md must stay
small: it is injected into every subagent prompt).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))

from wf386a3_atomic_write import atomic_write_file  # noqa: E402

AGENTS = REPO_ROOT / "AGENTS.md"

OLD = (
    "§6.269.13 is the HOST-BOUND-PATH defect class (four censuses of one class "
    "returned four counts)**"
)
NEW = OLD + (
    "; §6.269.17 is the DEAD-GATE class — a check that cannot fail (7 launch "
    "gates repaired; the shared verdict consumer is "
    "`molmetal/scripts/wf_launch_gate_verdict.py`)**"
)

TABLE_OLD = (
    "| `molmetal/scripts/pre_flight_check.py` | Pre-flight v2 (TODO-207) |\n"
)
TABLE_NEW = (
    TABLE_OLD
    + "| `molmetal/scripts/wf_launch_gate_verdict.py` | Launch-gate verdict "
    "consumer (Wf-A269 wave-10) — reads a pre-flight / checklist / budget "
    "REPORT and decides PASS/FAIL; a gate that cannot fail is the defect |\n"
)


def main() -> int:
    original = AGENTS.read_text(encoding="utf-8")
    n = original.count(OLD)
    assert n == 3, f"expected 3 shared gloss sites, found {n}"
    assert "§6.269.17" not in original, "the pointer is already present"
    assert TABLE_OLD in original, "the §7 table row anchor is missing"

    merged = original.replace(OLD, NEW)
    assert merged.count("§6.269.17") == 3, "the pointer did not land at 3 sites"

    merged = merged.replace(TABLE_OLD, TABLE_NEW, 1)

    atomic_write_file(AGENTS, merged, encoding="utf-8")

    after = AGENTS.read_text(encoding="utf-8")
    assert after.count("§6.269.17") == 3
    assert TABLE_NEW in after
    assert len(after) > len(original)
    print(
        f"AGENTS.md {len(original)} -> {len(after)} chars (+{len(after) - len(original)}); "
        f"3 gloss sites + 1 table row"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
