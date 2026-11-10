#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Is a synthetic active/inactive cohort JSON actually balanced?

WHY (Wf-A269 wave-9, 2026-11-02).  Two launchers checked their cohort's
active/inactive split with a GREP for a literal that the file does not
contain, so the check could never pass:

  * ``scripts/launch_wf310e_wf135_phase2_gpu.sh`` grepped for
    ``"active_inactive": "active"``.  The cohort file it points at
    (``molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json``)
    is a JSON object with top-level ``actives`` / ``inactives`` ARRAYS and a
    ``stats`` block (``n_active`` / ``n_inactive``).  The string
    ``active_inactive`` occurs NOWHERE in it.  Both greps returned 0, and
    because ``grep -c`` prints "0" AND exits 1 on no match, the ``|| echo 0``
    fallback appended a SECOND "0", so the captured value was "0\\n0".  The
    wrapper therefore reported "Phase 1 cohort may be unbalanced
    (active=0\\n0, inactive=0\\n0)" on EVERY run - while the cohort is in
    fact perfectly balanced (MEASURED: ``stats.n_active=250``,
    ``stats.n_inactive=250``, ``len(actives)=250``, ``len(inactives)=250``).
    The "cohort verified: 250 active + 250 inactive balanced" branch was
    unreachable: a dead PASS branch, the mirror image of a dead FAIL branch.
  * the archived ``launch_wf338v2_wf176_m19_lomo_cpu.sh`` greps the same
    literal the same way.

The counts are read with a JSON parser, from the first source that exists:
``stats.n_active`` / ``stats.n_inactive``, else ``len(actives)`` /
``len(inactives)``, else a row-level label field.

Usage
-----
    python molmetal/scripts/wf_cohort_balance_check.py \\
        --json molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json \\
        --expect-active 250 --expect-inactive 250

Stdout invariant: ``PASS`` / ``FAIL`` / ``SKIP`` (AGENTS.md §4).  Exit 0=PASS,
1=FAIL (unbalanced), 2=SKIP (file missing / not JSON).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Row-level label fields to try when the file has no stats block and no
# actives/inactives arrays.
ROW_LABEL_KEYS = ("active_inactive", "label", "class", "activity")
ACTIVE_LABELS = {"active", "actives", "1", "positive", "pos"}
INACTIVE_LABELS = {"inactive", "inactives", "0", "negative", "neg"}


def measure(path: Path) -> Tuple[Optional[int], Optional[int], str]:
    """Return (n_active, n_inactive, source_description)."""
    with path.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        stats = data.get("stats")
        if isinstance(stats, dict) and "n_active" in stats and "n_inactive" in stats:
            return (
                int(stats["n_active"]),
                int(stats["n_inactive"]),
                "stats.n_active / stats.n_inactive",
            )
        if "actives" in data and "inactives" in data:
            return (
                len(data.get("actives") or []),
                len(data.get("inactives") or []),
                "len(actives) / len(inactives)",
            )

    rows = data.get("rows") if isinstance(data, dict) else data
    if isinstance(rows, list) and rows:
        for key in ROW_LABEL_KEYS:
            if not isinstance(rows[0], dict) or key not in rows[0]:
                continue
            n_act = n_inact = 0
            for r in rows:
                v = str(r.get(key, "")).strip().lower()
                if v in ACTIVE_LABELS:
                    n_act += 1
                elif v in INACTIVE_LABELS:
                    n_inact += 1
            return n_act, n_inact, f"row-level field {key!r}"

    raise ValueError("no recognisable active/inactive source in the cohort JSON")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--json", required=True, help="cohort JSON path")
    ap.add_argument("--expect-active", type=int, default=250)
    ap.add_argument("--expect-inactive", type=int, default=250)
    ap.add_argument("--json-out", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    tag = "[cohort-balance-check]"
    path = Path(args.json)
    if not path.is_file():
        print(f"{tag} SKIP: cohort missing: {path}")
        return 2
    try:
        n_act, n_inact, source = measure(path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"{tag} SKIP: cannot measure {path}: {exc}")
        return 2

    ok = n_act == args.expect_active and n_inact == args.expect_inactive
    payload = {
        "path": str(path),
        "n_active": n_act,
        "n_inactive": n_inact,
        "expect_active": args.expect_active,
        "expect_inactive": args.expect_inactive,
        "source": source,
        "balanced": ok,
    }
    if args.json_out:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"{tag} {path.name}: active={n_act} inactive={n_inact} "
            f"(expected {args.expect_active}/{args.expect_inactive}) "
            f"[source: {source}]"
        )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
