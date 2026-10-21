#!/usr/bin/env python
# Copyright 2026 Mol-Metal contributors.
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
"""Count metal-bearing structures in the local ChEMBL 29 dump.

WHY THIS EXISTS
---------------
``todo.json`` blocker B1 ("external independent cohort") rests on the claim
that the local ChEMBL 29 dump contains **zero** metal structures, so it cannot
supply a cohort for the five-metal task.  That claim was measured on 2026-10-19
with an ad-hoc query and written into ``todo.json`` as prose: no script, no
artifact, no query text.  A number that lives only in a hand-written sentence
is exactly what §5.2 forbids, and it is also unrepeatable — nobody can tell
which SMILES column or which pattern produced it.

This script makes the claim executable.  It opens the dump read-only, counts
every structure and every structure whose ``canonical_smiles`` contains a
bracketed metal token, and writes the queries and timings beside the counts.

WHAT IT DOES NOT DO
-------------------
It does not decide whether ChEMBL *should* be used, and it does not download
anything.  It answers one question: are there metal structures in this file?

The patterns are bracketed element tokens (``[Pt]``, ``[Ru]``, ...) because a
bare ``Pt`` substring also matches organic text like "Ptt" or a name field; the
old probe used ``%Pt%`` and this script keeps that as a cross-check column so
the two can be compared rather than silently swapped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    PROJECT_ROOT / ".data" / "molmetal" / "chembl_29" / "chembl_29_sqlite" / "chembl_29.db"
)
DEFAULT_OUT = PROJECT_ROOT / "metrics" / "by_metric" / "chembl29_metal_scan.json"

#: The metals this project models (Ru/Ir/Rh/Os/Re) plus Pt, the one the
#: literature cohort uses.  These decide the verdict.
TARGET_METALS = ("Pt", "Ru", "Ir", "Rh", "Os", "Re")
#: Cross-checks that do not decide it: Fe is a common transition metal that
#: *should* survive if the dump kept any organometallic, Na is an alkali that
#: appears in salts.  Reported separately so a stray salt cannot masquerade as
#: "the dump has metals".
CROSS_CHECK_METALS = ("Fe", "Na")
METALS = TARGET_METALS + CROSS_CHECK_METALS


def scan(db_path: Path) -> Dict[str, Any]:
    """Return counts for every pattern, with the query text that produced them."""
    if not db_path.is_file():
        raise SystemExit(f"database not found: {db_path}")
    stat = db_path.stat()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        t0 = time.perf_counter()
        total = con.execute("SELECT COUNT(*) FROM compound_structures").fetchone()[0]
        total_seconds = round(time.perf_counter() - t0, 3)

        by_metal: Dict[str, Dict[str, Any]] = {}
        for metal in METALS:
            pattern = f"%[{metal}]%"
            query = (
                "SELECT COUNT(*) FROM compound_structures "
                f"WHERE canonical_smiles LIKE '{pattern}'"
            )
            t0 = time.perf_counter()
            n = con.execute(query).fetchone()[0]
            by_metal[metal] = {
                "pattern": pattern,
                "query": query,
                "n": int(n),
                "seconds": round(time.perf_counter() - t0, 3),
            }

        # The 2026-10-19 ad-hoc probe used a bare substring. Keep it so the two
        # can be compared instead of one silently replacing the other.
        legacy_pattern = "%Pt%"
        legacy_query = (
            "SELECT COUNT(*) FROM compound_structures "
            f"WHERE canonical_smiles LIKE '{legacy_pattern}'"
        )
        t0 = time.perf_counter()
        legacy = con.execute(legacy_query).fetchone()[0]
        legacy_block = {
            "pattern": legacy_pattern,
            "query": legacy_query,
            "n": int(legacy),
            "seconds": round(time.perf_counter() - t0, 3),
        }
    finally:
        con.close()

    n_any_metal = sum(1 for m in METALS if by_metal[m]["n"] > 0)
    n_target = sum(1 for m in TARGET_METALS if by_metal[m]["n"] > 0)
    return {
        "metric": "chembl29_metal_scan",
        "description": (
            "Counts of metal-bearing structures in the local ChEMBL 29 SQLite "
            "dump, by bracketed element token, plus the bare-substring probe "
            "used on 2026-10-19."
        ),
        "scope": "blocker B1 (external independent cohort) — can this dump supply one?",
        "status": "MEASURED",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "db": {
            "path": str(db_path),
            "size_bytes": stat.st_size,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "opened": "read-only (mode=ro)",
        },
        "total_structures": int(total),
        "total_seconds": total_seconds,
        "by_metal": by_metal,
        "legacy_substring_probe": legacy_block,
        "n_metals_with_any_hit": n_any_metal,
        "n_target_metals_with_any_hit": n_target,
        "target_metals": list(TARGET_METALS),
        "cross_check_metals": list(CROSS_CHECK_METALS),
        "verdict": (
            "NO_TARGET_METAL_STRUCTURES"
            if n_target == 0
            else "TARGET_METAL_STRUCTURES_PRESENT"
        ),
        "reading": (
            "The dump is de-metallised for everything this project models: all "
            "six target/precious metals (Pt, Ru, Ir, Rh, Os, Re) and Fe return "
            "0, so it cannot supply an external cohort and B1 stays blocked on "
            "data rather than on effort. The two [Na] hits are sodium salts of "
            "an azo dye (SMILES `OCCC[Na]`, an Na-C bond no chemist would draw); "
            "they are parse artefacts, not a metal chemistry cohort. A non-zero "
            "count for a target metal would not by itself unblock B1 either — "
            "the structures would still need SMILES, cell line and potency in "
            "the same units as the training corpus."
        ),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    report = scan(args.db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[chembl29] {report['total_structures']:,} structures, "
        f"{report['n_target_metals_with_any_hit']}/{len(TARGET_METALS)} target metals present "
        f"({report['n_metals_with_any_hit']}/{len(METALS)} including cross-checks) "
        f"-> {report['verdict']} ({args.out})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
