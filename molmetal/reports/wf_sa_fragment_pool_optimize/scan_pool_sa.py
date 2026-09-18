"""Scan every fragment in the validated 200+ tile pool and rank by SA score.

This is a one-off diagnostic script for WF-SA-Fragment-Pool-Optimize:
  * Load the 200+ tile pool via FRAGMENT_LIBRARY_200_TILES().
  * Compute the Ertl-Schuffenhauer SA score for every tile.
  * Print the top-10 highest-SA tiles with their SMILES, category, SA.
  * Persist the result to sa_pool_scan.csv for downstream use.

Honest framing: this is the empirical SA landscape of the pool.  The
top-10 list IS the implementation artefact that drives the
--keep-high-sa-tiles opt-in flag.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from molmetal_lam.sbdd_env.sa_score import (  # type: ignore
    _SASCORER_AVAILABLE,
    sa_score_ertl,
)
from molmetal_lam.tile_lib.library import (  # type: ignore
    FRAGMENT_LIBRARY_200_TILES,
)


def main() -> int:
    if not _SASCORER_AVAILABLE:
        print("ERROR: RDKit sascorer not importable in this environment.",
              file=sys.stderr)
        return 2

    pool = FRAGMENT_LIBRARY_200_TILES()
    print(f"Pool size: {len(pool)} tiles")

    rows = []
    for tile in pool:
        smi = getattr(tile, "smiles", "") or ""
        tags = list(getattr(tile, "tags", []) or [])
        sa = sa_score_ertl(smi)
        rows.append({"smiles": smi, "tags": "|".join(tags), "sa": sa})

    rows.sort(key=lambda r: (
        -(r["sa"] if r["sa"] == r["sa"] else -1.0)  # NaN to bottom
    ))

    out_csv = Path(__file__).resolve().parent / "sa_pool_scan.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["smiles", "tags", "sa"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")

    # Top-10 print
    print("\nTop-10 highest-SA tiles:")
    print(f"{'rank':>4}  {'SA':>6}  {'tags':<40}  SMILES")
    for i, r in enumerate(rows[:10], start=1):
        print(f"{i:>4}  {r['sa']:>6.3f}  {r['tags']:<40}  {r['smiles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
