#!/usr/bin/env python3
"""Build the eRAC descriptor table (UP-4).

Writes ``molmetal/data/metal_chemistry_5_erac.csv``: the five original columns
plus an effective-nuclear-charge block.  Nothing here is curated by hand — the
new columns are computed from the electron configuration by Slater's rules and
from RDKit's periodic table, so the table can be regenerated exactly.

Why: ``msrs_metal_descriptor_level.json`` (EXT-2) fed IE1/IE2/Pauling-EN into a
leave-one-metal-out ridge and failed (permutation p=0.167).  eRAC's claim is
that the *nuclear* charge is the wrong coordinate for same-group metals: Ru
(Z=44) and Os (Z=76) are both group 8, but a raw Z makes them maximally distant.
Z_eff compresses them (4.90 vs 4.25) without collapsing them.

Slater's rules for an nd electron: S = 1.00 x (electrons in groups to the left)
+ 0.35 x (other electrons in the same nd group).  "To the left" includes the
core and the outer ns electrons, which is why Z_eff falls from 4d to 5d.

Usage:
    .venv/bin/python molmetal/scripts/build_metal_erac_table.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from rdkit import Chem

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "molmetal" / "data" / "metal_chemistry_5.csv"
DST = REPO / "molmetal" / "data" / "metal_chemistry_5_erac.csv"

# (Z, core electrons, n_f, outer ns, n_d)  -- ground-state configurations
CONFIG = {
    "Ru": (44, 36, 0, 1, 7),   # [Kr] 4d7 5s1
    "Rh": (45, 36, 0, 1, 8),   # [Kr] 4d8 5s1
    "Re": (75, 54, 14, 2, 5),  # [Xe] 4f14 5d5 6s2
    "Os": (76, 54, 14, 2, 6),  # [Xe] 4f14 5d6 6s2
    "Ir": (77, 54, 14, 2, 7),  # [Xe] 4f14 5d7 6s2
}

# d-electron count in the M(III) state used throughout MSRS
D_COUNT_M3 = {"Ru": 5, "Rh": 6, "Re": 4, "Os": 5, "Ir": 6}


def slater_z_eff_nd(z: int, n_d: int) -> float:
    """Z_eff for an nd electron, Slater's rules.

    Every electron outside the nd subshell sits in a group "to the left" of it
    (core, 4f, and the outer ns) and shields 1.00 each; the other nd electrons
    shield 0.35 each.  So ``left = Z - n_d`` — no shell bookkeeping needed.
    """
    s = 1.00 * (z - n_d) + 0.35 * (n_d - 1)
    return z - s


def main() -> int:
    pt = Chem.GetPeriodicTable()
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    fields = list(rows[0].keys()) + [
        "z_eff_nd", "z_eff_over_rcov", "ie_gap_kj",
        "en_over_rcov", "d_count_M3",
    ]
    out = []
    for r in rows:
        m = r["metal_symbol"]
        z, core, n_f, n_s, n_d = CONFIG[m]
        z_eff_nd = slater_z_eff_nd(z, n_d)
        rcov = pt.GetRcovalent(z)
        ie1, ie2, en = float(r["ie1_kj"]), float(r["ie2_kj"]), float(r["pauling_en"])
        r.update(
            z_eff_nd=round(z_eff_nd, 4),
            z_eff_over_rcov=round(z_eff_nd / rcov, 4),
            ie_gap_kj=round(ie2 - ie1, 2),
            en_over_rcov=round(en / rcov, 4),
            d_count_M3=D_COUNT_M3[m],
        )
        out.append(r)

    with DST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    for r in out:
        print(f"{r['metal_symbol']}: Z_eff(nd)={r['z_eff_nd']:.2f} "
              f"Z_eff/rcov={r['z_eff_over_rcov']:.2f} d_M3={r['d_count_M3']}")
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
