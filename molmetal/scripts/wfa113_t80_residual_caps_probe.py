#!/usr/bin/env python
"""Wf-A113 (T80/T81) — what still cannot be rebuilt after T77-T79?

T77-T79 removed the three unit errors in the selector's accounting (a metal's
budget is a bond count, an aromatic bond at a heteroatom is Kekule-single, a
metal bond is dative at the donor).  This probe measures what is left: for each
ligand element, does the CORPUS's own bond graph still cost more than the
decoder's cap, i.e. does the decoder still drop a real bond?

Two residuals were found and fixed:

T80 — **S, Se and As caps were below RDKit's own limit.**  ``_COVALENT_VALENCE``
had S 2, Se 2, As 3.  RDKit accepts S 6, Se 6, As 5 (verified on 2026.03.6).
The corpus is not thioether-only: measured bond counts in the six-metal cohort
are S 1:198, 2:7,120, 3:1,720, 4:2,090, 5:2, 6:29 and As 3:37, 4:364, 5:4,
6:2.  A cap of 2 refuses every sulfoxide, sulfone and sulfonium; a cap of 3
refuses 370 of 409 As atoms.  This is the same shape as the P cap 3 -> 5 fix in
T78.

T81 — **a ring-fusion carbon's third aromatic bond was charged 1.5.**  The 1.5
is the average of one Kekule double and one single, which assumes exactly two
aromatic bonds at the atom.  A fusion carbon has three, and no Kekule structure
gives it two doubles, so the true sum is 4.  Charging 4.5 put it over its cap
of 4 and the selector dropped the third bond, breaking the fusion.

``BEFORE`` = caps {S:2, Se:2, As:3} and 1.5 for every aromatic bond at carbon,
i.e. the state after T79.  ``AFTER`` = the current tables.

Writes ``metrics/by_metric/wfa113_t80_residual_caps.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t80_residual_caps_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t80_residual_caps.json"

BO_PATHS = [
    ROOT / ".data/molmetal/tmQM/tmQM_X1.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X2.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X3.BO.gz",
]

_Z_OF = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "As": 33, "Se": 34, "Br": 35, "I": 53,
    "Fe": 26, "Ni": 28, "Cu": 29, "Zn": 30, "Ru": 44, "Rh": 45, "Pd": 46,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
}
_SIX = {"Ru", "Rh", "Ir", "Os", "Re", "Pt"}
_METAL_ZS = frozenset({26, 28, 29, 30, 44, 45, 46, 75, 76, 77, 78, 79})

#: Pre-T80 caps for the three elements T80 raised.  Everything else is
#: unchanged, so the difference is attributable.
CAP_BEFORE = {16: 2, 33: 3, 34: 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-complexes", type=int, default=0,
                    help="per shard; 0 = all")
    args = ap.parse_args(argv)

    from molmetal.data.tmqm_clouds import bo_to_bond_class
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        _COVALENT_VALENCE as CAP_NOW,
        _endpoint_valence_cost,
    )
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks

    def is_metal_z(z: int) -> bool:
        return z in _METAL_ZS

    class_name = {1: "single", 2: "double", 3: "triple", 4: "aromatic"}
    n_atoms: Counter = Counter()
    over_before: Counter = Counter()
    over_after: Counter = Counter()
    pat_before: Dict[str, Counter] = defaultdict(Counter)
    pat_after: Dict[str, Counter] = defaultdict(Counter)
    worst: Dict[str, List[dict]] = defaultdict(list)
    n_complexes = 0
    cx_before = 0
    cx_after = 0
    # Boron clusters are the one residual that cannot be fixed: a carborane
    # boron has 5-6 neighbours and RDKit's limit for neutral B is 3, so the
    # decode is refused whatever the selector keeps.  Counting the over-cap
    # complexes that contain no B separates "our table is wrong" from "RDKit
    # cannot represent this molecule".
    cx_after_no_b = 0
    n_complexes_no_b = 0

    for path in BO_PATHS:
        if not path.is_file():
            continue
        n = 0
        for code, atoms in iter_bo_blocks([path]):
            syms = {idx: sym for idx, sym, _ in atoms}
            if not (set(syms.values()) & _SIX):
                continue
            syms = {i: s for i, s in syms.items() if s != "H"}
            bonds: Dict[Tuple[int, int], float] = {}
            for idx, _sym, nbrs in atoms:
                if idx not in syms:
                    continue
                for nbr_idx, bo in nbrs:
                    if nbr_idx == idx or bo < 0.3 or nbr_idx not in syms:
                        continue
                    key = (min(idx, nbr_idx), max(idx, nbr_idx))
                    if key not in bonds or bo > bonds[key]:
                        bonds[key] = bo

            cb: Dict[int, float] = defaultdict(float)
            ca: Dict[int, float] = defaultdict(float)
            arom_b: Counter = Counter()
            arom_a: Counter = Counter()
            pat: Dict[int, List[str]] = defaultdict(list)
            # Aromatic bonds are charged in a fixed order (by atom index) so
            # the "third aromatic bond at a carbon" counter is deterministic.
            for (i, j), bo in sorted(bonds.items()):
                cls = bo_to_bond_class(bo)
                zi, zj = _Z_OF.get(syms.get(i, "")), _Z_OF.get(syms.get(j, ""))
                if zi is None or zj is None:
                    continue
                # BEFORE: the ligand endpoint pays, and every aromatic bond at
                # carbon costs 1.5 whatever its position at the atom.
                for idx, z, other in ((i, zi, zj), (j, zj, zi)):
                    if is_metal_z(z):
                        cb[idx] += 1.0
                    elif is_metal_z(other):
                        cb[idx] += 0.0
                    elif cls == 4 and z in {5, 7, 8, 15, 16, 33, 34}:
                        cb[idx] += 1.0
                    else:
                        cb[idx] += {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}[cls]
                ca[i] += _endpoint_valence_cost(cls, zi, zj, is_metal_z, arom_a[i])
                ca[j] += _endpoint_valence_cost(cls, zj, zi, is_metal_z, arom_a[j])
                if cls == 4:
                    arom_a[i] += 1
                    arom_a[j] += 1
                pat[i].append(class_name[cls])
                pat[j].append(class_name[cls])

            cx_b = cx_a = False
            for idx, sym in syms.items():
                z = _Z_OF.get(sym)
                if z is None or z not in CAP_NOW or is_metal_z(z):
                    continue
                n_atoms[sym] += 1
                cap_b = float(CAP_BEFORE.get(z, CAP_NOW[z]))
                cap_a = float(CAP_NOW[z])
                key = tuple(sorted(pat[idx]))
                if cb[idx] > cap_b + 1e-9:
                    over_before[sym] += 1
                    pat_before[sym][key] += 1
                    cx_b = True
                    if len(worst[sym]) < 3:
                        worst[sym].append(
                            {"csd": code, "cost": round(cb[idx], 2), "cap": cap_b}
                        )
                if ca[idx] > cap_a + 1e-9:
                    over_after[sym] += 1
                    pat_after[sym][key] += 1
                    cx_a = True
            cx_before += int(cx_b)
            cx_after += int(cx_a)
            has_b = "B" in syms.values()
            if not has_b:
                n_complexes_no_b += 1
                cx_after_no_b += int(cx_a)
            n += 1
            n_complexes += 1
            if args.max_complexes and n >= args.max_complexes:
                break

    per_element = {}
    for sym in sorted(n_atoms, key=lambda s: -over_after[s]):
        c = n_atoms[sym]
        per_element[sym] = {
            "cap_before": CAP_BEFORE.get(_Z_OF.get(sym, -1), CAP_NOW.get(_Z_OF.get(sym, -1))),
            "cap_after": CAP_NOW.get(_Z_OF.get(sym, -1)),
            "n_atoms": c,
            "frac_over_before": over_before[sym] / c if c else 0.0,
            "frac_over_after": over_after[sym] / c if c else 0.0,
            "top_over_patterns_after": [
                {"pattern": "".join(k), "n": v}
                for k, v in pat_after[sym].most_common(3)
            ],
            "examples_before": worst[sym],
        }

    payload = {
        "wave": "Wf-A113",
        "task": "T80/T81",
        "date": "2026-10-28",
        "question": (
            "after T77-T79, which ligand atoms still exceed their cap on the "
            "CORPUS's own bond graph, i.e. where can the decoder still not "
            "rebuild the truth it was trained on?"
        ),
        "fix": {
            "T80_caps": {
                "S": "2 -> 6 (RDKit's limit; corpus has sulfoxide/sulfone)",
                "Se": "2 -> 6 (RDKit's limit; same group as S)",
                "As": "3 -> 5 (RDKit's limit; corpus is As(V))",
            },
            "T81_aromatic_degree": (
                "a carbon's third and later aromatic bond costs 1.0, not 1.5: "
                "a ring-fusion carbon's Kekule sum is 4, not 4.5"
            ),
        },
        "measured": {
            "cohort": "six project metals",
            "max_complexes_per_shard": args.max_complexes,
            "n_complexes": n_complexes,
            "n_atoms_total": sum(n_atoms.values()),
            "n_complexes_with_a_ligand_atom_over_cap_before": cx_before,
            "n_complexes_with_a_ligand_atom_over_cap_after": cx_after,
            "frac_complexes_over_before": (
                cx_before / n_complexes if n_complexes else 0.0
            ),
            "frac_complexes_over_after": (
                cx_after / n_complexes if n_complexes else 0.0
            ),
            "n_complexes_without_boron": n_complexes_no_b,
            "n_complexes_without_boron_over_cap_after": cx_after_no_b,
            "frac_complexes_over_after_excluding_boron": (
                cx_after_no_b / n_complexes_no_b if n_complexes_no_b else 0.0
            ),
            "per_element": per_element,
        },
        "script": "molmetal/scripts/wfa113_t80_residual_caps_probe.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    m = payload["measured"]
    print(
        f"complexes {n_complexes}  over-cap: {cx_before} "
        f"({m['frac_complexes_over_before']:.4f}) -> {cx_after} "
        f"({m['frac_complexes_over_after']:.4f})"
    )
    print(f"{'el':3s} {'cap':>7s} {'atoms':>8s} {'before':>8s} {'after':>8s}  top pattern after")
    for s, v in per_element.items():
        if v["n_atoms"] < 200:
            continue
        top = v["top_over_patterns_after"][:1]
        print(
            f"{s:3s} {v['cap_before']}->{v['cap_after']:<4d} {v['n_atoms']:8d} "
            f"{v['frac_over_before']:8.4f} {v['frac_over_after']:8.4f}  {top}"
        )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
