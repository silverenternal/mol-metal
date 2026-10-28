#!/usr/bin/env python
"""Wf-A113 (T78) — does ``_CLASS_VALENCE_COST`` fit the corpus for EVERY element?

T77 fixed the metal case (a count ceiling charged in bond-order units).  This
probe asks the same question of the ligand table, where the cap really is an
order sum: for each element, does the CORPUS's own bond graph ever cost more
than ``_COVALENT_VALENCE[Z]``?  If it does, the decoder cannot rebuild the
truth it was trained on — it will drop that bond, split the molecule, and the
loss in ``bond_exact_type_rate`` and ``gen_validity`` follows.

Two things were measured and fixed (T78):

1. **Aromatic bonds at heteroatoms cost 1.5.**  1.5 is the Kekulé average,
   which is right for carbon (benzene C: 2 x 1.5 + 1 H = 4 = cap) and wrong
   for N/O/S, which carry the aromatic bond as a lone pair or as the double
   bond's partner with no second substituent.  Pyrrole N (2 x 1.5 + H = 4.0 >
   3), furan O (3.0 > 2) and an aryl amide N (1.5 + 1 + 1 = 3.5 > 3) all
   lose a real bond.  The decoder now charges such bonds 1.0.

2. **The P cap was 3.**  The corpus is phosphate — P with four single bonds —
   so 98.8% of P atoms exceeded the cap and the 4th P-O bond was refused.
   P is present in 7.7% of the six-metal complexes.  Cap is now 5 (RDKit
   permits 3 or 5).

``BEFORE`` below is the pre-T78 rule, kept local so the artifact states
before/after without a git checkout.

Writes ``metrics/by_metric/wfa113_t78_element_cost_fit.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t78_element_cost_fit_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t78_element_cost_fit.json"

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
_HETERO = {5, 7, 8, 15, 16, 33, 34}

#: Pre-T78: aromatic cost 1.5 at every element, P cap 3.
COST_BEFORE = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
CAP_BEFORE = {15: 3}


def _charge(z: int, cls: int, cost: Dict[int, float], aromatic_hetero: bool) -> float:
    if aromatic_hetero and cls == 4 and z in _HETERO:
        return 1.0
    return cost[cls]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-complexes", type=int, default=0,
                    help="per shard; 0 = all")
    ap.add_argument("--six-metal-only", action="store_true", default=True,
                    help="restrict to complexes containing a project metal "
                         "(default on: that is the training cohort)")
    ap.add_argument("--all-metals", dest="six_metal_only", action="store_false")
    ap.add_argument("--include-hydrogens", action="store_true",
                    help="keep H.  Default OFF, because the trainer loads with "
                         "load_tmqm_clouds(include_hydrogens=False): the "
                         "decoder never sees an H, and a cap measured on a "
                         "graph that has them is measuring a different "
                         "molecule.  Including H inflates the N figure (69.7% "
                         "vs 61.7%) because a pyrrole N carries an N-H the "
                         "cloud does not have.")
    args = ap.parse_args(argv)

    from molmetal.data.tmqm_clouds import bo_to_bond_class
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        _CLASS_VALENCE_COST as COST_NOW,
        _COVALENT_VALENCE as CAP_NOW,
    )
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks

    class_name = {1: "single", 2: "double", 3: "triple", 4: "aromatic"}
    n_atoms: Counter = Counter()
    over_before: Counter = Counter()
    over_after: Counter = Counter()
    class_mix: Dict[str, Counter] = defaultdict(Counter)
    pat_before: Dict[str, Counter] = defaultdict(Counter)
    worst: Dict[str, List[dict]] = defaultdict(list)
    n_complexes = 0

    for path in BO_PATHS:
        if not path.is_file():
            continue
        n = 0
        for code, atoms in iter_bo_blocks([path]):
            syms = {idx: sym for idx, sym, _ in atoms}
            if args.six_metal_only and not (set(syms.values()) & _SIX):
                continue
            if not args.include_hydrogens:
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
            pat: Dict[int, List[str]] = defaultdict(list)
            for (i, j), bo in bonds.items():
                cls = bo_to_bond_class(bo)
                zi, zj = _Z_OF.get(syms.get(i, "")), _Z_OF.get(syms.get(j, ""))
                for idx, z in ((i, zi), (j, zj)):
                    if z is None:
                        continue
                    cb[idx] += _charge(z, cls, COST_BEFORE, False)
                    ca[idx] += _charge(z, cls, COST_NOW, True)
                    pat[idx].append(class_name[cls])
                if syms.get(i):
                    class_mix[syms[i]][class_name[cls]] += 1
                if syms.get(j):
                    class_mix[syms[j]][class_name[cls]] += 1
            for idx, sym in syms.items():
                z = _Z_OF.get(sym)
                if z is None or z not in CAP_NOW:
                    continue
                cap_b = float(CAP_BEFORE.get(z, CAP_NOW[z]))
                cap_a = float(CAP_NOW[z])
                n_atoms[sym] += 1
                if cb[idx] > cap_b + 1e-9:
                    over_before[sym] += 1
                    pat_before[sym][tuple(sorted(pat[idx]))] += 1
                    if len(worst[sym]) < 3:
                        worst[sym].append(
                            {"csd": code, "cost": round(cb[idx], 2), "cap": cap_b}
                        )
                if ca[idx] > cap_a + 1e-9:
                    over_after[sym] += 1
            n += 1
            n_complexes += 1
            if args.max_complexes and n >= args.max_complexes:
                break

    per_element = {}
    for sym in sorted(n_atoms, key=lambda s: -over_before[s]):
        c = n_atoms[sym]
        per_element[sym] = {
            "cap_now": CAP_NOW.get(_Z_OF.get(sym, -1)),
            "n_atoms": c,
            "frac_over_before": over_before[sym] / c if c else 0.0,
            "frac_over_after": over_after[sym] / c if c else 0.0,
            "class_mix": dict(class_mix[sym]),
            "top_over_patterns_before": [
                {"pattern": "".join(k), "n": v}
                for k, v in pat_before[sym].most_common(3)
            ],
            "examples": worst[sym],
        }

    payload = {
        "wave": "Wf-A113",
        "task": "T78",
        "date": "2026-10-28",
        "question": (
            "does _CLASS_VALENCE_COST exceed _COVALENT_VALENCE on the "
            "CORPUS's own bond graph, i.e. can the decoder rebuild the truth "
            "it was trained on?"
        ),
        "fix": {
            "aromatic_cost_at_heteroatoms": "1.5 -> 1.0 for Z in {5,7,8,15,16,33,34}",
            "P_cap": "3 -> 5 (corpus is phosphate: P with four single bonds)",
        },
        "measured": {
            "cohort": "six project metals" if args.six_metal_only else "all metals",
            "max_complexes_per_shard": args.max_complexes,
            "n_complexes": n_complexes,
            "n_atoms_total": sum(n_atoms.values()),
            "per_element": per_element,
        },
        "script": "molmetal/scripts/wfa113_t78_element_cost_fit_probe.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{'el':3s} {'atoms':>8s} {'before':>8s} {'after':>8s}")
    for s, v in per_element.items():
        if v["n_atoms"] < 200:
            continue
        print(
            f"{s:3s} {v['n_atoms']:8d} {v['frac_over_before']:8.4f} "
            f"{v['frac_over_after']:8.4f}"
        )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
