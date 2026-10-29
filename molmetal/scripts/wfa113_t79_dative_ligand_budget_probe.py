#!/usr/bin/env python
"""Wf-A113 (T79) — does the ligand endpoint pay for its bond to the metal?

The selector's cap table is a LIGAND-valence table.  A metal-ligand bond is
dative from the donor's side: the electron pair is the ligand's lone pair, so
the donor's valence is unchanged.  RDKit implements exactly this in
``cleanupOrganometallics()``, which ``SanitizeMol`` runs on every mol the
decoder produces — it rewrites a single bond between a metal and a hypervalent
atom into ``BondType.DATIVE``, and a dative bond does not count towards the
donor's explicit valence (rdkit/rdkit#8312; verified on RDKit 2026.03.6, see
``test_wfa113_t79_dative_ligand_budget.py``).

Charging the ligand for that bond therefore makes the SELECTOR stricter than
the validator it feeds: it drops a bond RDKit would have kept, which opens a
ring and turns a one-fragment decode into a multi-fragment one.  This probe
measures how often that happens on the corpus's own bond graph — i.e. how
often the decoder cannot rebuild the truth it was trained on.

``BEFORE`` = the ligand endpoint pays its class cost for a metal bond (the
pre-T79 rule).  ``AFTER`` = a bond to a metal costs the ligand endpoint 0.
Both are computed from the same graph, so the difference is the rule alone.

Writes ``metrics/by_metric/wfa113_t79_dative_ligand_budget.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t79_dative_ligand_budget_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t79_dative_ligand_budget.json"

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
_METALS = {"Fe", "Ni", "Cu", "Zn", "Ru", "Rh", "Pd", "Re", "Os", "Ir", "Pt", "Au"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-complexes", type=int, default=0,
                    help="per shard; 0 = all")
    ap.add_argument("--include-hydrogens", action="store_true",
                    help="keep H.  Default OFF: load_tmqm_clouds drops them, "
                         "so the decoder never sees an H and a cap measured on "
                         "an H-bearing graph measures a different molecule.")
    args = ap.parse_args(argv)

    from molmetal.data.tmqm_clouds import bo_to_bond_class
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        _CLASS_VALENCE_COST as COST,
        _COVALENT_VALENCE as CAP,
        _endpoint_valence_cost,
    )
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks

    def is_metal_z(z: int) -> bool:
        return z in (44, 45, 46, 75, 76, 77, 78, 26, 28, 29, 30, 79)

    def charge_before(order: int, z_self: int, z_other: int) -> float:
        """Pre-T79: the ligand endpoint pays its own class cost."""
        if is_metal_z(z_self):
            return 1.0
        if int(order) == 4 and z_self in {5, 7, 8, 15, 16, 33, 34}:
            return 1.0
        return COST.get(int(order), 1.0)

    class_name = {1: "single", 2: "double", 3: "triple", 4: "aromatic"}
    n_atoms: Counter = Counter()
    over_before: Counter = Counter()
    over_after: Counter = Counter()
    only_by_metal: Counter = Counter()
    metal_degree: Counter = Counter()
    pat_before: Dict[str, Counter] = defaultdict(Counter)
    worst: Dict[str, List[dict]] = defaultdict(list)
    n_complexes = 0
    cx_over_before = 0
    cx_over_after = 0

    for path in BO_PATHS:
        if not path.is_file():
            continue
        n = 0
        for code, atoms in iter_bo_blocks([path]):
            syms = {idx: sym for idx, sym, _ in atoms}
            if not (set(syms.values()) & _SIX):
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
            n_metal_nbr: Counter = Counter()
            pat: Dict[int, List[str]] = defaultdict(list)
            for (i, j), bo in bonds.items():
                cls = bo_to_bond_class(bo)
                zi, zj = _Z_OF.get(syms.get(i, "")), _Z_OF.get(syms.get(j, ""))
                if zi is None or zj is None:
                    continue
                cb[i] += charge_before(cls, zi, zj)
                cb[j] += charge_before(cls, zj, zi)
                ca[i] += _endpoint_valence_cost(cls, zi, zj, is_metal_z)
                ca[j] += _endpoint_valence_cost(cls, zj, zi, is_metal_z)
                if is_metal_z(zj) and not is_metal_z(zi):
                    n_metal_nbr[i] += 1
                if is_metal_z(zi) and not is_metal_z(zj):
                    n_metal_nbr[j] += 1
                pat[i].append(class_name[cls])
                pat[j].append(class_name[cls])

            cx_b = cx_a = False
            for idx, sym in syms.items():
                z = _Z_OF.get(sym)
                if z is None or z not in CAP or is_metal_z(z):
                    continue
                cap = float(CAP[z])
                n_atoms[sym] += 1
                metal_degree[n_metal_nbr[idx]] += 1
                b = cb[idx] > cap + 1e-9
                a = ca[idx] > cap + 1e-9
                if b:
                    over_before[sym] += 1
                    pat_before[sym][tuple(sorted(pat[idx]))] += 1
                    cx_b = True
                    if len(worst[sym]) < 3:
                        worst[sym].append(
                            {"csd": code, "cost": round(cb[idx], 2), "cap": cap}
                        )
                if a:
                    over_after[sym] += 1
                    cx_a = True
                if b and not a:
                    only_by_metal[sym] += 1
            cx_over_before += int(cx_b)
            cx_over_after += int(cx_a)
            n += 1
            n_complexes += 1
            if args.max_complexes and n >= args.max_complexes:
                break

    per_element = {}
    for sym in sorted(n_atoms, key=lambda s: -over_before[s]):
        c = n_atoms[sym]
        per_element[sym] = {
            "cap": CAP.get(_Z_OF.get(sym, -1)),
            "n_atoms": c,
            "frac_over_before": over_before[sym] / c if c else 0.0,
            "frac_over_after": over_after[sym] / c if c else 0.0,
            "frac_over_only_due_to_metal": only_by_metal[sym] / c if c else 0.0,
            "top_over_patterns_before": [
                {"pattern": "".join(k), "n": v}
                for k, v in pat_before[sym].most_common(3)
            ],
            "examples": worst[sym],
        }

    payload = {
        "wave": "Wf-A113",
        "task": "T79",
        "date": "2026-10-28",
        "question": (
            "does the selector charge the LIGAND endpoint for a bond to a "
            "metal, i.e. is it stricter than RDKit's cleanupOrganometallics, "
            "which makes such a bond dative and free at the donor?"
        ),
        "fix": {
            "rule": (
                "a bond whose partner is a metal costs the ligand endpoint 0; "
                "the metal endpoint still pays 1.0 (T77)"
            ),
            "where": (
                "molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py::"
                "_endpoint_valence_cost, shared by _select_bonds and "
                "_bridge_components"
            ),
        },
        "measured": {
            "cohort": "six project metals",
            "include_hydrogens": bool(args.include_hydrogens),
            "max_complexes_per_shard": args.max_complexes,
            "n_complexes": n_complexes,
            "n_atoms_total": sum(n_atoms.values()),
            "n_complexes_with_a_ligand_atom_over_cap_before": cx_over_before,
            "n_complexes_with_a_ligand_atom_over_cap_after": cx_over_after,
            "frac_complexes_over_before": (
                cx_over_before / n_complexes if n_complexes else 0.0
            ),
            "frac_complexes_over_after": (
                cx_over_after / n_complexes if n_complexes else 0.0
            ),
            "ligand_atoms_by_metal_neighbour_count": {
                str(k): v for k, v in sorted(metal_degree.items())
            },
            "per_element": per_element,
        },
        "script": "molmetal/scripts/wfa113_t79_dative_ligand_budget_probe.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    m = payload["measured"]
    print(
        f"complexes {n_complexes}  over-cap: {cx_over_before} "
        f"({m['frac_complexes_over_before']:.4f}) -> {cx_over_after} "
        f"({m['frac_complexes_over_after']:.4f})"
    )
    print(f"{'el':3s} {'atoms':>8s} {'before':>8s} {'after':>8s} {'metal-only':>11s}")
    for s, v in per_element.items():
        if v["n_atoms"] < 200:
            continue
        print(
            f"{s:3s} {v['n_atoms']:8d} {v['frac_over_before']:8.4f} "
            f"{v['frac_over_after']:8.4f} {v['frac_over_only_due_to_metal']:11.4f}"
        )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
