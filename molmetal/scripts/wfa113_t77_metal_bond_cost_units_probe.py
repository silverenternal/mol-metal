#!/usr/bin/env python
"""Wf-A113 (T77) — the decoder charges metal bonds in bond-ORDER units against
a COORDINATION-NUMBER ceiling.

``_select_bonds`` builds ``remaining = _cap_for_z(z)`` and then subtracts
``_CLASS_VALENCE_COST[class]`` per accepted bond, where aromatic costs 1.5.
For a carbon that is right: ``_COVALENT_VALENCE[6] = 4`` is a bond-ORDER sum, so
two aromatic bonds (1.5 + 1.5) plus one single (1.0) is a legal sp2 carbon.

For a metal it is not.  Wf-A113 (T72) set the metal caps from the corpus
coordination number (MND p99: Ru 10, Rh 10, Os 10, Ir 9, Re 9), which is a
COUNT of metal-ligand bonds.  Charging 1.5 for an aromatic-class metal bond
therefore spends 1.5 of a budget that is denominated in bonds: six aromatic
metal bonds consume 9 of a 10-bond budget, so the selector stops at 7 bonds.

The same file's own test (``test_cfm_metal_valence_caps.py``) documents this
category error in the other direction — deriving a bond-ORDER cap from a
coordination number made caps "roughly 2x too high".  The decoder now has the
mirror image: a COUNT cap spent in bond-order units.

This probe measures how much of the cohort is exposed: the class mix of
metal-incident bonds in the corpus, and per metal the p99 of
``sum(cost)`` vs ``count`` over the true bond graph.  If the two are equal the
defect is latent; the gap is what a mis-classed aromatic metal bond would cost.

Writes ``metrics/by_metric/wfa113_t77_metal_bond_cost_units.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t77_metal_bond_cost_units_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t77_metal_bond_cost_units.json"

BO_PATHS = [
    ROOT / ".data/molmetal/tmQM/tmQM_X1.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X2.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X3.BO.gz",
]

METALS = {"Ru", "Rh", "Ir", "Os", "Re", "Pt", "Pd"}

#: Mirrors ``cfm_decoder_yuelbond._CLASS_VALENCE_COST``.
COST = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[i])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-complexes", type=int, default=3000)
    args = ap.parse_args(argv)

    from molmetal.data.tmqm_clouds import bo_to_bond_class
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        TMQM_MEASURED_METAL_COORDINATION_CAPS,
    )
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks

    class_of = {1: "single", 2: "double", 3: "triple", 4: "aromatic"}
    metal_class: Counter = Counter()
    per_metal_cost: Dict[str, List[float]] = defaultdict(list)
    per_metal_count: Dict[str, List[float]] = defaultdict(list)
    over_ceiling = Counter()
    over_cap_after = Counter()
    n_clouds = Counter()
    worst = []
    z_of = {"Ru": 44, "Rh": 45, "Ir": 77, "Os": 76, "Re": 75, "Pt": 78, "Pd": 46}

    for path in BO_PATHS:
        if not path.is_file():
            continue
        n = 0
        for code, atoms in iter_bo_blocks([path]):
            syms = {idx: sym for idx, sym, _ in atoms}
            bonds: Dict[Tuple[int, int], float] = {}
            for idx, _sym, nbrs in atoms:
                for nbr_idx, bo in nbrs:
                    if nbr_idx == idx or bo < 0.3:
                        continue
                    key = (min(idx, nbr_idx), max(idx, nbr_idx))
                    if key not in bonds or bo > bonds[key]:
                        bonds[key] = bo
            # Per metal atom: count and cost of its incident bonds.
            mcount: Dict[int, float] = defaultdict(float)
            mcost: Dict[int, float] = defaultdict(float)
            for (i, j), bo in bonds.items():
                cls = bo_to_bond_class(bo)
                si, sj = syms.get(i, "?"), syms.get(j, "?")
                if si in METALS:
                    mcount[i] += 1
                    mcost[i] += COST[cls]
                    metal_class[(si, class_of[cls])] += 1
                if sj in METALS:
                    mcount[j] += 1
                    mcost[j] += COST[cls]
                    metal_class[(sj, class_of[cls])] += 1
            for idx, sym in syms.items():
                if sym not in METALS:
                    continue
                per_metal_count[sym].append(mcount[idx])
                per_metal_cost[sym].append(mcost[idx])
                n_clouds[sym] += 1
                if mcost[idx] > mcount[idx]:
                    over_ceiling[sym] += 1
                    if len(worst) < 6 and mcost[idx] - mcount[idx] > 0.4:
                        worst.append(
                            {
                                "csd": code,
                                "metal": sym,
                                "n_bonds": int(mcount[idx]),
                                "cost_sum": round(mcost[idx], 2),
                            }
                        )
                # AFTER the T77 fix the metal is charged 1.0 per bond, so the
                # only way to exceed the budget is to exceed the CAP itself.
                _cap = TMQM_MEASURED_METAL_COORDINATION_CAPS.get(z_of.get(sym, -1))
                if _cap is not None and mcount[idx] > _cap:
                    over_cap_after[sym] += 1
            n += 1
            if args.max_complexes and n >= args.max_complexes:
                break

    summary = {}
    for sym in sorted(per_metal_count):
        summary[sym] = {
            "n_complexes": n_clouds[sym],
            "count_p50": _pct(per_metal_count[sym], 0.50),
            "count_p99": _pct(per_metal_count[sym], 0.99),
            "cost_p50": _pct(per_metal_cost[sym], 0.50),
            "cost_p99": _pct(per_metal_cost[sym], 0.99),
            "n_cost_exceeds_count": over_ceiling[sym],
            "frac_cost_exceeds_count": (
                over_ceiling[sym] / n_clouds[sym] if n_clouds[sym] else 0.0
            ),
            "coord_cap": TMQM_MEASURED_METAL_COORDINATION_CAPS.get(z_of.get(sym, -1)),
            "n_count_exceeds_cap_after_fix": over_cap_after[sym],
            "frac_count_exceeds_cap_after_fix": (
                over_cap_after[sym] / n_clouds[sym] if n_clouds[sym] else 0.0
            ),
        }

    payload = {
        "wave": "Wf-A113",
        "task": "T77",
        "date": "2026-10-28",
        "question": (
            "does the decoder spend a COUNT ceiling (metal MND p99) in "
            "bond-ORDER units (_CLASS_VALENCE_COST, aromatic = 1.5)?"
        ),
        "measured": {
            "max_complexes_per_shard": args.max_complexes,
            "metal_bond_class_counts": {
                f"{m}:{c}": n for (m, c), n in sorted(metal_class.items())
            },
            "per_metal": summary,
            "examples_cost_gt_count": worst,
        },
        "verdict": (
            "the units differ whenever a metal-incident bond is not classed "
            "single; the corpus shows how often that is, and the p99 cost vs "
            "count gap is the budget the selector loses"
        ),
        "script": "molmetal/scripts/wfa113_t77_metal_bond_cost_units_probe.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["measured"]["per_metal"], indent=2))
    print(json.dumps(payload["measured"]["metal_bond_class_counts"], indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
