#!/usr/bin/env python3
"""Multi-seed scaffold-split stability for the metal-transfer probe (BLK-PA1).

`metal_activity_transfer_probe.py --with-scaffold-split` reports a single-seed
scaffold-split AUC. The within-metal column of Table 1 is a 5-seed range, so the
scaffold column needs one too. The scaffold grouping is deterministic
(largest-first, no RNG), so the only seed-dependence is the XGBoost fit; this
wrapper runs the probe over a seed list and reports per-metal
mean / min / max / spread for the random-CV, scaffold-split and cross-metal
columns **on the same seeds**, plus the per-seed net metal effect.

Usage
-----
::

    .venv/bin/python molmetal/scripts/metal_activity_transfer_scaffold_stability.py \
        --out-metric metrics/by_metric/metal_activity_transfer_scaffold_seed_stability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from molmetal.scripts.metal_activity_transfer_probe import METALS, run

# Same seeds the existing metal_activity_transfer_seed_stability.json uses, so
# the scaffold column and the within-metal column are quoted over one seed set.
DEFAULT_SEEDS = [42, 43, 44, 49, 123]


def _stats(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "n_seeds": int(len(a)),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "spread": float(a.max() - a.min()),
        "std": float(a.std()),
    }


def _fold_auc(rep: Dict[str, object], section: str, metal: str) -> Optional[float]:
    """AUC for `metal` out of a report's within/cross section, or None."""
    if section == "within_metal_cv":
        d = rep.get("within_metal_cv", {}).get(metal, {})
        return float(d["roc_auc"]) if d.get("status") == "MEASURED" else None
    if section == "within_metal_scaffold_split":
        d = rep.get("within_metal_scaffold_split", {}).get(metal, {})
        return float(d["roc_auc"]) if d.get("status") == "MEASURED" else None
    if section == "cross_metal":
        for f in rep.get("cross_metal_lomo_dedup", {}).get("per_fold", []):
            if f.get("metal") == metal and f.get("status") == "MEASURED":
                return float(f["roc_auc"])
        return None
    raise ValueError(section)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="5-seed scaffold-split stability for the metal-transfer probe."
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/metal_activity_transfer_scaffold_seed_stability.json",
    )
    args = p.parse_args(argv)

    seeds = list(args.seeds)
    reports: Dict[int, Dict[str, object]] = {}
    for s in seeds:
        reports[s] = run(
            metals=list(args.metals),
            seed=s,
            radius=args.radius,
            nbits=args.nbits,
            with_scaffold_split=True,
        )

    per_seed_columns: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        "within_random": {},
        "within_scaffold": {},
        "cross_metal": {},
    }
    net_per_seed: Dict[str, List[float]] = {}
    n_scaffolds: Dict[str, int] = {}

    for m in args.metals:
        for col, section in (
            ("within_random", "within_metal_cv"),
            ("within_scaffold", "within_metal_scaffold_split"),
            ("cross_metal", "cross_metal"),
        ):
            per_seed_columns[col][m] = {
                str(s): _fold_auc(reports[s], section, m) for s in seeds
            }
        # deterministic scaffold count (identical across seeds)
        sc = reports[seeds[0]].get("within_metal_scaffold_split", {}).get(m, {})
        if sc.get("status") == "MEASURED":
            n_scaffolds[m] = int(sc["n_scaffolds"])
        # per-seed net metal effect = scaffold - cross
        vals = []
        for s in seeds:
            a = _fold_auc(reports[s], "within_metal_scaffold_split", m)
            b = _fold_auc(reports[s], "cross_metal", m)
            if a is not None and b is not None:
                vals.append(a - b)
        net_per_seed[m] = vals

    stability: Dict[str, object] = {}
    for m in args.metals:
        entry: Dict[str, object] = {
            "n_scaffolds": n_scaffolds.get(m),
            "within_random": _stats(
                [v for v in per_seed_columns["within_random"][m].values() if v is not None]
            ),
            "within_scaffold": _stats(
                [v for v in per_seed_columns["within_scaffold"][m].values() if v is not None]
            ),
            "cross_metal": _stats(
                [v for v in per_seed_columns["cross_metal"][m].values() if v is not None]
            ),
            "net_metal_effect": _stats(net_per_seed[m]),
            "per_seed": {
                "within_random": per_seed_columns["within_random"][m],
                "within_scaffold": per_seed_columns["within_scaffold"][m],
                "cross_metal": per_seed_columns["cross_metal"][m],
            },
        }
        stability[m] = entry

    report = {
        "metric": "metal_activity_transfer_scaffold_seed_stability",
        "description": (
            "Per-metal seed range for the BLK-PA1 scaffold-split control, over the "
            "same seeds as metal_activity_transfer_seed_stability.json. The "
            "scaffold grouping is deterministic (largest-first, no RNG), so the "
            "seed spread here is XGBoost-fit noise only; the random within-metal "
            "column additionally varies the fold assignment. net_metal_effect = "
            "within_scaffold - cross_metal."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/metal_activity_transfer_scaffold_stability.py",
        "wraps": "molmetal/scripts/metal_activity_transfer_probe.py --with-scaffold-split",
        "model": f"morgan_fp(radius={args.radius}, {args.nbits}bit) + xgboost",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026)",
        "seeds": seeds,
        "scaffold_rule": (
            "Bemis-Murcko scaffold of the desalted largest organic fragment "
            "(FragmentParent -> re-parse -> MurckoScaffold); largest-first, "
            "row-count-greedy, 5 folds. Raw multi-fragment metal-complex SMILES "
            "yield an empty Murcko for 100% of rows and must be desalted first."
        ),
        "per_metal": stability,
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(report, indent=2, default=float))

    print(f"{'metal':>5} {'n_scaf':>7} {'random':>18} {'scaffold':>18} "
          f"{'cross':>18} {'net_metal':>18}")
    for m in args.metals:
        e = stability[m]
        def fmt(k):
            s = e[k]
            return f"{s['mean']:.4f}[{s['min']:.3f},{s['max']:.3f}]" if s else "n/a"
        print(f"{m:>5} {str(e['n_scaffolds']):>7} {fmt('within_random'):>18} "
              f"{fmt('within_scaffold'):>18} {fmt('cross_metal'):>18} "
              f"{fmt('net_metal_effect'):>18}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
