#!/usr/bin/env python
"""L3 — K-shot adaptation shot-curve.

How many labelled ligands from a held-out metal recover the zero-shot
cross-metal collapse?  Four arms share the cohort, the ligand-disjoint LOMO
split, the backbone and 10 seeds; only the support-set size K changes.  Each
arm's zero-shot baseline is scored on the IDENTICAL query rows in-run, so the
delta is paired.

Reports, per K: zero-shot query mean, k-shot query mean, paired delta with a
bootstrap CI and a one-sided p, and the per-metal delta.  Holm-Bonferroni
across the four K values.

MEASURED only; every number comes from the arm reports on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KS = (5, 10, 20, 40)
METALS = ("Ru", "Ir", "Rh", "Os", "Re")


def _cells(k: int):
    rep = json.loads((PROJECT_ROOT / f"molmetal/reports/l3_kshot_k{k}/metrics.json").read_text())
    out: Dict[str, Dict[str, float]] = {}
    for fold in rep["per_fold"]:
        m = str(fold["metal"])
        out[m] = {}
        for s, rec in fold["per_seed"].items():
            if "kshot" in rec:
                out[m][str(s)] = {
                    "zs": float(rec["kshot"]["zero_shot_query_auc"]),
                    "ks": float(rec["kshot"]["kshot_query_auc"]),
                }
    return out


def _paired_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n = delta.size
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = float(delta[rng.integers(0, n, n)].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_greater = float(np.mean(means <= 0.0))
    return {
        "n_cells": int(n),
        "mean_delta": float(delta.mean()),
        "delta_std": float(delta.std(ddof=1)) if n > 1 else 0.0,
        "ci95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(float(lo) > 0.0 or float(hi) < 0.0),
        "p_one_sided_greater": p_greater,
    }


def _holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(items)
    adj: Dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, float(p) * (n - rank)))
        adj[k] = running
    return adj


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=42)
    p.add_argument("--out-metric", default="metrics/by_metric/l3_kshot_curve.json")
    args = p.parse_args(argv)

    curve: Dict[str, object] = {}
    pvals: Dict[str, float] = {}
    for k in KS:
        try:
            cells = _cells(k)
        except FileNotFoundError:
            continue
        delta = np.asarray(
            [cells[m][s]["ks"] - cells[m][s]["zs"] for m in METALS for s in cells[m]],
            dtype=np.float64,
        )
        bt = _paired_bootstrap(delta, args.n_bootstrap, args.boot_seed)
        zs = np.asarray([cells[m][s]["zs"] for m in METALS for s in cells[m]])
        ks = np.asarray([cells[m][s]["ks"] for m in METALS for s in cells[m]])
        curve[str(k)] = {
            "k": k,
            "zero_shot_query_mean": float(zs.mean()),
            "kshot_query_mean": float(ks.mean()),
            "paired_delta": bt,
            "per_metal_delta": {
                m: float(np.mean([cells[m][s]["ks"] - cells[m][s]["zs"] for s in cells[m]]))
                for m in METALS
            },
            "per_metal_kshot": {
                m: float(np.mean([cells[m][s]["ks"] for s in cells[m]])) for m in METALS
            },
            "n_seeds": len(cells[METALS[0]]),
        }
        pvals[f"k{k}"] = float(bt["p_one_sided_greater"])

    holm = _holm(pvals) if pvals else {}
    report = {
        "metric": "l3_kshot_curve",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/l3_kshot_curve_analysis.py",
        "description": (
            "L3 shot-curve: the frozen zero-shot cross-metal model is fine-tuned "
            "on K labelled support ligands from the held-out metal and re-scored "
            "on the disjoint query set.  K in {5,10,20,40}, 10 seeds, 5 metals."
        ),
        "protocol": {
            "cohort": "MetalCytotoxDataset + CytotoxFilter(24h, ic50>=0.01uM) + collapse_duplicates, per metal",
            "split": "leave-one-metal-out, ligand-disjoint",
            "backbone": "chemprop v2 AtomMessagePassing d_h=300 depth=3",
            "ft_epochs": 20,
            "seeds": [42, 43, 44, 49, 123, 7, 8, 9, 10, 11],
            "significance": "paired bootstrap over the (metal, seed) cells per K, 10000 resamples, seed 42; Holm-Bonferroni across the 4 K values",
        },
        "curve": curve,
        "holm": {
            "correction": "Holm-Bonferroni (family = 4 K values vs their own paired zero-shot query baseline)",
            "n_family": len(pvals),
            "p_one_sided_raw": pvals,
            "p_one_sided_holm": holm,
            "significant_after_holm_0.05": [a for a, v in holm.items() if v < 0.05],
        },
        "honest_framing_note": (
            "MEASURED only.  This is an ADAPTATION result: the support labels come "
            "from the held-out metal.  The zero-shot LOMO column stays the paper's "
            "Table-1 cross number; this curve quantifies how much a few known "
            "compounds recover."
        ),
    }
    out = PROJECT_ROOT / args.out_metric
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    for k in sorted(curve, key=int):
        c = curve[k]
        d = c["paired_delta"]
        print(
            f"K={c['k']:3d} zs={c['zero_shot_query_mean']:.4f} "
            f"ks={c['kshot_query_mean']:.4f} delta={d['mean_delta']:+.4f} "
            f"CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
            f"p_raw={d['p_one_sided_greater']:.4f} p_holm={holm.get('k'+str(c['k']), float('nan')):.4f}"
        )
    print("significant after Holm:", report["holm"]["significant_after_holm_0.05"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
