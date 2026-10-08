#!/usr/bin/env python
"""Wf-fix3lines m2 R^2 per-cohort REAL verification at n=4078.

Combines two existing canonical surfaces without duplicating either:

* the cohort loader + on-the-fly 32-d RDKit featurizer from
  ``wf331r5_r2_w2_chemprop.load_cohort(max_cohort_size=...)``
  (the Phase-3a expansion, ``_load_platinai_expanded``), and
* the per-cohort tier config + multi-seed K-fold Ridge CV from
  ``wf388a1_m2_per_cohort_real.train_eval_per_cohort``.

Running ``wf388a1_m2_per_cohort_real`` alone cannot produce these numbers: its
``load_platinai_4cell`` calls ``_wf331r5_load_cohort`` WITHOUT
``max_cohort_size``, so the cohort is pinned at the 200-row pre-computed
extended-features CSV and the tier resolves to ``n_lt_300`` (12-d, alpha=1.0).
This runner threads the expansion flag through and keeps every other part of
the Wf-388-A1 protocol byte-identical (same tier table, same
``ridge_r2_cv``, same alias + seed-offset handling).

CPU-only diagnostic. No retrain / panel / ablation, so the pre-flight v2 gate
is not required (per AGENTS.md §5.1 and the Wf-331-R5 result.json convention).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from molmetal.scripts.wf331r5_r2_w2_chemprop import load_cohort as _wf331r5_load_cohort
from molmetal.scripts.wf388a1_m2_per_cohort_real import (
    aggregate_per_cohort_r2,
    train_eval_per_cohort,
)


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def run(
    out_path: str,
    cohorts: Optional[List[str]] = None,
    max_cohort_size: Optional[int] = 4078,
    n_seeds: int = 5,
    n_folds: int = 5,
) -> Dict[str, Any]:
    if cohorts is None:
        cohorts = ["platinai_4cell", "hilbert"]
    started = time.time()
    per_cohort: List[Dict[str, Any]] = []
    for cohort in cohorts:
        t0 = time.time()
        X, y, smiles, target_name = _wf331r5_load_cohort(
            cohort=cohort,
            max_samples=max_cohort_size or 2000,
            seed=0,
            max_cohort_size=max_cohort_size,
        )
        res = train_eval_per_cohort(
            cohort_data=(X, y, smiles, target_name),
            n_seeds=n_seeds,
            n_folds=n_folds,
            wass_w=0.05,
            epochs=30,
            cohort=cohort,
        )
        res["cohort"] = cohort
        res["target_name"] = target_name
        res["loader_max_cohort_size"] = max_cohort_size
        res["loader"] = (
            "wf331r5_r2_w2_chemprop.load_cohort("
            "max_cohort_size=%s) -> _load_platinai_expanded" % max_cohort_size
        )
        res["elapsed_sec"] = float(time.time() - t0)
        per_cohort.append(res)

    agg = aggregate_per_cohort_r2(per_cohort)
    final = {
        "wf": "wf-fix3lines (m2 R^2 per-cohort REAL verification at n=4078)",
        "task": (
            "m2 R^2 per-cohort REAL verification with the platinai_4cell cohort "
            "expanded from the 200-row pre-computed extended-features CSV to the "
            "full %s-row 12-d CSV (on-the-fly 32-d RDKit featurization). "
            "Per-cohort tier config + Ridge alpha selected by cohort size, "
            "%d seeds x %d-fold CV." % (max_cohort_size, n_seeds, n_folds)
        ),
        "protocol": (
            "Loader = wf331r5_r2_w2_chemprop.load_cohort(max_cohort_size=...) "
            "(the Phase-3a expansion). Tier table + Ridge CV = "
            "wf388a1_m2_per_cohort_real.train_eval_per_cohort, unmodified. "
            "No truncation/random subsampling: rows are the CSV's own row "
            "order, capped at max_cohort_size."
        ),
        "cohorts_tested": cohorts,
        "max_cohort_size": max_cohort_size,
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "per_cohort": per_cohort,
        "aggregate": agg,
        "honest_framing": (
            "MEASURED (not PROJECTED). CPU-only Ridge closed-form, no torch "
            "autograd, no encoder forward pass, no GPU retrain; pre-flight v2 "
            "not required for a CPU-only diagnostic. NOTE: the n_gte_1000 tier "
            "requests 50 features (12-d RDKit + 12-d chemprop v3 + 24-d "
            "Hilbert spectral) but the loader supplies only the 32-d RDKit "
            "block, so n_features_used=32 — this is a 32-d measurement scored "
            "under the n_gte_1000 tier's alpha=0.1, NOT a full 50-d ensemble."
        ),
        "elapsed_sec_total": float(time.time() - started),
    }
    out = _resolve(out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=False)
    return final


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--out-path",
        default="molmetal/reports/wf_fix3lines_plat_expand_2026_10_09/result.json",
    )
    p.add_argument("--cohorts", nargs="+", default=None)
    p.add_argument("--max-cohort-size", type=int, default=4078)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args(argv)
    cohorts = args.cohorts
    if cohorts is not None and len(cohorts) == 1 and "," in cohorts[0]:
        cohorts = [c.strip() for c in cohorts[0].split(",") if c.strip()]
    final = run(
        out_path=args.out_path,
        cohorts=cohorts,
        max_cohort_size=args.max_cohort_size,
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
    )
    for c in final["per_cohort"]:
        print(
            f"  cohort={c['cohort']:>16s}  n={c['cohort_size']:>5d}  "
            f"tier={c['tier']:>12s}  alpha={c['alpha_used']:<4} "
            f"n_feat={c['n_features_used']:>3d} (avail {c['n_features_available']})  "
            f"R2={c['r2_mean']:+.4f} +/- {c['r2_std']:.4f}  "
            f"splits={c['n_splits_effective']}  {c['verdict']}"
        )
    print(f"  elapsed={final['elapsed_sec_total']:.1f}s  out={final['elapsed_sec_total']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
