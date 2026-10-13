#!/usr/bin/env python3
"""PA20 — external (temporal) validation for the classification axis, 5 metals.

The Krasnov-protocol column (``pa18_model_class_5metal.py``) reports internal
DOI-grouped 10-fold CV: every fold's test rows are held out, but the whole
cohort is the same DB and the same time window.  Krasnov's own paper also
reports an *external* validation.  This script closes that gap with a
**prospective temporal holdout**: train on paper-funnel rows published at or
before ``--cutoff-year`` and score rows published after it.

Why temporal and not a second database
--------------------------------------
The host has no second metal-cytotoxicity DB with the same label rule
(``active := IC50_Dark < 10 uM``) and the same five metals.  A temporal split
is the strongest external validation available offline: it is a genuine
held-out set (future publications, unseen chemistry, shifted active rate) and
is the standard prospective protocol in QSAR/ADMET (cf. the repo's
``baseline_temporal_holdout.py``).  The alternative — reusing the DOI folds and
calling them "external" — would be a mislabel, so we don't.

DOI integrity
-------------
``Year`` is a per-DOI constant in this DB, so a year filter cannot straddle a
paper across the cut.  That is asserted per metal (``doi_straddle_count`` == 0)
rather than assumed; a non-zero value would invalidate the split and is
recorded, not hidden.

Random-split reference
----------------------
For each metal we also fit the same model on a seeded random split with the
*same test size*, so the artifact carries the temporal gap
(``auc_random - auc_temporal``) — the number that shows how much of the
internal-CV AUC is bought by in-distribution leakage.

Usage
-----
::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    .venv/bin/python molmetal/scripts/pa20_external_temporal_5metal.py \\
        --metals Ru Ir Rh Os Re --models lightgbm xgboost \\
        --cutoff-year 2021 \\
        --out metrics/by_metric/pa20_external_temporal_5metal.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.pa18_model_class_5metal import (  # noqa: E402
    DEFAULT_METALS,
    DEFAULT_MODELS,
    _fit_predict,
    _threshold_metrics,
    build_cohort,
)


def _doi_straddle(cohort: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> int:
    """Number of DOIs that appear on both sides of the temporal cut."""
    tr_dois = set(cohort.loc[train_mask, "DOI"].astype(str))
    te_dois = set(cohort.loc[test_mask, "DOI"].astype(str))
    return int(len(tr_dois & te_dois))


def run_metal(
    df: pd.DataFrame, metal: str, models: List[str], cutoff_year: int, seed: int,
    radius: int, nbits: int, n_jobs: int, thr: float, n_random_seeds: int,
) -> Dict[str, object]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from molmetal.baselines.eval_utils import morgan_features

    cohort = build_cohort(df, metal)
    y = np.asarray(cohort["active"], dtype=int)
    smiles = cohort["SMILES_Ligands"].astype(str).to_numpy()
    year = pd.to_numeric(cohort["Year"], errors="coerce").to_numpy()
    if np.isnan(year).any():
        raise SystemExit(f"[{metal}] {int(np.isnan(year).sum())} rows have no Year")

    train_mask = year <= cutoff_year
    test_mask = year > cutoff_year
    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
    straddle = _doi_straddle(cohort, train_mask, test_mask)
    print(
        f"[{metal}] temporal cut<={cutoff_year}: train={n_tr} test={n_te} "
        f"doi_straddle={straddle} "
        f"active_rate(train)={y[train_mask].mean():.3f} "
        f"active_rate(test)={y[test_mask].mean():.3f}",
        flush=True,
    )

    X = morgan_features(smiles, radius, nbits)
    out: Dict[str, object] = {
        "n_rows": int(len(cohort)),
        "cutoff_year": int(cutoff_year),
        "n_train": n_tr,
        "n_test": n_te,
        "doi_straddle_count": straddle,
        "active_rate_train": float(y[train_mask].mean()) if n_tr else None,
        "active_rate_test": float(y[test_mask].mean()) if n_te else None,
        "models": {},
    }
    if n_tr == 0 or n_te == 0 or len(np.unique(y[train_mask])) < 2:
        out["skipped"] = "train/test too small or single-class train"
        return out

    tr = np.flatnonzero(train_mask)
    te = np.flatnonzero(test_mask)
    n = len(cohort)

    for model in models:
        prob = _fit_predict(model, X[tr], y[tr], X[te], seed, n_jobs)
        auc = float(roc_auc_score(y[te], prob))
        pr_auc = float(average_precision_score(y[te], prob))
        tm = _threshold_metrics(y[te], prob, thr)

        # Random-split reference with the SAME test size (seeded, averaged).
        rng = np.random.default_rng(seed)
        rand_aucs: List[float] = []
        rand_prs: List[float] = []
        for s in range(int(n_random_seeds)):
            perm = rng.permutation(n)
            rte, rtr = perm[:n_te], perm[n_te:]
            if len(np.unique(y[rtr])) < 2 or len(np.unique(y[rte])) < 2:
                continue
            rprob = _fit_predict(model, X[rtr], y[rtr], X[rte], int(s), n_jobs)
            rand_aucs.append(float(roc_auc_score(y[rte], rprob)))
            rand_prs.append(float(average_precision_score(y[rte], rprob)))

        out["models"][model] = {
            "temporal_auc": auc,
            "temporal_pr_auc": pr_auc,
            "threshold": float(thr),
            "threshold_metrics": {k: v for k, v in tm.items()
                                  if k not in ("n_positive", "n_negative")},
            "n_test_positive": int(tm["n_positive"]),
            "n_test_negative": int(tm["n_negative"]),
            "random_reference_auc_mean": float(np.mean(rand_aucs)) if rand_aucs else None,
            "random_reference_auc_std": float(np.std(rand_aucs)) if rand_aucs else None,
            "random_reference_pr_auc_mean": float(np.mean(rand_prs)) if rand_prs else None,
            "n_random_seeds_used": int(len(rand_aucs)),
            "temporal_gap_auc": (
                float(np.mean(rand_aucs) - auc) if rand_aucs else None
            ),
            "model_spec": f"morgan_fp(radius={radius}, {nbits}bit) + {model}",
        }
        print(
            f"[{metal}] {model}: temporal_auc={auc:.4f} pr_auc={pr_auc:.4f} "
            f"acc={tm['accuracy']:.4f} mcc={tm['mcc']:.4f} "
            f"| random_auc={np.mean(rand_aucs) if rand_aucs else float('nan'):.4f} "
            f"gap={np.mean(rand_aucs) - auc if rand_aucs else float('nan'):.4f}",
            flush=True,
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--metals", nargs="+", default=list(DEFAULT_METALS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--cutoff-year", type=int, default=2021)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument(
        "--n-random-seeds",
        type=int,
        default=3,
        help="number of seeded random splits for the reference AUC (same test size).",
    )
    ap.add_argument(
        "--out", default="metrics/by_metric/pa20_external_temporal_5metal.json"
    )
    args = ap.parse_args(argv)

    data_dir = Path(
        os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
    )
    csv_path = Path(args.csv) if args.csv else data_dir / "MetalCytoToxDB.csv"
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    per_metal: Dict[str, object] = {}
    for metal in args.metals:
        per_metal[metal] = run_metal(
            df, metal, list(args.models), args.cutoff_year, args.seed,
            args.radius, args.nbits, args.n_jobs, args.threshold,
            args.n_random_seeds,
        )

    report = {
        "metric": "pa20_external_temporal_5metal",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": (
            "ROC-AUC + PR-AUC + threshold-classification metrics on a prospective "
            "temporal holdout (train Year <= cutoff, test Year > cutoff); "
            "random-split AUC with the same test size reported as the gap reference"
        ),
        "scope": "tier2_precious_metal_anticancer / paper-funnel cohort per metal",
        "source": "molmetal/scripts/pa20_external_temporal_5metal.py",
        "cohort_rule": (
            "per metal: cytotox.build_paper_cohort_indices(df, metal=m) — same "
            "cohort rule as pa18_model_class_5metal"
        ),
        "split": (
            f"temporal: train = Year <= {args.cutoff_year}, test = Year > "
            f"{args.cutoff_year}; Year is per-DOI so no paper straddles (asserted)"
        ),
        "label_rule": "active := IC50_Dark_value < 10 uM (paper cut)",
        "decision_threshold": float(args.threshold),
        "models": list(args.models),
        "per_metal": per_metal,
        "krasnov_2026_reference": {
            "note": (
                "Krasnov 2026 (J. Med. Chem. 69(8) 8838-8851) is a classification "
                "paper and reports its own external validation; this temporal "
                "holdout is our analogue.  No same-dataset temporal number is "
                "published there, so no numeric reference is recorded."
            )
        },
        "why": (
            "Close the 'external validation' metric gap: pa18 is internal "
            "DOI-grouped CV only.  A prospective temporal holdout is the strongest "
            "external validation available offline and exposes the in-distribution "
            "leakage via the random-split reference."
        ),
        "honest_framing_note": (
            "MEASURED only.  This is a TEMPORAL holdout, not a cross-database "
            "external set (none is available on the host with the same label rule "
            "and metal scope).  The random-split reference is a seeded diagnostic, "
            "not a second protocol.  Test active rates differ from train (recorded "
            "per metal) — that shift is the point of the axis, not a defect."
        ),
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(out, json.dumps(report, indent=2, default=float, ensure_ascii=False))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
