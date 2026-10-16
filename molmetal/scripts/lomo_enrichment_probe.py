#!/usr/bin/env python3
"""Pooled leave-one-metal-out enrichment curve — the resolvability probe.

=====================================================================
Why this file exists
=====================================================================
``baseline_lomo_cv.py`` reports ``hit_rate_top5pct`` PER FOLD. That
number is not resolvable on this dataset and must not be quoted.
MEASURED (same script, ``--model xgb``, seed 42):

    metal  n_test  k@5%  hit@5%  base   lift
    Ru        3668   183   0.530  0.263  2.02
    Ir        1242    62   0.565  0.448  1.26
    Rh         108     5   1.000  0.324  3.09   <- k=5, all five hit
    Os          87     4   1.000  0.437  2.29   <- k=4, all four hit
    Re         166     8   0.000  0.163  0.00   <- k=8, none hit

Rh / Os / Re folds are 87-166 rows, so their top-5% is 4-8 molecules.
A hit rate of 1.0 on k=4 and 0.0 on k=8 is a coin landing twice, not a
ranking ability. The aggregate ``hit_rate_top5pct_mean`` is 0.619 with a
std of 0.414 -- the folds disagree far more than any method would move
them, and the mean is an artefact of k, not of chemistry.

Pooling all folds into ONE enrichment curve fixes the resolution
problem: the pooled cohort is n=5271, so top-1% is k=53 rather than k=4.
This is also the framing the comparison work uses -- MetalCytoToxDB
reports a hit-rate improvement over random sampling, not a per-fold
top-k.

What this script answers
------------------------
Can a cross-metal scorer rank held-out-metal candidates well enough for
a "retained hit-rate" generation metric to mean anything? If the pooled
enrichment at a defensible k is ~1.0, the metric is dead no matter how
much generation engineering follows it.

Honest framing
--------------
The model here is Morgan-FP + XGBoost with baseline_lomo_cv's exact
hyperparameters, NOT our D-MPNN. That is deliberate. The D-MPNN LOMO run
measures 0.5575 +- 0.0949 (``molmetal/reports/dmpnn_clean5_lomo/
metrics.json``) and is slower; this probe asks the cheaper prior
question -- is the signal present in the data at all -- so that we do not
spend GPU time if the answer is no. If the cheap baseline has no signal,
the expensive one will not either.

Usage
-----
::

    .venv/bin/python molmetal/scripts/lomo_enrichment_probe.py \
        --out molmetal/reports/lomo_hitrate_probe_2026_10_10/enrichment.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# k fractions for the enrichment curve. 1% and 5% are the decision-relevant
# region -- a generator produces a candidate pool you want to be selective
# about. 25%/50% show where the curve crosses 1.0, i.e. where ranking stops
# adding anything over sampling the training prior.
K_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50)

DEFAULT_METALS = ["Ru", "Ir", "Rh", "Os", "Re"]

# Mirrors baseline_lomo_cv._fit_eval's XGBClassifier params exactly, so the
# two scripts' numbers are comparable. Changing one without the other would
# make the "the cheap baseline is as good as the D-MPNN" comparison invalid.
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "tree_method": "hist",
}


def _load_env_default() -> None:
    """``cytotox.DEFAULT_CSV`` honours MOLMETAL_DATA_DIR; mirror env.sh 1b.

    Without this the loader falls back to the /mnt/storage mount, which does
    not exist on this host -- measured FileNotFoundError on 2026-10-10.
    """
    if not os.environ.get("MOLMETAL_DATA_DIR"):
        local = PROJECT_ROOT / ".data" / "molmetal"
        if (local / "MetalCytoToxDB.csv").exists():
            os.environ["MOLMETAL_DATA_DIR"] = str(local)


def _fit_predict(
    train_smiles: np.ndarray,
    y_train_pool: np.ndarray,
    test_smiles: np.ndarray,
    seed: int,
    morgan_radius: int,
    morgan_nbits: int,
) -> np.ndarray:
    """Fit on the train pool and return P(active) for the held-out metal.

    The held-out metal is NEVER used for validation, since it is OOD by
    construction -- same contract as baseline_lomo_cv._fit_eval.
    """
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    from molmetal.baselines.eval_utils import morgan_features

    x_all = morgan_features(
        np.concatenate([train_smiles, test_smiles]), morgan_radius, morgan_nbits
    )
    n_pool = len(train_smiles)
    x_pool, x_test = x_all[:n_pool], x_all[n_pool:]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_pool, y_train_pool, test_size=0.2, random_state=seed, stratify=y_train_pool
    )

    params = dict(XGB_PARAMS, random_state=seed)
    pos = max(1, int(y_tr.sum()))
    neg = max(1, int(len(y_tr) - pos))
    params["scale_pos_weight"] = float(neg) / float(pos)

    clf = XGBClassifier(**params)
    clf.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
    return np.asarray(clf.predict_proba(x_test)[:, 1], dtype=float)


def _curve_at(prob: np.ndarray, y: np.ndarray, base: float) -> Dict[str, Dict[str, float]]:
    """Cumulative hit rate at increasing k, from ONE shared ranking.

    A single stable argsort is reused for every k. Sorting per-k with
    ``np.argpartition`` is wrong here: this cohort has ~3,358 tied
    probability pairs out of 5,271 rows, so at small k the arbitrary
    pick inside a tie block decides the number. Measured 2026-10-10 --
    argpartition reported top-1% hit rate 0.472 vs the stable-sort 0.491,
    and an earlier run gave 0.472 where a stable sort gives a monotone
    curve. The ranking, not the tie-break, is what is being measured.
    """
    order = np.argsort(-prob, kind="stable")
    y_sorted = y[order]
    cum_hits = np.cumsum(y_sorted)
    n = len(y)
    out: Dict[str, Dict[str, float]] = {}
    for frac in K_FRACTIONS:
        k = max(1, int(round(frac * n)))
        hit = float(cum_hits[k - 1] / k)
        se = (hit * (1.0 - hit) / k) ** 0.5
        out[f"top_{int(frac * 100)}pct"] = {
            "k": int(k),
            "n_hits": int(cum_hits[k - 1]),
            "hit_rate": hit,
            "base_rate": base,
            "lift": (hit / base) if base > 0 else float("nan"),
            "hit_rate_ci95": [
                max(0.0, hit - 1.96 * se),
                min(1.0, hit + 1.96 * se),
            ],
        }
    return out


def run_probe(
    metals: List[str],
    seed: int,
    min_test_rows: int = 50,
    morgan_radius: int = 2,
    morgan_nbits: int = 2048,
) -> Dict[str, object]:
    _load_env_default()

    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    per_metal: Dict[str, object] = {}
    skipped: List[Dict[str, str]] = []
    for m in metals:
        flt = CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m])
        ds = MetalCytotoxDataset.from_csv(filters=flt)
        if len(ds) == 0:
            skipped.append({"metal": m, "reason": "no_rows_after_filter"})
            continue
        if len(ds) < min_test_rows:
            skipped.append({"metal": m, "reason": f"too_few_rows (n={len(ds)})"})
            continue
        per_metal[m] = ds

    if len(per_metal) < 2:
        return {"status": "INSUFFICIENT_FOLDS", "skipped": skipped}

    folds: List[Dict[str, object]] = []
    pooled_p: List[np.ndarray] = []
    pooled_y: List[np.ndarray] = []
    pooled_m: List[np.ndarray] = []

    for held_out, ds in per_metal.items():
        pool = pd.concat(
            [d.df for m, d in per_metal.items() if m != held_out], ignore_index=True
        )
        pool_ds = MetalCytotoxDataset(df=pool)
        prob = _fit_predict(
            np.asarray(pool_ds.smiles),
            np.asarray(pool_ds.active, dtype=int),
            np.asarray(ds.smiles),
            seed,
            morgan_radius,
            morgan_nbits,
        )
        y = np.asarray(ds.active, dtype=int)
        base = float(y.mean())
        folds.append(
            {
                "metal": held_out,
                "roc_auc": float(roc_auc_score(y, prob)),
                "pr_auc": float(average_precision_score(y, prob)),
                "n_train": int(len(pool_ds)),
                "n_test": int(len(y)),
                "pos_rate_test": base,
                "enrichment": _curve_at(prob, y, base),
            }
        )
        pooled_p.append(prob)
        pooled_y.append(y)
        pooled_m.append(np.array([held_out] * len(y), dtype=object))

    # ---- pooled ----
    p_all = np.concatenate(pooled_p)
    y_all = np.concatenate(pooled_y)
    m_all = np.concatenate(pooled_m)
    n_all = len(y_all)
    base_all = float(y_all.mean())
    pooled_curve = _curve_at(p_all, y_all, base_all)

    order = np.argsort(-p_all)
    y_sorted = y_all[order]
    k5 = pooled_curve["top_5pct"]["k"]
    top5_metal, top5_active = m_all[order[:k5]], y_sorted[:k5]

    per_metal_recovery = {}
    for m in per_metal:
        mask = m_all == m
        n_act = int(y_all[mask].sum())
        rec = int(((top5_metal == m) & (top5_active == 1)).sum())
        per_metal_recovery[m] = {
            "n_total": int(mask.sum()),
            "n_active": n_act,
            "n_active_in_pooled_top5pct": rec,
            "recall_of_active_at_5pct": (rec / n_act) if n_act > 0 else float("nan"),
        }

    return {
        "status": "MEASURED",
        "model": f"morgan_fp(r={morgan_radius},{morgan_nbits}bit) + xgb",
        "data_source": "MetalCytoToxDB (Krasnov et al. 2026) via molmetal/data/cytotox.py",
        "protocol": "leave-one-metal-out; train on 4 metals, test on the held-out one",
        "note_why_not_dmpnn": (
            "our D-MPNN LOMO measures 0.5575 +- 0.0949 with w2_real=False "
            "(reports/dmpnn_clean5_lomo/metrics.json). This probe asks the "
            "cheaper prior question with baseline_lomo_cv's exact XGB params."
        ),
        "seed": seed,
        "per_fold": folds,
        "pooled": {
            "n": int(n_all),
            "base_rate": base_all,
            "roc_auc": float(roc_auc_score(y_all, p_all)),
            "pr_auc": float(average_precision_score(y_all, p_all)),
            "n_tied_probability_rows": int(n_all - len(np.unique(p_all))),
            "enrichment": pooled_curve,
            "peak_enrichment": max(
                pooled_curve.items(), key=lambda kv: kv[1]["lift"]
            )[0],
        },
        "per_metal_recovery_at_pooled_top5pct": per_metal_recovery,
        "skipped": skipped,
        "verdict": _verdict(pooled_curve, base_all),
    }


def _verdict(pooled_curve: Dict[str, Dict[str, float]], base: float) -> Dict[str, object]:
    """A retained-hit-rate generation metric needs a lift that clears its own CI."""
    t5 = pooled_curve["top_5pct"]
    lo, hi = t5["hit_rate_ci95"]
    lift_lo = lo / base if base > 0 else float("nan")
    lift_hi = hi / base if base > 0 else float("nan")
    return {
        "pooled_top5pct_lift": t5["lift"],
        "lift_ci95": [lift_lo, lift_hi],
        "usable_for_generation_metric": bool(lift_lo > 1.0),
        "criterion": (
            "the lower 95% bound of the pooled top-5% lift must exceed 1.0, "
            "otherwise a 'retained hit-rate' number would be indistinguishable "
            "from sampling the training prior"
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pooled LOMO enrichment probe.")
    p.add_argument("--metals", nargs="+", default=DEFAULT_METALS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-test-rows", type=int, default=50)
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-nbits", type=int, default=2048)
    p.add_argument(
        "--out",
        default="molmetal/reports/lomo_hitrate_probe_2026_10_10/enrichment.json",
    )
    args = p.parse_args(argv)

    rep = run_probe(
        metals=list(args.metals),
        seed=args.seed,
        min_test_rows=args.min_test_rows,
        morgan_radius=args.morgan_radius,
        morgan_nbits=args.morgan_nbits,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")

    if rep.get("status") != "MEASURED":
        print(f"[probe] status={rep['status']} {rep.get('reason', '')}")
        return 3

    pool = rep["pooled"]
    print(f"[probe] pooled n={pool['n']} base={pool['base_rate']:.3f} "
          f"AUC={pool['roc_auc']:.4f} PR-AUC={pool['pr_auc']:.4f}\n")
    print(f"{'k':>7} {'hit':>7} {'base':>7} {'lift':>6} {'95% CI on lift':>18}")
    for _key, c in pool["enrichment"].items():
        lo, hi = c["hit_rate_ci95"]
        print(f"{c['k']:7d} {c['hit_rate']:7.3f} {c['base_rate']:7.3f} "
              f"{c['lift']:6.2f}   [{lo / c['base_rate']:5.2f}, {hi / c['base_rate']:5.2f}]")
    v = rep["verdict"]
    print(f"\n[verdict] pooled top-5% lift = {v['pooled_top5pct_lift']:.2f} "
          f"(95% CI {v['lift_ci95'][0]:.2f}-{v['lift_ci95'][1]:.2f})")
    print(f"[verdict] usable_for_generation_metric = {v['usable_for_generation_metric']}")
    print(f"[probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())