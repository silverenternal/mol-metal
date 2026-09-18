"""Temporal holdout baseline evaluation.

Trains a Morgan-FP + GBDT classifier on rows published **before** a
configurable cutoff year and evaluates on rows published **at or after**
the cutoff.  This is the standard "data drift" / "post-cutoff hit-rate"
protocol from TODO/06_milestones/milestones.md §Phase 4 and matches the
post-2024 hit-rate evaluation referenced in Phase 4 §validation.

Honest framing
--------------
The temporal gap is the **single most informative number** in our
classical-ML baseline suite, because it quantifies *real* generalisation
to unseen chemistry over time — not the seen-SMILES overlap that plagues
the random split (cf. ``molmetal/reports/leakage_diagnosis.md``).

We report both the absolute post-cutoff AUC and the **temporal gap**
(``Δ = AUC_random − AUC_temporal``); a large positive gap means the
random-split number was inflated by leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from molmetal.baselines.eval_utils import compute_metrics, morgan_features
from molmetal.baselines.morgan_lightgbm import lightgbm_available
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import TemporalSplitter


# Default metals mirror the LOMO script
DEFAULT_METALS = ("Ru", "Ir", "Rh", "Os", "Re")


# ---------------------------------------------------------------------------
# Per-metal temporal split
# ---------------------------------------------------------------------------
def _per_metal_temporal(
    metal: str,
    model: str,
    train_pool: MetalCytotoxDataset,
    test_ds: MetalCytotoxDataset,
    seed: int,
    morgan_radius: int,
    morgan_nbits: int,
) -> Dict[str, float]:
    """Train on ``train_pool`` (pre-cutoff) and evaluate on ``test_ds`` (post-cutoff)."""
    # 1. Use the TemporalSplitter on the train pool to carve a val set
    val_splitter = TemporalSplitter(cutoff_year=2050, val_fraction=0.1)
    split = val_splitter(train_pool)
    train_idx = split.train_idx
    val_idx = split.val_idx

    y_pool = train_pool.active.astype(int)

    # 2. Featurise on union of train pool + post-cutoff test
    all_smiles = np.concatenate([train_pool.smiles, test_ds.smiles])
    x_all = morgan_features(all_smiles, morgan_radius, morgan_nbits)
    n_train_pool = len(train_pool)
    x_pool = x_all[:n_train_pool]
    x_test = x_all[n_train_pool:]
    y_train = y_pool[train_idx]
    y_val = y_pool[val_idx]
    y_test = test_ds.active.astype(int)

    x_train = x_pool[train_idx]
    x_val = x_pool[val_idx]

    # 3. Train
    if model == "xgb":
        from xgboost import XGBClassifier

        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_jobs": -1,
            "random_state": seed,
            "tree_method": "hist",
        }
        pos = max(1, int(y_train.sum()))
        neg = max(1, int(len(y_train) - pos))
        params["scale_pos_weight"] = float(neg) / float(pos)
        clf = XGBClassifier(**params)
        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    elif model == "lightgbm":
        if not lightgbm_available():
            raise RuntimeError("lightgbm not installed; install via `uv pip install lightgbm`.")
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "metric": "auc",
            "n_jobs": -1,
            "random_state": seed,
            "verbose": -1,
            "min_child_samples": 5,
        }
        pos = max(1, int(y_train.sum()))
        neg = max(1, int(len(y_train) - pos))
        params["scale_pos_weight"] = float(neg) / float(pos)
        clf = LGBMClassifier(**params)
        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="auc")
    else:
        raise ValueError(f"unknown model {model!r}")

    score = clf.predict_proba(x_test)[:, 1]
    cls_metrics = compute_metrics(y_test, score)

    return {
        "metal": metal,
        "model": model,
        "roc_auc": cls_metrics["roc_auc"],
        "pr_auc": cls_metrics["pr_auc"],
        "hit_rate_top5pct": cls_metrics["hit_rate_top5pct"],
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "pos_rate_test": float(y_test.mean()),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def run_temporal(
    metals: Tuple[str, ...] = DEFAULT_METALS,
    model: str = "xgb",
    cutoff_year: int = 2023,
    seed: int = 42,
    min_test_rows: int = 50,
    min_train_rows: int = 100,
) -> Dict[str, object]:
    """Run temporal-holdout CV across ``metals``.

    For each metal we (a) filter rows pre-cutoff for training and
    post-cutoff for testing, then (b) report the post-cutoff AUC +
    hit@5%.

    Parameters
    ----------
    cutoff_year : int
        Rows with ``Year < cutoff_year`` go into train+val, rows with
        ``Year >= cutoff_year`` go into test.  Default 2023 so the test
        fold has at least 1-2 years of post-cutoff data (2024-2025).
    """
    folds: List[Dict[str, float]] = []
    skipped: List[Dict[str, str]] = []
    for m in metals:
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=[m],
        )
        ds = MetalCytotoxDataset.from_csv(filters=flt)
        if len(ds) == 0:
            skipped.append({"metal": m, "reason": "no_rows_after_filter"})
            continue
        years = ds.years
        n_pre = int((years < cutoff_year).sum())
        n_post = int((years >= cutoff_year).sum())
        if n_post < min_test_rows:
            skipped.append(
                {
                    "metal": m,
                    "reason": f"too_few_post_rows (n_post={n_post} < min_test_rows={min_test_rows})",
                }
            )
            continue
        if n_pre < min_train_rows:
            skipped.append(
                {
                    "metal": m,
                    "reason": f"too_few_pre_rows (n_pre={n_pre} < min_train_rows={min_train_rows})",
                }
            )
            continue
        # Build pre-cutoff and post-cutoff datasets
        pre_mask = years < cutoff_year
        post_mask = years >= cutoff_year
        ds_pre = MetalCytotoxDataset(df=ds.df[pre_mask].reset_index(drop=True).copy())
        ds_post = MetalCytotoxDataset(df=ds.df[post_mask].reset_index(drop=True).copy())
        try:
            fold = _per_metal_temporal(
                metal=m,
                model=model,
                train_pool=ds_pre,
                test_ds=ds_post,
                seed=seed,
                morgan_radius=2,
                morgan_nbits=2048,
            )
        except Exception as e:
            skipped.append({"metal": m, "reason": f"run_failed: {e!r}"})
            continue
        fold["n_train_pre"] = n_pre
        fold["n_test_post"] = n_post
        fold["cutoff_year"] = cutoff_year
        folds.append(fold)

    return {
        "model": model,
        "cutoff_year": cutoff_year,
        "metals_evaluated": [f["metal"] for f in folds],
        "metals_skipped": skipped,
        "folds": folds,
        "aggregate": _aggregate(folds),
        "seed": seed,
    }


def _aggregate(folds: List[Dict[str, float]]) -> Dict[str, float]:
    if not folds:
        return {}
    keys = ("roc_auc", "pr_auc", "hit_rate_top5pct")
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([f[k] for f in folds], dtype=float)
        finite = vals[~np.isnan(vals)]
        out[f"{k}_mean"] = float(finite.mean()) if len(finite) else float("nan")
        out[f"{k}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        out[f"{k}_n"] = int(len(finite))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal holdout baseline.")
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(DEFAULT_METALS),
        help=f"Metals to evaluate (default: {list(DEFAULT_METALS)})",
    )
    p.add_argument(
        "--model",
        default="xgb",
        choices=["xgb", "lightgbm"],
        help="Which GBDT baseline to use (default: xgb)",
    )
    p.add_argument(
        "--cutoff-year",
        type=int,
        default=2023,
        help="Rows with Year < cutoff_year train, Year >= cutoff_year test (default: 2023)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-test-rows",
        type=int,
        default=50,
        help="Skip a metal if its post-cutoff row count is below this (default 50)",
    )
    p.add_argument(
        "--min-train-rows",
        type=int,
        default=100,
        help="Skip a metal if its pre-cutoff row count is below this (default 100)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path.  Default: molmetal/reports/baseline_temporal_<cutoff>_<model>.json",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        f"[temporal] model={args.model} cutoff_year={args.cutoff_year} "
        f"metals={args.metals} seed={args.seed}"
    )
    t0 = time.time()
    report = run_temporal(
        metals=tuple(args.metals),
        model=args.model,
        cutoff_year=args.cutoff_year,
        seed=args.seed,
        min_test_rows=args.min_test_rows,
        min_train_rows=args.min_train_rows,
    )
    elapsed = time.time() - t0

    print(f"[temporal] folds completed: {len(report['folds'])}")
    for fold in report["folds"]:
        print(
            f"    {fold['metal']:>4}  AUC={fold['roc_auc']:.4f}  "
            f"AP={fold['pr_auc']:.4f}  Hit@5%={fold['hit_rate_top5pct']:.3f}  "
            f"n_pre={fold['n_train_pre']}  n_post={fold['n_test_post']}"
        )
    agg = report["aggregate"]
    if agg:
        print(
            f"[temporal] mean ROC-AUC = {agg['roc_auc_mean']:.4f} "
            f"± {agg['roc_auc_std']:.4f} (n={agg['roc_auc_n']})"
        )
    print(f"[temporal] elapsed={elapsed:.1f}s")

    out_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT
        / "molmetal"
        / "reports"
        / f"baseline_temporal_{args.cutoff_year}_{args.model}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["elapsed_seconds"] = elapsed
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[temporal] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())