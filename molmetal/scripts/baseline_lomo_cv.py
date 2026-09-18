"""Leave-One-Metal-Out (LOMO) CV for classical ML baselines.

Trains a single model on **all metals except one** then evaluates on the
held-out metal.  This is the cross-metal generalisation protocol from
TODO/06_milestones/milestones.md §Phase 4 (Krasnov 2026 multi-metal model).

Reporting
---------
For each held-out metal we report Pearson r between the predicted
probability and the binary ``active`` label on the held-out fold.  The
final output is the mean ± std across held-out folds.

Honest framing
--------------
This is a *baseline* LOMO protocol — we are measuring the floor set by
Morgan-FP + GBDT on cross-metal transfer.  Our Lambda / CFM work
replaces this signal (de novo design, not activity prediction) so the
two should be read as complementary: the LOMO score here bounds how
much signal the *activity prediction* side has at all when chemistry is
shifting across the periodic table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make ``molmetal.*`` importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from molmetal.baselines.eval_utils import compute_metrics, morgan_features
from molmetal.baselines.morgan_xgb import MorganXGBBaseline
from molmetal.baselines.morgan_lightgbm import (
    MorganLightGBMBaseline,
    lightgbm_available,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import RandomSplitter


# Default metals: Krasnov 2026 + our Lambda/CFM work centres on these.
DEFAULT_METALS = ("Ru", "Ir", "Rh", "Os", "Re")


# ---------------------------------------------------------------------------
# Per-metal fold
# ---------------------------------------------------------------------------
def _per_metal_metrics(
    metal: str,
    model: str,
    train_mols: MetalCytotoxDataset,
    test_mol: MetalCytotoxDataset,
    seed: int,
    morgan_radius: int,
    morgan_nbits: int,
) -> Dict[str, float]:
    """Train on ``train_mols`` and evaluate on ``test_mol``.

    Returns a dict with ``{roc_auc, pr_auc, hit_rate_top5pct, pearson_r,
    spearman_r, n_train, n_test, pos_rate_test}`` — note that the
    Pearson / Spearman block is the headline LOMO metric.
    """
    # 1. Split the train pool into train/val (the held-out metal is *only*
    # used for testing, never for validation, since it's OOD).
    y_pool = train_mols.active.astype(int)
    splitter = RandomSplitter(seed=seed)
    split = splitter(train_mols)
    train_idx, val_idx = split.train_idx, split.val_idx

    # 2. Featurise on the union of train pool + held-out test
    all_smiles = np.concatenate([train_mols.smiles, test_mol.smiles])
    x_all = morgan_features(all_smiles, morgan_radius, morgan_nbits)
    n_train_pool = len(train_mols)
    x_pool = x_all[:n_train_pool]
    x_test = x_all[n_train_pool:]
    y_train = y_pool[train_idx]
    y_val = y_pool[val_idx]
    y_test = test_mol.active.astype(int)

    x_train = x_pool[train_idx]
    x_val = x_pool[val_idx]

    # 3. Train chosen model
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
        # Class-weight balancing
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

    # 4. Compute metrics — both classification metrics and (rank) corr.
    cls_metrics = compute_metrics(y_test, score)
    pearson_r, spearman_r = _safe_correlations(y_test, score)

    return {
        "metal": metal,
        "model": model,
        "roc_auc": cls_metrics["roc_auc"],
        "pr_auc": cls_metrics["pr_auc"],
        "hit_rate_top5pct": cls_metrics["hit_rate_top5pct"],
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "pos_rate_test": float(y_test.mean()),
    }


def _safe_correlations(y: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    """Pearson / Spearman with graceful NaN handling."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    if len(y) < 2 or np.std(y) == 0 or np.std(s) == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr, spearmanr

        pr, _ = pearsonr(y, s)
        sp, _ = spearmanr(y, s)
        return float(pr), float(sp)
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_lomo(
    metals: Tuple[str, ...] = DEFAULT_METALS,
    model: str = "xgb",
    seed: int = 42,
    max_rows_per_metal: Optional[int] = None,
    min_test_rows: int = 50,
) -> Dict[str, object]:
    """Run LOMO CV across ``metals``.  Returns a JSON-able report."""
    folds: List[Dict[str, float]] = []
    skipped: List[Dict[str, str]] = []
    pool_dfs: List[pd.DataFrame] = []
    per_metal_dfs: Dict[str, pd.DataFrame] = {}

    # 1. Load each metal once
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
        if len(ds) < min_test_rows:
            skipped.append(
                {
                    "metal": m,
                    "reason": f"too_few_rows (n={len(ds)} < min_test_rows={min_test_rows})",
                }
            )
            continue
        if max_rows_per_metal is not None and len(ds) > max_rows_per_metal:
            ds = MetalCytotoxDataset(df=ds.df.head(max_rows_per_metal).copy())
        per_metal_dfs[m] = ds.df.copy()
        pool_dfs.append(ds.df.copy())

    if not pool_dfs:
        raise RuntimeError(
            f"LOMO aborted: no usable metal cohorts.  Skipped={skipped!r}"
        )

    pool = MetalCytotoxDataset(df=pd.concat(pool_dfs, ignore_index=True))

    # 2. For each held-out metal: build a test_ds and a pool minus it.
    for held_out, df in per_metal_dfs.items():
        test_ds = MetalCytotoxDataset(df=df.copy())
        # Build train pool = pool minus the held-out rows
        keep_mask = pool.metals != held_out
        train_pool = MetalCytotoxDataset(df=pool.df[keep_mask].copy())

        try:
            fold = _per_metal_metrics(
                metal=held_out,
                model=model,
                train_mols=train_pool,
                test_mol=test_ds,
                seed=seed,
                morgan_radius=2,
                morgan_nbits=2048,
            )
        except Exception as e:
            skipped.append({"metal": held_out, "reason": f"run_failed: {e!r}"})
            continue
        folds.append(fold)

    # 3. Aggregate mean / std
    metrics_summary = _aggregate_folds(folds)

    return {
        "model": model,
        "metals_evaluated": [f["metal"] for f in folds],
        "metals_skipped": skipped,
        "folds": folds,
        "aggregate": metrics_summary,
        "seed": seed,
    }


def _aggregate_folds(folds: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean / std across folds for the headline metrics."""
    if not folds:
        return {}
    keys = ("roc_auc", "pr_auc", "hit_rate_top5pct", "pearson_r", "spearman_r")
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
    p = argparse.ArgumentParser(description="Leave-One-Metal-Out CV.")
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(DEFAULT_METALS),
        help=f"Metals to rotate over (default: {list(DEFAULT_METALS)})",
    )
    p.add_argument(
        "--model",
        default="xgb",
        choices=["xgb", "lightgbm"],
        help="Which GBDT baseline to use (default: xgb)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-rows-per-metal",
        type=int,
        default=None,
        help=(
            "Optional cap on rows-per-metal (for quick smoke tests).  "
            "Defaults to None = use all rows."
        ),
    )
    p.add_argument(
        "--min-test-rows",
        type=int,
        default=50,
        help="Skip a metal if its filtered row count is below this (default 50)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path.  Default: molmetal/reports/baseline_lomo_<model>.json",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        f"[lomo] model={args.model} metals={args.metals} "
        f"seed={args.seed} max_rows_per_metal={args.max_rows_per_metal}"
    )
    t0 = time.time()
    report = run_lomo(
        metals=tuple(args.metals),
        model=args.model,
        seed=args.seed,
        max_rows_per_metal=args.max_rows_per_metal,
        min_test_rows=args.min_test_rows,
    )
    elapsed = time.time() - t0

    # 4. Pretty-print
    print(f"[lomo] folds completed: {len(report['folds'])}")
    for fold in report["folds"]:
        print(
            f"    hold-out={fold['metal']:>4}  AUC={fold['roc_auc']:.4f}  "
            f"AP={fold['pr_auc']:.4f}  Pearson r={fold['pearson_r']:+.4f}  "
            f"n_train={fold['n_train']}  n_test={fold['n_test']}"
        )
    agg = report["aggregate"]
    if agg:
        print(
            f"[lomo] mean Pearson r = {agg['pearson_r_mean']:+.4f} "
            f"± {agg['pearson_r_std']:.4f} (n={agg['pearson_r_n']})"
        )
        print(
            f"[lomo] mean ROC-AUC = {agg['roc_auc_mean']:.4f} "
            f"± {agg['roc_auc_std']:.4f}"
        )
    print(f"[lomo] elapsed={elapsed:.1f}s")

    # 5. Write JSON
    out_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "molmetal" / "reports" / f"baseline_lomo_{args.model}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["elapsed_seconds"] = elapsed
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[lomo] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())