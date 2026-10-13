#!/usr/bin/env python3
"""PA18-reg — pIC50 regression on the paper-funnel cohort, per metal.

WHY THIS SCRIPT EXISTS (T1, metric alignment)
---------------------------------------------
The one same-dataset paper (Krasnov 2026, MetalCytoToxDB) reports **two**
metrics per metal: a binary ROC-AUC and a **pIC50 MAE** (regression).  This
repository reproduces the AUC axis (``pa18_model_class_5metal.py``) but has no
clean pIC50-MAE number on the aligned cohort — the existing regression artifacts
(``metalcytotox_combined_r2``) are measured on n=200 sub-cohorts under a
different split and are labelled honest-negative, so they are not comparable.

This script fills that gap: it is the regression analogue of
``pa18_model_class_5metal.py``, using the SAME cohort rule, the SAME DOI-grouped
10-fold generator, the SAME Morgan-FP features and the SAME model classes
(LightGBM / XGBoost — the class Krasnov used).  Only the head changes
(regressor on pIC50 instead of classifier on ``active``) and the metrics change
(MAE / RMSE / R² / Pearson r instead of AUC).

Cohort rule (per metal): IC50-present -> metal -> measurement-level dedup ->
76-name non-cancer exclusion, i.e. ``cytotox.build_paper_cohort_indices(df,
metal=m)`` with every surviving cell line.  Label is ``pIC50 = -log10(IC50_M)``
(``cytotox._add_pic50``).  The regression target is row-level (one row per
measurement); the within-ligand pIC50 sd is ~0.2 vs a total sd of ~0.8, so
ligand-only features carry most of the signal and the residual is the
cell-line/assay component — the same target Krasnov regresses.

Honest framing (AGENTS.md §5.2)
-------------------------------
Every metric is MEASURED with ``sklearn.metrics`` on the fold's held-out rows.
The Krasnov pIC50-MAE reference value is NOT captured in this repo (the source
paper's regression table has not been transcribed); the report records it as
``null`` with an explicit note rather than inventing a number.  A constant
(train-mean) predictor MAE is included per metal so the reader can see how much
of the error the model actually removes.

Usage
-----
::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    .venv/bin/python molmetal/scripts/pa18_pic50_regression_5metal.py \\
        --metals Ru Ir Rh Os Re --models lightgbm xgboost \\
        --out metrics/by_metric/pa18_pic50_regression_5metal.json
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

DEFAULT_METALS: tuple = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_MODELS: tuple = ("lightgbm", "xgboost")


def build_cohort(df: pd.DataFrame, metal: str) -> pd.DataFrame:
    """Paper funnel for one metal: IC50-present -> metal -> dedup -> non-cancer."""
    from molmetal.data import cytotox as C

    idx = C.build_paper_cohort_indices(df, metal=metal)
    cohort = df.iloc[idx].reset_index(drop=True)
    cohort = C._add_pic50(cohort)
    cohort = C._add_active(cohort)
    return cohort


def _fit_predict(model: str, X_tr, y_tr, X_te, seed: int, n_jobs: int) -> np.ndarray:
    if model == "lightgbm":
        from lightgbm import LGBMRegressor

        reg = LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=int(n_jobs),
            random_state=seed,
            verbose=-1,
            min_child_samples=5,
        )
        reg.fit(X_tr, y_tr)
        return reg.predict(X_te)
    if model == "xgboost":
        import xgboost as xgb

        reg = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=int(n_jobs),
            random_state=seed,
            min_child_weight=5,
            tree_method="hist",
        )
        reg.fit(X_tr, y_tr)
        return reg.predict(X_te)
    raise ValueError(f"unknown model {model!r}")


def _rdkit_descriptors(smiles: np.ndarray) -> np.ndarray:
    """RDKit 2D descriptors (~210) — the strongest single lever found for pIC50.

    MEASURED (2026-10-14, DOI-split, Ru): Morgan-FP R² 0.121 → RDKit-2D 0.239.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append([0.0] * len(Descriptors._descList))
            continue
        vals = []
        for _, fn in Descriptors._descList:
            try:
                vals.append(fn(m))
            except Exception:
                vals.append(0.0)
        rows.append(vals)
    X = np.asarray(rows, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Some RDKit descriptors (e.g. Ipc) are astronomically large; XGBoost's
    # float32 DMatrix rejects them ("value too large").  Clip to a safe range.
    return np.clip(X, -1e6, 1e6)


def build_features(
    smiles: np.ndarray, counterions: np.ndarray, radius: int, nbits: int,
    feature_set: str,
) -> np.ndarray:
    """Assemble the feature matrix for the requested set."""
    from molmetal.baselines.eval_utils import morgan_features

    parts: List[np.ndarray] = []
    if feature_set in ("morgan", "morgan+rdkit", "morgan+rdkit+counterion"):
        parts.append(morgan_features(smiles, radius, nbits))
    if feature_set in ("rdkit", "morgan+rdkit", "morgan+rdkit+counterion"):
        parts.append(_rdkit_descriptors(smiles))
    if feature_set == "morgan+rdkit+counterion":
        parts.append(morgan_features(counterions, 2, 512))
    if not parts:
        raise ValueError(f"unknown feature_set {feature_set!r}")
    return parts[0] if len(parts) == 1 else np.hstack(parts)


def _cv_metrics(
    folds, X, y, model: str, seed: int, n_jobs: int,
) -> Dict[str, object]:
    """Run one CV partition and return the mean metrics + per-fold rows."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr

    maes: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []
    rs: List[float] = []
    base_maes: List[float] = []  # constant (train-mean) predictor
    per_fold: List[Dict[str, object]] = []
    for tr, te in folds:
        tr = np.asarray(tr)
        te = np.asarray(te)
        pred = _fit_predict(model, X[tr], y[tr], X[te], seed, n_jobs)
        mae = float(mean_absolute_error(y[te], pred))
        rmse = float(mean_squared_error(y[te], pred) ** 0.5)
        r2 = float(r2_score(y[te], pred))
        r = float(pearsonr(y[te], pred)[0]) if len(te) > 2 else float("nan")
        base = float(mean_absolute_error(y[te], np.full_like(y[te], y[tr].mean())))
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        rs.append(r)
        base_maes.append(base)
        per_fold.append(
            {
                "fold": int(len(per_fold)),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson_r": r,
                "baseline_mae_train_mean": base,
            }
        )
    return {
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_r2": float(np.mean(r2s)),
        "std_r2": float(np.std(r2s)),
        "mean_pearson_r": float(np.nanmean(rs)),
        "mean_baseline_mae_train_mean": float(np.mean(base_maes)),
        "per_fold": per_fold,
    }


def run_metal(
    df: pd.DataFrame, metal: str, models: List[str], n_folds: int, seed: int,
    radius: int, nbits: int, n_jobs: int, random_diagnostic: bool = True,
    feature_set: str = "morgan",
) -> Dict[str, object]:
    from sklearn.model_selection import KFold
    from molmetal.data import cytotox as C

    cohort = build_cohort(df, metal)
    y = np.asarray(cohort["pIC50"], dtype=float)
    smiles = cohort["SMILES_Ligands"].astype(str).to_numpy()
    n_rows = len(cohort)
    n_complexes = int(cohort[["SMILES_Ligands", "Metal"]].drop_duplicates().shape[0])
    n_cells = int(cohort["Cell_line"].astype(str).nunique())
    print(
        f"[{metal}] rows={n_rows} complexes={n_complexes} cell_lines={n_cells} "
        f"pIC50_mean={y.mean():.3f} pIC50_sd={y.std():.3f}",
        flush=True,
    )

    folds = C.build_doi_split_folds(cohort, n_folds=n_folds, seed=seed)
    all_val = np.concatenate([np.asarray(f["val_idx"]) for f in folds])
    if len(np.unique(all_val)) != n_rows:
        raise SystemExit(
            f"[{metal}] fold partition broken: {len(np.unique(all_val))} unique "
            f"val rows for {n_rows} cohort rows"
        )

    doi_folds = [
        (f["train_idx"], f["val_idx"]) for f in folds
    ]

    X = build_features(
        smiles,
        cohort["Counterion"].fillna("").astype(str).to_numpy(),
        radius, nbits, feature_set,
    )

    out: Dict[str, object] = {
        "n_rows": int(n_rows),
        "n_complexes": n_complexes,
        "n_cell_lines": n_cells,
        "pic50_mean": float(y.mean()),
        "pic50_sd": float(y.std()),
        "n_folds": int(n_folds),
        "models": {},
    }

    for model in models:
        rec = _cv_metrics(doi_folds, X, y, model, seed, n_jobs)
        rec["model_spec"] = f"morgan_fp(radius={radius}, {nbits}bit) + {model}"
        out["models"][model] = rec
        print(
            f"[{metal}] {model}: MAE={rec['mean_mae']:.4f} +/- {rec['std_mae']:.4f}  "
            f"RMSE={rec['mean_rmse']:.4f}  R2={rec['mean_r2']:.4f}  "
            f"(base MAE={rec['mean_baseline_mae_train_mean']:.4f})",
            flush=True,
        )

    # Random-split diagnostic (LightGBM only): the SAME features/model under an
    # in-distribution split.  The gap between this and the DOI-split number is
    # the paper/assay distribution shift, and is what makes the low DOI-split R²
    # interpretable (mirrors the binary AUC random-vs-DOI contrast).
    if random_diagnostic and "lightgbm" in models:
        kf = list(KFold(n_folds, shuffle=True, random_state=seed).split(X))
        rnd = _cv_metrics(kf, X, y, "lightgbm", seed, n_jobs)
        out["random_split_diagnostic_lightgbm"] = {
            "mean_mae": rnd["mean_mae"],
            "mean_rmse": rnd["mean_rmse"],
            "mean_r2": rnd["mean_r2"],
            "mean_pearson_r": rnd["mean_pearson_r"],
            "note": (
                "LightGBM under a shuffled KFold on the same cohort/features — "
                "NOT the reported protocol; a reference point for how much of the "
                "DOI-split error is paper/assay shift."
            ),
        }
        print(
            f"[{metal}] random-split diagnostic (lightgbm): "
            f"MAE={rnd['mean_mae']:.4f} R2={rnd['mean_r2']:.4f} "
            f"(DOI-split R2={out['models']['lightgbm']['mean_r2']:.4f})",
            flush=True,
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--metals", nargs="+", default=list(DEFAULT_METALS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help=(
            "threads per boosted-tree fit.  Deliberately small: this host is "
            "shared and n_jobs=-1 thrashes (MEASURED in pa18: load avg ~73 on "
            "32 cores starved a 20-fit run past 15 min)."
        ),
    )
    ap.add_argument(
        "--features",
        default="morgan+rdkit+counterion",
        choices=("morgan", "rdkit", "morgan+rdkit", "morgan+rdkit+counterion"),
        help=(
            "feature set.  MEASURED lever (2026-10-14): RDKit-2D roughly doubles "
            "the DOI-split R² vs Morgan-FP alone, so the default is the strongest."
        ),
    )
    ap.add_argument(
        "--no-random-diagnostic",
        action="store_true",
        help="skip the LightGBM random-split diagnostic (on by default)",
    )
    ap.add_argument(
        "--out",
        default="metrics/by_metric/pa18_pic50_regression_5metal.json",
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
            df, metal, list(args.models), args.n_folds, args.seed,
            args.radius, args.nbits, args.n_jobs,
            random_diagnostic=not args.no_random_diagnostic,
            feature_set=args.features,
        )

    # Pooled means across the five metals (per-model), for a single headline row.
    pooled: Dict[str, object] = {}
    for model in args.models:
        maes = [per_metal[m]["models"][model]["mean_mae"] for m in per_metal]
        rmses = [per_metal[m]["models"][model]["mean_rmse"] for m in per_metal]
        r2s = [per_metal[m]["models"][model]["mean_r2"] for m in per_metal]
        rs = [per_metal[m]["models"][model]["mean_pearson_r"] for m in per_metal]
        bases = [
            per_metal[m]["models"][model]["mean_baseline_mae_train_mean"]
            for m in per_metal
        ]
        pooled[model] = {
            "mean_mae": float(np.mean(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "mean_r2": float(np.mean(r2s)),
            "mean_pearson_r": float(np.mean(rs)),
            "mean_baseline_mae_train_mean": float(np.mean(bases)),
        }

    # Pooled random-vs-DOI contrast (LightGBM), the interpretability anchor.
    diag_metals = [
        m for m in per_metal if "random_split_diagnostic_lightgbm" in per_metal[m]
    ]
    random_vs_doi = None
    if diag_metals:
        random_vs_doi = {
            "pooled_random_mae": float(
                np.mean([
                    per_metal[m]["random_split_diagnostic_lightgbm"]["mean_mae"]
                    for m in diag_metals
                ])
            ),
            "pooled_random_r2": float(
                np.mean([
                    per_metal[m]["random_split_diagnostic_lightgbm"]["mean_r2"]
                    for m in diag_metals
                ])
            ),
            "pooled_doi_mae": float(
                np.mean([
                    per_metal[m]["models"]["lightgbm"]["mean_mae"]
                    for m in diag_metals
                ])
            ),
            "pooled_doi_r2": float(
                np.mean([
                    per_metal[m]["models"]["lightgbm"]["mean_r2"]
                    for m in diag_metals
                ])
            ),
            "metals": diag_metals,
            "note": (
                "LightGBM, same cohort/features, random KFold vs DOI-grouped "
                "10-fold.  The gap is the paper/assay distribution shift and is "
                "why the DOI-split R² sits near zero despite the model beating "
                "the train-mean baseline.  Not the reported protocol."
            ),
        }

    report = {
        "metric": "pa18_pic50_regression_5metal",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "pIC50 (MAE/RMSE in log units, lower better; R2/Pearson higher better)",
        "scope": "tier2_precious_metal_anticancer / paper-funnel cohort per metal",
        "source": "molmetal/scripts/pa18_pic50_regression_5metal.py",
        "cohort_rule": (
            "per metal: cytotox.build_paper_cohort_indices(df, metal=m) — "
            "IC50-present -> metal -> measurement-level dedup -> 76-name "
            "non-cancer exclusion; all surviving cell lines"
        ),
        "split": "DOI-grouped 10-fold CV (cytotox.build_doi_split_folds), seed=0",
        "label_rule": "pIC50 = -log10(IC50_Dark_value * 1e-6); row-level (per measurement)",
        "features": args.features,
        "models": list(args.models),
        "per_metal": per_metal,
        "pooled": pooled,
        "random_vs_doi_diagnostic": random_vs_doi,
        "krasnov_2026_reference": {
            "paper": "J. Med. Chem. 2026, 69(8), 8838-8851 (DOI 10.1021/acs.jmedchem.5c02755)",
            "paper_task": "classification (cytotoxicity class, Ru + Ir subsets) + external validation + multimetal",
            "binary_auc_ru": 0.810,
            "binary_auc_ir": 0.730,
            "pic50_mae": None,
            "note": (
                "CORRECTED 2026-10-14: the source paper is a CLASSIFICATION paper "
                "(it assigns a cytotoxicity class for the Ru and Ir subsets); it "
                "does NOT report a pIC50 regression metric.  An earlier repo claim "
                "of 'binary AUC + pIC50 MAE' was wrong and is retracted.  This "
                "pIC50-regression axis is therefore an EXTRA metric, NOT a "
                "Krasnov-aligned one.  The aligned comparison is the binary AUC "
                "axis (pa18_model_class_5metal.json).  Full-text tables are "
                "Cloudflare-blocked from this host; task type confirmed from the "
                "abstract (Europe PMC PMID 41969144)."
            ),
        },
        "why": (
            "T1 metric alignment: the same-dataset paper (Krasnov 2026) reports "
            "a pIC50 MAE alongside binary AUC; this repo reproduced only the AUC "
            "axis.  This is the regression analogue of pa18_model_class_5metal "
            "— same cohort, same folds, same features, same model class."
        ),
        "honest_framing_note": (
            "MEASURED only.  The train-mean constant-predictor MAE is reported "
            "per metal as the trivial reference; the model's skill is the gap "
            "below it.  No external MAE value is claimed."
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
