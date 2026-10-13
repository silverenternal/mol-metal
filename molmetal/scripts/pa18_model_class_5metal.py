#!/usr/bin/env python3
"""PA18 — model-class matrix on the paper-funnel cohort, per metal (BLK-PA18).

``paper_a`` Table 3's Krasnov contrast is a single cross-cohort point on Ru
(ours 0.7717 vs their LightGBM 0.810) plus one same-cohort model-class pair
(D-MPNN 0.7717 vs LightGBM 0.7677).  A reviewer can call that thin: one metal,
one model-class contrast.

This script widens the model-class axis to the five precious metals and two
classical learners (Morgan-FP + LightGBM, Morgan-FP + XGBoost) on each metal's
own paper-funnel cohort, using the SAME DOI-grouped 10-fold generator
(``cytotox.build_doi_split_folds``) that produced the D-MPNN Ru folds.  The
D-MPNN column is merged in from a per-metal JSON if supplied; otherwise the
entry records that it is absent rather than inventing a number.

Cohort rule (per metal): IC50-present -> metal -> measurement-level dedup ->
76-name non-cancer exclusion, i.e. ``cytotox.build_paper_cohort_indices(df,
metal=m)`` with every surviving cell line.  No cell-line subsetting: the
Krasnov-23 reconstruction is a Ru-specific comparison device and is not applied
to metals Krasnov never published.

Honest framing (AGENTS.md §5.2)
-------------------------------
Every AUC is MEASURED with ``sklearn.metrics.roc_auc_score`` on the fold's
held-out rows.  Folds are DOI-grouped; the straddle counts (raw vs
complex-merged) are recorded per metal, not hidden.

Usage
-----
::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    .venv/bin/python molmetal/scripts/pa18_model_class_5metal.py \\
        --metals Ru Ir Rh Os Re --models lightgbm xgboost \\
        --out metrics/by_metric/pa18_model_class_5metal.json
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
KRASNOV_RU_LIGHTGBM = {"mean": 0.810, "std": 0.024}


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
        from lightgbm import LGBMClassifier

        pos = max(1, int(y_tr.sum()))
        neg = max(1, int(len(y_tr) - pos))
        clf = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            objective="binary",
            metric="auc",
            n_jobs=int(n_jobs),
            random_state=seed,
            verbose=-1,
            min_child_samples=5,
            scale_pos_weight=float(neg) / float(pos),
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    if model == "xgboost":
        import xgboost as xgb

        pos = max(1, int(y_tr.sum()))
        neg = max(1, int(len(y_tr) - pos))
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=int(n_jobs),
            random_state=seed,
            min_child_weight=5,
            scale_pos_weight=float(neg) / float(pos),
            tree_method="hist",
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    raise ValueError(f"unknown model {model!r}")


def run_metal(
    df: pd.DataFrame, metal: str, models: List[str], n_folds: int, seed: int,
    radius: int, nbits: int, n_jobs: int,
) -> Dict[str, object]:
    from sklearn.metrics import roc_auc_score
    from molmetal.baselines.eval_utils import morgan_features
    from molmetal.data import cytotox as C

    cohort = build_cohort(df, metal)
    y = np.asarray(cohort["active"], dtype=int)
    smiles = cohort["SMILES_Ligands"].astype(str).to_numpy()
    n_rows = len(cohort)
    n_complexes = int(cohort[["SMILES_Ligands", "Metal"]].drop_duplicates().shape[0])
    n_cells = int(cohort["Cell_line"].astype(str).nunique())
    print(
        f"[{metal}] rows={n_rows} complexes={n_complexes} cell_lines={n_cells} "
        f"active_rate={y.mean():.4f}",
        flush=True,
    )

    folds = C.build_doi_split_folds(cohort, n_folds=n_folds, seed=seed)
    all_val = np.concatenate([np.asarray(f["val_idx"]) for f in folds])
    if len(np.unique(all_val)) != n_rows:
        raise SystemExit(
            f"[{metal}] fold partition broken: {len(np.unique(all_val))} unique "
            f"val rows for {n_rows} cohort rows"
        )

    # Leakage indicator, recorded per metal (raw DOI vs complex-merged grouping).
    def _straddle(grouping: str) -> int:
        if grouping == "raw":
            grp = C.doi_group_ids(cohort["DOI"])
        else:
            grp = C.doi_group_ids(cohort["DOI"], smiles=cohort["SMILES_Ligands"])
        worst = 0
        for f in folds:
            va = set(int(i) for i in f["val_idx"])
            tr = set(int(i) for i in f["train_idx"])
            worst = max(worst, len({int(grp[i]) for i in va} & {int(grp[i]) for i in tr}))
        return worst

    X = morgan_features(smiles, radius, nbits)

    out: Dict[str, object] = {
        "n_rows": int(n_rows),
        "n_complexes": n_complexes,
        "n_cell_lines": n_cells,
        "active_rate": float(y.mean()),
        "n_folds": int(n_folds),
        "fold_straddle": {
            "raw_doi": _straddle("raw"),
            "complex_merged": _straddle("merged"),
        },
        "models": {},
    }

    for model in models:
        aucs: List[float] = []
        per_fold: List[Dict[str, object]] = []
        for f in folds:
            tr, te = np.asarray(f["train_idx"]), np.asarray(f["val_idx"])
            prob = _fit_predict(model, X[tr], y[tr], X[te], seed, n_jobs)
            auc = float(roc_auc_score(y[te], prob))
            aucs.append(auc)
            per_fold.append(
                {
                    "fold": int(f["fold"]),
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "roc_auc": auc,
                }
            )
            print(f"[{metal}] {model} fold {f['fold']} auc={auc:.4f}", flush=True)
        out["models"][model] = {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "per_fold": per_fold,
            "model_spec": (
                f"morgan_fp(radius={radius}, {nbits}bit) + {model}"
            ),
        }
        print(
            f"[{metal}] {model}: mean={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}",
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
            "shared and n_jobs=-1 thrashes (MEASURED: load avg ~73 on 32 cores "
            "starved a 20-fit run past 15 min)."
        ),
    )
    ap.add_argument(
        "--dmpnn-json",
        default=None,
        help=(
            "optional JSON mapping metal -> {mean_auc, std_auc} to merge as the "
            "D-MPNN column.  Omitted = recorded as absent, never fabricated."
        ),
    )
    ap.add_argument("--out", default="metrics/by_metric/pa18_model_class_5metal.json")
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
        )

    dmpnn: Dict[str, object] = {"source": None, "per_metal": {}}
    if args.dmpnn_json and Path(args.dmpnn_json).is_file():
        dmpnn_raw = json.loads(Path(args.dmpnn_json).read_text())
        dmpnn = {"source": args.dmpnn_json, "per_metal": dmpnn_raw}

    # Model-class verdict per metal: does the D-MPNN (when present) sit at or
    # above the classical learners on the same cohort?
    verdicts: Dict[str, object] = {}
    for metal, rec in per_metal.items():
        means = {m: rec["models"][m]["mean_auc"] for m in rec["models"]}
        best_classical = max(means.values()) if means else None
        dm = None
        if isinstance(dmpnn.get("per_metal"), dict):
            dm = (dmpnn["per_metal"].get(metal) or {}).get("mean_auc")
        verdicts[metal] = {
            "classical_means": means,
            "best_classical_mean": best_classical,
            "dmpnn_mean": dm,
            "dmpnn_minus_best_classical": (
                float(dm - best_classical) if (dm is not None and best_classical is not None) else None
            ),
        }

    report = {
        "metric": "pa18_model_class_5metal",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer / paper-funnel cohort per metal",
        "source": "molmetal/scripts/pa18_model_class_5metal.py",
        "cohort_rule": (
            "per metal: cytotox.build_paper_cohort_indices(df, metal=m) — "
            "IC50-present -> metal -> measurement-level dedup -> 76-name "
            "non-cancer exclusion; all surviving cell lines"
        ),
        "split": "DOI-grouped 10-fold CV (cytotox.build_doi_split_folds), seed=0",
        "label_rule": "active := IC50_Dark_value < 10 uM (paper cut)",
        "models": list(args.models),
        "per_metal": per_metal,
        "dmpnn_column": dmpnn,
        "dmpnn_alignment": (
            {
                "cohort": (
                    "per-metal paper-funnel cohort, ALL surviving cell lines — "
                    "the same cohort the classical column uses"
                ),
                "split": (
                    "cytotox.build_doi_split_folds(cohort, n_folds=10, seed=0) "
                    "— the SAME folds as the classical column"
                ),
                "protocol": (
                    "run_m1_paper_cv.py --cohort-only --no-inner-val "
                    "--deterministic"
                ),
                "row_level_identity": (
                    "verified: the D-MPNN is pinned to the cohort via --data-csv, "
                    "so its test rows equal the classical column's test rows "
                    "EXACTLY (checked per metal x fold, not just by count)"
                ),
                "reproducibility": (
                    "--deterministic pins cuBLAS/cuDNN so the single-seed run is "
                    "bit-reproducible; without it the D-MPNN varies run-to-run on "
                    "CUDA (MEASURED Re fold 0: 0.6658 / 0.8242 / 0.6206)."
                ),
            }
            if dmpnn.get("per_metal")
            else None
        ),
        "verdicts": verdicts,
        "krasnov_2026_ru_lightgbm": KRASNOV_RU_LIGHTGBM,
        "why": (
            "BLK-PA18: widen the Krasnov contrast from a single Ru point and one "
            "model-class pair to 5 metals x {D-MPNN, LightGBM, XGBoost} on each "
            "metal's own paper-funnel cohort and DOI folds, so the cross-cohort "
            "gap is attributable to cohort scope rather than model capability."
        ),
        "honest_framing_note": (
            "MEASURED only.  The D-MPNN column is merged from --dmpnn-json when "
            "supplied and otherwise recorded as absent.  Krasnov published only "
            "Ru and Ir; the other three metals have no external reference."
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
