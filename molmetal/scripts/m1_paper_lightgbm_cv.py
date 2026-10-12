#!/usr/bin/env python3
"""BLK-PA5: Morgan-FP + LightGBM on the Krasnov-comparable paper cohort.

Why this exists
---------------
``metalcytotox_ru_auc.json::paper_protocol_measurement`` records our D-MPNN at
0.7717 +/- 0.0380 on the paper-funnel Ru cohort (15,799 rows / 4,793 complexes
/ 591 cell lines, DOI-grouped 10-fold CV).  The paper reports 0.810 +/- 0.024
(LightGBM).  Claim 3 calls that "the only like-for-like comparison" while the
method section admits the cohort differs — so two things move at once (model
class AND cohort) and the honest-negative cannot be attributed.

This script removes one of the two confounds.  It reuses the EXACT cohort and
the EXACT fold partition of the D-MPNN measurement and swaps only the model
class (D-MPNN -> Morgan-FP + LightGBM).  Same rows, same folds, same label cut
(IC50 < 10 uM).  Any residual gap is then attributable to the model class on
THIS cohort, not to the cohort.

Fold provenance (important)
---------------------------
The folds default to the on-disk partition under ``--folds-dir`` — the ones the
D-MPNN number was scored against — so the two arms are fold-identical.  Those
stored folds were built with RAW-DOI grouping (MEASURED 2026-10-12: 0 straddles
under raw DOI, 18 under the complex-merged grouping), i.e. a complex appearing
in two articles can land on both sides.  That is a caveat on BOTH arms equally,
so the model-class contrast is unaffected; it is recorded in the output rather
than hidden.  ``--regenerate-folds`` instead rebuilds a leak-free (complex-
merged) partition with the current code, which no longer matches the stored
D-MPNN number.

Usage::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    .venv/bin/python molmetal/scripts/m1_paper_lightgbm_cv.py \\
        --folds-dir /tmp/wf_m1_paper_cv/folds \\
        --out metrics/by_metric/metalcytotox_ru_paper_lightgbm_cv.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAPER_RU_LIGHTGBM = {"mean": 0.810, "std": 0.024}
DMPNN_OURS = {"mean": 0.7717467464034218, "std": 0.037955584760391604}


def build_cohort(df: pd.DataFrame):
    """Paper funnel: IC50-present -> Ru -> dedup -> 76-name non-cancer exclusion."""
    from molmetal.data import cytotox as C

    idx = C.build_paper_cohort_indices(df)
    cohort = df.iloc[idx].reset_index(drop=True)
    cohort = C._add_pic50(cohort)
    cohort = C._add_active(cohort)
    return cohort


def load_folds(folds_dir: Path, n_folds: int) -> List[Dict[str, np.ndarray]]:
    folds = []
    for k in range(n_folds):
        fp = folds_dir / f"fold_{k}.json"
        if not fp.is_file():
            raise SystemExit(f"missing {fp}; pass --regenerate-folds or a valid --folds-dir")
        d = json.loads(fp.read_text())
        folds.append(
            {
                "fold": int(d["fold"]),
                "train_idx": np.asarray(d["train_idx"], dtype=np.int64),
                "val_idx": np.asarray(d["val_idx"], dtype=np.int64),
            }
        )
    return folds


def regenerate_folds(cohort: pd.DataFrame, n_folds: int, seed: int):
    from molmetal.data import cytotox as C

    folds = C.build_doi_split_folds(cohort, n_folds=n_folds, seed=seed)
    return [
        {
            "fold": int(f["fold"]),
            "train_idx": np.asarray(f["train_idx"], dtype=np.int64),
            "val_idx": np.asarray(f["val_idx"], dtype=np.int64),
        }
        for f in folds
    ]


def fold_group_straddle(cohort: pd.DataFrame, folds, grouping: str) -> int:
    """Count DOI groups appearing on both sides of any fold (leak indicator)."""
    from molmetal.data import cytotox as C

    if grouping == "raw":
        grp = C.doi_group_ids(cohort["DOI"])
    else:
        grp = C.doi_group_ids(cohort["DOI"], smiles=cohort["SMILES_Ligands"])
    worst = 0
    for f in folds:
        va = set(int(i) for i in f["val_idx"])
        tr = set(int(i) for i in f["train_idx"])
        gv = {int(grp[i]) for i in va}
        gt = {int(grp[i]) for i in tr}
        worst = max(worst, len(gv & gt))
    return worst


def run(args) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from lightgbm import LGBMClassifier

    from molmetal.baselines.eval_utils import morgan_features

    data_dir = Path(
        os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
    )
    csv_path = Path(args.csv) if args.csv else data_dir / "MetalCytoToxDB.csv"
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    cohort = build_cohort(df)
    y = np.asarray(cohort["active"], dtype=int)
    smiles = cohort["SMILES_Ligands"].astype(str).to_numpy()
    n_rows = len(cohort)
    print(
        f"[cohort] rows={n_rows} "
        f"complexes={cohort[['SMILES_Ligands', 'Metal']].drop_duplicates().shape[0]} "
        f"cell_lines={cohort['Cell_line'].astype(str).nunique()} "
        f"active_rate={y.mean():.4f}  (paper: 11294 / 4399 / 23)",
        flush=True,
    )

    if args.regenerate_folds:
        folds = regenerate_folds(cohort, args.n_folds, args.seed)
        fold_source = f"regenerated (build_doi_split_folds n_folds={args.n_folds} seed={args.seed})"
    else:
        folds = load_folds(Path(args.folds_dir), args.n_folds)
        fold_source = str(args.folds_dir)

    # Partition sanity: every row exactly once across held-out folds.
    all_val = np.concatenate([f["val_idx"] for f in folds])
    if not (len(all_val) == n_rows and len(np.unique(all_val)) == n_rows):
        raise SystemExit(
            f"fold partition broken: {len(all_val)} val rows for {n_rows} cohort rows"
        )

    X = morgan_features(smiles, args.radius, args.nbits)

    per_fold = []
    aucs, aps = [], []
    for f in folds:
        tr, te = f["train_idx"], f["val_idx"]
        y_tr = y[tr]
        pos = max(1, int(y_tr.sum()))
        neg = max(1, int(len(y_tr) - pos))
        params = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            objective="binary",
            metric="auc",
            n_jobs=-1,
            random_state=args.seed,
            verbose=-1,
            min_child_samples=5,
            scale_pos_weight=float(neg) / float(pos),
        )
        clf = LGBMClassifier(**params)
        clf.fit(X[tr], y_tr)
        prob = clf.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y[te], prob))
        ap = float(average_precision_score(y[te], prob))
        aucs.append(auc)
        aps.append(ap)
        per_fold.append(
            {
                "fold": int(f["fold"]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "roc_auc": auc,
                "pr_auc": ap,
            }
        )
        print(f"[fold {f['fold']}] n_train={len(tr)} n_test={len(te)} auc={auc:.4f}", flush=True)

    mean = float(np.mean(aucs))
    std = float(np.std(aucs))

    report = {
        "metric": "metalcytotox_ru_paper_lightgbm_cv",
        "description": (
            "Morgan-FP + LightGBM on the paper-funnel Ru cohort and the SAME "
            "DOI-grouped 10-fold partition used by the D-MPNN paper-protocol "
            "measurement. Isolates model class from cohort (BLK-PA5)."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier1_precious_metal_anticancer / paper-protocol Ru cohort",
        "source": "molmetal/scripts/m1_paper_lightgbm_cv.py",
        "model": f"morgan_fp(radius={args.radius}, {args.nbits}bit) + lightgbm",
        "seed": args.seed,
        "label_rule": "active := IC50_Dark_value < 10 uM (paper cut)",
        "cohort": {
            "rows": int(n_rows),
            "complexes": int(cohort[["SMILES_Ligands", "Metal"]].drop_duplicates().shape[0]),
            "cell_lines": int(cohort["Cell_line"].astype(str).nunique()),
            "active_rate": float(y.mean()),
        },
        "fold_source": fold_source,
        "fold_leakage_check": {
            "straddling_groups_raw_doi": fold_group_straddle(cohort, folds, "raw"),
            "straddling_groups_complex_merged": fold_group_straddle(cohort, folds, "merged"),
            "note": (
                "Straddling groups = DOI groups appearing on both sides of a "
                "fold. Non-zero under the complex-merged grouping means a "
                "complex can sit in train and test; this applies to BOTH arms "
                "equally and does not bias the model-class contrast."
            ),
        },
        "n_folds": len(folds),
        "per_fold": per_fold,
        "mean_auc": mean,
        "std_auc": std,
        "comparison": {
            "lightgbm_this_cohort": {"mean": mean, "std": std},
            "dmpnn_this_cohort": DMPNN_OURS,
            "delta_lightgbm_minus_dmpnn": mean - DMPNN_OURS["mean"],
            "paper_lightgbm_reference_cross_cohort": PAPER_RU_LIGHTGBM,
        },
        "verdict": (
            "Like-for-like model-class comparison on the SAME cohort and folds: "
            f"LightGBM {mean:.4f} +/- {std:.4f} vs D-MPNN "
            f"{DMPNN_OURS['mean']:.4f} +/- {DMPNN_OURS['std']:.4f} "
            f"(delta {mean - DMPNN_OURS['mean']:+.4f})."
        ),
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--folds-dir", default="/tmp/wf_m1_paper_cv/folds")
    ap.add_argument("--regenerate-folds", action="store_true")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument(
        "--out",
        default="metrics/by_metric/metalcytotox_ru_paper_lightgbm_cv.json",
    )
    args = ap.parse_args(argv)

    rep = run(args)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(out, json.dumps(rep, indent=2, default=float))
    print(
        f"\n[BLK-PA5] LightGBM mean={rep['mean_auc']:.4f} +/- {rep['std_auc']:.4f} "
        f"| D-MPNN {DMPNN_OURS['mean']:.4f} | paper(cross-cohort) "
        f"{PAPER_RU_LIGHTGBM['mean']:.3f}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
