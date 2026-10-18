#!/usr/bin/env python3
"""AUD-M4 — learning curve: is the binding constraint data volume or features?

Before adding features, measure how much of the surface is already recoverable
from 10/25/50/100% of the training ligands.  A curve that is still climbing at
100% says "get more data"; a curve that has plateaued says "the features are the
constraint, and more rows will not help".

Protocol: the production design (ECFP4 + metal + cell), the production folds
(ligand-grouped 5-fold), the production LightGBM hyper-parameters.  Subsampling
is by ligand group, never by row, so a ligand cannot be half-in and half-out.

Usage::

    .venv/bin/python molmetal/scripts/msrs_learning_curve.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_code_levers import base_blocks, folds_for  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    LGB_PARAMS,
    N_FOLDS,
    SEED,
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ENDPOINT = "potency"
FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def oof_at_fraction(
    X, y: np.ndarray, groups: np.ndarray, fold_id: np.ndarray, fraction: float, seed: int
):
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(seed)
    oof = np.full(len(y), np.nan)
    n_train = []
    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = ~tr
        ligands = np.unique(groups[tr])
        n_keep = max(1, int(round(len(ligands) * fraction)))
        keep = set(rng.choice(ligands, size=n_keep, replace=False).tolist())
        rows = tr & np.isin(groups, list(keep))
        n_train.append(int(rows.sum()))
        model = LGBMRegressor(**LGB_PARAMS)
        model.fit(X[rows], y[rows])
        oof[te] = model.predict(X[te])
    assert not np.isnan(oof).any()
    return oof, n_train


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--metal-desc", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_learning_curve.json"
    )
    args = parser.parse_args(argv)

    for p in (args.long, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    long = pd.read_parquet(args.long)
    descriptors = pd.read_csv(args.metal_desc)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy().reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    groups = sub["ligand_main_fragment"].to_numpy()
    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    lig, met, cell = base_blocks(sub, descriptors, cell_vocab)
    from scipy import sparse

    X = sparse.hstack([lig, met, cell], format="csr")
    fold_id = folds_for(groups)
    print(f"[curve] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()}")

    curve: Dict[str, object] = {}
    for f in FRACTIONS:
        t0 = time.time()
        oof, n_train = oof_at_fraction(X, y, groups, fold_id, f, SEED)
        entry = {
            **score(y, oof),
            "fraction": f,
            "n_train_rows_per_fold": n_train,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        curve[f"{f:.2f}"] = entry
        print(f"  {f:5.2f}  R2={entry['r2']:.4f}  MAE={entry['mae']:.4f}  ({entry['elapsed_seconds']}s)")

    full = curve["1.00"]["r2"]
    plateau = None
    for f in FRACTIONS:
        if curve[f"{f:.2f}"]["r2"] >= 0.95 * full:
            plateau = f
            break

    out = {
        "artifact": "msrs_learning_curve",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "endpoint": ENDPOINT,
            "design": "ECFP4 + metal(onehot+3 descriptors) + cell(top40+other)",
            "cv": f"GroupKFold({N_FOLDS}) grouped by ligand_main_fragment",
            "subsample_unit": "ligand group (never row)",
            "fractions": list(FRACTIONS),
            "lightgbm": LGB_PARAMS,
            "seed": SEED,
        },
        "n_rows": int(len(sub)),
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "curve": curve,
        "plateau_fraction_95pct_of_full": plateau,
        "reading": (
            "If the curve is still climbing steeply at 1.00, more data is the lever. "
            "If 0.50 already reaches 95% of the full R2, rows are not the constraint and "
            "feature/structure work is."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}; plateau at {plateau}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
