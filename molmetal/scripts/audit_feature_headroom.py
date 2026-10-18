#!/usr/bin/env python
"""Information audit — how much potency variance each feature block can carry.

This is a diagnostic, not a model.  It answers one question: the current
production design is ECFP4 + metal one-hot + 3 periodic descriptors + cell-line
one-hot (top 40).  Which of the covariates that already exist in
``measurements.parquet`` but never reach the model would add information, and how
much?

Protocol: identical LightGBM hyper-parameters and identical ligand-grouped
5-fold CV for every stage, adding one block at a time.  R² is out-of-fold, so a
block only earns credit if it generalises to unseen ligands.

Stages are cumulative and ordered from "already in the model" to "not in the
model at all".  ``doi`` is reported last and flagged: a paper-level target
encoding is legitimate for interpolation but cannot transfer to a new paper, so
its R² is an upper bound on what an assay-offset model could recover, not a
deployable gain.

Also audits two information losses that are not features at all:
  * censored potency rows, which every current model drops;
  * ``n_exact`` (replicate count), which could weight the loss.

Usage::

    .venv/bin/python molmetal/scripts/audit_feature_headroom.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    LGB_PARAMS,
    N_FOLDS,
    SEED,
    _git_commit,
    _sha256,
    morgan_matrix,
)

ENDPOINT = "potency"
CATEGORICALS = [
    "metal",
    "oxidation_state",
    "charge_complex",
    "counterion",
    "cell_line_key",
    "time_bucket",
    "year",
    "doi",
]
#: Cumulative stages.  The first three are the production MSRS-2 design.
STAGES: List[Tuple[str, List[str]]] = [
    ("ligand_only", ["ligand_ecfp4"]),
    ("+metal", ["ligand_ecfp4", "metal"]),
    ("+metal+cell (production)", ["ligand_ecfp4", "metal", "cell_line_key"]),
    ("+time", ["ligand_ecfp4", "metal", "cell_line_key", "time_bucket"]),
    (
        "+metal chemistry (ox/charge/counterion)",
        ["ligand_ecfp4", "metal", "cell_line_key", "time_bucket",
         "oxidation_state", "charge_complex", "counterion"],
    ),
    (
        "+year",
        ["ligand_ecfp4", "metal", "cell_line_key", "time_bucket",
         "oxidation_state", "charge_complex", "counterion", "year"],
    ),
    (
        "+doi (paper offset; interpolation only)",
        ["ligand_ecfp4", "metal", "cell_line_key", "time_bucket",
         "oxidation_state", "charge_complex", "counterion", "year", "doi"],
    ),
    (
        "+metal chemistry AFTER doi (is it just a paper proxy?)",
        ["ligand_ecfp4", "metal", "cell_line_key", "time_bucket", "year", "doi",
         "oxidation_state", "charge_complex", "counterion"],
    ),
]


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------
def time_bucket(values: pd.Series) -> pd.Series:
    """Collapse 30 durations onto the four that carry most of the data."""
    out = pd.Series("other", index=values.index, dtype=object)
    out[values == 24] = "24h"
    out[values == 48] = "48h"
    out[values == 72] = "72h"
    out[values == 96] = "96h"
    out[values.isna()] = "missing"
    return out


def categorical_codes(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Integer codes for categorical columns; missing becomes its own level."""
    blocks, levels = [], {}
    for col in columns:
        values = df[col].astype(object).where(df[col].notna(), "__missing__")
        codes, uniques = pd.factorize(values.astype(str), sort=True)
        blocks.append(codes.astype(np.int32))
        levels[col] = int(len(uniques))
    if not blocks:
        return np.zeros((len(df), 0), dtype=np.int32), levels
    return np.column_stack(blocks).astype(np.int32), levels


def build_stage(
    df: pd.DataFrame, features: Sequence[str], fp: sparse.csr_matrix
) -> Tuple[sparse.csr_matrix, List[int], Dict[str, int]]:
    """Return (X, categorical indices, level counts) for one cumulative stage."""
    cat_cols = [f for f in features if f in CATEGORICALS]
    codes, levels = categorical_codes(df, cat_cols)
    blocks = []
    cat_idx: List[int] = []
    if "ligand_ecfp4" in features:
        blocks.append(fp)
        n_lig = fp.shape[1]
    else:
        n_lig = 0
    if codes.shape[1]:
        blocks.append(sparse.csr_matrix(codes.astype(np.float32)))
        cat_idx = list(range(n_lig, n_lig + codes.shape[1]))
    X = sparse.hstack(blocks, format="csr")
    return X, cat_idx, levels


def oof_regression(
    X: sparse.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    categorical_feature: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    # ``categorical_feature`` must be a fit() argument, not a booster param: sklearn's
    # LGBMRegressor forwards unknown kwargs to the C++ layer, which rejects an int list.
    cat = list(categorical_feature) if categorical_feature else "auto"
    for k, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups=groups)):
        model = LGBMRegressor(**LGB_PARAMS)
        model.fit(X[tr], y[tr], categorical_feature=cat)
        oof[te] = model.predict(X[te])
        fold_id[te] = k
    assert not np.isnan(oof).any()
    return oof, fold_id


def score(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    resid = y - pred
    return {
        "r2": round(float(1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)), 4),
        "mae": round(float(np.mean(np.abs(resid))), 4),
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 4),
    }


# ---------------------------------------------------------------------------
# Information losses that are not features
# ---------------------------------------------------------------------------
def censoring_audit(long: pd.DataFrame) -> Dict[str, object]:
    """Censored targets: are they dropped, or carried as if exact?

    MSRS-0 records a right-censored potency at its detection bound, so the row is
    *kept* with ``value_log10 = log10(bound)``.  The target is therefore a lower
    bound treated as an exact value.  The audit reports the point mass at the
    bound and, more importantly, whether censoring is **differential by metal** —
    if one metal is censored more often, its mean is compressed more, and the
    metal ordering is biased by an amount comparable to the effects being ranked.
    """
    sub = long[long["endpoint"] == ENDPOINT]
    exact = sub[sub["value_log10"].notna()]
    cens = sub[sub["censoring"].astype(str) != "none"]
    bounds = cens["censor_bound"].dropna()
    by_metal = {}
    for metal, g in sub.groupby("metal"):
        by_metal[str(metal)] = {
            "n_rows": int(len(g)),
            "n_censored": int((g["censoring"].astype(str) != "none").sum()),
            "censored_share": round(float((g["censoring"].astype(str) != "none").mean()), 4),
            "mean_value_log10": round(float(g["value_log10"].mean()), 4),
        }
    shares = [v["censored_share"] for v in by_metal.values()]
    return {
        "n_rows_total": int(len(sub)),
        "n_exact": int(len(exact)),
        "n_censored": int(len(cens)),
        "censored_share": round(float(len(cens) / max(1, len(sub))), 4),
        "censoring_kinds": {str(k): int(v) for k, v in cens["censoring"].value_counts().items()},
        "n_all_censored_groups": int(cens["all_censored"].sum()) if "all_censored" in cens else None,
        "n_rows_at_the_ceiling": int((sub["value_log10"] == 2.0).sum()),
        "exact_median_log10": round(float(exact["value_log10"].median()), 4),
        "censored_bound_median_log10": (
            round(float(np.log10(bounds.median())), 4) if len(bounds) else None
        ),
        "censored_share_by_metal": by_metal,
        "censored_share_spread": round(float(max(shares) - min(shares)), 4) if shares else None,
        "interpretation": (
            "Censored potency rows are NOT dropped: MSRS-0 carries them at the "
            "detection bound, so the target is a lower bound treated as exact. "
            "That compresses the inactive tail. Because the censored share differs "
            "across metals (spread above), the compression is differential and "
            "biases the metal ordering by roughly share_spread x tail_shortfall."
        ),
    }


def reliability_audit(dedup: pd.DataFrame) -> Dict[str, object]:
    """``n_exact`` is a replicate count and could weight the loss; is it informative?"""
    sub = dedup[(dedup["endpoint"] == ENDPOINT) & dedup["value_log10"].notna()]
    counts = sub["n_exact"].value_counts().sort_index()
    return {
        "n_rows": int(len(sub)),
        "n_exact_distribution": {str(int(k)): int(v) for k, v in counts.items()},
        "share_single_measurement": round(float((sub["n_exact"] == 1).mean()), 4),
        "note": "n_exact >= 2 rows are medians of replicate measurements and are the least noisy targets",
    }


def coverage_audit(df: pd.DataFrame) -> Dict[str, object]:
    """Cardinality and missingness of every covariate the audit uses."""
    out = {}
    for col in CATEGORICALS + ["ligand_main_fragment", "ligand_canonical"]:
        if col not in df.columns:
            continue
        out[col] = {
            "n_unique": int(df[col].nunique()),
            "missing_share": round(float(df[col].isna().mean()), 4),
        }
    cell = df["cell_line_key"].value_counts()
    out["cell_line_top40_share"] = round(float(cell.head(40).sum() / len(df)), 4)
    out["time_top_values"] = {str(k): int(v) for k, v in df["time_h"].value_counts().head(6).items()}
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def select_rows(long: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Two target conventions, and they are not the same experiment.

    ``exact_only`` is what MSRS-2 actually fits: ``all_censored`` groups are
    dropped.  ``censored_at_bound`` keeps every row, with the censored ones
    sitting at ``log10(bound)``.  The long table carries the bound; the dedup
    table sets those rows to NaN and MSRS-2 drops them (4,813 of 26,219).
    """
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ]
    if mode == "exact_only":
        sub = sub[sub["censoring"].astype(str) == "none"]
    elif mode != "censored_at_bound":
        raise ValueError(f"unknown row mode: {mode}")
    out = sub.copy().reset_index(drop=True)
    out["time_bucket"] = time_bucket(out["time_h"])
    return out


def run_ladder(sub: pd.DataFrame) -> Tuple[Dict[str, object], float]:
    y = sub["value_log10"].to_numpy(dtype=float)
    groups = sub["ligand_main_fragment"].to_numpy()
    fp = morgan_matrix(sub["ligand_main_fragment"].tolist())
    print(f"[audit] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()}")
    stages: Dict[str, object] = {}
    for name, features in STAGES:
        t0 = time.time()
        X, cat_idx, levels = build_stage(sub, features, fp)
        oof, _ = oof_regression(X, y, groups, cat_idx)
        stages[name] = {
            **score(y, oof),
            "n_features": int(X.shape[1]),
            "categorical_levels": levels,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        print(f"  {name:56s} R2={stages[name]['r2']:.4f} MAE={stages[name]['mae']:.4f}")

    base = stages["+metal+cell (production)"]["r2"]
    for name, entry in stages.items():
        entry["r2_delta_vs_production"] = round(float(entry["r2"] - base), 4)
    return stages, base


def headroom_block(stages: Dict[str, object], base: float) -> Dict[str, object]:
    return {
        "production_r2": base,
        "time_only_delta": stages["+time"]["r2_delta_vs_production"],
        "metal_chemistry_delta": stages["+metal chemistry (ox/charge/counterion)"][
            "r2_delta_vs_production"
        ],
        "year_delta": stages["+year"]["r2_delta_vs_production"],
        "paper_offset_delta": stages["+doi (paper offset; interpolation only)"][
            "r2_delta_vs_production"
        ],
        # Incremental value of metal-state chemistry once the paper offset is
        # already in the model.  If this collapses, the chemistry block was a
        # paper/ligand proxy, not a metal signal.
        "metal_chemistry_after_doi_delta": round(
            float(
                stages["+metal chemistry AFTER doi (is it just a paper proxy?)"]["r2"]
                - stages["+doi (paper offset; interpolation only)"]["r2"]
            ),
            4,
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "feature_headroom_audit.json"
    )
    args = parser.parse_args(argv)

    for p in (args.long, args.dedup):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    long = pd.read_parquet(args.long)
    dedup = pd.read_parquet(args.dedup)

    rowsets = {}
    for mode in ("exact_only", "censored_at_bound"):
        sub = select_rows(long, mode)
        stages, base = run_ladder(sub)
        rowsets[mode] = {
            "n_rows": int(len(sub)),
            "n_ligands": int(sub["ligand_main_fragment"].nunique()),
            "coverage": coverage_audit(sub),
            "stages": stages,
            "headroom": headroom_block(stages, base),
        }

    # The production convention is the one the paper's numbers come from.
    stages = rowsets["exact_only"]["stages"]
    base = rowsets["exact_only"]["headroom"]["production_r2"]

    out = {
        "artifact": "feature_headroom_audit",
        "version": "1.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "endpoint": ENDPOINT,
            "target": "value_log10 (log10 IC50, uM)",
            "cv": f"GroupKFold({N_FOLDS}) grouped by ligand_main_fragment",
            "learner": "LightGBM, identical hyper-parameters at every stage",
            "stages": [name for name, _ in STAGES],
            "rowsets": {
                "exact_only": "censored rows dropped (this is what MSRS-2 fits)",
                "censored_at_bound": "censored rows kept at log10(bound)",
            },
            "warning": (
                "doi is a paper-level categorical. Its R2 is an upper bound on an assay-offset "
                "model and is not deployable to an unseen paper."
            ),
            "lightgbm": LGB_PARAMS,
            "seed": SEED,
        },
        "coverage": rowsets["exact_only"]["coverage"],
        "censoring": censoring_audit(long),
        "reliability": reliability_audit(dedup),
        "rowsets": rowsets,
        "stages": stages,
        "headroom": rowsets["exact_only"]["headroom"],
        "caveats": [
            "Cumulative stages: each R2 is the whole feature set, not the marginal value of the block in isolation.",
            "Two row sets are reported because the long and dedup tables disagree on censored rows; "
            "`exact_only` matches MSRS-2, `censored_at_bound` is what the long table carries.",
            "The DOI stage can memorise paper-specific offsets and is reported as an upper bound only.",
            "Counterion is 33.0% missing; missing is encoded as its own level.",
            "The last stage tests whether the metal-chemistry block survives a paper "
            "offset. It largely does not: the block is a paper/ligand proxy here.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    for mode, block in rowsets.items():
        print(f"  [{mode}] n={block['n_rows']} production R2={block['headroom']['production_r2']:.4f}")
        print(f"           headroom = {json.dumps(block['headroom'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
