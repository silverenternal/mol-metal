#!/usr/bin/env python3
"""MSRS code levers — AUD-F1 / AUD-D1 / AUD-M1 under one fixed protocol.

The audit (`audit_feature_headroom.py`) said where the information is not.  This
script changes the code and measures the three levers that the audit left open,
each against the same ligand-grouped 5-fold split and the same LightGBM
hyper-parameters as MSRS-2:

  AUD-F1  ligand representation   ECFP4 -> ECFP4 + RDKit2D-32
  AUD-D1  censored target         drop censored rows -> right-censored normal
                                  likelihood (Tobit), trained on every row and
                                  evaluated on the exact rows only
  AUD-M1  interaction structure   hard additive constraints -> rank-r bilinear
                                  ligand x metal features, so the metal order can
                                  depend on the ligand

Every arm also gets the decision-layer read (top-1 regret against a fold-safe
``global_best``, DOI-clustered), because MSRS-7 showed that a better MAE does
not imply a better ranking.

Usage::

    .venv/bin/python molmetal/scripts/msrs_code_levers.py
    .venv/bin/python molmetal/scripts/msrs_code_levers.py --arms additive,f1_rdkit --limit 4000
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
from scipy import sparse, stats
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.msrs_recommendation import (  # noqa: E402
    build_cohorts,
    cohort_metrics,
    doi_clustered_test,
    fold_baseline_scores,
    summarise_cohorts,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    LGB_PARAMS,
    METALS,
    N_FOLDS,
    SEED,
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    cell_matrix,
    metal_matrix,
    morgan_matrix,
    score,
)

ENDPOINT = "potency"
SIGMA = 0.5  # censored-normal scale, log10 units (baseline residual sd is ~0.5)
COHORT_KEYS = ["ligand_canonical", "cell_line_key", "doi", "time_h"]
RDKIT_DIM = 32

#: name -> (uses_rdkit, interaction_rank, constraints, categorical_mode)
#: categorical_mode: "none" = one-hot; "top40" = native categorical at the same
#: cardinality as the one-hot design; "full" = native categorical over every level.
ARMS: Dict[str, Tuple[bool, int, bool, str]] = {
    "additive": (False, 0, True, "none"),
    "interaction": (False, 0, False, "none"),
    "f1_rdkit": (True, 0, False, "none"),
    "f1_rdkit_additive": (True, 0, True, "none"),
    "m1_rank2": (False, 2, False, "none"),
    "m1_rank4": (False, 4, False, "none"),
    "combined_rank4_rdkit": (True, 4, False, "none"),
    # Cell line and metal as native categorical splits instead of one-hot.
    # "top40" holds cardinality fixed, so the difference is the split rule;
    # "full" also drops the top-40 cap and lets LightGBM see all 733 levels.
    "cat_cell": (False, 0, False, "top40"),
    "cat_cell_full": (False, 0, False, "full"),
    "cat_cell_rdkit": (True, 0, False, "top40"),
    # Combinations of the two levers that actually moved R2.
    "cat_full_rank2": (False, 2, False, "full"),
    "best_v2": (True, 4, False, "full"),
}


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------
_RDKIT_CACHE: Dict[str, Optional[np.ndarray]] = {}


def rdkit_block(smiles: Sequence[str]) -> np.ndarray:
    """RDKit2D-32 per ligand; NaN where the descriptor could not be computed.

    LightGBM handles NaN natively, so no imputation is applied.  Imputing with a
    global median would inject a value the molecule does not have.
    """
    from molmetal.data.rdkit_2d_extended import compute_rdkit_2d_extended

    out = np.full((len(smiles), RDKIT_DIM), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles):
        if smi not in _RDKIT_CACHE:
            try:
                _RDKIT_CACHE[smi] = compute_rdkit_2d_extended(smi)
            except Exception:
                _RDKIT_CACHE[smi] = None
        vec = _RDKIT_CACHE[smi]
        if vec is not None and len(vec) == RDKIT_DIM:
            out[i] = vec
    return out


def native_codes(
    df: pd.DataFrame, cell_vocabulary: Sequence[str], mode: str
) -> np.ndarray:
    """Integer codes for metal + cell line, for LightGBM's categorical splits."""
    metal_index = {m: i for i, m in enumerate(METALS)}
    if mode == "full":
        cell_values = df["cell_line_key"].astype(str)
        levels = {c: i for i, c in enumerate(sorted(cell_values.unique()))}
    elif mode == "top40":
        levels = {c: i for i, c in enumerate(cell_vocabulary)}
        levels["__other__"] = len(cell_vocabulary)
    else:
        raise ValueError(f"unknown categorical mode: {mode}")
    other = levels.get("__other__", len(levels))
    metal = np.array([metal_index[m] for m in df["metal"]], dtype=np.int32)
    cell = np.array([levels.get(c, other) for c in df["cell_line_key"].astype(str)], dtype=np.int32)
    return np.column_stack([metal, cell])


def base_blocks(
    df: pd.DataFrame, descriptors: pd.DataFrame, cell_vocabulary: Sequence[str]
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    lig = morgan_matrix(df["ligand_main_fragment"].tolist())
    met = sparse.csr_matrix(metal_matrix(df["metal"].tolist(), descriptors))
    cell = cell_matrix(df["cell_line_key"].tolist(), cell_vocabulary)
    return lig, met, cell


def interaction_block(
    lig: sparse.csr_matrix, met: sparse.csr_matrix, svd: TruncatedSVD
) -> sparse.csr_matrix:
    """Rank-r bilinear term: u_r(ligand) * 1[metal = m] for every r, m.

    The additive surface is f(x) + g(m) + h(c).  A rank-r outer product lets the
    metal coefficient vary with the ligand, which is exactly the structure the
    hard interaction constraints forbid.
    """
    u = svd.transform(lig)  # (n, r)
    onehot = met[:, : len(METALS)].toarray()
    cols = [u[:, [i]] * onehot for i in range(u.shape[1])]
    return sparse.csr_matrix(np.hstack(cols).astype(np.float32))


# ---------------------------------------------------------------------------
# Censored-normal objective
# ---------------------------------------------------------------------------
def tobit_objective(censored: np.ndarray, sigma: float):
    """Right-censored normal NLL: exact rows are squared error, censored rows
    are penalised only for predicting *below* the detection bound.

    For a censored row at bound ``b`` the likelihood is ``P(y > b) = S(z)`` with
    ``z = (b - mu) / sigma``, so ``dNLL/dmu = -phi(z) / (sigma * S(z))``.  The
    gradient is negative (LightGBM moves ``mu`` up) and its magnitude grows as
    the prediction falls further below the bound — which is the whole point:
    a compound that was inactive at the top dose must not be fitted as potent.
    """
    from scipy.stats import norm

    def obj(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y = dataset.get_label()
        z = np.clip((y - preds) / sigma, -10.0, 10.0)
        sf = np.clip(norm.sf(z), 1e-12, None)
        grad = np.where(censored, -norm.pdf(z) / (sigma * sf), (preds - y) / sigma ** 2)
        hess = np.full_like(preds, 1.0 / sigma ** 2)
        return grad, hess

    return obj


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def folds_for(groups: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    fold_id = np.full(len(groups), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_folds)
    for k, (_, te) in enumerate(splitter.split(np.zeros((len(groups), 1)), groups=groups)):
        fold_id[te] = k
    assert (fold_id >= 0).all()
    return fold_id


def fit_arm(
    lig: sparse.csr_matrix,
    met: sparse.csr_matrix,
    cell: sparse.csr_matrix,
    rdkit: Optional[np.ndarray],
    rank: int,
    constraints: bool,
    y: np.ndarray,
    fold_id: np.ndarray,
    censored: Optional[np.ndarray] = None,
    cat_codes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """OOF predictions for one arm.  SVD is fitted inside each training fold.

    ``cat_codes`` switches metal and cell line from one-hot to LightGBM's native
    categorical splits; the codes are appended after the ligand block.
    """
    from lightgbm import LGBMRegressor, train as lgb_train
    import lightgbm as lgb

    oof = np.full(len(y), np.nan)
    params = dict(LGB_PARAMS)
    if constraints and rank == 0:
        n_lig, n_met, n_cell = lig.shape[1], met.shape[1], cell.shape[1]
        n_extra = 0 if rdkit is None else rdkit.shape[1]
        params["interaction_constraints"] = [
            list(range(0, n_lig + n_extra)),
            list(range(n_lig + n_extra, n_lig + n_extra + n_met)),
            list(range(n_lig + n_extra + n_met, n_lig + n_extra + n_met + n_cell)),
        ]

    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = ~tr
        if cat_codes is not None:
            parts_tr, parts_te = [lig[tr]], [lig[te]]
            if rdkit is not None:
                parts_tr.append(sparse.csr_matrix(rdkit[tr]))
                parts_te.append(sparse.csr_matrix(rdkit[te]))
            parts_tr.append(sparse.csr_matrix(cat_codes[tr].astype(np.float32)))
            parts_te.append(sparse.csr_matrix(cat_codes[te].astype(np.float32)))
            n_lig = lig.shape[1]
            n_extra = 0 if rdkit is None else rdkit.shape[1]
            cat_idx = [n_lig + n_extra + i for i in range(cat_codes.shape[1])]
        else:
            parts_tr, parts_te = [lig[tr], met[tr], cell[tr]], [lig[te], met[te], cell[te]]
            cat_idx = []
            if rdkit is not None:
                parts_tr.append(sparse.csr_matrix(rdkit[tr]))
                parts_te.append(sparse.csr_matrix(rdkit[te]))
            if rank:
                svd = TruncatedSVD(n_components=rank, random_state=SEED)
                svd.fit(lig[tr])
                parts_tr.append(interaction_block(lig[tr], met[tr], svd))
                parts_te.append(interaction_block(lig[te], met[te], svd))
        X_tr = sparse.hstack(parts_tr, format="csr")
        X_te = sparse.hstack(parts_te, format="csr")

        if censored is None:
            model = LGBMRegressor(**params)
            if cat_idx:
                model.fit(X_tr, y[tr], categorical_feature=cat_idx)
            else:
                model.fit(X_tr, y[tr])
            oof[te] = model.predict(X_te)
        else:
            # ``lgb.train`` is the documented path for a callable objective; the
            # sklearn wrapper forwards it to the C++ layer instead.
            p = dict(params)
            p.pop("objective", None)
            n_rounds = p.pop("n_estimators", 500)
            p["objective"] = tobit_objective(censored[tr], SIGMA)
            init = float(np.mean(y[tr]))
            booster = lgb_train(
                p,
                lgb.Dataset(
                    X_tr,
                    label=y[tr],
                    init_score=np.full(int(tr.sum()), init),
                    categorical_feature=cat_idx if cat_idx else "auto",
                ),
                num_boost_round=n_rounds,
            )
            oof[te] = booster.predict(X_te) + init
    assert not np.isnan(oof).any()
    return oof


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------
def order_diversity(cohorts: pd.DataFrame, scorer: str) -> Dict[str, object]:
    """How many distinct metal orders does this scorer produce inside a fold?

    Cohorts whose predicted scores are tied to machine precision are skipped:
    their "order" is decided by row order, not by the model, and counting them
    would manufacture diversity that is not there.
    """
    out: Dict[str, object] = {"by_fold": {}, "min_distinct_orders": None, "n_tied_cohorts": 0}
    counts = []
    for k, g in cohorts.groupby("fold"):
        orders = []
        for _cid, x in g.groupby("cohort"):
            vals = x[scorer].to_numpy(dtype=float)
            if len(vals) > 1 and float(vals.max() - vals.min()) < 1e-9:
                out["n_tied_cohorts"] = int(out["n_tied_cohorts"]) + 1
                continue
            orders.append(tuple(x.sort_values(scorer, kind="stable")["metal"]))
        n = int(len(set(orders)))
        counts.append(n)
        out["by_fold"][str(int(k))] = {
            "n_cohorts_strict": len(orders),
            "n_distinct_orders": n,
        }
    out["min_distinct_orders"] = int(min(counts)) if counts else None
    out["constant_within_fold"] = bool(counts and min(counts) == 1)
    return out


def decision_report(
    sub: pd.DataFrame, y: np.ndarray, fold_id: np.ndarray, preds: Dict[str, np.ndarray]
) -> Dict[str, object]:
    frame = sub[COHORT_KEYS + ["metal"]].copy()
    frame["y"] = y
    frame["fold"] = fold_id
    frame["cohort"] = frame.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup().to_numpy()
    for name, pred in preds.items():
        frame[name] = pred

    scorers = list(preds)
    cohorts = build_cohorts(frame, scorers)
    records = []
    for cid, g in cohorts.groupby("cohort"):
        y_obs = g["y"].to_numpy()
        for s in scorers:
            rec = cohort_metrics(y_obs, g[s].to_numpy())
            records.append({"cohort": int(cid), "scorer": s, "doi": g["doi"].iloc[0], **rec})
    rec = pd.DataFrame(records)

    out: Dict[str, object] = {"n_cohorts": int(cohorts["cohort"].nunique()), "scorers": {}}
    for s in scorers:
        entry = {
            "summary": summarise_cohorts(rec[rec["scorer"] == s]),
            "order_diversity": order_diversity(cohorts, s),
        }
        if s != "global_best":
            entry["vs_global_best"] = doi_clustered_test(rec, s, "global_best")
        out["scorers"][s] = entry
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--metal-desc", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv"
    )
    parser.add_argument("--arms", type=str, default="", help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N ligands only")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_code_levers.json"
    )
    args = parser.parse_args(argv)

    for p in (args.long, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    long = pd.read_parquet(args.long)
    descriptors = pd.read_csv(args.metal_desc)

    # Production target convention: exact measurements only (what MSRS-2 fits).
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy().reset_index(drop=True)

    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    groups = sub["ligand_main_fragment"].to_numpy()
    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    lig, met, cell = base_blocks(sub, descriptors, cell_vocab)
    rdkit = rdkit_block(sub["ligand_main_fragment"].tolist())
    cat_modes = {
        "top40": native_codes(sub, cell_vocab, "top40"),
        "full": native_codes(sub, cell_vocab, "full"),
    }
    fold_id = folds_for(groups)
    print(
        f"[levers] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()} "
        f"rdkit_nan_rows={int(np.isnan(rdkit).any(axis=1).sum())} "
        f"cat_levels top40={int(cat_modes['top40'].max()) + 1} "
        f"full={int(cat_modes['full'].max()) + 1}"
    )

    global_pred, cell_pred, baseline_meta = fold_baseline_scores(
        sub["metal"].to_numpy(), sub["cell_line_key"].to_numpy(), y, fold_id
    )
    preds: Dict[str, np.ndarray] = {"global_best": global_pred, "cell_best": cell_pred}

    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
    arm_results: Dict[str, object] = {}
    for name, (use_rdkit, rank, constraints, cat_mode) in ARMS.items():
        if wanted and name not in wanted:
            continue
        codes = None if cat_mode == "none" else cat_modes[cat_mode]
        t0 = time.time()
        pred = fit_arm(
            lig, met, cell,
            rdkit if use_rdkit else None,
            rank, constraints, y, fold_id,
            cat_codes=codes,
        )
        preds[name] = pred
        entry = {
            **score(y, pred),
            "paired_vs_additive": None,
            "elapsed_seconds": round(time.time() - t0, 1),
            "design": {
                "rdkit2d": use_rdkit,
                "interaction_rank": rank,
                "interaction_constraints": constraints,
                "categorical_mode": cat_mode,
                "n_features": int(
                    lig.shape[1]
                    + (RDKIT_DIM if use_rdkit else 0)
                    + (codes.shape[1] if codes is not None else met.shape[1] + cell.shape[1])
                    + len(METALS) * rank
                ),
            },
        }
        arm_results[name] = entry
        print(f"  {name:22s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f} ({entry['elapsed_seconds']}s)")

    # AUD-D1: train on every row, evaluate on the exact rows only.
    d1 = None
    if not wanted or "d1_tobit" in wanted:
        allrows = long[
            (long["endpoint"] == ENDPOINT)
            & long["value_log10"].notna()
            & long["ligand_main_fragment"].notna()
            & long["ligand_canonical"].notna()
        ].copy().reset_index(drop=True)
        if args.limit:
            allrows = allrows[allrows["ligand_main_fragment"].isin(set(groups))].reset_index(drop=True)
        censored = (allrows["censoring"].astype(str) != "none").to_numpy()
        y_all = allrows["value_log10"].to_numpy(dtype=float)
        cell_vocab_all = allrows["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
        lig_a, met_a, cell_a = base_blocks(allrows, descriptors, cell_vocab_all)
        fold_a = folds_for(allrows["ligand_main_fragment"].to_numpy())
        t0 = time.time()
        pred_all = fit_arm(
            lig_a, met_a, cell_a, None, 0, False, y_all, fold_a, censored=censored
        )
        exact = ~censored
        entry = {
            **score(y_all[exact], pred_all[exact]),
            "n_train_rows": int(len(allrows)),
            "n_censored_rows": int(censored.sum()),
            "n_eval_rows": int(exact.sum()),
            "share_pred_at_or_above_bound_censored": round(
                float((pred_all[censored] >= y_all[censored]).mean()), 4
            ),
            "share_pred_at_or_above_bound_exact": round(
                float((pred_all[exact] >= y_all[exact]).mean()), 4
            ),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        d1 = entry
        print(f"  d1_tobit               R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
              f"P(pred>=bound | censored)={entry['share_pred_at_or_above_bound_censored']:.3f}")

    decision = decision_report(sub, y, fold_id, preds)

    # Pair each arm against the production additive surface on per-ligand MAE.
    from molmetal.scripts.msrs_surface_model import paired_ligand_test

    base_err = np.abs(y - preds["additive"]) if "additive" in preds else None
    if base_err is not None:
        for name, entry in arm_results.items():
            if name == "additive":
                continue
            entry["paired_vs_additive"] = paired_ligand_test(sub, base_err, np.abs(y - preds[name]))

    out = {
        "artifact": "msrs_code_levers",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "metal_descriptors": str(args.metal_desc),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "endpoint": ENDPOINT,
            "target": "value_log10 (log10 IC50, uM), exact rows only",
            "cv": f"GroupKFold({N_FOLDS}) grouped by ligand_main_fragment",
            "lightgbm": LGB_PARAMS,
            "sigma_censored": SIGMA,
            "cohort_keys": COHORT_KEYS,
            "seed": SEED,
        },
        "n_rows": int(len(sub)),
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "baseline_meta": baseline_meta,
        "arms": arm_results,
        "d1_tobit": d1,
        "decision": decision,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
