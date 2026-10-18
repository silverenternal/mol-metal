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
    # AUD-D4: the same design with LightGBM's minimum per-category count relaxed.
    # At the default 100 a cell line needs 100 rows before it can be split on,
    # which silently re-creates the top-40 cap the "full" mode removed.
    "cat_full_mdpg10": (False, 0, False, "full"),
}

#: Arm name -> LightGBM parameter overrides applied on top of ``LGB_PARAMS``.
PARAM_OVERRIDES: Dict[str, Dict[str, object]] = {
    "cat_full_mdpg10": {"min_data_per_group": 10, "cat_smooth": 5.0},
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
    """Integer codes for metal + cell line, for LightGBM's categorical splits.

    The metal vocabulary is ``METALS`` plus any extra metal in ``df`` (e.g. the
    auxiliary ``Pt`` arm), appended in sorted order so the codes are stable.
    """
    extras = sorted(set(df["metal"].astype(str)) - set(METALS))
    metals = tuple(METALS) + tuple(extras)
    metal_index = {m: i for i, m in enumerate(metals)}
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
    sample_weight: Optional[np.ndarray] = None,
    params_override: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """OOF predictions for one arm.  SVD is fitted inside each training fold.

    ``cat_codes`` switches metal and cell line from one-hot to LightGBM's native
    categorical splits; the codes are appended after the ligand block.

    Rows with ``fold_id < 0`` are auxiliary: they train in every fold and are
    never evaluated, so their OOF slot stays NaN.
    """
    from lightgbm import LGBMRegressor, train as lgb_train
    import lightgbm as lgb

    oof = np.full(len(y), np.nan)
    params = dict(LGB_PARAMS)
    if params_override:
        params.update(params_override)
    if constraints and rank == 0:
        n_lig, n_met, n_cell = lig.shape[1], met.shape[1], cell.shape[1]
        n_extra = 0 if rdkit is None else rdkit.shape[1]
        params["interaction_constraints"] = [
            list(range(0, n_lig + n_extra)),
            list(range(n_lig + n_extra, n_lig + n_extra + n_met)),
            list(range(n_lig + n_extra + n_met, n_lig + n_extra + n_met + n_cell)),
        ]

    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
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
        w_tr = None if sample_weight is None else sample_weight[tr]

        if censored is None:
            model = LGBMRegressor(**params)
            fit_kwargs = {} if w_tr is None else {"sample_weight": w_tr}
            if cat_idx:
                model.fit(X_tr, y[tr], categorical_feature=cat_idx, **fit_kwargs)
            else:
                model.fit(X_tr, y[tr], **fit_kwargs)
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
                    weight=w_tr,
                    init_score=np.full(int(tr.sum()), init),
                    categorical_feature=cat_idx if cat_idx else "auto",
                ),
                num_boost_round=n_rounds,
            )
            oof[te] = booster.predict(X_te) + init
    assert not np.isnan(oof[fold_id >= 0]).any()
    return oof


# ---------------------------------------------------------------------------
# Two-part censoring (AUD-D1b)
# ---------------------------------------------------------------------------
#: The classifier shares the regressor's structure but is shallower: censoring
#: is driven by a handful of features (metal, cell line, dose range), not by
#: ligand substructure.
LGB_CLF_PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.3,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def censoring_classifier(
    lig: sparse.csr_matrix,
    met: sparse.csr_matrix,
    cell: sparse.csr_matrix,
    censored: np.ndarray,
    fold_id: np.ndarray,
    cat_codes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Out-of-fold ``P(censored | x)``, the first half of a two-part model.

    Cross-fitted: each row is scored by a classifier that did not train on its
    fold.  The label is ``censored``, which is observed at prediction time, so
    using these probabilities as weights later is a nuisance-model cross-fit
    (double-ML style), not leakage of the regression target.
    """
    from lightgbm import LGBMClassifier

    prob = np.full(len(censored), np.nan)
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        tr = fold_id != k
        te = fold_id == k
        if cat_codes is not None:
            X_tr = sparse.hstack(
                [lig[tr], sparse.csr_matrix(cat_codes[tr].astype(np.float32))], format="csr"
            )
            X_te = sparse.hstack(
                [lig[te], sparse.csr_matrix(cat_codes[te].astype(np.float32))], format="csr"
            )
            cat_idx = list(range(lig.shape[1], lig.shape[1] + cat_codes.shape[1]))
        else:
            X_tr = sparse.hstack([lig[tr], met[tr], cell[tr]], format="csr")
            X_te = sparse.hstack([lig[te], met[te], cell[te]], format="csr")
            cat_idx = []
        clf = LGBMClassifier(**LGB_CLF_PARAMS)
        if cat_idx:
            clf.fit(X_tr, censored[tr], categorical_feature=cat_idx)
        else:
            clf.fit(X_tr, censored[tr])
        prob[te] = clf.predict_proba(X_te)[:, 1]
    assert not np.isnan(prob[fold_id >= 0]).any()
    return prob


def truncated_mean(mu: np.ndarray, bound: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """``E[y | y > bound]`` under a normal with mean ``mu`` and scale ``sigma``.

    ``mu + sigma * phi(z) / S(z)`` with ``z = (bound - mu) / sigma``.  The
    inverse Mills ratio is always above ``z``, so the result is always above the
    bound -- which is the property a censored row requires.

    The ratio is evaluated in log space.  Clipping ``S(z)`` at a small positive
    number looks harmless but destroys the identity once ``z`` is more than a
    few sigma: at ``z = 8`` the true ratio is 8.17 and a ``1e-12`` clip gives
    0.005, so the "conditional mean" would fall below the bound it conditions on.
    """
    from scipy.special import log_ndtr

    z = (bound - mu) / sigma
    log_phi = -0.5 * z ** 2 - 0.5 * np.log(2.0 * np.pi)
    log_sf = log_ndtr(-z)
    return mu + sigma * np.exp(log_phi - log_sf)


# ---------------------------------------------------------------------------
# Auxiliary labels from a different metal (AUD-M3b)
# ---------------------------------------------------------------------------
PT_AUX_DOI = "PLATINAI_4CELL"


def platinum_aux_frame(path: Path, cell_vocabulary: Sequence[str]) -> pd.DataFrame:
    """Melt ``platinai_4cell`` into MSRS-shaped rows with ``metal = "Pt"``.

    The file carries four IC50 columns per Pt complex.  Only the cell lines that
    also exist in the MSRS vocabulary are kept: an auxiliary row for a cell line
    the surface cannot place teaches a metal offset for a context it will never
    be asked about.  IC50 is nM, so ``log10(uM) = log10(nM) - 3``.
    """
    from molmetal.scripts.msrs_build_endpoints import main_fragment

    frame = pd.read_csv(path, low_memory=False)
    cell_cols = {
        "ic50_mcf7_nm": "MCF7",
        "ic50_a549_nm": "A549",
        "ic50_a2780_nm": "A2780",
        "ic50_a2780cis_nm": "A2780CIS",
    }
    keep = [c for c in cell_cols if c in frame.columns]
    long = frame.melt(
        id_vars=["smiles"], value_vars=keep, var_name="cell_col", value_name="ic50_nm"
    ).dropna(subset=["ic50_nm"])
    long["cell_line_key"] = long["cell_col"].map(cell_cols)
    long = long[long["cell_line_key"].isin(set(cell_vocabulary))].copy()
    long["ligand_main_fragment"] = long["smiles"].map(main_fragment)
    long = long[long["ligand_main_fragment"].notna()].copy()
    return pd.DataFrame(
        {
            "ligand_main_fragment": long["ligand_main_fragment"].to_numpy(),
            "cell_line_key": long["cell_line_key"].to_numpy(),
            "metal": "Pt",
            "doi": PT_AUX_DOI,
            "time_h": np.nan,
            "value_log10": np.log10(long["ic50_nm"].to_numpy(dtype=float)) - 3.0,
        }
    )


# ---------------------------------------------------------------------------
# Nested early stopping (AUD-T1)
# ---------------------------------------------------------------------------
def fit_arm_nested(
    lig: sparse.csr_matrix,
    cat_codes: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    groups: np.ndarray,
    max_rounds: int = 1500,
    patience: int = 50,
    inner_folds: int = 3,
) -> Tuple[np.ndarray, List[int]]:
    """OOF predictions with the tree count chosen inside each training fold.

    ``n_estimators=500`` is a constant nobody tuned.  Picking it on the outer
    test fold would leak; picking it on an inner split of the training fold is
    the standard fix.  The median inner best-iteration is used rather than the
    mean so one pathological inner fold cannot inflate the refit.
    """
    from lightgbm import LGBMRegressor, early_stopping

    oof = np.full(len(y), np.nan)
    best_iters: List[int] = []
    cat_idx = list(range(lig.shape[1], lig.shape[1] + cat_codes.shape[1]))

    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = fold_id == k
        X_tr = sparse.hstack(
            [lig[tr], sparse.csr_matrix(cat_codes[tr].astype(np.float32))], format="csr"
        )
        X_te = sparse.hstack(
            [lig[te], sparse.csr_matrix(cat_codes[te].astype(np.float32))], format="csr"
        )
        inner_iters: List[int] = []
        for itr, iva in GroupKFold(n_splits=inner_folds).split(
            X_tr, y[tr], groups[tr]
        ):
            model = LGBMRegressor(**{**LGB_PARAMS, "n_estimators": max_rounds})
            model.fit(
                X_tr[itr],
                y[tr][itr],
                categorical_feature=cat_idx,
                # ``eval_X``/``eval_y`` reject a list of CSR matrices, so the
                # deprecated tuple form stays until LightGBM accepts sparse here.
                eval_set=[(X_tr[iva], y[tr][iva])],
                callbacks=[early_stopping(patience, verbose=False)],
            )
            inner_iters.append(int(model.best_iteration_ or max_rounds))
        chosen = max(1, int(np.median(inner_iters)))
        best_iters.append(chosen)
        final = LGBMRegressor(**{**LGB_PARAMS, "n_estimators": chosen})
        final.fit(X_tr, y[tr], categorical_feature=cat_idx)
        oof[te] = final.predict(X_te)
    assert not np.isnan(oof).any()
    return oof, best_iters


# ---------------------------------------------------------------------------
# Paper-level offset (AUD-D2)
# ---------------------------------------------------------------------------
def doi_offset_calibration(
    oof: np.ndarray,
    y: np.ndarray,
    doi: np.ndarray,
    fold_id: np.ndarray,
    shrink: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Add a fold-safe per-paper offset to the predictions.

    The audit measured +0.1109 in-sample R2 from a ``doi`` feature.  A DOI is
    useless for a compound nobody has assayed, so the honest version is: learn
    the paper's residual level from the training rows of that paper and apply it
    to its test rows, shrinking by ``n / (n + shrink)`` so a two-row paper
    cannot move its own predictions by a log unit.

    Residuals are taken from the OOF predictions, not from an in-sample fit --
    in-sample residuals are far too small and would understate the offset.
    """
    out = oof.copy()
    applied = np.zeros(len(y), dtype=bool)
    per_fold: Dict[str, object] = {}
    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = fold_id == k
        resid = pd.DataFrame({"doi": doi[tr], "r": (y - oof)[tr]})
        stats_ = resid.groupby("doi")["r"].agg(["mean", "size"])
        adj = stats_["mean"] * stats_["size"] / (stats_["size"] + shrink)
        mapped = pd.Series(doi[te]).map(adj)
        hit = mapped.notna().to_numpy()
        out[te] = oof[te] + np.where(hit, mapped.fillna(0.0).to_numpy(), 0.0)
        applied[te] = hit
        per_fold[str(int(k))] = {
            "n_dois_in_train": int(len(adj)),
            "share_test_rows_with_a_known_doi": round(float(hit.mean()), 4),
        }
    meta = {
        "shrink": shrink,
        "share_rows_offset_applied": round(float(applied.mean()), 4),
        "per_fold": per_fold,
    }
    return out, meta


# ---------------------------------------------------------------------------
# Duplicate measurements (AUD-D3)
# ---------------------------------------------------------------------------
def duplicate_weights(
    frame: pd.DataFrame, keys: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """``1 / n`` for each measured cell, so a repeat counts once.

    The unit is ``cohort x metal``: a cohort (ligand, cell line, paper, time) is
    compared across metals, so the repeated measurement of one ligand-metal pair
    is what deserves down-weighting.

    Missing key columns raise.  pandas returns an all-ones group size when a key
    is absent instead of failing, which would silently turn this lever into a
    no-op.
    """
    cols = list(COHORT_KEYS) + ["metal"] if keys is None else list(keys)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"duplicate_weights: missing key columns {missing}")
    counts = (
        frame.groupby(cols, dropna=False)["metal"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    return 1.0 / counts, counts


# ---------------------------------------------------------------------------
# Per-metal models (AUD-M2)
# ---------------------------------------------------------------------------
def per_metal_fit(
    lig: sparse.csr_matrix,
    cat_codes: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    metals: np.ndarray,
) -> np.ndarray:
    """One model per metal, same folds, cell line still categorical.

    Pooling assumes a shared ligand response; five separate models assume none.
    Under ligand-grouped CV every metal has rows in every fold, so the two
    designs are comparable row for row.
    """
    from lightgbm import LGBMRegressor

    oof = np.full(len(y), np.nan)
    for metal in sorted(set(metals.tolist())):
        positions = np.flatnonzero(metals == metal)
        # The metal column is constant inside the subset, so only the cell code
        # carries information; keeping it avoids a useless constant feature.
        codes_m = cat_codes[positions][:, 1:2]
        lig_m = lig[positions]
        y_m, fold_m = y[positions], fold_id[positions]
        cat_idx = list(range(lig_m.shape[1], lig_m.shape[1] + codes_m.shape[1]))
        for k in sorted(set(fold_m.tolist())):
            tr = fold_m != k
            te = fold_m == k
            X_tr = sparse.hstack(
                [lig_m[tr], sparse.csr_matrix(codes_m[tr].astype(np.float32))], format="csr"
            )
            X_te = sparse.hstack(
                [lig_m[te], sparse.csr_matrix(codes_m[te].astype(np.float32))], format="csr"
            )
            model = LGBMRegressor(**LGB_PARAMS)
            model.fit(X_tr, y_m[tr], categorical_feature=cat_idx)
            oof[positions[te]] = model.predict(X_te)
    assert not np.isnan(oof).any()
    return oof


def nested_doi_offset(
    lig: sparse.csr_matrix,
    cat_codes: np.ndarray,
    y: np.ndarray,
    fold_id: np.ndarray,
    groups: np.ndarray,
    doi: np.ndarray,
    inner_folds: int = 2,
    shrink: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """DOI offset with the residuals produced by an inner cross-fit.

    ``doi_offset_calibration`` takes residuals from the outer OOF predictions.
    A training row's OOF prediction comes from a model trained on every fold
    *except its own* -- which includes the current test fold.  The offset then
    carries a little of the test fold's labels.  Here the residuals come from a
    cross-fit that never touches the test fold, so the gain is measured without
    that path.
    """
    from lightgbm import LGBMRegressor

    oof = np.full(len(y), np.nan)
    cat_idx = list(range(lig.shape[1], lig.shape[1] + cat_codes.shape[1]))
    n_hit = 0
    per_fold: Dict[str, object] = {}

    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = fold_id == k
        X_tr = sparse.hstack(
            [lig[tr], sparse.csr_matrix(cat_codes[tr].astype(np.float32))], format="csr"
        )
        X_te = sparse.hstack(
            [lig[te], sparse.csr_matrix(cat_codes[te].astype(np.float32))], format="csr"
        )
        y_tr, g_tr = y[tr], groups[tr]
        resid = np.full(int(tr.sum()), np.nan)
        for itr, iva in GroupKFold(n_splits=inner_folds).split(X_tr, y_tr, g_tr):
            inner = LGBMRegressor(**LGB_PARAMS)
            inner.fit(X_tr[itr], y_tr[itr], categorical_feature=cat_idx)
            resid[iva] = y_tr[iva] - inner.predict(X_tr[iva])
        assert not np.isnan(resid).any()

        outer = LGBMRegressor(**LGB_PARAMS)
        outer.fit(X_tr, y_tr, categorical_feature=cat_idx)
        pred_te = outer.predict(X_te)

        stats_ = pd.DataFrame({"doi": doi[tr], "r": resid}).groupby("doi")["r"].agg(
            ["mean", "size"]
        )
        adj = stats_["mean"] * stats_["size"] / (stats_["size"] + shrink)
        mapped = pd.Series(doi[te]).map(adj)
        hit = mapped.notna().to_numpy()
        oof[te] = pred_te + np.where(hit, mapped.fillna(0.0).to_numpy(), 0.0)
        n_hit += int(hit.sum())
        per_fold[str(int(k))] = {
            "n_dois_in_train": int(len(adj)),
            "share_test_rows_with_a_known_doi": round(float(hit.mean()), 4),
        }

    return oof, {
        "shrink": shrink,
        "inner_folds": inner_folds,
        "share_rows_offset_applied": round(n_hit / len(y), 4),
        "per_fold": per_fold,
    }


def rare_cell_analysis(
    y: np.ndarray, pred: np.ndarray, cell_key: np.ndarray, threshold: int = 100
) -> Dict[str, object]:
    """Per-cell-line MAE, split into cells above and below ``threshold`` rows.

    The "full" categorical mode gives LightGBM every level, but a level with 30
    rows still cannot be split on without overfitting, so the question is
    whether rare cells are actually worse than common ones.
    """
    frame = pd.DataFrame(
        {"cell": np.asarray(cell_key), "err": np.abs(np.asarray(y) - np.asarray(pred))}
    )
    per_cell = frame.groupby("cell")["err"].agg(["mean", "size"])
    rare = per_cell[per_cell["size"] < threshold]
    common = per_cell[per_cell["size"] >= threshold]
    n_rare_rows = int(rare["size"].sum())
    return {
        "n_cells": int(len(per_cell)),
        "n_rare_cells": int(len(rare)),
        "n_common_cells": int(len(common)),
        "share_rows_in_rare_cells": round(float(n_rare_rows / len(frame)), 4),
        "mae_rare_cells": round(float(rare["mean"].mean()), 4) if len(rare) else None,
        "mae_common_cells": round(float(common["mean"].mean()), 4) if len(common) else None,
        "mae_gap_rare_minus_common": (
            round(float(rare["mean"].mean() - common["mean"].mean()), 4)
            if len(rare) and len(common)
            else None
        ),
    }


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
    parser.add_argument(
        "--aux-pt",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv",
        help="Pt complexes used as an auxiliary sixth metal (AUD-M3b)",
    )
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
            params_override=PARAM_OVERRIDES.get(name),
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
                "params_override": PARAM_OVERRIDES.get(name, {}),
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

    # AUD-D1b: two-part censoring.  The Tobit arm above failed because a single
    # fixed-scale likelihood lets censored rows dominate.  Two-part separates the
    # two questions: is this row censored at all (classifier), and where does an
    # uncensored row sit (regressor).  The regressor never sees a censored row,
    # so the exact-row R2 is the baseline by construction -- the deliverables are
    # the classifier's quality and the IPW-corrected fit.
    d1b = None
    if not wanted or wanted & {"d1b_twopart", "d1b_ipw"}:
        allrows = long[
            (long["endpoint"] == ENDPOINT)
            & long["value_log10"].notna()
            & long["ligand_main_fragment"].notna()
            & long["ligand_canonical"].notna()
        ].copy().reset_index(drop=True)
        if args.limit:
            allrows = allrows[allrows["ligand_main_fragment"].isin(set(groups))].reset_index(drop=True)
        censored = (allrows["censoring"].astype(str) != "none").to_numpy()
        exact = ~censored
        y_all = allrows["value_log10"].to_numpy(dtype=float)
        cell_vocab_all = allrows["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
        lig_a, met_a, cell_a = base_blocks(allrows, descriptors, cell_vocab_all)
        fold_a = folds_for(allrows["ligand_main_fragment"].to_numpy())
        codes_all = native_codes(allrows, cell_vocab_all, "full")

        t0 = time.time()
        p_censored = censoring_classifier(
            lig_a, met_a, cell_a, censored.astype(int), fold_a, cat_codes=codes_all
        )
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(censored.astype(int), p_censored))
        brier = float(np.mean((p_censored - censored.astype(float)) ** 2))
        recall = float((p_censored[censored] >= 0.5).mean()) if censored.any() else None
        specificity = float((p_censored[exact] < 0.5).mean())
        # A 0.5 threshold is arbitrary for an 18.7%-positive classifier: with
        # 300 shallow trees the probabilities rarely cross 0.5, so recall@0.5
        # measures calibration, not ranking.  Also report the operating point
        # that flags exactly as many rows as are actually censored.
        base_rate = float(censored.mean())
        thr = float(np.quantile(p_censored, 1.0 - base_rate))
        flagged = p_censored >= thr
        recall_matched = float((flagged[censored]).mean())
        precision_matched = float(censored[flagged].mean()) if flagged.any() else None

        # Regressor on exact rows only, same folds.  Physically dropping the
        # censored rows keeps LightGBM's min_child_samples honest; zero weights
        # would still count those rows toward the leaf-size constraint.
        lig_e, met_e, cell_e = lig_a[exact], met_a[exact], cell_a[exact]
        y_e, fold_e, codes_e = y_all[exact], fold_a[exact], codes_all[exact]
        pred_exact = fit_arm(
            lig_e, met_e, cell_e, None, 0, False, y_e, fold_e, cat_codes=codes_e
        )

        # Inverse-censoring-probability weights: if censoring is predictable from
        # x, the exact rows are a non-random sample and the metal coefficients
        # are biased toward the potent side.
        w_ipw = 1.0 / np.clip(1.0 - p_censored[exact], 0.05, 1.0)
        pred_ipw = fit_arm(
            lig_e, met_e, cell_e, None, 0, False, y_e, fold_e,
            cat_codes=codes_e, sample_weight=w_ipw,
        )

        bound = y_all[censored]
        # Score the censored rows with a zero-weighted fit.  LightGBM counts
        # zero-weight rows toward min_child_samples, so this is not bit-identical
        # to dropping them; the exact-row R2 of this fit is reported alongside
        # the subsetted one so the approximation is visible rather than assumed.
        pred_all_rows = fit_arm(
            lig_a, met_a, cell_a, None, 0, False, y_all, fold_a, cat_codes=codes_all,
            sample_weight=exact.astype(float),
        )
        mu_censored = pred_all_rows[censored]
        pred_censored = truncated_mean(mu_censored, bound)

        base_r2 = score(y_e, pred_exact)["r2"]
        entry = {
            "design": "classifier P(censored|x) + regressor on exact rows",
            "n_train_rows": int(len(allrows)),
            "n_censored_rows": int(censored.sum()),
            "n_exact_rows": int(exact.sum()),
            "classifier_auc": round(auc, 4),
            "classifier_brier": round(brier, 4),
            "classifier_recall_at_0_5": None if recall is None else round(recall, 4),
            "classifier_specificity_at_0_5": round(specificity, 4),
            "classifier_threshold_base_rate": round(thr, 4),
            "classifier_recall_at_base_rate": round(recall_matched, 4),
            "classifier_precision_at_base_rate": (
                None if precision_matched is None else round(precision_matched, 4)
            ),
            "mean_p_censored_on_censored": round(float(p_censored[censored].mean()), 4),
            "mean_p_censored_on_exact": round(float(p_censored[exact].mean()), 4),
            "r2_exact_rows_regressor": base_r2,
            "r2_exact_rows_ipw": score(y_e, pred_ipw)["r2"],
            "mae_exact_rows_regressor": score(y_e, pred_exact)["mae"],
            "mae_exact_rows_ipw": score(y_e, pred_ipw)["mae"],
            "ipw_weight_p50": round(float(np.median(w_ipw)), 3),
            "ipw_weight_p95": round(float(np.quantile(w_ipw, 0.95)), 3),
            "ipw_weight_max": round(float(w_ipw.max()), 3),
            # The truncated mean is above the bound by construction, so this is
            # a property of the estimator, not evidence that the model works.
            # The informative number is ``share_mu_at_or_above_bound_censored``.
            "share_truncated_mean_at_or_above_bound_censored": round(
                float((pred_censored >= bound).mean()), 4
            ),
            "truncated_mean_is_above_bound_by_construction": True,
            "share_mu_at_or_above_bound_censored": round(
                float((mu_censored >= bound).mean()), 4
            ),
            "mean_overshoot_censored": round(float((pred_censored - bound).mean()), 4),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        d1b = entry
        print(
            f"  d1b_twopart           AUC={entry['classifier_auc']:.3f} "
            f"recall={entry['classifier_recall_at_0_5']:.3f} "
            f"R2_exact={entry['r2_exact_rows_regressor']:.4f} "
            f"R2_ipw={entry['r2_exact_rows_ipw']:.4f}"
        )

    # AUD-M3b: can a sixth metal's labels teach the ligand main effect?
    # platinai_4cell carries 4,078 Pt complexes with no MSRS overlap, measured on
    # MCF7 / A549 / A2780 -- three cell lines the surface already knows.  The
    # additive surface says the ligand term is metal-independent, so these rows
    # should help; the interaction story says they should not.
    pt_aux = None
    if args.aux_pt.exists() and (not wanted or "pt_aux" in wanted):
        aux = platinum_aux_frame(args.aux_pt, cell_vocab)
        cols = ["ligand_main_fragment", "cell_line_key", "metal", "doi", "time_h", "value_log10"]
        comb = pd.concat([sub[cols], aux[cols]], ignore_index=True)
        y_c = comb["value_log10"].to_numpy(dtype=float)
        fold_c = np.concatenate([fold_id, np.full(len(aux), -1, dtype=int)])
        t0 = time.time()
        lig_c = morgan_matrix(comb["ligand_main_fragment"].tolist())
        codes_c = native_codes(comb, cell_vocab, "full")
        dummy = sparse.csr_matrix((len(comb), 1))
        pred_c = fit_arm(
            lig_c, dummy, dummy, None, 0, False, y_c, fold_c, cat_codes=codes_c
        )
        n_real = len(sub)
        entry = {
            "n_aux_rows": int(len(aux)),
            "n_aux_ligands": int(aux["ligand_main_fragment"].nunique()),
            "aux_metals": sorted(aux["metal"].unique().tolist()),
            "aux_cells": sorted(aux["cell_line_key"].unique().tolist()),
            "aux_target_median": round(float(aux["value_log10"].median()), 3),
            "msrs_target_median": round(float(np.median(y)), 3),
            "r2_with_aux": score(y, pred_c[:n_real])["r2"],
            "mae_with_aux": score(y, pred_c[:n_real])["mae"],
            "delta_r2_vs_cat_cell_full": None,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        if "cat_cell_full" in arm_results:
            entry["delta_r2_vs_cat_cell_full"] = round(
                entry["r2_with_aux"] - arm_results["cat_cell_full"]["r2"], 4
            )
        pt_aux = entry
        print(
            f"  pt_aux                R2={entry['r2_with_aux']:.4f} "
            f"MAE={entry['mae_with_aux']:.4f} aux_rows={entry['n_aux_rows']}"
        )

    # AUD-T1: choose the tree count inside each training fold instead of fixing
    # it at 500.  Reported separately from the arms because the comparison is
    # "same design, better n_estimators", not a new feature block.
    t1 = None
    if not wanted or "t1_nested" in wanted:
        t0 = time.time()
        pred_t1, best_iters = fit_arm_nested(
            lig, cat_modes["full"], y, fold_id, groups
        )
        entry = {
            **score(y, pred_t1),
            "n_estimators_fixed_baseline": LGB_PARAMS["n_estimators"],
            "best_iters_per_fold": best_iters,
            "median_best_iter": int(np.median(best_iters)),
            "delta_r2_vs_cat_cell_full": None,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        if "cat_cell_full" in arm_results:
            entry["delta_r2_vs_cat_cell_full"] = round(
                entry["r2"] - arm_results["cat_cell_full"]["r2"], 4
            )
        t1 = entry
        print(
            f"  t1_nested             R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
            f"iters={best_iters} ({entry['elapsed_seconds']}s)"
        )

    # AUD-D2: paper-level offset, estimated from training rows of the same DOI.
    need_base = {"doi_offset", "doi_offset_nested", "rare_cell"}
    base_pred = preds.get("cat_cell_full")
    if base_pred is None and (not wanted or wanted & need_base):
        base_pred = fit_arm(
            lig, met, cell, None, 0, False, y, fold_id, cat_codes=cat_modes["full"]
        )
    doi_off = None
    if not wanted or "doi_offset" in wanted:
        t0 = time.time()
        pred_doi, meta = doi_offset_calibration(
            base_pred, y, sub["doi"].astype(str).to_numpy(), fold_id
        )
        entry = {
            **score(y, pred_doi),
            "base_design": "cat_cell_full",
            "base_r2": score(y, base_pred)["r2"],
            "delta_r2": round(score(y, pred_doi)["r2"] - score(y, base_pred)["r2"], 4),
            "residual_source": "outer OOF (cross-fold; see doi_offset_nested)",
            **meta,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        doi_off = entry
        print(
            f"  doi_offset            R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
            f"delta={entry['delta_r2']:+.4f} covered={entry['share_rows_offset_applied']:.3f}"
        )

    # The same offset with residuals that never saw the test fold.
    doi_off_nested = None
    if not wanted or "doi_offset_nested" in wanted:
        t0 = time.time()
        pred_dn, meta_n = nested_doi_offset(
            lig, cat_modes["full"], y, fold_id, groups,
            sub["doi"].astype(str).to_numpy(),
        )
        entry = {
            **score(y, pred_dn),
            "base_design": "cat_cell_full",
            "base_r2": score(y, base_pred)["r2"],
            "delta_r2": round(score(y, pred_dn)["r2"] - score(y, base_pred)["r2"], 4),
            "residual_source": "inner cross-fit inside each training fold",
            **meta_n,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        doi_off_nested = entry
        print(
            f"  doi_offset_nested     R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
            f"delta={entry['delta_r2']:+.4f} covered={entry['share_rows_offset_applied']:.3f}"
        )

    # AUD-D4: are rare cell lines actually worse, and does relaxing the
    # per-category minimum fix them?
    rare = None
    if not wanted or "rare_cell" in wanted:
        cells = sub["cell_line_key"].astype(str).to_numpy()
        entry = {"cat_cell_full": rare_cell_analysis(y, base_pred, cells)}
        if "cat_full_mdpg10" in preds:
            entry["cat_full_mdpg10"] = rare_cell_analysis(y, preds["cat_full_mdpg10"], cells)
            entry["delta_mae_rare"] = round(
                entry["cat_full_mdpg10"]["mae_rare_cells"]
                - entry["cat_cell_full"]["mae_rare_cells"],
                4,
            )
        rare = entry
        print(
            f"  rare_cell             gap_rare_minus_common="
            f"{entry['cat_cell_full']['mae_gap_rare_minus_common']:+.4f} "
            f"rare_rows={entry['cat_cell_full']['share_rows_in_rare_cells']:.3f}"
        )

    # AUD-D3: does a repeated measurement deserve more weight than a single one?
    dup = None
    if not wanted or "dup_weight" in wanted:
        w, counts = duplicate_weights(sub)
        t0 = time.time()
        pred_dup = fit_arm(
            lig, met, cell, None, 0, False, y, fold_id,
            cat_codes=cat_modes["full"], sample_weight=w,
        )
        entry = {
            **score(y, pred_dup),
            "n_rows": int(len(sub)),
            "n_rows_in_duplicated_cohorts": int((counts > 1).sum()),
            "share_rows_in_duplicated_cohorts": round(float((counts > 1).mean()), 4),
            "max_cohort_size": int(counts.max()),
            "delta_r2_vs_cat_cell_full": None,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        if "cat_cell_full" in arm_results:
            entry["delta_r2_vs_cat_cell_full"] = round(
                entry["r2"] - arm_results["cat_cell_full"]["r2"], 4
            )
        dup = entry
        print(
            f"  dup_weight            R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
            f"dup_rows={entry['share_rows_in_duplicated_cohorts']:.3f}"
        )

    # AUD-M2: one model per metal instead of one pooled surface.
    per_metal = None
    if not wanted or "per_metal" in wanted:
        t0 = time.time()
        pred_pm = per_metal_fit(
            lig, cat_modes["full"], y, fold_id, sub["metal"].to_numpy()
        )
        by_metal = {
            str(m): score(y[sub["metal"].to_numpy() == m], pred_pm[sub["metal"].to_numpy() == m])
            for m in sorted(sub["metal"].unique())
        }
        entry = {
            **score(y, pred_pm),
            "by_metal": by_metal,
            "delta_r2_vs_cat_cell_full": None,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        if "cat_cell_full" in arm_results:
            entry["delta_r2_vs_cat_cell_full"] = round(
                entry["r2"] - arm_results["cat_cell_full"]["r2"], 4
            )
        per_metal = entry
        print(
            f"  per_metal             R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
            f"delta={entry['delta_r2_vs_cat_cell_full']:+.4f}"
        )

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
        "d1b_twopart": d1b,
        "pt_aux": pt_aux,
        "t1_nested": t1,
        "doi_offset": doi_off,
        "doi_offset_nested": doi_off_nested,
        "rare_cell": rare,
        "dup_weight": dup,
        "per_metal": per_metal,
        "decision": decision,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
