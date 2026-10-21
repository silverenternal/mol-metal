#!/usr/bin/env python3
"""MSRS-2 — the response surface, CPU-first: does ligand x metal interaction pay?

=====================================================================
The question
=====================================================================
MSRS-1 established that the interaction share is large where it can be
measured (Ru-Os 71%, Ru-Ir 65% on the loose key). That is a statement about
variance. This work package asks the operational question:

    given a ligand and a metal, can we predict log10 IC50 well enough that
    the *ranking* of metals is right — and does allowing ligand x metal
    interaction beat a strictly additive surface?

=====================================================================
Why a CPU tree model before the D-MPNN
=====================================================================
The acceptance criterion is qualitative ("interaction significantly beats
additive under ligand-grouped CV"), not "beat the state of the art". A
gradient-boosted tree on ECFP4 can answer it in minutes. If a flexible tree
cannot exploit the interaction, a D-MPNN is unlikely to; if it can, the
D-MPNN arm becomes an absolute-performance question that can wait for GPU.

The comparison is deliberately **within one model class**:

    additive     LightGBM with interaction_constraints = [[ligand], [metal], [cell]]
    interaction  LightGBM, same hyper-parameters, no constraints

Both see the same features, the same folds, the same seeds. The only thing
that changes is which feature crosses the trees are allowed to make. Comparing
Ridge (additive by construction) with GBDT would confound structure with
capacity.

=====================================================================
The estimand-level read
=====================================================================
An additive surface has a sharp consequence: for a matched pair (same ligand,
same cell line, same paper, same duration) the predicted delta between two
metals is

    y_hat(m') - y_hat(m) = g(m') - g(m)

i.e. **the same number for every ligand**. The additive model cannot rank
metals differently for different ligands. So its delta-sign accuracy is a
constant, and the fraction of pairs whose observed sign disagrees with the
metal main effect is exactly the share of decisions the additive model gets
wrong by construction. That fraction is reported directly.

=====================================================================
Data decisions
=====================================================================
- Targets: potency (log10 IC50 dark) and cisplatin_ratio (RI). Phototoxicity
  is excluded as a regression target: MSRS-5 measured negative out-of-fold R2
  for PI, so fitting it here would manufacture a number nobody should use.
- Rows with all measurements censored have no numeric y and are dropped
  (4,813 of 26,219 potency rows). This is a real selection: censored rows are
  mostly inactive compounds, so the surface is fitted on the potent side.
  The censored share is reported with every endpoint.
- CV groups are `ligand_main_fragment`, not the strict complex key. Grouping
  by the strict key would let the same organic ligand leak across folds via a
  different counterion.
- Cell-line vocabulary = top 40 lines by row count + "other", computed
  unsupervised from the endpoint's rows (no labels involved).

=====================================================================
Usage
=====================================================================
::

    .venv/bin/python molmetal/scripts/msrs_surface_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir", "Rh", "Os", "Re")
ENDPOINTS = ("potency", "cisplatin_ratio")
FP_BITS = 2048
FP_RADIUS = 2
TOP_CELL_LINES = 40
N_FOLDS = 5
SEED = 20261018
NOISE_FLOOR = 0.2091  # MSRS-1 pooled within-group sigma on potency, log10
SIGN_THRESHOLDS = (0.0, 0.1, NOISE_FLOOR, 0.5)

LGB_PARAMS = dict(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
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


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def morgan_matrix(smiles: Sequence[str], n_bits: int = FP_BITS, radius: int = FP_RADIUS) -> sparse.csr_matrix:
    """ECFP4 bits for each SMILES; an unparseable SMILES yields an all-zero row."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows, cols = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        for bit in fp.GetOnBits():
            rows.append(i)
            cols.append(int(bit))
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(smiles), n_bits), dtype=np.float32)


def metal_matrix(
    metals: Sequence[str],
    descriptors: pd.DataFrame,
    columns: Sequence[str] = ("ie1_kj", "ie2_kj", "pauling_en"),
) -> np.ndarray:
    """One-hot metal + the tabulated descriptor block.

    ``columns`` defaults to IE1 / IE2 / Pauling EN, which is what every artifact
    produced before 2026-10-21 used.  UP-4 passes an eRAC block instead
    (``z_eff_slater`` / ``z_eff_over_rcov`` / ``d_count_M3``) to test whether the
    nuclear charge is the wrong coordinate for same-group metals.
    """
    desc = descriptors.set_index("metal_symbol")
    onehot = np.zeros((len(metals), len(METALS)), dtype=np.float32)
    for i, m in enumerate(metals):
        onehot[i, METALS.index(m)] = 1.0
    extra = np.array(
        [
            [desc.at[m, c] for c in columns]
            if m in desc.index
            else [np.nan] * len(columns)
            for m in metals
        ],
        dtype=np.float32,
    )
    return np.hstack([onehot, extra])


def cell_matrix(cells: Sequence[str], vocabulary: Sequence[str]) -> sparse.csr_matrix:
    index = {c: i for i, c in enumerate(vocabulary)}
    other = len(vocabulary)
    rows = np.arange(len(cells))
    cols = np.array([index.get(c, other) for c in cells], dtype=np.int32)
    return sparse.csr_matrix(
        (np.ones(len(cells), dtype=np.float32), (rows, cols)),
        shape=(len(cells), len(vocabulary) + 1),
        dtype=np.float32,
    )


def build_design(
    df: pd.DataFrame, descriptors: pd.DataFrame, cell_vocabulary: Sequence[str]
) -> Tuple[sparse.csr_matrix, List[str], List[List[int]]]:
    """Return X, feature names, and the three interaction groups."""
    lig = morgan_matrix(df["ligand_main_fragment"].tolist())
    met = sparse.csr_matrix(metal_matrix(df["metal"].tolist(), descriptors))
    cell = cell_matrix(df["cell_line_key"].tolist(), cell_vocabulary)
    X = sparse.hstack([lig, met, cell], format="csr")

    n_lig, n_met, n_cell = lig.shape[1], met.shape[1], cell.shape[1]
    names = (
        [f"ecfp4_{i}" for i in range(n_lig)]
        + [f"metal_onehot_{m}" for m in METALS]
        + ["metal_ie1", "metal_ie2", "metal_en"]
        + [f"cell_{c}" for c in cell_vocabulary]
        + ["cell_other"]
    )
    groups = [
        list(range(0, n_lig)),
        list(range(n_lig, n_lig + n_met)),
        list(range(n_lig + n_met, n_lig + n_met + n_cell)),
    ]
    assert len(names) == X.shape[1]
    return X, names, groups


def fit_predict(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_test: sparse.csr_matrix,
    constraints: Optional[List[List[int]]],
) -> np.ndarray:
    from lightgbm import LGBMRegressor

    params = dict(LGB_PARAMS)
    if constraints is not None:
        params["interaction_constraints"] = constraints
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def oof_predict(
    X: sparse.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    constraints: Optional[List[List[int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.full(len(y), np.nan, dtype=float)
    fold_id = np.full(len(y), -1, dtype=int)
    splitter = GroupKFold(n_splits=N_FOLDS)
    for k, (tr, te) in enumerate(splitter.split(X, y, groups)):
        oof[te] = fit_predict(X[tr], y[tr], X[te], constraints)
        fold_id[te] = k
    assert not np.isnan(oof).any()
    return oof, fold_id


def oof_means(y: np.ndarray, keys: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fold-safe global-mean and per-key-mean baselines (no test row touches its own mean)."""
    global_pred = np.full(len(y), np.nan)
    key_pred = np.full(len(y), np.nan)
    splitter = GroupKFold(n_splits=N_FOLDS)
    for tr, te in splitter.split(np.zeros((len(y), 1)), y, groups):
        gmean = float(np.mean(y[tr]))
        global_pred[te] = gmean
        means = pd.Series(y[tr]).groupby(keys[tr]).mean()
        key_pred[te] = pd.Series(keys[te]).map(means).fillna(gmean).to_numpy()
    assert not np.isnan(global_pred).any() and not np.isnan(key_pred).any()
    return global_pred, key_pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def score(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        "r2": round(float(r2_score(y, pred)), 4),
        "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 4),
    }


def paired_ligand_test(
    df: pd.DataFrame, err_a: np.ndarray, err_b: np.ndarray
) -> Dict[str, object]:
    """Paired one-sided t-test on per-ligand mean absolute error (b better than a)."""
    frame = pd.DataFrame(
        {"lig": df["ligand_main_fragment"].to_numpy(), "a": err_a, "b": err_b}
    )
    per = frame.groupby("lig")[["a", "b"]].mean()
    diff = per["a"] - per["b"]  # positive => b has lower error
    if float(diff.std(ddof=1)) == 0.0:
        t, p_one = 0.0, 1.0
    else:
        t, p_two = stats.ttest_rel(per["a"], per["b"])
        p_one = float(p_two / 2.0) if t > 0 else float(1.0 - p_two / 2.0)
    return {
        "n_ligands": int(len(per)),
        "mean_mae_a": round(float(per["a"].mean()), 4),
        "mean_mae_b": round(float(per["b"].mean()), 4),
        "mean_delta": round(float(diff.mean()), 4),
        "t": round(float(t), 4),
        "p_one_sided": p_one,
        "direction": "b better" if diff.mean() > 0 else "a better",
    }


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def matched_delta_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """All unordered cross-metal pairs sharing (ligand fragment, cell, DOI, duration)."""
    key = ["ligand_main_fragment", "cell_line_key", "doi", "time_h"]
    rank = {m: i for i, m in enumerate(METALS)}
    rows = []
    for _, g in df.groupby(key, dropna=False):
        if g["metal"].nunique() < 2:
            continue
        g = g.reset_index()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                ma, mb = g.at[i, "metal"], g.at[j, "metal"]
                if ma == mb:
                    continue
                # Canonical order so "Ru-Ir" never appears as "Ir-Ru".
                if rank[ma] > rank[mb]:
                    lo, hi, mlo, mhi = j, i, mb, ma
                else:
                    lo, hi, mlo, mhi = i, j, ma, mb
                rows.append(
                    {
                        "idx_a": int(g.at[lo, "index"]),
                        "idx_b": int(g.at[hi, "index"]),
                        "metal_a": mlo,
                        "metal_b": mhi,
                        "cell_line_key": g.at[i, "cell_line_key"],
                        "doi": g.at[i, "doi"],
                    }
                )
    return pd.DataFrame(rows)


def _sign(x: np.ndarray) -> np.ndarray:
    return np.sign(np.round(x, 10))


def delta_sign_report(
    y: np.ndarray,
    pred_add: np.ndarray,
    pred_int: np.ndarray,
    pairs: pd.DataFrame,
    fold_id: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> Dict[str, object]:
    if pairs.empty:
        return {"n_pairs": 0, "status": "NO_MATCHED_PAIRS"}

    ia, ib = pairs["idx_a"].to_numpy(), pairs["idx_b"].to_numpy()
    obs = y[ib] - y[ia]
    d_add = pred_add[ib] - pred_add[ia]
    d_int = pred_int[ib] - pred_int[ia]

    out: Dict[str, object] = {
        "n_pairs": int(len(pairs)),
        "n_dois": int(pairs["doi"].nunique()),
        "by_metal_pair": {},
        "by_threshold": {},
    }

    # Structural check: holding the cell line AND the fold fixed, the additive
    # delta must be a constant. Each fold is a separate model, so the delta is
    # allowed to change across folds; metal x cell is allowed (cell is its own
    # group), so the check conditions on the cell line.
    pairs = pairs.assign(fold_a=fold_id[pairs["idx_a"].to_numpy()])
    spreads = []
    for _, g in pairs.groupby(["metal_a", "metal_b", "cell_line_key", "fold_a"], dropna=False):
        ia_g = g["idx_a"].to_numpy()
        ib_g = g["idx_b"].to_numpy()
        d = pred_add[ib_g] - pred_add[ia_g]
        spreads.append(float(np.max(np.abs(d - d[0]))))
    add_spread = max(spreads) if spreads else 0.0
    out["additive_delta_ligand_independent"] = bool(add_spread < 1e-4)
    out["additive_delta_ligand_independent_given_cell"] = bool(add_spread < 1e-4)
    out["additive_delta_max_within_pair_spread"] = round(add_spread, 8)
    out["additive_delta_sd"] = round(float(np.std(d_add)), 6)
    out["interaction_delta_sd"] = round(float(np.std(d_int)), 6)

    for th in SIGN_THRESHOLDS:
        keep = np.abs(obs) >= th
        if keep.sum() == 0:
            continue
        sa, si, so = _sign(d_add[keep]), _sign(d_int[keep]), _sign(obs[keep])
        ok_a, ok_i = sa == so, si == so
        b = int((ok_i & ~ok_a).sum())
        c = int((ok_a & ~ok_i).sum())
        out["by_threshold"][f"{th:g}"] = {
            "n_pairs": int(keep.sum()),
            "additive_accuracy": round(float(ok_a.mean()), 4),
            "interaction_accuracy": round(float(ok_i.mean()), 4),
            "delta_accuracy": round(float(ok_i.mean() - ok_a.mean()), 4),
            "n_interaction_right_additive_wrong": b,
            "n_additive_right_interaction_wrong": c,
            "mcnemar_p_one_sided": _mcnemar_one_sided(b, c),
        }

    primary = out["by_threshold"].get(f"{NOISE_FLOOR:g}")
    if primary is not None:
        keep = np.abs(obs) >= NOISE_FLOOR
        ok_i = _sign(d_int[keep]) == _sign(obs[keep])
        ok_a = _sign(d_add[keep]) == _sign(obs[keep])
        out["primary_threshold"] = NOISE_FLOOR
        out["bootstrap_ci_interaction_accuracy"] = _clustered_bootstrap(
            ok_i.astype(float), pairs.loc[keep, "doi"].to_numpy(), n_boot, seed
        )
        out["bootstrap_ci_additive_accuracy"] = _clustered_bootstrap(
            ok_a.astype(float), pairs.loc[keep, "doi"].to_numpy(), n_boot, seed
        )
        # Is the interaction model useless, or merely over-confident? Blend the
        # two deltas and scan the weight. alpha=0 is the additive surface.
        curve = {}
        for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
            blended = d_add[keep] + alpha * (d_int[keep] - d_add[keep])
            curve[f"{alpha:g}"] = round(float((_sign(blended) == _sign(obs[keep])).mean()), 4)
        out["shrinkage_curve"] = curve
        out["best_shrinkage_alpha"] = max(curve, key=curve.get)
        out["shrinkage_note"] = (
            "alpha is scanned on the same out-of-fold pairs with one free parameter; "
            "treat the curve as a diagnostic, not a validated procedure."
        )

    for (ma, mb), g in pairs.groupby(["metal_a", "metal_b"]):
        idx = g.index.to_numpy()
        keep = np.abs(obs[idx]) >= NOISE_FLOOR
        if keep.sum() == 0:
            continue
        sel = idx[keep]
        out["by_metal_pair"][f"{ma}-{mb}"] = {
            "n_pairs": int(keep.sum()),
            "additive_accuracy": round(float((_sign(d_add[sel]) == _sign(obs[sel])).mean()), 4),
            "interaction_accuracy": round(float((_sign(d_int[sel]) == _sign(obs[sel])).mean()), 4),
        }
    return out


def _mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar: P(X >= b) with X ~ Bin(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binom.sf(b - 1, n, 0.5))


def _clustered_bootstrap(
    values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int
) -> List[float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    by_cluster = {c: values[clusters == c] for c in uniq}
    stats_out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sample = np.concatenate([by_cluster[c] for c in pick])
        stats_out.append(sample.mean())
    lo, hi = np.percentile(stats_out, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# ---------------------------------------------------------------------------
# One endpoint
# ---------------------------------------------------------------------------
def run_endpoint(
    df: pd.DataFrame,
    endpoint: str,
    descriptors: pd.DataFrame,
    verbose: bool = True,
) -> Dict[str, object]:
    t0 = time.time()
    sub = df[df["endpoint"] == endpoint].copy()
    n_total = len(sub)
    n_censored = int(sub["value_log10"].isna().sum())
    sub = sub[sub["value_log10"].notna() & sub["ligand_main_fragment"].notna()].copy()
    sub = sub.reset_index(drop=True)
    if sub.empty:
        return {"status": "EMPTY"}

    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    X, names, groups = build_design(sub, descriptors, cell_vocab)
    y = sub["value_log10"].to_numpy(dtype=float)
    group_ids = sub["ligand_main_fragment"].to_numpy()

    if verbose:
        print(
            f"[{endpoint}] rows={len(sub)} (dropped {n_censored} censored of {n_total}), "
            f"ligands={sub['ligand_main_fragment'].nunique()}, features={X.shape[1]}"
        )

    preds: Dict[str, np.ndarray] = {}
    preds["global_mean"], preds["per_metal_mean"] = oof_means(
        y, sub["metal"].to_numpy(), group_ids
    )
    n_lig = groups[0][-1] + 1
    preds["ligand_only"] = oof_predict(X[:, :n_lig], y, group_ids, None)[0]
    preds["additive"] = oof_predict(X, y, group_ids, groups)[0]
    preds["interaction"], fold_id = oof_predict(X, y, group_ids, None)

    models = {}
    for name, pred in preds.items():
        entry = score(y, pred)
        entry["mae_by_metal"] = {
            m: round(float(mean_absolute_error(y[sub["metal"].to_numpy() == m], pred[sub["metal"].to_numpy() == m])), 4)
            for m in METALS
            if (sub["metal"].to_numpy() == m).sum() > 0
        }
        entry["mae_by_fold"] = {
            int(k): round(float(mean_absolute_error(y[fold_id == k], pred[fold_id == k])), 4)
            for k in sorted(set(fold_id))
        }
        models[name] = entry

    err = {k: np.abs(y - v) for k, v in preds.items()}
    paired = {
        "interaction_vs_additive": paired_ligand_test(sub, err["additive"], err["interaction"]),
        "interaction_vs_ligand_only": paired_ligand_test(sub, err["ligand_only"], err["interaction"]),
        "additive_vs_per_metal_mean": paired_ligand_test(sub, err["per_metal_mean"], err["additive"]),
    }

    pairs = matched_delta_pairs(sub)
    delta = delta_sign_report(y, preds["additive"], preds["interaction"], pairs, fold_id)

    if verbose:
        print(
            f"  mae: additive={models['additive']['mae']} interaction={models['interaction']['mae']} "
            f"ligand_only={models['ligand_only']['mae']} per_metal={models['per_metal_mean']['mae']}"
        )
        print(
            f"  paired (ligand-level): delta={paired['interaction_vs_additive']['mean_delta']} "
            f"p={paired['interaction_vs_additive']['p_one_sided']:.3g}"
        )
        if delta.get("n_pairs"):
            prim = delta["by_threshold"].get(f"{NOISE_FLOOR:g}", {})
            print(
                f"  delta sign @|d|>={NOISE_FLOOR}: n={prim.get('n_pairs')} "
                f"add={prim.get('additive_accuracy')} int={prim.get('interaction_accuracy')} "
                f"mcnemar_p={prim.get('mcnemar_p_one_sided')}"
            )

    return {
        "status": "OK",
        "n_rows": int(len(sub)),
        "n_dropped_censored": n_censored,
        "censored_share": round(n_censored / n_total, 4) if n_total else None,
        "n_ligands": int(sub["ligand_main_fragment"].nunique()),
        "n_cell_lines": int(sub["cell_line_key"].nunique()),
        "n_dois": int(sub["doi"].nunique()),
        "n_features": int(X.shape[1]),
        "feature_groups": {
            "ligand_ecfp4": len(groups[0]),
            "metal_onehot_plus_descriptors": len(groups[1]),
            "cell_line_onehot": len(groups[2]),
        },
        "models": models,
        "paired_tests": paired,
        "delta_sign": delta,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument(
        "--metal-desc", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_surface_model.json"
    )
    args = parser.parse_args(argv)

    for p in (args.dedup, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    df = pd.read_parquet(args.dedup)
    descriptors = pd.read_csv(args.metal_desc)

    results = {}
    for endpoint in ENDPOINTS:
        results[endpoint] = run_endpoint(df, endpoint, descriptors)

    # Holm across endpoints for the primary structural comparison.
    raw = {
        ep: r["paired_tests"]["interaction_vs_additive"]["p_one_sided"]
        for ep, r in results.items()
        if r.get("status") == "OK"
    }
    adjusted = holm(raw)
    for ep in adjusted:
        results[ep]["paired_tests"]["interaction_vs_additive"]["holm_p_across_endpoints"] = adjusted[ep]

    out = {
        "artifact": "msrs_surface_model",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 response surface (CPU baseline)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "metal_descriptors": str(args.metal_desc),
            "metal_descriptors_sha256": _sha256(args.metal_desc),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "backbone": "LightGBM on ECFP4(2048) + metal one-hot(5) + metal descriptors(3) + cell one-hot(top 40 + other)",
            "additive_definition": "interaction_constraints = [[ligand bits], [metal features], [cell features]]",
            "interaction_definition": "same features, same hyper-parameters, no constraints",
            "cv": f"GroupKFold({N_FOLDS}) grouped by ligand_main_fragment",
            "seed": SEED,
            "lightgbm": LGB_PARAMS,
            "targets": list(ENDPOINTS),
            "excluded_target": "phototoxicity (MSRS-5: out-of-fold R2 < 0)",
            "sign_thresholds": list(SIGN_THRESHOLDS),
            "primary_sign_threshold": NOISE_FLOOR,
        },
        "endpoints": results,
        "caveats": [
            "Censored rows (all measurements right- or left-censored) carry no numeric y and are "
            "dropped. They are disproportionately inactive compounds, so every fit is on the potent "
            "side of the cohort; the dropped share is reported per endpoint.",
            "Matched delta pairs use ligand_main_fragment, i.e. the loose key: the organic ligand is "
            "held fixed but counterions/co-ligands may differ. This is the same estimand as MSRS-1's "
            "loose sensitivity analysis, not the strict single-variable substitution.",
            "ECFP4 has no explicit metal, so the metal enters only through the metal block. That is "
            "the intended metal-aware design, but it means the model cannot represent coordination "
            "geometry.",
            "The additive model's delta is ligand-independent by construction; its sign accuracy is "
            "therefore a constant per metal pair and is reported as the null, not as a fitted result.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
