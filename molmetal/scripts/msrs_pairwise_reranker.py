#!/usr/bin/env python3
"""FIX-M4 — a symmetric pairwise reranker for within-cohort metal choice.

Why this is not a rerun of FIX-E3
=================================
FIX-E3's ``pair_delta`` arm described a pair as ``[X[a], cell_a, metal_a code,
metal_b code]``.  That is asymmetric: metal B contributes only its identity,
so the model can never see that Ru(IV) differs from Pt(II) in oxidation state
or coordination.  It was also trained with 500 boosted trees on ~700 pairs.

FIX-M3 supplies the motive.  Within a cohort only the cohort x metal
interaction changes the ranking; the metal main effect is 3.4% of within-cohort
variance and the current models sit exactly at that level.  The correct
inductive bias for "which metal wins here" is a **difference** representation:
swapping the two metals must negate both the features and the target.

Arms
====
  main_effect    fold-safe metal mean (``global_best``)          — the bar
  pair_identity  d(metal one-hot)                                — main effect only
  pair_state     + d(oxidation state, charge, d-electrons, ...)  — metal chemistry
  pair_full      + ligand-context x metal interactions           — ligand-specific

All pair models are ridge regressions fitted in closed form (no trees): with
~700 training pairs, capacity is the enemy.  Alpha is chosen by an inner
GroupKFold on the training folds only.

Two protocols, both out of sample (``requires_dual_split``):
  canonical_grouped  — folds split on ``ligand_canonical``
  fragment_grouped   — folds split on ``ligand_main_fragment``

Acceptance: on ``canonical_grouped`` the best pair arm's mean within-cohort
regret must beat ``main_effect`` with a DOI-clustered one-sided p < 0.05.
Anything else is logged as negative.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_pairwise_reranker.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from molmetal.features.build import build
from molmetal.features.coordination import coordination_block
from molmetal.features.metal import state_block
from molmetal.features.spec import CellSpec, FeatureSpec, LigandSpec, MetalSpec
from molmetal.scripts.msrs_code_levers import COHORT_KEYS, ENDPOINT, folds_for
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_recommendation import (
    cohort_metrics,
    doi_clustered_test,
    fold_baseline_scores,
)
from molmetal.scripts.msrs_surface_model import _git_commit, _sha256

LIGAND_ONLY = FeatureSpec(
    ligand=LigandSpec(source="ligand_canonical", include=("canonical",)),
    metal=MetalSpec(one_hot=False, columns=()),
    cell=CellSpec(mode="none"),
)
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)
GROUP_COL = {
    SPLIT_FRAGMENT_GROUPED: "ligand_main_fragment",
    SPLIT_CANONICAL_GROUPED: "ligand_canonical",
}
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
N_SVD = 32
SEED = 0


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_blocks(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    bundle = build(frame, LIGAND_ONLY)
    state, state_names = state_block(frame)
    coord, coord_names, coverage = coordination_block(frame, include_geometry=True)
    coord_dense = np.asarray(coord.todense(), dtype=np.float64)
    svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
    ligand_ctx = svd.fit_transform(bundle.X).astype(np.float64)

    metals = sorted(frame["metal"].astype(str).unique())
    onehot = np.zeros((len(frame), len(metals)), dtype=np.float64)
    index = {m: i for i, m in enumerate(metals)}
    for i, m in enumerate(frame["metal"].astype(str)):
        onehot[i, index[m]] = 1.0

    return {
        "onehot": onehot,
        "state": np.asarray(state, dtype=np.float64),
        "coord": coord_dense,
        "ligand": ligand_ctx,
        "_names": np.array(
            [f"metal={m}" for m in metals]
            + list(state_names)
            + list(coord_names)
            + [f"lig_svd{i}" for i in range(N_SVD)],
            dtype=object,
        ),
        "_coverage": coverage,
        "_spec_hash": bundle.spec_hash,
    }


def pair_design(
    blocks: Dict[str, np.ndarray],
    a: int,
    b: int,
    arm: str,
) -> np.ndarray:
    """Antisymmetric pair features for "b relative to a"."""
    d_onehot = blocks["onehot"][b] - blocks["onehot"][a]
    parts = [d_onehot]
    if arm in ("pair_state", "pair_full"):
        parts.append(blocks["state"][b] - blocks["state"][a])
        parts.append(blocks["coord"][b] - blocks["coord"][a])
    if arm == "pair_full":
        # Ligand context enters only through interactions with the metal
        # difference, so the whole design stays antisymmetric.
        ctx = 0.5 * (blocks["ligand"][a] + blocks["ligand"][b])
        parts.append(np.outer(ctx, d_onehot).ravel())
    return np.concatenate(parts)


def pair_matrix(
    blocks: Dict[str, np.ndarray],
    rows: np.ndarray,
    cohorts: np.ndarray,
    metals: np.ndarray,
    arm: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All ordered within-cohort pairs in ``rows``: X, target, cohort, (a, b).

    Cohorts with fewer than two distinct metals are skipped: pairing two
    replicate rows of the same metal would teach the model that identical
    features carry a non-zero delta, which is pure noise.
    """
    feats: List[np.ndarray] = []
    targets: List[float] = []
    groups: List[int] = []
    idx_a: List[int] = []
    idx_b: List[int] = []
    table = pd.DataFrame({"c": cohorts[rows], "i": rows, "m": metals[rows]})
    for cid, g in table.groupby("c", sort=False):
        if g["m"].nunique() < 2:
            continue
        members = g["i"].to_numpy()
        for a, b in permutations(members, 2):
            feats.append(pair_design(blocks, int(a), int(b), arm))
            targets.append(float(blocks["_y"][b] - blocks["_y"][a]))
            groups.append(int(cid))
            idx_a.append(int(a))
            idx_b.append(int(b))
    if not feats:
        raise ValueError("no within-cohort pairs in this fold")
    return (
        np.vstack(feats),
        np.asarray(targets, dtype=np.float64),
        np.asarray(groups, dtype=int),
        np.column_stack([idx_a, idx_b]),
    )


# ---------------------------------------------------------------------------
# Ridge with an inner DOI-grouped alpha search
# ---------------------------------------------------------------------------
def _ridge_fit(
    X: np.ndarray, t: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    t_mean = float(t.mean())
    Z = (X - mu) / sd
    gram = Z.T @ Z + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(gram, Z.T @ (t - t_mean))
    return coef, mu, sd, t_mean


def _ridge_predict(
    X: np.ndarray, coef: np.ndarray, mu: np.ndarray, sd: np.ndarray, t_mean: float
) -> np.ndarray:
    return ((X - mu) / sd) @ coef + t_mean


def choose_alpha(
    X: np.ndarray, t: np.ndarray, groups: np.ndarray
) -> float:
    """Inner GroupKFold(3) over DOI; never sees the outer test fold."""
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return 100.0
    rng = np.random.default_rng(SEED)
    order = rng.permutation(uniq)
    buckets = np.array_split(order, 3)
    best, best_mse = 100.0, np.inf
    for alpha in ALPHAS:
        errs = []
        for bucket in buckets:
            te = np.isin(groups, bucket)
            if te.all() or (~te).sum() < 10:
                continue
            coef, mu, sd, t_mean = _ridge_fit(X[~te], t[~te], alpha)
            pred = _ridge_predict(X[te], coef, mu, sd, t_mean)
            errs.append(float(np.mean((pred - t[te]) ** 2)))
        if errs and np.mean(errs) < best_mse:
            best, best_mse = alpha, float(np.mean(errs))
    return best


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def run_pair_arm(
    blocks: Dict[str, np.ndarray],
    fold_id: np.ndarray,
    cohorts: np.ndarray,
    arm: str,
    multi_metal: np.ndarray,
    metals: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """OOF per-row score = mean predicted delta against the cohort's metals."""
    scores = np.full(len(fold_id), np.nan)
    alphas: List[float] = []
    n_pairs = 0
    for k in sorted(np.unique(fold_id)):
        tr = np.where((fold_id != k) & multi_metal)[0]
        te = np.where((fold_id == k) & multi_metal)[0]
        X, t, g, _ = pair_matrix(blocks, tr, cohorts, metals, arm)
        n_pairs += len(t)
        alpha = choose_alpha(X, t, g)
        alphas.append(alpha)
        coef, mu, sd, t_mean = _ridge_fit(X, t, alpha)

        test_table = pd.DataFrame({"c": cohorts[te], "i": te, "m": metals[te]})
        for cid, grp in test_table.groupby("c", sort=False):
            if grp["m"].nunique() < 2:
                continue
            members = grp["i"].to_numpy()
            acc = {int(i): [] for i in members}
            for a, b in permutations(members, 2):
                feat = pair_design(blocks, int(a), int(b), arm)[None, :]
                delta = float(_ridge_predict(feat, coef, mu, sd, t_mean)[0])
                # delta = f(b) - f(a); each row's score is its own value.
                acc[int(a)].append(-delta)
                acc[int(b)].append(delta)
            for i, vals in acc.items():
                scores[i] = float(np.mean(vals))
    return scores, {"alphas": alphas, "n_training_pairs": n_pairs}


def pair_accuracy(scores: np.ndarray, cohorts: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    correct, total = 0, 0
    for cid in np.unique(cohorts):
        idx = np.where((cohorts == cid) & ~np.isnan(scores))[0]
        for a, b in permutations(idx, 2):
            true = y[b] - y[a]
            pred = scores[b] - scores[a]
            if true == 0:
                continue
            correct += int(np.sign(true) == np.sign(pred))
            total += 1
    return {"pairwise_accuracy": round(correct / total, 4) if total else None, "n_ordered_pairs": total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_pairwise_reranker.json",
    )
    args = parser.parse_args(argv)

    long = pd.read_parquet(args.long)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy().reset_index(drop=True)
    sub["cohort"] = sub.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup()
    multi_metal = (
        sub.groupby("cohort")["metal"].transform("nunique").to_numpy() >= 2
    )
    y = sub["value_log10"].to_numpy(dtype=float)

    blocks = build_blocks(sub)
    blocks["_y"] = y
    cohorts = sub["cohort"].to_numpy()

    print(
        f"[m4] rows={len(sub)} cohorts={sub['cohort'].nunique()} "
        f"multi_metal={int(sub.loc[multi_metal, 'cohort'].nunique())} "
        f"donor_cov={blocks['_coverage']['has_donor']:.3f}"
    )

    splits: Dict[str, object] = {}
    for split in SPLITS:
        groups = sub[GROUP_COL[split]].to_numpy()
        fold_id = folds_for(groups)
        base_pred, cell_pred, _ = fold_baseline_scores(
            sub["metal"].to_numpy(),
            sub["cell_line_key"].to_numpy(),
            y,
            fold_id,
        )
        arms: Dict[str, np.ndarray] = {"main_effect": base_pred, "cell_effect": cell_pred}
        meta: Dict[str, object] = {}
        for arm in ("pair_identity", "pair_state", "pair_full"):
            print(f"[m4] split={split} arm={arm}")
            scores, info = run_pair_arm(
                blocks, fold_id, cohorts, arm, multi_metal, sub["metal"].astype(str).to_numpy()
            )
            arms[arm] = scores
            meta[arm] = info

        frame = pd.DataFrame(
            {
                "cohort": cohorts,
                "y": y,
                "doi": sub["doi"].to_numpy(),
                "metal": sub["metal"].to_numpy(),
                "fold": fold_id,
            }
        )
        for name, pred in arms.items():
            frame[name] = pred
        valid = frame[multi_metal].dropna(subset=list(arms)).reset_index(drop=True)

        records = []
        for cid, g in valid.groupby("cohort", sort=False):
            for name in arms:
                rec = cohort_metrics(g["y"].to_numpy(), g[name].to_numpy())
                records.append({"cohort": int(cid), "scorer": name, "doi": g["doi"].iloc[0], **rec})
        rec = pd.DataFrame(records)

        table: Dict[str, object] = {}
        for name in arms:
            per = rec[rec["scorer"] == name]
            table[name] = {
                "mean_regret": round(float(per["regret"].mean()), 4),
                "median_regret": round(float(per["regret"].median()), 4),
                "top1_accuracy": round(float(per["top1_correct"].mean()), 4),
                **pair_accuracy(
                    valid[name].to_numpy(), valid["cohort"].to_numpy(), valid["y"].to_numpy()
                ),
            }

        tests = {}
        for name in ("pair_identity", "pair_state", "pair_full"):
            tests[name] = {
                "vs_main_effect": doi_clustered_test(rec, name, "main_effect"),
                "vs_cell_effect": doi_clustered_test(rec, name, "cell_effect"),
            }
        splits[split] = {"n_cohorts": int(valid["cohort"].nunique()), "arms": table, "tests": tests, "fit": meta}
        print(f"[m4] {split}: " + " | ".join(f"{k}={v['mean_regret']:.4f}" for k, v in table.items()))

    canonical = splits[SPLIT_CANONICAL_GROUPED]
    best_pair = max(
        ("pair_identity", "pair_state", "pair_full"),
        key=lambda a: -canonical["arms"][a]["mean_regret"],
    )
    p_one = float(
        canonical["tests"][best_pair]["vs_main_effect"].get("p_one_sided_doi", 1.0)
    )
    delta = float(
        canonical["tests"][best_pair]["vs_main_effect"].get("mean_regret_delta", 0.0)
    )
    verdict = "POSITIVE" if (delta > 0 and p_one < 0.05) else "NEGATIVE"

    payload = {
        "artifact": "msrs_pairwise_reranker",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "ligand_spec_hash": blocks["_spec_hash"],
        },
        "design": {
            "rows": int(len(sub)),
            "cohorts": int(sub["cohort"].nunique()),
            "ligand_source": "ligand_canonical",
            "model": "ridge on antisymmetric within-cohort pair differences",
            "alphas": list(ALPHAS),
            "n_svd": N_SVD,
            "arms": {
                "main_effect": "fold-safe metal mean (global_best)",
                "cell_effect": "fold-safe (metal, cell line) mean",
                "pair_identity": "d(metal one-hot)",
                "pair_state": "+ d(oxidation state, charge, d-electrons, valence, coordination)",
                "pair_full": "+ ligand-context x metal interactions",
            },
            "arm_ligand_source": {
                "main_effect": "ligand_canonical",
                "cell_effect": "ligand_canonical",
                "pair_identity": "ligand_canonical",
                "pair_state": "ligand_canonical",
                "pair_full": "ligand_canonical",
            },
        },
        "splits": splits,
        "best_pair_arm": best_pair,
        "delta_regret_vs_main_effect": round(delta, 4),
        "p_one_sided_doi": p_one,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(f"[m4] verdict={verdict} best={best_pair} delta={delta:+.4f} p={p_one:.4g}")
    print(f"[m4] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
