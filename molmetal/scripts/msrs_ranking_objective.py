#!/usr/bin/env python3
"""FIX-E3 — align the training objective with the decision objective.

NEG-17 measured the mismatch directly: FIX-L1 complex features raised regression
R2 from 0.4368 to 0.5012 but *worsened* within-cohort top-1 regret from 0.1212
to 0.1314.  A model trained on absolute log IC50 is optimising "how potent is
this compound", while the recommendation asks "which metal is best in this
cohort".  Those are different functionals, and better fit to the first does not
imply better ordering on the second.

This script keeps the representation fixed at the best available joint block
(FIX-L1 + L2 + L3, measured 0.5556 canonical-grouped) and changes only the
target:

  regression   y                          — the E2 arm, for reference
  centered     y - mean(y in cohort)      — learn relative potency
  pair_delta   y_b - y_a for metal pairs  — learn the metal swap directly

``centered`` is the cheapest honest test of the hypothesis.  Cohort means are
computed on training folds only; a cohort with a single training row carries no
relative information and is dropped (weight 0) rather than taught a zero.
At prediction time the missing cohort mean is a constant within the cohort, so
it cannot change the argmin — ranking centred scores is ranking absolute ones.

``pair_delta`` builds one training row per ordered metal pair in a cohort:
features are ``[ligand bits, cell code, metal_a code, metal_b code]`` and the
target is ``y_b - y_a``; reversing the pair negates the target, so the model
cannot learn a directional bias.  At prediction time every metal's score is its
mean predicted delta against the other metals in the cohort.

Acceptance (same as FIX-E2, so the two are comparable): full-cohort mean regret
< 0.1198 **and** DOI-clustered Holm-adjusted p < 0.05.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_ranking_objective.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse

from molmetal.features.build import build
from molmetal.features.cell import cell_vocabulary, native_codes
from molmetal.features.coordination import coordination_block
from molmetal.features.metal import state_block
from molmetal.features.spec import CellSpec, FeatureSpec, LigandSpec, MetalSpec
from molmetal.scripts.msrs_code_levers import (
    COHORT_KEYS,
    ENDPOINT,
    LGB_PARAMS,
    fit_arm,
    folds_for,
)
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
from molmetal.scripts.msrs_surface_model import (
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    holm,
)

BASELINE_FULL_REGRET = 0.1198
ALPHA = 0.05
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)

LIGAND_ONLY = FeatureSpec(
    ligand=LigandSpec(source="ligand_canonical", include=("canonical",)),
    metal=MetalSpec(one_hot=False, columns=()),
    cell=CellSpec(mode="none"),
)


def _centered_oof(
    X: sparse.csr_matrix,
    y: np.ndarray,
    fold_id: np.ndarray,
    cohorts: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    """OOF predictions of within-cohort centred potency.

    A cohort lies entirely inside one fold because folds are grouped by ligand
    and a cohort is one ligand, so its training mean is the full cohort mean.
    Singleton cohorts carry no relative signal and get weight 0.
    """
    from lightgbm import LGBMRegressor

    oof = np.full(len(y), np.nan)
    params = dict(LGB_PARAMS)
    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = ~tr
        frame = pd.DataFrame({"c": cohorts[tr], "y": y[tr]})
        stats = frame.groupby("c")["y"].agg(["mean", "size"])
        mean = stats["mean"].reindex(frame["c"]).to_numpy()
        size = stats["size"].reindex(frame["c"]).to_numpy()
        yc = y[tr] - mean
        w = (size >= 2).astype(np.float32)

        Xtr = sparse.hstack(
            [X[tr], sparse.csr_matrix(codes[tr].astype(np.float32))], format="csr"
        )
        Xte = sparse.hstack(
            [X[te], sparse.csr_matrix(codes[te].astype(np.float32))], format="csr"
        )
        n_lig = X.shape[1]
        cat_idx = [n_lig, n_lig + 1]
        model = LGBMRegressor(**params)
        model.fit(Xtr, yc, sample_weight=w, categorical_feature=cat_idx)
        oof[te] = model.predict(Xte)
    return oof


def _pair_rows(
    X: sparse.csr_matrix,
    y: np.ndarray,
    cohorts: np.ndarray,
    metals: np.ndarray,
    codes: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Ordered within-cohort metal pairs: features, target, pair cohort id."""
    n_lig = X.shape[1]
    rows: List[sparse.csr_matrix] = []
    targets: List[float] = []
    groups: List[int] = []
    for cid in np.unique(cohorts):
        idx = np.where(cohorts == cid)[0]
        if len(idx) < 2:
            continue
        for a in idx:
            for b in idx:
                if a == b:
                    continue
                # [ligand_a, cell_a, metal_a, metal_b] — the pair is described
                # by A's chemistry and both metal identities.
                feat = sparse.hstack(
                    [
                        X[a],
                        sparse.csr_matrix(
                            np.array([[codes[a, 1], codes[a, 0], codes[b, 0]]],
                                     dtype=np.float32)
                        ),
                    ],
                    format="csr",
                )
                rows.append(feat)
                targets.append(float(y[b] - y[a]))
                groups.append(int(cid))
    if not rows:
        raise ValueError("_pair_rows: no cohort has two metals")
    return (
        sparse.vstack(rows, format="csr"),
        np.asarray(targets, dtype=np.float32),
        np.asarray(groups, dtype=int),
    )


def _pair_scores(
    X: sparse.csr_matrix,
    y: np.ndarray,
    fold_id: np.ndarray,
    cohorts: np.ndarray,
    metals: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    """Per-row score = mean predicted delta against the cohort's other metals."""
    from lightgbm import LGBMRegressor

    params = dict(LGB_PARAMS)
    n_lig = X.shape[1]
    scores = np.full(len(y), np.nan)
    for k in sorted(set(fold_id.tolist())):
        tr = fold_id != k
        te = ~tr
        Xp, yp, _ = _pair_rows(X[tr], y[tr], cohorts[tr], metals[tr], codes[tr])
        model = LGBMRegressor(**params)
        model.fit(Xp, yp)

        for cid in np.unique(cohorts[te]):
            idx = np.where(cohorts == cid)[0]
            if len(idx) < 2:
                continue
            acc = {i: [] for i in idx}
            for a in idx:
                for b in idx:
                    if a == b:
                        continue
                    feat = sparse.hstack(
                        [
                            X[a],
                            sparse.csr_matrix(
                                np.array(
                                    [[codes[a, 1], codes[a, 0], codes[b, 0]]],
                                    dtype=np.float32,
                                )
                            ),
                        ],
                        format="csr",
                    )
                    delta = float(model.predict(feat)[0])
                    acc[a].append(delta)
                    acc[b].append(-delta)
            for i in idx:
                scores[i] = float(np.mean(acc[i]))
    return scores


def _evaluate(
    frame: pd.DataFrame, y: np.ndarray, preds: Dict[str, np.ndarray], cohorts: np.ndarray
) -> Dict[str, object]:
    work = frame.copy()
    work["y"] = y
    work["cohort"] = cohorts
    for name, pred in preds.items():
        work[name] = pred
    # A cohort with one metal cannot support a recommendation; counting it as
    # a zero-regret success would inflate top-1 accuracy (the same filter
    # ``build_cohorts(min_size=2)`` applies in the decision layer).
    sizes = work.groupby("cohort")["metal"].nunique()
    keep = sizes[sizes >= 2].index
    work = work[work["cohort"].isin(keep)]

    records = []
    for cid, g in work.groupby("cohort", sort=False):
        for name in preds:
            rec = cohort_metrics(g["y"].to_numpy(), g[name].to_numpy())
            records.append(
                {
                    "cohort": int(cid),
                    "scorer": name,
                    "doi": g["doi"].iloc[0],
                    **rec,
                }
            )
    rec_df = pd.DataFrame(records)
    summary = {
        name: {
            "mean_regret": round(float(rec_df[rec_df["scorer"] == name]["regret"].mean()), 4),
            "top1_accuracy": round(
                float(rec_df[rec_df["scorer"] == name]["top1_correct"].mean()), 4
            ),
            "n_cohorts": int((rec_df["scorer"] == name).sum()),
        }
        for name in preds
    }
    tests = {
        name: doi_clustered_test(rec_df, name, "global_best")
        for name in preds
        if name != "global_best"
    }
    pvals = {k: float(v.get("p_one_sided_doi", 1.0)) for k, v in tests.items()}
    adjusted = holm(pvals)
    for k in tests:
        tests[k]["p_holm"] = round(float(adjusted[k]), 6)
    return {"summary": summary, "clustered_tests": tests}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_ranking_objective.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    long = pd.read_parquet(args.long)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    vocab = cell_vocabulary(
        sub["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES
    )
    codes = native_codes(
        sub["metal"].astype(str).tolist(),
        sub["cell_line_key"].astype(str).tolist(),
        vocab,
        mode="full",
    )

    bundle = build(sub, LIGAND_ONLY)
    state, _ = state_block(sub)
    coord, _, coverage = coordination_block(sub, include_geometry=True)
    X = sparse.hstack([bundle.X, sparse.csr_matrix(state), coord], format="csr")

    cohorts = (
        sub[COHORT_KEYS].groupby(COHORT_KEYS, dropna=False, sort=False).ngroup().to_numpy()
    )

    splits: Dict[str, object] = {}
    for split, col in (
        (SPLIT_FRAGMENT_GROUPED, "ligand_main_fragment"),
        (SPLIT_CANONICAL_GROUPED, "ligand_canonical"),
    ):
        print(f"[rank] split={split}")
        fold_id = folds_for(sub[col].to_numpy())
        global_pred, _, _ = fold_baseline_scores(
            sub["metal"].to_numpy(), sub["cell_line_key"].to_numpy(), y, fold_id
        )
        empty = sparse.csr_matrix((len(sub), 0), dtype=np.float32)
        preds = {
            "global_best": global_pred,
            "regression": fit_arm(
                X, empty, empty, None, 0, False, y, fold_id, cat_codes=codes
            ),
            "centered": _centered_oof(X, y, fold_id, cohorts, codes),
        }
        if split == SPLIT_CANONICAL_GROUPED:
            preds["pair_delta"] = _pair_scores(
                X,
                y,
                fold_id,
                cohorts,
                sub["metal"].astype(str).to_numpy(),
                codes,
            )
        splits[split] = _evaluate(sub, y, preds, cohorts)
        for name, s in splits[split]["summary"].items():
            print(f"    {name:12s} regret={s['mean_regret']:.4f} top1={s['top1_accuracy']:.4f}")

    canonical = splits[SPLIT_CANONICAL_GROUPED]["summary"]
    candidates = {k: v for k, v in canonical.items() if k != "global_best"}
    best_arm = min(candidates, key=lambda k: candidates[k]["mean_regret"])
    best = canonical[best_arm]
    p_holm = splits[SPLIT_CANONICAL_GROUPED]["clustered_tests"][best_arm]["p_holm"]
    passed = best["mean_regret"] < BASELINE_FULL_REGRET and p_holm < ALPHA

    payload = {
        "artifact": "msrs_ranking_objective",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "ligand_spec_hash": bundle.spec_hash,
        },
        "design": {
            "rows": int(len(sub)),
            "n_cohorts": int(len(np.unique(cohorts))),
            "features": "ECFP4(ligand_canonical) | metal state (4) | coordination",
            "ligand_source": "ligand_canonical",
            "ligand_block": "ECFP4(ligand_canonical), 2048 bits",
            "objectives": {
                "regression": "y",
                "centered": "y - mean(y | cohort, training folds); singleton cohorts weight 0",
                "pair_delta": "y_b - y_a for ordered within-cohort metal pairs",
            },
            "coverage": coverage,
            "baseline_full_regret": BASELINE_FULL_REGRET,
            "alpha": ALPHA,
            "correction": "Holm over the objectives tested per split",
        },
        "splits": splits,
        "best_arm": best_arm,
        "headline": {
            "split": SPLIT_CANONICAL_GROUPED,
            "arm": best_arm,
            "mean_regret": best["mean_regret"],
            "p_holm_vs_global_best": p_holm,
        },
        "verdict": "POSITIVE" if passed else "NEGATIVE",
        "reading": (
            f"canonical 分组最佳目标 {best_arm}：全量 regret {best['mean_regret']:.4f}"
            f"（基线 {BASELINE_FULL_REGRET:.4f}），Holm p={p_holm:.4f}。"
            "pair_delta 只在 canonical 分组跑（成本最高）；它若不胜过 centered，说明"
            "「相对强度」这一层就够，不必显式建模金属对。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(f"[rank] verdict={payload['verdict']} best={best_arm} regret={best['mean_regret']:.4f}")
    print(f"[rank] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
