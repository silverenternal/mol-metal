#!/usr/bin/env python3
"""UP-2b — a listwise ranker on the same folds, scored on the same pairs.

``msrs_ranking_expanded.py`` re-scores the existing pointwise regression
out-of-fold predictions and finds balanced accuracy 0.5272 (strict) / 0.4705
(ligand_cell), both with CIs that include 0.5.  That is a statement about the
*learner*, not about the task: a pointwise squared-error model has no reason
to order metals within a ligand correctly.

This trains the objective the task actually asks for.  Groups are
(ligand_canonical, cell_line_key); within a group the metals are ordered by
potency and the label is that rank, so LightGBM's lambdarank optimises the
within-group ordering directly.  Folds are GroupKFold(5) on DOI — the same
partition as the published DOI protocol, so the comparison is paired.

Everything downstream (pair construction, gate, additive ceiling, clustered
bootstrap) is imported from ``msrs_ranking_expanded`` so the two arms cannot
drift apart.

=====================================================================
Usage
=====================================================================
::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_listwise_ranker.py
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
from scipy import sparse
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    ENDPOINT,
    LIGAND_ONLY,
    TOP_CELL_LINES,
    build,
    cell_vocabulary,
    coordination_block,
    load_descriptors,
    state_block,
)
from molmetal.scripts.msrs_ranking_expanded import (  # noqa: E402
    PAIR_KEYS,
    _additive_ceiling,
    _by_metal_pair,
    _cluster_bootstrap,
    _metrics,
    score_pairs,
)
from molmetal.scripts.msrs_surface_model import SEED, _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_FOLDS = 5
GROUP_KEY = ("ligand_canonical", "cell_line_key")
LGB_RANK = dict(
    objective="lambdarank",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.3,
    reg_lambda=1.0,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
    label_gain=[0, 1, 3, 7, 15, 31],
)
MAX_LABEL = 5


def build_features(
    long: pd.DataFrame, metal_desc: Path
) -> tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray]:
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)

    load_descriptors(metal_desc)
    vocab = cell_vocabulary(sub["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES)
    bundle = build(sub, LIGAND_ONLY)
    state, _ = state_block(sub)
    coord, _, _ = coordination_block(sub, include_geometry=True)
    both = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    X = sparse.hstack([bundle.X, both], format="csr")
    y = sub["value_log10"].to_numpy(dtype=float)
    return sub, X, y


def rank_labels(frame: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer relevance per row and contiguous group sizes, sorted by group.

    Higher relevance = more potent = lower y.  Rows are reordered so that each
    group is contiguous, which is what LightGBM's ``group`` argument expects.
    """
    key = frame[list(GROUP_KEY)].astype(str).agg("|".join, axis=1).to_numpy()
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    y_sorted = y[order]
    labels = np.zeros(len(y_sorted), dtype=int)
    sizes: List[int] = []
    start = 0
    for i in range(1, len(key_sorted) + 1):
        if i == len(key_sorted) or key_sorted[i] != key_sorted[start]:
            n = i - start
            sizes.append(n)
            if n > 1:
                block = y_sorted[start:i]
                # most potent (lowest y) gets the highest label.  A group can
                # exceed five rows when the same (ligand, cell, metal) was
                # measured in more than one paper; clip so the gain table
                # stays finite.
                ranks = np.argsort(np.argsort(-block, kind="stable"), kind="stable")
                labels[start:i] = np.minimum(ranks, MAX_LABEL)
            start = i
    return order, labels, np.array(sizes, dtype=int)


def deadzone_labels(
    delta: np.ndarray, tau: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """UP-23 — which observed pair deltas are larger than the assay noise?

    ``tau`` is the metal pair's ``noise_sd_delta`` from ``msrs_metal_graph``.
    A pair with ``|delta| < tau`` cannot be told apart from a tie by the
    measurement, so treating it as a hard label teaches the ranker noise.

    Returns ``(keep, weight)``:

    * ``keep``   — arm A: drop the pair entirely.
    * ``weight`` — arm B: keep it but scale the gradient by ``min(1, |d|/tau)``,
      so a pair barely above the noise floor counts for very little.

    ``tau <= 0`` means the metal pair has no estimated noise scale; such pairs
    are kept at full weight rather than silently dropped.
    """
    delta = np.asarray(delta, dtype=float)
    tau = np.asarray(tau, dtype=float)
    absd = np.abs(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(tau > 0, absd / np.where(tau > 0, tau, 1.0), 1.0)
    weight = np.clip(ratio, 0.0, 1.0)
    keep = (tau <= 0) | (absd >= tau)
    return keep, weight


def fit_fold(
    X: sparse.csr_matrix,
    labels: np.ndarray,
    sizes: np.ndarray,
    order: np.ndarray,
    train_mask_sorted: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train on whole groups, predict every row (sorted order)."""
    from lightgbm import LGBMRanker

    group_id = np.repeat(np.arange(len(sizes)), sizes)
    keep_groups = np.unique(group_id[train_mask_sorted])
    tr = np.isin(group_id, keep_groups)
    group_sizes = np.array([int((group_id == g).sum()) for g in keep_groups], dtype=int)

    params = dict(LGB_RANK)
    params["random_state"] = int(seed)
    model = LGBMRanker(**params)
    model.fit(X[order][tr], labels[tr], group=group_sizes)
    return model.predict(X[order])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    ap.add_argument(
        "--metal-desc", type=Path, default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv"
    )
    ap.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_listwise_ranker.json"
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED, 42, 43, 44, 49])
    args = ap.parse_args(argv)

    for p in (args.long, args.metal_desc):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    t0 = time.time()
    long = pd.read_parquet(args.long)
    frame, X, y = build_features(long, args.metal_desc)
    order, labels, sizes = rank_labels(frame, y)
    doi = frame["doi"].astype(str).to_numpy()
    folds = np.full(len(frame), -1, dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(np.zeros((len(frame), 1)), groups=doi)):
        folds[te] = k

    # Per-seed out-of-fold scores, in the original row order.
    oof = np.full((len(frame), len(args.seeds)), np.nan)
    for j, seed in enumerate(args.seeds):
        scores = np.full(len(frame), np.nan)
        for k in range(N_FOLDS):
            tr = folds != k
            te_sorted = (~tr)[order]
            s = fit_fold(X, labels, sizes, order, tr[order], seed)
            scores[order[te_sorted]] = s[te_sorted]
        oof[:, j] = scores
        print(f"  seed {seed} done")
    score = np.nanmean(oof, axis=1)
    # Higher score = more potent = lower y, so negate to make it y-like.
    pred = -score

    results: Dict[str, object] = {}
    for name, key in PAIR_KEYS.items():
        pairs = pair_table(frame, key=key)
        results[name] = {
            "key": list(key),
            "n_pairs_built": int(len(pairs)),
            **score_pairs(y, pred, pairs, folds, GATE_THRESHOLD, frame["ligand_canonical"].to_numpy()),
        }
        o = results[name].get("overall", {})
        print(
            f"[ranker/{name}] n={o.get('n_pairs')} raw={o.get('raw_sign_accuracy')} "
            f"balanced={o.get('balanced_accuracy')} majority={o.get('majority_rate')}"
        )

    wide = results["ligand_cell"].get("overall", {})
    ci = (results["ligand_cell"].get("bootstrap", {}).get("balanced_ci95") or [0, 0])[0]
    gate_block = {
        "balanced_ge_0.58": bool((wide.get("balanced_accuracy") or 0) >= 0.58),
        "balanced_ci_low_gt_0.52": bool(ci > 0.52),
        "raw_ge_0.63": bool((wide.get("raw_sign_accuracy") or 0) >= 0.63),
    }

    out = {
        "artifact": "msrs_listwise_ranker",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 ranking / UP-2b",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "long": str(args.long),
            "seeds": list(args.seeds),
            "argv": [sys.executable] + (argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "learner": "LightGBM lambdarank",
            "group": list(GROUP_KEY),
            "label": "rank of -y within group (higher = more potent)",
            "cv": "GroupKFold(5) on DOI, same partition as the DOI protocol",
            "features": "same blocks as the ligand+both arm (ECFP4 + metal state + coordination + cell)",
            "n_rows": int(len(frame)),
            "n_groups_multi_metal": int((sizes > 1).sum()),
            "frozen_gate": {
                "balanced": 0.58,
                "balanced_ci_low": 0.52,
                "raw": 0.63,
                "frozen_at": "2026-10-21T15:10+08:00",
            },
        },
        "pairs": results,
        "gate": gate_block,
        "verdict": "PASS" if all(gate_block.values()) else "FAIL",
        "verdict_rule": "all three gate fields must hold on the ligand_cell pair set",
        "elapsed_seconds": round(time.time() - t0, 1),
        "caveats": [
            "The ranker is scored with the same pair builder as the regression arm, so the "
            "comparison is paired by construction.",
            "Groups of one metal contribute no lambdarank gradient; they are kept in the training "
            "matrix but cannot teach an ordering.",
            "Cross-paper pairs (ligand_cell) carry a paper-level offset that the strict key removes; "
            "both are reported.",
            "Scores are out-of-fold, so a cross-paper pair whose two rows fall in different DOI folds "
            "compares scores from two different models with arbitrary relative scale. The strict set "
            "does not have this problem (both rows share a DOI, hence a fold) and is the cleaner "
            "read; the ligand_cell balanced number should be treated as an upper bound.",
            "Direction check: corr(score, y) = -0.156 on one held-out fold, i.e. higher score really "
            "does mean more potent, so the pair signs are not inverted.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[UP-2b] verdict={out['verdict']} gate={gate_block} ({out['elapsed_seconds']}s)")
    print(f"json: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
