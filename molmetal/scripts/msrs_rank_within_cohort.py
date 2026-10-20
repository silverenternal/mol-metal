#!/usr/bin/env python3
"""C3 — a within-cohort ranker, the one ordering mechanism not yet tried.

The three routes already measured all *derive* an ordering from a level model
or from a pair regression: differencing (0.5826), fusion (0.5505), direct delta
regression (0.5183).  None of them ever optimised the ordering itself.

This arm does.  Each cohort is a query group; the metals inside it are the
items; relevance is the within-cohort rank of the observed potency.  LightGBM's
``lambdarank`` objective then maximises the probability that the higher-ranked
item scores higher — the quantity the gate actually measures.

Budget match: the ranker gets the same 500 trees / 31 leaves / 0.05 learning
rate as the level regressor (``LGB_PARAMS``).  It is not given more capacity,
only a different loss.

Evaluation is unchanged from STR-5, so the numbers are comparable: the same
427 pairs, the same |delta| >= 0.2091 gate, the same DOI-clustered bootstrap.

Usage::

    OMP_NUM_THREADS=4 .venv/bin/python molmetal/scripts/msrs_rank_within_cohort.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.features.cell import cell_vocabulary, native_codes  # noqa: E402
from molmetal.scripts.msrs_delta_direct import (  # noqa: E402
    SIGN_GATE,
    baseline as level_baseline,
    cluster_bootstrap,
    delta_stats,
)
from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_feature_expansion import (  # noqa: E402
    maccs_rdkit_block,
    select_cohort,
)
from molmetal.scripts.msrs_generalization import pair_table, scheme_folds  # noqa: E402
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    LIGAND_ONLY,
    TOP_CELL_LINES,
    build,
    coordination_block,
    load_descriptors,
    state_block,
)
from molmetal.scripts.msrs_surface_model import LGB_PARAMS, _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOL = "doi"
COHORT_KEYS = ("ligand_canonical", "cell_line_key", "doi", "time_h")
N_ESTIMATORS = int(LGB_PARAMS["n_estimators"])
NUM_LEAVES = int(LGB_PARAMS["num_leaves"])
LEARNING_RATE = float(LGB_PARAMS["learning_rate"])


def cohort_keys(frame: pd.DataFrame) -> np.ndarray:
    """One string per row identifying its (ligand, cell line, paper, time) cohort.

    ``DataFrame.astype(str)`` is a NO-OP on object columns, so a NaN in
    ``time_h`` survived as a float and blew up the ``"\\x1f".join`` below.
    Map element-wise instead.
    """
    cols = [frame[c].map(lambda v: "NA" if pd.isna(v) else str(v)) for c in COHORT_KEYS]
    return pd.concat(cols, axis=1).agg("\x1f".join, axis=1).to_numpy()


def relevance(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Within-cohort rank of the potency (0 = lowest), as a LightGBM label."""
    out = np.zeros(len(frame), dtype=np.int64)
    keys = cohort_keys(frame)
    for _, rows in pd.Series(np.arange(len(frame))).groupby(keys, sort=False):
        idx = rows.to_numpy()
        if len(idx) < 2:
            continue
        order = np.argsort(y[idx], kind="stable")
        ranks = np.empty(len(idx), dtype=np.int64)
        ranks[order] = np.arange(len(idx))
        out[idx] = ranks
    return out


def build_features(frame: pd.DataFrame) -> sparse.csr_matrix:
    """The STR-4 winning block: ECFP4 + state/coordination + MACCS + RDKit."""
    base = build(frame, LIGAND_ONLY)
    state, _ = state_block(frame)
    coord, _, _ = coordination_block(frame, include_geometry=True)
    state_coord = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    base_ligand = sparse.hstack([base.X, state_coord], format="csr")
    extra = maccs_rdkit_block(frame)
    return sparse.hstack([base_ligand, sparse.csr_matrix(extra)], format="csr")


def group_sizes(keys: np.ndarray) -> np.ndarray:
    _, counts = np.unique(keys, return_counts=True)
    return counts.astype(np.int32)


def tie_stats(obs: np.ndarray, pred: np.ndarray, gate: float) -> Dict[str, object]:
    """Separate "the model tied" from "the model got it backwards".

    A lambdarank score is a per-query quantity.  If the trees never split on the
    metal block, both rows of a within-cohort metal pair land in the same leaf
    and get the SAME score; ``sign(pred)`` is then 0 and ``delta_stats`` counts
    the pair as wrong.  That is not an inversion, it is a tie, and the two must
    not be reported as the same thing.
    """
    keep = np.abs(obs) >= gate
    tie = np.isclose(pred, 0.0, atol=1e-12)
    n_gated = int(keep.sum())
    n_tied = int((tie & keep).sum())
    correct = np.sign(pred[keep]) == np.sign(obs[keep])
    tie_aware = np.where(tie[keep], 0.5, correct.astype(float))
    return {
        "n_gated": n_gated,
        "n_tied": n_tied,
        "tie_share": round(float(n_tied / max(1, n_gated)), 4),
        "sign_accuracy_ties_half": round(float(tie_aware.mean()), 4) if n_gated else None,
        "reading": (
            "tie_share is the share of gated pairs the model scores identically. "
            "High tie_share means the learned function is metal-blind: it ranks "
            "ligands, not metals, and the ordering claim has no mechanism behind it."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        type=Path,
        default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet",
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scores-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_rank_within_cohort" / "oof_scores.npz",
        help=(
            "Keep the OOF ranker scores. A sign accuracy this far below chance "
            "has to be attributable before it is reportable, and that needs the "
            "per-row scores, not just the aggregate."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_rank_within_cohort.json",
    )
    args = parser.parse_args(argv)

    from lightgbm import LGBMRanker

    t0 = time.time()
    frame = select_cohort(args.long)
    y = frame["value_log10"].to_numpy(dtype=float)
    load_descriptors(args.metal_desc)
    X = build_features(frame)
    print(f"[c3] rows={len(frame)} features={X.shape[1]} in {time.time() - t0:.0f}s")

    fold_id, groups, info = scheme_folds(frame, PROTOCOL)
    keys = cohort_keys(frame)
    rel = relevance(frame, y)
    sizes = pd.Series(keys).value_counts()
    n_comparable = int(sizes[sizes >= 2].sum())

    scores = np.full(len(frame), np.nan)
    for k in sorted(np.unique(fold_id)):
        tr = np.where(fold_id != k)[0]
        te = np.where(fold_id == k)[0]
        # lambdarank needs the rows of a query contiguous in the matrix
        order = np.argsort(keys[tr], kind="stable")
        tr_sorted = tr[order]
        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            num_leaves=NUM_LEAVES,
            min_child_samples=int(LGB_PARAMS["min_child_samples"]),
            subsample=float(LGB_PARAMS["subsample"]),
            subsample_freq=int(LGB_PARAMS["subsample_freq"]),
            colsample_bytree=float(LGB_PARAMS["colsample_bytree"]),
            reg_lambda=float(LGB_PARAMS["reg_lambda"]),
            random_state=args.seed,
            n_jobs=4,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        model.fit(
            X[tr_sorted],
            rel[tr_sorted],
            group=group_sizes(keys[tr_sorted]),
        )
        scores[te] = model.predict(X[te])

    pairs = pair_table(frame)
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.scores_out,
        score=np.asarray(scores, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        rel=np.asarray(rel, dtype=np.int64),
        fold=np.asarray(fold_id, dtype=np.int64),
        metal=np.asarray(frame["metal"].to_numpy(), dtype=str),
        doi=np.asarray(frame["doi"].to_numpy(), dtype=str),
        cohort=np.asarray(keys, dtype=str),
        pair_a=np.asarray(ia, dtype=np.int64),
        pair_b=np.asarray(ib, dtype=np.int64),
    )
    obs = y[ib] - y[ia]
    pred = scores[ib] - scores[ia]
    clusters = frame["doi"].astype(str).to_numpy()[ia]
    scored = np.isfinite(pred)

    gated = delta_stats(obs[scored], pred[scored], GATE_THRESHOLD)
    ungated = delta_stats(obs[scored], pred[scored], None)
    boot = cluster_bootstrap(obs[scored], pred[scored], clusters[scored], GATE_THRESHOLD)
    base = level_baseline(PROJECT_ROOT / "metrics" / "by_metric" / "msrs_seed_ensemble.json")

    # A lambdarank score is a per-query quantity; ties are not inversions.
    ties = tie_stats(obs[scored], pred[scored], GATE_THRESHOLD)

    sign = gated.get("sign_accuracy")
    sign_tie = ties["sign_accuracy_ties_half"]
    ci_low = boot["sign_accuracy"]["ci95"][0]
    passed = sign_tie >= SIGN_GATE and ci_low > 0.55
    verdict = "PASS" if passed else ("KILL" if sign_tie < 0.55 else "PARTIAL")

    payload = {
        "artifact": "msrs_rank_within_cohort",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_rank_within_cohort.py",
            "git_commit": _git_commit(),
        },
        "design": {
            "protocol": PROTOCOL,
            "objective": "lambdarank on within-cohort relevance",
            "features": "ECFP4 + state/coordination + MACCS(167) + 20 RDKit descriptors",
            "n_features": int(X.shape[1]),
            "budget_match": {
                "n_estimators": N_ESTIMATORS,
                "num_leaves": NUM_LEAVES,
                "learning_rate": LEARNING_RATE,
                "note": "same capacity as LGB_PARAMS; only the loss changes",
            },
            "folds": info,
            "n_cohorts": int(len(sizes)),
            "n_cohorts_comparable": int((sizes >= 2).sum()),
            "n_rows_in_comparable_cohorts": n_comparable,
            "cohort_definition": list(COHORT_KEYS),
            "gate_threshold": GATE_THRESHOLD,
            "sign_gate": SIGN_GATE,
            "n_pairs_scored": int(scored.sum()),
            "wall_s": round(time.time() - t0, 1),
        },
        "gated": {**gated, "bootstrap": boot},
        "ungated": ungated,
        "ties": ties,
        "baseline_level_model": base,
        "verdict": verdict,
        "verdict_detail": {
            "sign": sign,
            "sign_ties_half": sign_tie,
            "tie_share": ties["tie_share"],
            "sign_ci95": boot["sign_accuracy"]["ci95"],
            "sign_gate": bool(sign is not None and sign >= SIGN_GATE),
            "ci_gate": bool(ci_low > 0.55),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        f"[c3] pairs={int(scored.sum())} gated={gated.get('n_pairs')} sign={sign} "
        f"ties={ties['tie_share']} sign_ties_half={sign_tie} "
        f"ci={boot['sign_accuracy']['ci95']} mae={gated.get('mae_delta')}"
    )
    print(f"[c3] level baseline sign={base['sign_accuracy'] if base else None}")
    print(f"[c3] verdict={verdict} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
