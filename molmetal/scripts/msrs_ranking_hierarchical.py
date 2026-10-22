#!/usr/bin/env python3
"""UP-23 — are the ranking labels noisy, and does a hierarchy help?

UP-2b's listwise ranker reaches balanced accuracy 0.5985 on the 360
ligand+cell pairs.  ``msrs_support_audit`` says Ir-Rh has separability 0.115,
i.e. almost every |delta| inside that metal pair is smaller than the assay
noise.  A pair whose true delta is zero but whose observed delta is not is a
coin flip presented to the model as a hard label, so it costs accuracy twice:
once when the model is trained on it, once when it is scored on it.

This script separates those two costs.

Arms
----
``lambdarank_none``      the UP-2b model, unchanged.  Reproducing 0.5985 here
                         is the check that nothing else moved.
``lambdarank_deadzone``  same model, tie-aware labels: within a group, rows
                         whose |delta| is below the metal pair's noise SD are
                         merged into one relevance level, so lambdarank sees no
                         gradient for them.
``bt_*``                 a pairwise logistic (Bradley-Terry) ranker on the
                         feature difference x_a - x_b, which is where a
                         per-pair *weight* is expressible at all:

    ``bt_none``     every pair at weight 1
    ``bt_exclude``  arm A — deadzone pairs dropped
    ``bt_weight``   arm B — w = min(1, |delta| / tau)
    ``bt_hier``     ``bt_weight`` plus per-metal indicators, i.e. the score is
                    f(x) + u_metal.  On the 4-node metal graph this is the
                    additive model of ``msrs_metal_graph.gls_levels``; the
                    fitted u are reported and compared to that artifact.

Scoring is ``msrs_ranking_expanded.score_pairs`` throughout — same pair
builder, same |delta| >= 0.2091 gate, same folds — so the arms are paired by
construction.  Everything is reported twice: on all 360 gated pairs (the UP-2b
number) and on the deadzone-retained subset (the UP-23 number), because a
number on a cleaner subset is not comparable to a number on the full set
unless both are shown.

Usage
-----
::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_ranking_hierarchical.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_listwise_ranker import (  # noqa: E402
    GROUP_KEY,
    MAX_LABEL,
    build_features,
    deadzone_labels,
    fit_fold,
    rank_labels,
)
from molmetal.scripts.msrs_ranking_expanded import (  # noqa: E402
    _metrics,
    score_pairs,
)
from molmetal.scripts.msrs_surface_model import SEED, _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_FOLDS = 5
METALS = ["Ir", "Os", "Re", "Rh", "Ru"]
GATE_BALANCED = 0.65
GATE_MIN_RETAINED = 240


# --------------------------------------------------------------------------
# deadzone bookkeeping
# --------------------------------------------------------------------------
def tau_lookup(edges: Dict[str, dict]) -> Dict[frozenset, float]:
    """Canonical (unordered) metal pair -> noise SD of the delta."""
    out: Dict[frozenset, float] = {}
    for name, block in edges.items():
        a, b = name.split("-")
        tau = block.get("noise_sd_delta")
        out[frozenset((a, b))] = float(tau) if tau else 0.0
    return out


def pair_taus(pairs: pd.DataFrame, tau: Dict[frozenset, float]) -> np.ndarray:
    return np.array(
        [tau.get(frozenset((a, b)), 0.0) for a, b in zip(pairs["metal_a"], pairs["metal_b"])],
        dtype=float,
    )


def tie_labels(
    frame: pd.DataFrame, y: np.ndarray, tau: Dict[frozenset, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``rank_labels`` but rows within one noise SD of each other are tied.

    Transitive closure matters: in a group {Ru, Os, Ir} with |y_Ru - y_Os| <
    tau and |y_Os - y_Ir| < tau but |y_Ru - y_Ir| > tau, all three end up in
    one cluster.  That is the correct behaviour for a noise model — the
    measurement cannot resolve any of the three — but it does mean a group can
    collapse to a single label and contribute no gradient at all.
    """
    key = frame[list(GROUP_KEY)].astype(str).agg("|".join, axis=1).to_numpy()
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    y_sorted = y[order]
    metals = frame["metal"].astype(str).to_numpy()[order]
    labels = np.zeros(len(y_sorted), dtype=int)
    sizes: List[int] = []
    start = 0
    for i in range(1, len(key_sorted) + 1):
        if i == len(key_sorted) or key_sorted[i] != key_sorted[start]:
            n = i - start
            sizes.append(n)
            if n > 1:
                block_y = y_sorted[start:i]
                block_m = metals[start:i]
                parent = list(range(n))

                def find(u: int) -> int:
                    while parent[u] != u:
                        parent[u] = parent[parent[u]]
                        u = parent[u]
                    return u

                for u in range(n):
                    for v in range(u + 1, n):
                        if block_m[u] == block_m[v]:
                            continue
                        t = tau.get(frozenset((block_m[u], block_m[v])), 0.0)
                        if t > 0 and abs(block_y[u] - block_y[v]) < t:
                            ru, rv = find(u), find(v)
                            if ru != rv:
                                parent[rv] = ru
                roots = np.array([find(u) for u in range(n)])
                uniq = sorted(set(roots.tolist()))
                cluster_mean = np.array(
                    [float(np.mean(block_y[roots == r])) for r in uniq]
                )
                # lowest y (most potent) -> highest relevance, same as rank_labels
                ranks = np.argsort(np.argsort(-cluster_mean, kind="stable"), kind="stable")
                rank_of = {r: int(ranks[j]) for j, r in enumerate(uniq)}
                labels[start:i] = np.minimum(
                    np.array([rank_of[r] for r in roots]), MAX_LABEL
                )
            start = i
    return order, labels, np.array(sizes, dtype=int)


# --------------------------------------------------------------------------
# Bradley-Terry arms
# --------------------------------------------------------------------------
def pair_design(
    X: sparse.csr_matrix, ia: np.ndarray, ib: np.ndarray
) -> sparse.csr_matrix:
    """Row-wise feature difference, the design of a pairwise logistic model."""
    return sparse.csr_matrix(X[ia] - X[ib])


def fit_bt(
    D: sparse.csr_matrix,
    target: np.ndarray,
    weight: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Ridge-penalised logistic regression with no intercept.

    ``target = +1`` when row a is the more potent of the two.  With no
    intercept the model cannot learn a per-pair bias, which is what makes the
    ``bt_hier`` arm's metal indicators meaningful rather than redundant.
    """
    if len(target) == 0:
        return np.zeros(D.shape[1])
    clf = LogisticRegression(
        fit_intercept=False,
        C=1.0 / max(alpha, 1e-6),
        solver="liblinear",
        max_iter=2000,
        tol=1e-6,
    )
    clf.fit(D, target, sample_weight=weight)
    return np.asarray(clf.coef_).ravel()


def metal_onehot(metals: np.ndarray) -> sparse.csr_matrix:
    idx = np.array([METALS.index(m) if m in METALS else -1 for m in metals])
    rows = np.arange(len(idx))
    ok = idx >= 0
    data = np.ones(int(ok.sum()))
    return sparse.csr_matrix(
        (data, (rows[ok], idx[ok])), shape=(len(idx), len(METALS))
    )


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_pair_wilson(obs: np.ndarray, pred: np.ndarray, label: np.ndarray) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name in sorted(set(label.tolist())):
        m = label == name
        if m.sum() == 0:
            continue
        correct = int(np.sum((pred[m] > 0) == (obs[m] > 0)))
        lo, hi = wilson_ci(correct, int(m.sum()))
        out[name] = {
            "n": int(m.sum()),
            "correct": correct,
            "raw_accuracy": round(correct / int(m.sum()), 4),
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
        }
    return out


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    ap.add_argument(
        "--metal-desc", type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    ap.add_argument(
        "--metal-graph", type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_graph.json",
    )
    ap.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_ranking_hierarchical.json",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED, 42, 43, 44, 49])
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty for the BT arms")
    ap.add_argument("--skip-lambdarank", action="store_true", help="smoke: BT arms only")
    args = ap.parse_args(argv)

    for p in (args.long, args.metal_desc, args.metal_graph):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    t0 = time.time()
    long = pd.read_parquet(args.long)
    frame, X, y = build_features(long, args.metal_desc)
    metals = frame["metal"].astype(str).to_numpy()

    edges = json.loads(args.metal_graph.read_text())["edges_by_pair"]
    tau = tau_lookup(edges)

    pairs = pair_table(frame, key=("ligand_canonical", "cell_line_key")).reset_index(drop=True)
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    delta = y[ia] - y[ib]
    tau_p = pair_taus(pairs, tau)
    keep, weight = deadzone_labels(delta, tau_p)
    pairs["tau"] = tau_p
    pairs["delta"] = delta
    pairs["keep"] = keep
    pairs["weight"] = weight
    pairs["label"] = pairs["metal_a"] + "-" + pairs["metal_b"]

    retained = pairs[pairs["keep"]]
    n_retained_scored = int((retained["delta"].abs() >= GATE_THRESHOLD).sum())
    n_scored = int((pairs["delta"].abs() >= GATE_THRESHOLD).sum())
    print(
        f"[UP-23] pairs={len(pairs)} kept={int(keep.sum())} "
        f"scored={n_scored} scored_kept={n_retained_scored}"
    )

    doi = frame["doi"].astype(str).to_numpy()
    folds = np.full(len(frame), -1, dtype=int)
    for k, (_, te) in enumerate(
        GroupKFold(n_splits=N_FOLDS).split(np.zeros((len(frame), 1)), groups=doi)
    ):
        folds[te] = k

    # ---------------- lambdarank arms ----------------
    preds: Dict[str, np.ndarray] = {}
    label_info: Dict[str, dict] = {}
    if not args.skip_lambdarank:
        for arm, lab in (
            ("lambdarank_none", None),
            ("lambdarank_deadzone", "tie"),
        ):
            if lab is None:
                order, labels, sizes = rank_labels(frame, y)
            else:
                order, labels, sizes = tie_labels(frame, y, tau)
            oof = np.full((len(frame), len(args.seeds)), np.nan)
            for j, seed in enumerate(args.seeds):
                scores = np.full(len(frame), np.nan)
                for k in range(N_FOLDS):
                    tr = folds != k
                    te_sorted = (~tr)[order]
                    s = fit_fold(X, labels, sizes, order, tr[order], seed)
                    scores[order[te_sorted]] = s[te_sorted]
                oof[:, j] = scores
            preds[arm] = -np.nanmean(oof, axis=1)  # negate: higher score = more potent
            starts = np.concatenate([[0], np.cumsum(sizes)])
            n_flat = 0
            for g in range(len(sizes)):
                blk = labels[starts[g]:starts[g + 1]]
                if len(blk) > 1 and len(np.unique(blk)) == 1:
                    n_flat += 1
            label_info[arm] = {
                "n_label_levels": int(len(np.unique(labels))),
                "n_groups_all_tied": n_flat,
                "n_groups_multi_metal": int((sizes > 1).sum()),
            }
            print(f"  {arm}: label levels={label_info[arm]['n_label_levels']} "
                  f"all-tied groups={n_flat}")

    # ---------------- Bradley-Terry arms ----------------
    X_h = sparse.hstack([X, metal_onehot(metals)], format="csr")
    # binary target: 1 when idx_a is the more potent row (lower y).  Exact
    # ties carry no ordering information, so they are given zero weight rather
    # than a third class (which would make this a multiclass problem).
    pair_target = (delta < 0).astype(float)
    tie_w = (delta != 0).astype(float)

    bt_arms = {
        "bt_none": dict(design=X, keep=np.ones(len(pairs), dtype=bool),
                        weight=np.ones(len(pairs)), hier=False),
        "bt_exclude": dict(design=X, keep=keep, weight=np.ones(len(pairs)), hier=False),
        "bt_weight": dict(design=X, keep=np.ones(len(pairs), dtype=bool),
                          weight=weight, hier=False),
        "bt_hier": dict(design=X_h, keep=np.ones(len(pairs), dtype=bool),
                        weight=weight, hier=True),
    }
    bt_meta: Dict[str, dict] = {}
    for arm, spec in bt_arms.items():
        Xd = spec["design"]
        oof = np.full((len(frame), len(args.seeds)), np.nan)
        for j, seed in enumerate(args.seeds):
            scores = np.zeros(len(frame))
            hits = np.zeros(len(frame))
            for k in range(N_FOLDS):
                # a pair is trainable only if both rows are in training folds
                tr = (folds[ia] != k) & (folds[ib] != k) & spec["keep"]
                if tr.sum() == 0:
                    continue
                D = pair_design(Xd, ia[tr], ib[tr])
                w = fit_bt(D, pair_target[tr], spec["weight"][tr] * tie_w[tr], args.alpha)
                te = folds == k
                scores[te] += Xd[te] @ w
                hits[te] += 1
            scores = np.divide(scores, np.maximum(hits, 1))
            scores[hits == 0] = np.nan
            oof[:, j] = scores
        preds[arm] = -np.nanmean(oof, axis=1)
        # refit on every pair for the interpretable coefficients
        D_all = pair_design(Xd, ia, ib)
        w_all = fit_bt(D_all, pair_target, spec["weight"] * tie_w, args.alpha)
        meta = {"n_train_pairs_full": int(spec["keep"].sum())}
        if spec["hier"]:
            meta["metal_effects"] = {
                m: round(float(w_all[len(w_all) - len(METALS) + i]), 4)
                for i, m in enumerate(METALS)
            }
        bt_meta[arm] = meta
        print(f"  {arm}: fitted ({spec['keep'].sum()} pairs, hier={spec['hier']})")

    # ---------------- score every arm ----------------
    ligands = frame["ligand_canonical"].astype(str).to_numpy()
    results: Dict[str, dict] = {}
    for arm, pred in preds.items():
        full = score_pairs(y, pred, pairs, folds, GATE_THRESHOLD, ligands)
        sub = score_pairs(
            y, pred, retained.reset_index(drop=True), folds, GATE_THRESHOLD, ligands
        )
        block = {"all_pairs": full, "retained": sub}
        # per-metal-pair Wilson on the retained subset (what the gate reads)
        ia_r = retained["idx_a"].to_numpy()
        ib_r = retained["idx_b"].to_numpy()
        obs_r = y[ib_r] - y[ia_r]
        d_r = pred[ib_r] - pred[ia_r]
        sel = np.abs(obs_r) >= GATE_THRESHOLD
        block["wilson_retained"] = per_pair_wilson(
            obs_r[sel], d_r[sel], retained["label"].to_numpy()[sel]
        )
        results[arm] = block
        o_all = full.get("overall", {})
        o_ret = sub.get("overall", {})
        print(
            f"[{arm}] all n={o_all.get('n_pairs')} bal={o_all.get('balanced_accuracy')} "
            f"| retained n={o_ret.get('n_pairs')} bal={o_ret.get('balanced_accuracy')}"
        )

    # ---------------- gate ----------------
    def bal(arm: str, which: str = "retained") -> float:
        v = results[arm][which].get("overall", {}).get("balanced_accuracy")
        return float(v) if v is not None else float("-inf")

    best_arm = max(results, key=lambda a: bal(a, "retained"))
    n_pairs_lower = sum(
        1 for v in results[best_arm]["wilson_retained"].values() if v["wilson_lo"] > 0.5
    )
    gate_block = {
        "frozen_at": "2026-10-22T12:42+08:00",
        "checks": {
            "balanced_ge_0.65": {
                "arm": best_arm,
                "value": bal(best_arm, "retained"),
                "threshold": GATE_BALANCED,
                "baseline_up2b": 0.5985,
                "pass": bool(bal(best_arm, "retained") >= GATE_BALANCED),
            },
            "per_pair_wilson_lo_gt_0.5": {
                "arm": best_arm,
                "n_pairs_below_or_equal": n_pairs_lower,
                "required": 2,
                "pass": bool(n_pairs_lower >= 2),
            },
            "n_pairs_retained_ge_240": {
                "value": n_retained_scored,
                "value_full_pair_table": int(keep.sum()),
                "threshold": GATE_MIN_RETAINED,
                "pass": bool(n_retained_scored >= GATE_MIN_RETAINED),
            },
        },
    }
    gate_block["gate_pass"] = all(v["pass"] for v in gate_block["checks"].values())
    gate_block["kill"] = {
        "rule": "balanced < 0.62 on the retained subset -> ranking line closed (DP-8 yes)",
        "value": bal(best_arm, "retained"),
        "triggered": bool(bal(best_arm, "retained") < 0.62),
    }

    out = {
        "artifact": "msrs_ranking_hierarchical",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 ranking / UP-23",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "long": str(args.long),
            "metal_graph": str(args.metal_graph),
            "seeds": list(args.seeds),
            "argv": [sys.executable] + (argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "device": "cpu",
            "device_note": "LightGBM on 21k rows is faster on CPU than on GPU at this size; "
                           "no CUDA context is created.",
            "group": list(GROUP_KEY),
            "cv": "GroupKFold(5) on DOI, same partition as UP-2b",
            "gate_threshold": GATE_THRESHOLD,
            "alpha": args.alpha,
            "deadzone_tau_source": "msrs_metal_graph.json:/edges_by_pair/*/noise_sd_delta",
            "deadzone_tau": {name: tau[frozenset(name.split("-"))] for name in edges},
            "n_pairs": int(len(pairs)),
            "n_pairs_kept": int(keep.sum()),
            "n_pairs_scored": n_scored,
            "n_pairs_scored_kept": n_retained_scored,
            "retained_by_metal_pair": {
                str(k): {
                    "n": int(len(v)),
                    "kept": int(v["keep"].sum()),
                    "tau": round(float(v["tau"].iloc[0]), 4),
                }
                for k, v in pairs.groupby("label")
            },
            "label_info": label_info,
            "bt_meta": bt_meta,
        },
        "arms": results,
        "gate": gate_block,
        "verdict": "PASS" if gate_block["gate_pass"] else "FAIL",
        "caveats": [
            "A pair whose |delta| is below the noise SD is not known to be a tie — it is "
            "known to be unresolvable. Dropping it from training is the right move; dropping "
            "it from *evaluation* would be measuring on the easy subset, which is why both "
            "the full 360 and the retained subset are reported for every arm.",
            "The BT arms see only 651 pairs (520 per training fold), against 21k rows for "
            "lambdarank. A weaker BT number is a sample-size statement, not an architecture one.",
            "Cross-paper pairs (the ligand_cell key) compare out-of-fold scores from two "
            "different models with arbitrary relative scale; UP-2b already flags this. It "
            "applies to every arm here equally.",
            "The metal graph is a tree on 4 observed nodes, so the additive (hier) model is "
            "saturated: u_Ru - u_Rh reproduces the Ru-Rh mean delta exactly. 'Borrowing "
            "strength' has no residual degrees of freedom to work with; see UP-18.",
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[UP-23] verdict={out['verdict']} best={best_arm} gate={gate_block['checks']}")
    print(f"json: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
