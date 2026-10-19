#!/usr/bin/env python3
"""MET-5 — decision utility curves: at what coverage is the model useful?

MET-2/MET-3 established a single operating point: mean within-cohort regret
0.1191 vs 0.1198 for ``global_best``, not significant.  A single point cannot
answer the question a user actually asks — *may I abstain, and how much does
accuracy improve on the cases I keep?*  This script sweeps coverage.

Method
------
* Cohorts are built exactly as in MET-2 (one row per cohort x metal, replicates
  pooled by median, cohorts with >= 2 metals).
* Confidence is the **top-1 margin**: the gap between the best and second-best
  predicted potency.  Larger gap = more confident that the pick is distinct.
* For coverage q in 0.1..1.0, keep the top-q fraction of cohorts by model
  margin and score model / ``global_best`` / ``cell_best`` **on the same
  subset** (paired).
* Net benefit uses the standard decision-curve form with a hit defined as
  ``regret <= NOISE_FLOOR`` (the chosen metal is within measurement noise of
  the observed best):

      NB(p_t) = (TP - FP * p_t / (1 - p_t)) / N

* Significance: DOI-clustered one-sided t-test of the paired regret delta
  (model vs ``global_best``), Holm-corrected across the coverage grid.

Acceptance: at coverage 0.5, mean regret <= 0.10 and Holm p < 0.05.

Output: ``metrics/by_metric/msrs_decision_curve.json``.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_decision_curve.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_generalization import scheme_folds  # noqa: E402
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    ENDPOINT,
    SEED,
)
from molmetal.scripts.msrs_recommendation import (  # noqa: E402
    MIN_COHORT_SIZE,
    build_cohorts,
    cohort_metrics,
    doi_clustered_test,
    fold_baseline_scores,
    summarise_cohorts,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    NOISE_FLOOR,
    _git_commit,
    _sha256,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi")
ARM = "ligand+both"
COVERAGES = tuple(round(0.1 * i, 1) for i in range(1, 11))
THRESHOLDS = tuple(round(0.1 * i, 1) for i in range(1, 10))
REGRET_GATE = 0.10
COVERAGE_GATE = 0.5
ALPHA = 0.05
N_BOOT = 2000
SCORERS = ("model", "global_best", "cell_best")


def top1_margin(pred: np.ndarray) -> float:
    """Best minus second-best predicted potency (lower is better)."""
    p = np.sort(np.asarray(pred, dtype=float))
    if len(p) < 2:
        return 0.0
    return float(p[1] - p[0])


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return [float(v) for v in adj]


def net_benefit(hit: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    """Standard decision-curve net benefit; ``hit`` is regret <= NOISE_FLOOR."""
    n = len(hit)
    tp = float(hit.sum())
    fp = float(n - tp)
    out: Dict[str, float] = {}
    for t in thresholds:
        if t >= 1.0:
            continue
        out[f"{t:g}"] = round((tp - fp * t / (1.0 - t)) / n, 4)
    return out


def build_records(
    sub: pd.DataFrame,
    y: np.ndarray,
    pred_model: np.ndarray,
    pred_global: np.ndarray,
    pred_cell: np.ndarray,
    fold: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cohort": pd.factorize(
                sub["ligand_canonical"].astype(str)
                + "|"
                + sub["cell_line_key"].astype(str)
                + "|"
                + sub["doi"].astype(str)
                + "|"
                + sub["time_h"].astype(str)
            )[0],
            "metal": sub["metal"].astype(str).to_numpy(),
            "y": y,
            "doi": sub["doi"].astype(str).to_numpy(),
            "fold": fold,
            "model": pred_model,
            "global_best": pred_global,
            "cell_best": pred_cell,
        }
    )
    frame = frame[frame["fold"] >= 0].reset_index(drop=True)
    cohorts = build_cohorts(frame, scorers=list(SCORERS), min_size=MIN_COHORT_SIZE)

    records = []
    for cid, g in cohorts.groupby("cohort", sort=False):
        for scorer in SCORERS:
            rec = cohort_metrics(g["y"].to_numpy(), g[scorer].to_numpy())
            rec.update(
                {
                    "cohort": int(cid),
                    "scorer": scorer,
                    "doi": g["doi"].iloc[0],
                    "margin": top1_margin(g[scorer].to_numpy()),
                    "n_metals": int(g["metal"].nunique()),
                }
            )
            records.append(rec)
    return pd.DataFrame(records)


def run_protocol(
    sub: pd.DataFrame,
    y: np.ndarray,
    pred_model: np.ndarray,
    protocol: str,
    seed: int,
) -> Dict[str, object]:
    t0 = time.time()
    fold, _, info = scheme_folds(sub, protocol)
    metals = sub["metal"].astype(str).to_numpy()
    cells = sub["cell_line_key"].astype(str).to_numpy()
    pred_global, pred_cell, base_info = fold_baseline_scores(metals, cells, y, fold)

    rec = build_records(sub, y, pred_model, pred_global, pred_cell, fold)
    n_cohorts = int(rec["cohort"].nunique())
    model_rec = rec[rec["scorer"] == "model"].set_index("cohort")

    curve: List[Dict[str, object]] = []
    pvals: List[float] = []
    for q in COVERAGES:
        n_keep = max(1, int(np.ceil(q * n_cohorts)))
        keep = model_rec["margin"].sort_values(ascending=False).head(n_keep).index
        subset = rec[rec["cohort"].isin(keep)]
        row: Dict[str, object] = {
            "coverage": q,
            "n_cohorts": int(n_keep),
            "frac": round(n_keep / n_cohorts, 4),
        }
        for scorer in SCORERS:
            r = subset[subset["scorer"] == scorer]
            s = summarise_cohorts(r)
            row[scorer] = {
                "mean_regret": s["mean_regret"],
                "top1_accuracy": s["top1_accuracy"],
                "mean_ndcg": s["mean_ndcg"],
                f"hit@{NOISE_FLOOR:g}": s[f"hit@{NOISE_FLOOR:g}"],
                "net_benefit": net_benefit(
                    r[f"hit@{NOISE_FLOOR:g}"].to_numpy(dtype=bool), THRESHOLDS
                ),
            }
        test = doi_clustered_test(subset, "model", "global_best", n_boot=N_BOOT, seed=seed)
        row["model_vs_global_best"] = test
        pvals.append(float(test.get("p_one_sided_doi", 1.0)))
        curve.append(row)

    adjusted = holm(pvals)
    for row, p_adj in zip(curve, adjusted):
        row["model_vs_global_best"]["p_holm"] = round(p_adj, 4)

    full = rec[rec["scorer"] == "model"]
    gate_row = next(r for r in curve if abs(r["coverage"] - COVERAGE_GATE) < 1e-9)
    gate_regret = float(gate_row["model"]["mean_regret"])
    gate_p = float(gate_row["model_vs_global_best"].get("p_holm", 1.0))
    passed = bool(gate_regret <= REGRET_GATE and gate_p < ALPHA)

    return {
        "fold_info": info,
        "baselines": base_info,
        "n_cohorts": n_cohorts,
        "cohort_size": {str(k): int(v) for k, v in rec.groupby("cohort")["n_metals"].first().value_counts().items()},
        "full_coverage": {
            scorer: summarise_cohorts(rec[rec["scorer"] == scorer]) for scorer in SCORERS
        },
        "full_coverage_net_benefit": {
            scorer: net_benefit(
                rec[(rec["scorer"] == scorer)][f"hit@{NOISE_FLOOR:g}"].to_numpy(dtype=bool),
                THRESHOLDS,
            )
            for scorer in SCORERS
        },
        "curve": curve,
        "gate": {
            "coverage": COVERAGE_GATE,
            "mean_regret": round(gate_regret, 4),
            "p_holm": round(gate_p, 4),
            "passed": passed,
        },
        "margin_distribution": {
            scorer: {
                "mean": round(float(rec[rec["scorer"] == scorer]["margin"].mean()), 4),
                "median": round(float(rec[rec["scorer"] == scorer]["margin"].median()), 4),
                "p10": round(float(rec[rec["scorer"] == scorer]["margin"].quantile(0.10)), 4),
                "p90": round(float(rec[rec["scorer"] == scorer]["margin"].quantile(0.90)), 4),
            }
            for scorer in SCORERS
        },
        "wall_s": round(time.time() - t0, 1),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_generalization" / "oof.npz",
    )
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_decision_curve.json",
    )
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    data = np.load(args.oof, allow_pickle=False)
    y = data["y"].astype(float)
    sub = pd.DataFrame(
        {
            "ligand_canonical": data["ligand_canonical"].astype(str),
            "cell_line_key": data["cell_line_key"].astype(str),
            "doi": data["doi"].astype(str),
            "time_h": data["time_h"].astype(float),
            "metal": data["metal"].astype(str),
        }
    )

    protocols: Dict[str, object] = {}
    for protocol in args.protocols:
        key = f"{protocol}__{ARM}"
        if key not in data:
            protocols[protocol] = {"status": "MISSING_OOF", "key": key}
            continue
        protocols[protocol] = run_protocol(
            sub, y, data[key].astype(float), protocol, args.seed
        )
        g = protocols[protocol]["gate"]
        print(
            f"[met5] {protocol:7s} cohorts={protocols[protocol]['n_cohorts']} "
            f"q=0.5 regret={g['mean_regret']} p_holm={g['p_holm']} "
            f"passed={g['passed']}"
        )

    primary = protocols.get("doi", {}).get("gate", {})
    verdict = "PASS" if primary.get("passed") else "NEGATIVE"

    payload = {
        "artifact": "msrs_decision_curve",
        "version": "1.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "oof": str(args.oof),
            "oof_sha256": _sha256(args.oof),
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "split_schema": "consumes MET-2's out-of-fold predictions; two protocols, so the dual-split gate does not apply",
            "arm": ARM,
            "protocols": list(args.protocols),
            "primary_protocol": "doi",
            "confidence": "top-1 margin = second-best minus best predicted potency",
            "coverages": list(COVERAGES),
            "thresholds": list(THRESHOLDS),
            "hit_definition": f"regret <= {NOISE_FLOOR}",
            "net_benefit": "NB = (TP - FP * p_t/(1-p_t)) / N",
            "min_cohort_size": MIN_COHORT_SIZE,
            "n_boot": N_BOOT,
            "holm": "across the 10 coverage levels, per protocol",
            "seed": int(args.seed),
            "gate": f"coverage {COVERAGE_GATE}: mean regret <= {REGRET_GATE} AND Holm p < {ALPHA}",
        },
        "protocols": protocols,
        "verdict": verdict,
        "verdict_rule": (
            f"DOI protocol, coverage {COVERAGE_GATE}: mean regret <= {REGRET_GATE} "
            f"AND Holm-adjusted one-sided p < {ALPHA} vs global_best"
        ),
        "gate_measurement": primary,
    }
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[met5] verdict={verdict} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
