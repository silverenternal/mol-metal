#!/usr/bin/env python3
"""STR-5 — the delta as a direct target, read out of the FIX-M4 pair model.

Why this is a re-read and not a new model
=========================================
STR-5's plan ("antisymmetric pair features, regress the observed delta, DOI
grouped, capacity matched") is what ``msrs_pairwise_reranker.py`` already
trains as the ``pair_full`` arm: ridge on

    d(metal one-hot) + d(state) + d(coordination) + ligand_context (x) d(one-hot)

fitted on within-cohort ordered pairs, with the target being ``y[b] - y[a]``.
Every cohort in this corpus has k = 2 metals, so a row-potential model and a
direct pair regression are the same object here.  Training a second model
would be a rerun, not a new test.

What was missing is the readout: FIX-M4 reported ``pairwise_accuracy`` over
all 834 ordered pairs and mean regret.  The STR-5 gate is defined on the
218 gated pairs (|delta| >= 0.2091) with a DOI-clustered interval, which was
never computed.  This script computes exactly that, from the per-pair
predictions dumped by ``--pair-oof-out``.

Capacity matching: the pair model is linear with alpha chosen on an inner
DOI-grouped split, fitted on ~850 ordered pairs.  It cannot out-spend the
main LightGBM; that is the strongest form of the constraint, not a weaker one.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_pairwise_reranker.py \\
        --pair-oof-out results/msrs_pairwise_reranker/pair_oof_doi.npz
    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_delta_direct.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_delta_metrics import GATE_THRESHOLD  # noqa: E402
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SIGN_GATE = 0.65
CI_LOWER_GATE = 0.55
SIGN_KILL = 0.62
MAE_KILL = 0.50
N_BOOT = 2000
BOOT_SEED = 20261020


def canonical_pairs(
    pair_a: np.ndarray, pair_b: np.ndarray, metal: np.ndarray
) -> np.ndarray:
    """Keep one orientation per unordered pair: lower metal rank in ``a``.

    Same convention as ``pair_table`` (``idx_a`` = lower ranked metal), so the
    deltas here are directly comparable to the level model's.
    """
    rank = {m: i for i, m in enumerate(sorted(set(metal.tolist())))}
    ma = np.array([rank[m] for m in metal[pair_a]])
    mb = np.array([rank[m] for m in metal[pair_b]])
    return ma < mb


def delta_stats(obs: np.ndarray, pred: np.ndarray, gate: Optional[float]) -> Dict[str, object]:
    keep = np.isfinite(obs) & np.isfinite(pred)
    if gate is not None:
        keep &= np.abs(obs) >= gate
    if not keep.any():
        return {"n_pairs": 0}
    o, p = obs[keep], pred[keep]
    slope, intercept = (np.nan, np.nan)
    r2 = np.nan
    if np.std(p) > 0:
        slope, intercept = np.polyfit(p, o, 1)
        resid = o - (slope * p + intercept)
        r2 = 1.0 - float(np.sum(resid**2) / np.sum((o - o.mean()) ** 2))
    return {
        "n_pairs": int(keep.sum()),
        "sign_accuracy": round(float(np.mean(np.sign(p) == np.sign(o))), 4),
        "mae_delta": round(float(np.mean(np.abs(p - o))), 4),
        "rmse_delta": round(float(np.sqrt(np.mean((p - o) ** 2))), 4),
        "bias_delta": round(float(np.mean(p - o)), 4),
        "mean_abs_observed_delta": round(float(np.mean(np.abs(o))), 4),
        "mean_abs_predicted_delta": round(float(np.mean(np.abs(p))), 4),
        "calibration": {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "r2": round(float(r2), 4),
        },
    }


def cluster_bootstrap(
    obs: np.ndarray,
    pred: np.ndarray,
    clusters: np.ndarray,
    gate: float,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Dict[str, object]:
    """Resample whole DOIs, recompute the gated sign / MAE each time."""
    keep = np.isfinite(obs) & np.isfinite(pred) & (np.abs(obs) >= gate)
    o, p, c = obs[keep], pred[keep], clusters[keep]
    uniq = np.unique(c)
    index = {u: np.where(c == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    signs, maes = [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[d] for d in draw])
        signs.append(float(np.mean(np.sign(p[rows]) == np.sign(o[rows]))))
        maes.append(float(np.mean(np.abs(p[rows] - o[rows]))))
    return {
        "n_dois": int(len(uniq)),
        "n_boot": int(n_boot),
        "sign_accuracy": {
            "mean": round(float(np.mean(signs)), 4),
            "ci95": [round(float(np.percentile(signs, 2.5)), 4), round(float(np.percentile(signs, 97.5)), 4)],
        },
        "mae_delta": {
            "mean": round(float(np.mean(maes)), 4),
            "ci95": [round(float(np.percentile(maes, 2.5)), 4), round(float(np.percentile(maes, 97.5)), 4)],
        },
    }


def baseline(path: Path) -> Optional[Dict[str, object]]:
    """The level model's delta numbers on the same gated pair set."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    doi = data.get("protocols", {}).get("doi", {}).get("ensemble", {}).get("delta", {})
    entry = doi.get(str(GATE_THRESHOLD))
    if entry is None:
        return None
    return {
        "source": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
        "n_pairs": entry.get("n_pairs"),
        "sign_accuracy": entry.get("sign_accuracy"),
        "sign_ci95": entry.get("bootstrap", {}).get("sign_accuracy", {}).get("ci95"),
        "mae_delta": entry.get("mae_delta"),
        "mae_ci95": entry.get("bootstrap", {}).get("mae_delta", {}).get("ci95"),
        "bias_delta": entry.get("bias_delta"),
        "calibration_slope": entry.get("calibration", {}).get("slope"),
        "mean_abs_predicted_delta": entry.get("mean_abs_predicted_delta"),
        "model": "seed-averaged LightGBM on absolute potency, delta taken as a difference",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_pairwise_reranker" / "pair_oof_doi.npz",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_seed_ensemble.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_delta_direct.json",
    )
    args = parser.parse_args(argv)

    with np.load(args.pairs, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    pair_a = arrays["pair_a"].astype(int)
    pair_b = arrays["pair_b"].astype(int)
    metal = arrays["metal"].astype(str)
    y = arrays["y"].astype(float)
    doi = arrays["doi"].astype(str)
    fold = arrays["fold"].astype(int)

    keep = canonical_pairs(pair_a, pair_b, metal)
    a, b = pair_a[keep], pair_b[keep]
    obs = y[b] - y[a]
    pred = arrays["pair_pred"][keep]
    clusters = doi[a]
    n_cohorts = len(np.unique(arrays["cohort"]))

    gated = delta_stats(obs, pred, GATE_THRESHOLD)
    ungated = delta_stats(obs, pred, None)
    boot = cluster_bootstrap(obs, pred, clusters, GATE_THRESHOLD)
    base = baseline(args.baseline)

    sign = gated.get("sign_accuracy")
    ci_low = boot["sign_accuracy"]["ci95"][0]
    passed = sign is not None and sign >= SIGN_GATE and ci_low > CI_LOWER_GATE
    killed = (sign is not None and sign < SIGN_KILL) or (
        gated.get("mae_delta") is not None and gated["mae_delta"] >= MAE_KILL
    )
    verdict = "PASS" if passed else ("KILL" if killed else "PARTIAL")

    payload = {
        "artifact": "msrs_delta_direct",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_delta_direct.py",
            "git_commit": _git_commit(),
            "pair_oof": str(args.pairs.relative_to(PROJECT_ROOT)) if args.pairs.is_relative_to(PROJECT_ROOT) else str(args.pairs),
            "pair_model": "molmetal/scripts/msrs_pairwise_reranker.py :: pair_full",
        },
        "design": {
            "protocol": "doi (5 folds, whole papers held out)",
            "model": "ridge on antisymmetric pair differences; target = observed delta",
            "capacity_match": "linear, ~850 ordered training pairs, alpha by inner DOI-grouped CV; cannot out-spend the main model",
            "pair_orientation": "lower metal rank in a (same convention as pair_table)",
            "n_cohorts": int(n_cohorts),
            "n_unordered_pairs": int(len(obs)),
            "gate_threshold": GATE_THRESHOLD,
            "sign_gate": SIGN_GATE,
            "ci_lower_gate": CI_LOWER_GATE,
            "sign_kill": SIGN_KILL,
            "mae_kill": MAE_KILL,
            "bootstrap": {"cluster": "doi", "n_boot": N_BOOT, "seed": BOOT_SEED},
        },
        "gated": {**gated, "bootstrap": boot},
        "ungated": ungated,
        "baseline_level_model": base,
        "verdict": verdict,
        "verdict_detail": {
            "sign": sign,
            "sign_ci95": boot["sign_accuracy"]["ci95"],
            "sign_gate": bool(sign is not None and sign >= SIGN_GATE),
            "ci_gate": bool(ci_low > CI_LOWER_GATE),
            "mae_delta": gated.get("mae_delta"),
            "mae_gate": bool(gated.get("mae_delta") is not None and gated["mae_delta"] < MAE_KILL),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    print(
        f"[str5] pairs={len(obs)} gated={gated['n_pairs']} "
        f"sign={sign} ci={boot['sign_accuracy']['ci95']} "
        f"mae={gated['mae_delta']} bias={gated['bias_delta']} "
        f"slope={gated['calibration']['slope']}"
    )
    if base:
        print(f"[str5] level baseline: sign={base['sign_accuracy']} mae={base['mae_delta']}")
    print(f"[str5] verdict={verdict} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
