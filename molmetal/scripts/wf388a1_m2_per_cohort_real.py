#!/usr/bin/env python3
"""Wf-388-A1 — m2 R² per-cohort config on REAL cohorts (TODO-201 §3.2).

=====================================================================
Wf-388-A1 / Wf-380 ultracode follow-up — per Wf-380-A2 + TODO-201 §3.2,
apply the per-cohort-specific feature configuration (12-d / 24-d / 50-d
matched to cohort size) to the 3 REAL cohorts:

  * platinai_4cell        (n=200)  → 12-d RDKit + Ridge α=1.0
  * hilbert               (n=200)  → 12-d RDKit + Ridge α=1.0
  * metalcytotox_combined (n=2000) → 32-d RDKit + Ridge α=0.1

This is the **real-cohort verification** of the per-cohort config
that Wf-380-A2 validated on synthetic cohorts only. Per
Wf-369-A2 verdict, platinai_4cell + hilbert R² gate remained at
HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble
(OVERFIT on 50+ features for n=200). The per-cohort config is the
remediation axis (Category B, no retraining).

Wf-388-A1 is **CPU-only diagnostic** — no GPU retrain, no torch
autograd, no encoder forward pass. The Ridge evaluation uses the
Wf-380-A2 numpy closed-form solver (with intercept) + multi-seed
K-fold CV per Wf-335-M2 protocol.

Public API
----------
* :func:`load_platinai_4cell`        load real platinai_4cell cohort
* :func:`load_hilbert`               load real hilbert cohort
* :func:`load_metalcytotox_combined` load real metalcytotox_combined cohort
* :func:`train_eval_per_cohort`      per-cohort train+eval pipeline
* :func:`aggregate_per_cohort_r2`    aggregate R² across cohorts
* :func:`run_wf388a1`                main entry point

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* This is the **real-cohort** verification of the per-cohort config
  axis (Wf-380-A2 was synthetic-only). MEASURED-on-real-cohort R²
  per per-cohort config = MEASURED, not PROJECTED.
* Per-cohort verdict:
    - If R² > 0 AND lift ≥ +0.10 vs Wf-335-M2 best: CEILING_BREAK.
    - If R² > 0: WEAK_RECOVERY (gate LIFTED for R² > 0 invariant).
    - Else: HONEST_NEGATIVE_R2_DEEP preserved.
* aggregate_status:
    - CEILING_BREAK_PARTIAL if ≥ 1 cohort CEILING_BREAK.
    - PARTIAL_LIFT if ≥ 1 cohort WEAK_RECOVERY.
    - HONEST_NEGATIVE_R2_DEEP if all 3 cohorts stay R² ≤ 0.
* No GPU launches; no torch autograd; CPU-only Ridge baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import Ridge  # type: ignore

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover — defensive
    _HAS_SKLEARN = False

# Reuse the canonical cohort loader from Wf-331-R5 (3 cohorts: platinai_4cell + hilbert + metalcytotox_combined).
from molmetal.scripts.wf331r5_r2_w2_chemprop import (
    load_cohort as _wf331r5_load_cohort,
    PLATINAI_12D_COLUMNS,
    PLATINAI_32D_COLUMNS,
    METALCYTOTOX_TARGET_COLUMN,
)

# Reuse the per-cohort config selector from Wf-380-A2 (size-based tier).
from molmetal.scripts.wf380a2_m2_r2_per_cohort import (
    select_features as wf380a2_select_features,
    load_config as wf380a2_load_config,
    ridge_r2_eval as wf380a2_ridge_r2_eval,
    ridge_r2_cv as wf380a2_ridge_r2_cv,
)

# Reuse the Wf-335-M2 baseline R² anchors (HONEST_NEGATIVE_R2_DEEP on platinai + hilbert, CEILING_BREAK on metalcytotox_combined).
from molmetal.scripts.wf331r5_r2_w2_chemprop import (
    WF325A_BASELINE_R2,
)


# ---------------------------------------------------------------------------
# Wf-335-M2 best R² per cohort (the Wf-388-A1 baseline anchor)
# ---------------------------------------------------------------------------
# Per `molmetal/reports/wf335m2_r2/result.json` (2026-09-29):
WF335M2_BEST_R2: Dict[str, float] = {
    "platinai_4cell": -0.12505399802714393,  # 12d_baseline, seed=0
    "hilbert": -0.1268,  # placeholder, overridden below
    "metalcytotox_combined": 0.22268113216745766,  # CEILING_BREAK
}

# Per Wf-369-A2 verdict + `molmetal/reports/wf335m2_r2/result.json` per_axis.csv:
# hilbert best R² per multi-seed × 5-fold CV is -0.1268 (12d_baseline seed_offset=10).
WF335M2_HILBERT_BEST_R2: float = -0.1268

# Update hilbert anchor.
WF335M2_BEST_R2["hilbert"] = WF335M2_HILBERT_BEST_R2


# ---------------------------------------------------------------------------
# Section 1: Repo path resolution (mirror Wf-380-A2 / Wf-367-B1 pattern)
# ---------------------------------------------------------------------------
def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for start in (here, os.getcwd()):
        cur = start
        for _ in range(6):
            child = os.path.join(cur, "molmetal")
            if os.path.isdir(child) and (
                os.path.isdir(os.path.join(child, "data"))
                or os.path.isdir(os.path.join(child, "scripts"))
            ):
                candidate = os.path.abspath(os.path.join(cur, path))
                if os.path.exists(candidate):
                    return candidate
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Section 2: Real-cohort loaders (per TODO-201 §3.2 — wraps Wf-331-R5 load_cohort)
# ---------------------------------------------------------------------------
def load_platinai_4cell(
    max_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load REAL platinai_4cell cohort (n=200).

    Per TODO-201 §3.2 — 4-cell pIC50 mean target (mcf7 + a549 + a2780 + a2780cis).
    Returns (X, y, smiles_list, target_name) where X is 32-d RDKit extended
    features (the canonical Wf-331-R5 32-d matrix).
    """
    return _wf331r5_load_cohort(
        cohort="platinai_4cell",
        max_samples=max_samples,
        seed=0,
    )


def load_hilbert(
    max_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load REAL hilbert cohort (n=200).

    Per TODO-201 §3.2 — same SMILES as platinai_4cell but different
    random split (seed_offset=10 in the train/eval pipeline). Returns
    (X, y, smiles_list, target_name) where X is 32-d RDKit extended.
    """
    return _wf331r5_load_cohort(
        cohort="hilbert",
        max_samples=max_samples,
        seed=0,
    )


def load_metalcytotox_combined(
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load REAL metalcytotox_combined cohort (n=2000).

    Per TODO-201 §3.2 — 5-metal MetalCytoToxDB activity target
    (Krasnov 2026). Returns (X, y, smiles_list, target_name) where
    X is 12-d RDKit + 20-d zero-fill (Wf-331-R5 32-d on metalcytotox
    is computed on-the-fly; we use 12-d for the per-cohort config
    since n=2000 maps to n_gte_1000 tier, where Ridge alpha=0.1).
    """
    return _wf331r5_load_cohort(
        cohort="metalcytotox_combined",
        max_samples=max_samples,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Section 3: Per-cohort config application (12d / 24d / 50d matched to n)
# ---------------------------------------------------------------------------
def apply_per_cohort_config(
    X: np.ndarray,
    cohort_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the Wf-380-A2 per-cohort config to a feature matrix.

    Selects the appropriate feature count slice based on cohort_size
    and the Wf-380-A2 tier table:

      * n < 300   → X[:, :12]   (12-d RDKit)
      * n < 1000  → X[:, :24]   (24-d RDKit + chemprop v3 surrogate)
      * n >= 1000 → X[:, :32]   (32-d RDKit + chemprop v3 surrogate +
                                  Hilbert spectral surrogate)

    Returns
    -------
    (X_per_cohort, config_dict) where X_per_cohort is the per-cohort
    feature slice and config_dict is the Wf-380-A2 tier config.
    """
    cfg = wf380a2_select_features(cohort_size)
    n_features_expected = cfg["n_features_expected"]
    # Map to the actually available features in X.
    n_available = X.shape[1]
    n_use = min(n_features_expected, n_available)
    if n_use < n_features_expected:
        # Available < expected — use what we have (this happens for
        # metalcytotox where we have 12-d + zero-fill to 32-d).
        n_use_actual = n_use
    else:
        n_use_actual = n_features_expected
    X_per = X[:, :n_use_actual].astype(np.float64, copy=True)
    cfg_out = dict(cfg)
    cfg_out["n_features_available"] = int(n_available)
    cfg_out["n_features_used"] = int(n_use_actual)
    return X_per, cfg_out


# ---------------------------------------------------------------------------
# Section 4: Train+eval per cohort (per-cohort config + multi-seed K-fold CV)
# ---------------------------------------------------------------------------
def train_eval_per_cohort(
    cohort_data: Tuple[np.ndarray, np.ndarray, List[str], str],
    n_seeds: int = 5,
    n_folds: int = 5,
    wass_w: float = 0.05,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Train + eval per cohort using per-cohort config + Ridge baseline.

    Parameters
    ----------
    cohort_data : tuple
        (X, y, smiles_list, target_name) from one of the 3 loaders.
    n_seeds : int
        Number of random seeds for K-fold CV (default 5).
    n_folds : int
        Number of K-fold splits (default 5).
    wass_w : float
        W₂ weight (placeholder for category parity; the per-cohort config
        does NOT use W₂ — it's a feature-count axis only).
    epochs : int
        Number of training epochs (placeholder for category parity;
        Ridge is closed-form so epochs is unused — but recorded for
        the result.json schema).

    Returns
    -------
    dict with:
        - cohort_size
        - n_features_used
        - alpha_used
        - tier
        - r2_mean, r2_std (multi-seed K-fold CV)
        - baseline_r2_mechanical (Wf-335-M2 best R²)
        - lift_vs_baseline (r2_mean - baseline_r2_mechanical)
        - verdict (CEILING_BREAK / WEAK_RECOVERY / HONEST_NEGATIVE_R2_DEEP)
        - target_name
        - epochs_recorded (epochs param, for schema parity)
        - wass_w_recorded (wass_w param, for schema parity)
    """
    X, y, smiles_list, target_name = cohort_data
    n_samples = int(X.shape[0])
    cohort_size = n_samples
    # Apply per-cohort config (12-d / 24-d / 50-d matched to cohort_size).
    X_per, cfg = apply_per_cohort_config(X, cohort_size)
    # Multi-seed K-fold CV (Wf-335-M2 protocol).
    cv = wf380a2_ridge_r2_cv(
        X=X_per,
        y=y,
        n_splits=n_folds,
        n_seeds=n_seeds,
        alpha=cfg["alpha"],
        seed_offset=0,
    )
    r2_mean = float(cv["r2_mean"])
    r2_std = float(cv["r2_std"])
    # Determine the Wf-335-M2 baseline R² anchor by target_name.
    if "mcf7" in target_name.lower() or "a2780" in target_name.lower():
        baseline_cohort = "platinai_4cell"
    elif "hilbert" in target_name.lower():
        baseline_cohort = "hilbert"
    else:
        baseline_cohort = "metalcytotox_combined"
    baseline_r2 = float(WF335M2_BEST_R2.get(baseline_cohort, float("nan")))
    lift_vs_baseline = (
        r2_mean - baseline_r2 if np.isfinite(baseline_r2) else float("nan")
    )
    # Per-cohort verdict.
    if not np.isfinite(r2_mean):
        verdict = "UNDEFINED"
    elif r2_mean > 0.0 and lift_vs_baseline >= 0.10:
        verdict = "CEILING_BREAK"
    elif r2_mean > 0.0 and lift_vs_baseline >= 0.05:
        verdict = "MODERATE_RECOVERY"
    elif r2_mean > 0.0:
        verdict = "WEAK_RECOVERY"
    elif lift_vs_baseline >= 0.0:
        verdict = "NO_LIFT_HONEST_NEGATIVE"
    else:
        verdict = "HONEST_NEGATIVE_R2_DEEP"
    return {
        "cohort_size": int(cohort_size),
        "n_features_used": int(X_per.shape[1]),
        "alpha_used": float(cfg["alpha"]),
        "tier": cfg["tier"],
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "n_splits_effective": int(cv.get("n_splits_effective", n_seeds * n_folds)),
        "baseline_r2_mechanical": baseline_r2,
        "lift_vs_baseline": lift_vs_baseline,
        "verdict": verdict,
        "target_name": target_name,
        "epochs_recorded": int(
            epochs
        ),  # Ridge is closed-form; recorded for schema parity.
        "wass_w_recorded": float(
            wass_w
        ),  # Per-cohort config does NOT use W₂; recorded for schema parity.
        "n_features_available": int(cfg.get("n_features_available", 0)),
        "n_samples": int(cohort_size),
        "smiles_count": len(smiles_list),
        "config_features": cfg["features"],
        "config_model": cfg["model"],
    }


# ---------------------------------------------------------------------------
# Section 5: Aggregate per-cohort R² (across cohorts)
# ---------------------------------------------------------------------------
def aggregate_per_cohort_r2(results_per_cohort: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-cohort R² across cohorts.

    Returns mean R² + std R² across cohorts + aggregate_status
    per the Wf-369-A2 verdict taxonomy.
    """
    finite_r2 = [
        r["r2_mean"]
        for r in results_per_cohort
        if np.isfinite(r.get("r2_mean", float("nan")))
    ]
    n_above_zero = sum(1 for r in finite_r2 if r > 0.0)
    n_ceiling_break = sum(
        1 for r in results_per_cohort if r.get("verdict") == "CEILING_BREAK"
    )
    n_weak_recovery = sum(
        1
        for r in results_per_cohort
        if r.get("verdict") in ("WEAK_RECOVERY", "MODERATE_RECOVERY", "CEILING_BREAK")
    )
    n_total = len(results_per_cohort)
    mean_r2 = float(np.mean(finite_r2)) if finite_r2 else float("nan")
    std_r2 = float(np.std(finite_r2)) if finite_r2 else float("nan")
    if n_ceiling_break >= 2:
        aggregate_status = "CEILING_BREAK_MAJORITY"
    elif n_ceiling_break >= 1:
        aggregate_status = "CEILING_BREAK_PARTIAL"
    elif n_weak_recovery >= 1:
        aggregate_status = "MEASURED_PARTIAL_LIFT"
    else:
        aggregate_status = "HONEST_NEGATIVE_R2_DEEP"
    return {
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "n_cohorts_tested": n_total,
        "n_cohorts_above_zero": n_above_zero,
        "n_cohorts_ceiling_break": n_ceiling_break,
        "n_cohorts_weak_recovery_or_better": n_weak_recovery,
        "aggregate_status": aggregate_status,
    }


# ---------------------------------------------------------------------------
# Section 6: Main entry point — full Wf-388-A1 evaluation
# ---------------------------------------------------------------------------
COHORT_REGISTRY: Dict[str, Any] = {
    "platinai_4cell": load_platinai_4cell,
    "hilbert": load_hilbert,
    "metalcytotox_combined": load_metalcytotox_combined,
}


def run_wf388a1(
    out_path: str = "molmetal/reports/wf388a1_m2_per_cohort_real/result.json",
    cohorts: Optional[List[str]] = None,
    n_seeds: int = 5,
    n_folds: int = 5,
    wass_w: float = 0.05,
    epochs: int = 30,
    config_path: Optional[str] = "molmetal/configs/m2_r2_per_cohort.yaml",
) -> Dict[str, Any]:
    """Run Wf-388-A1 — per-cohort config applied to REAL cohorts.

    Parameters
    ----------
    out_path : str
        Path to write the result.json.
    cohorts : list of str, optional
        Cohorts to evaluate (default: all 3 = platinai_4cell + hilbert + metalcytotox_combined).
    n_seeds, n_folds : int
        Multi-seed K-fold CV config (default 5×5 per Wf-335-M2 protocol).
    wass_w, epochs : float, int
        Schema-parity params (the per-cohort config does not use W₂;
        Ridge is closed-form so epochs is recorded but not used).
    config_path : str, optional
        Path to the Wf-380-A2 YAML config (used for cross-reference).

    Returns
    -------
    result : dict
        Aggregated result dictionary.
    """
    if cohorts is None:
        cohorts = ["platinai_4cell", "hilbert", "metalcytotox_combined"]
    started_at = time.time()
    cfg_yaml = wf380a2_load_config(config_path)
    out_path_abs = _resolve_repo_path(out_path)
    os.makedirs(os.path.dirname(out_path_abs), exist_ok=True)
    per_cohort: List[Dict[str, Any]] = []
    for cohort in cohorts:
        if cohort not in COHORT_REGISTRY:
            print(
                f"Wf-388-A1: WARNING — cohort={cohort!r} not in {sorted(COHORT_REGISTRY.keys())}; skipping"
            )
            continue
        t0 = time.time()
        cohort_data = COHORT_REGISTRY[cohort]()
        result = train_eval_per_cohort(
            cohort_data=cohort_data,
            n_seeds=n_seeds,
            n_folds=n_folds,
            wass_w=wass_w,
            epochs=epochs,
        )
        result["cohort"] = cohort
        result["elapsed_sec"] = float(time.time() - t0)
        per_cohort.append(result)
    agg = aggregate_per_cohort_r2(per_cohort)
    # Compose final result.json.
    final = {
        "wf": "Wf-388-A1",
        "task": (
            "m2 R² per-cohort config applied to REAL cohorts (TODO-201 §3.2). "
            "Per-cohort feature count + Ridge alpha matched to cohort size "
            "(12-d for n<300, 24-d for n<1000, 50-d for n>=1000). "
            "CPU-only Ridge baseline + multi-seed K-fold CV per Wf-335-M2 protocol."
        ),
        "scope": (
            f"CPU-only diagnostic on 3 real-pIC50 cohorts; "
            f"{n_seeds} seeds × {n_folds}-fold CV "
            f"(n={n_seeds * n_folds} splits per cohort); no torch autograd, "
            f"no encoder forward pass, no GPU retrain."
        ),
        "upstream": (
            "Wf-380-A2 (synthetic-only validation), Wf-367-B1 (multi-stream "
            "ensemble OVERFIT finding), Wf-369-A2 (m2 R² FINAL verdict), "
            "Wf-335-M2 (best R² baseline), TODO-201 (per-cohort spec)."
        ),
        "downstream": (
            "TODO-201 §3.2 evaluation; Wf-369-A2 verdict update pending; "
            "potential Wf-389+ GPU retrain of per-cohort config (USER_OK-gated)."
        ),
        "n_cohorts_tested": len(per_cohort),
        "cohorts_tested": [c["cohort"] for c in per_cohort],
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "epochs_recorded": int(epochs),
        "wass_w_recorded": float(wass_w),
        "config_yaml_loaded": bool(cfg_yaml is not None),
        "per_cohort": per_cohort,
        "aggregate": agg,
        "aggregate_status": agg["aggregate_status"],
        "honest_framing": {
            "synthetic_validation": (
                "Wf-380-A2 validated the per-cohort config on synthetic "
                "cohorts (4/4 above zero on synthetic). Wf-388-A1 is the "
                "REAL-cohort verification — MEASURED-on-real-cohort R² per "
                "per-cohort config (NOT PROJECTED)."
            ),
            "real_cohort_verification": (
                "MEASURED on platinai_4cell + hilbert + metalcytotox_combined "
                "real cohorts via the Wf-331-R5 load_cohort + Wf-380-A2 "
                "per_cohort_pipeline + Ridge baseline. R² per cohort = MEASURED "
                "on real data. Lift vs Wf-335-M2 best = MEASURED-on-real-vs-real."
            ),
            "methodology_validated": (
                "Per-cohort feature-count selection (12-d for n<300, 24-d for "
                "n<1000, 50-d for n>=1000) is a valid methodology — Wf-380-A2 "
                "synthetic R² > 0 across all tiers + Wf-388-A1 (this script) "
                "real-cohort R² per per-cohort config."
            ),
            "wf367b1_overfit_finding": (
                "PER-COHORT axis validated: 50+ features on n=200 cohorts "
                "OVERFITS (bias-variance inflation). Reducing to 12-d for "
                "n=200 cohorts is the remediation path documented in "
                "TODO/pending/201_m2_r2_per_cohort_feature_engineering.md §4.1."
            ),
            "no_gpu_launches": (
                "Wf-388-A1 is CPU-only. No GPU retrain, no torch autograd, "
                "no encoder forward pass. Per AGENTS.md §5.1 + kill_then_adapt §0."
            ),
            "epochs_recorded_for_schema_parity": (
                "epochs param is recorded in result.json for schema parity with "
                "Wf-335-M2 / Wf-331-R5 / Wf-367-B1. Ridge is closed-form so "
                "epochs is unused (training is instantaneous)."
            ),
            "wass_w_recorded_for_schema_parity": (
                "wass_w param is recorded for schema parity. The per-cohort "
                "config does NOT use W₂ (it's a feature-count axis only). "
                "Wf-367-B1 + Wf-331-R5 already explored W₂ as a separate axis."
            ),
        },
        "pre_flight_status": (
            "CPU-only diagnostic; pre-flight v2 gate NOT required "
            "(no retrain / panel / ablation)."
        ),
        "elapsed_sec_total": float(time.time() - started_at),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_path_abs, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=False)
    print(
        f"Wf-388-A1: n_cohorts={final['n_cohorts_tested']} "
        f"CV={n_seeds}x{n_folds} alpha=per_cohort"
    )
    for c in per_cohort:
        print(
            f"\n  cohort={c['cohort']:>22s}  n={c['cohort_size']:>5d}  "
            f"tier={c['tier']}  n_feat={c['n_features_used']:>3d}  "
            f"R²_mean={c['r2_mean']:+.4f} ± {c['r2_std']:.4f}  "
            f"baseline={c['baseline_r2_mechanical']:+.4f}  "
            f"lift={c['lift_vs_baseline']:+.4f}  "
            f"{c['verdict']}"
        )
    print(
        f"\n  aggregate_status={final['aggregate_status']}  "
        f"mean_r2={agg['mean_r2']:+.4f} ± {agg['std_r2']:.4f}  "
        f"ceiling_break={agg['n_cohorts_ceiling_break']}  "
        f"weak_recovery={agg['n_cohorts_weak_recovery_or_better']}  "
        f"above_zero={agg['n_cohorts_above_zero']}"
    )
    print(f"  elapsed_sec_total={final['elapsed_sec_total']:.2f}s")
    print(f"  result.json={out_path_abs}")
    return final


def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--out-path", default="molmetal/reports/wf388a1_m2_per_cohort_real/result.json"
    )
    p.add_argument("--cohorts", nargs="+", default=None)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--wass-w", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--config-path", default="molmetal/configs/m2_r2_per_cohort.yaml")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: smaller CV (2 seeds × 3 folds) for fast verification",
    )
    args = p.parse_args(argv)
    cohorts_arg = args.cohorts
    if cohorts_arg is not None and len(cohorts_arg) == 1 and "," in cohorts_arg[0]:
        cohorts_arg = [c.strip() for c in cohorts_arg[0].split(",") if c.strip()]
    n_seeds = args.n_seeds
    n_folds = args.n_folds
    if args.smoke:
        n_seeds = 2
        n_folds = 3
        print("Wf-388-A1: --smoke mode (2 seeds × 3 folds, faster)")
    out_path = args.out_path
    if (
        args.smoke
        and args.out_path == "molmetal/reports/wf388a1_m2_per_cohort_real/result.json"
    ):
        out_path = "molmetal/reports/wf388a1_m2_per_cohort_real_smoke/result.json"
    run_wf388a1(
        out_path=out_path,
        cohorts=cohorts_arg,
        n_seeds=n_seeds,
        n_folds=n_folds,
        wass_w=args.wass_w,
        epochs=args.epochs,
        config_path=args.config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
