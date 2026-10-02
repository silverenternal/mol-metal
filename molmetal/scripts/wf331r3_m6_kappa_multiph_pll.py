"""Wf-331-R3 — m6_kappa full Multi-PH/PLL integration (CPU).

================================================================
Wf-331-R3 (2026-09-29) — CPU-only full integration of the
32-bin Fourier + Multi-PH + PLL feature extractor into the
m6_kappa measurement pipeline.

Per Wf-325D MEASURED baseline: m6_kappa cohort mean = 0.472
(gap -0.128 to ChemAxon PMC11387870 threshold 0.6).

Per Wf-329-W4 + WEAK_TO_STRONG §2.5: Multi-PH/PLL feature
extraction is the recovery path.  Wf-331-R3 wires the full
joint (Fourier + Multi-PH + PLL) feature matrix into the
m6_kappa pipeline with a *meaningful* recovery signal:

* baseline m6_kappa = mean RDKit Shape Kappa on the cohort
  (canonical Wf-252 metric, identical to Wf-325D).
* augmented m6_kappa = cross-validated Ridge prediction
  mean using the joint (Fourier + Multi-PH + PLL) design
  matrix (CPU sklearn ``Ridge`` + ``cross_val_predict``).
* cv_r2 = 5-fold cross-validated R² of the joint design
  matrix predicting the raw m6_kappa — the structural
  recovery signal (positive = joint features carry kappa
  information; this is the meaningful "lift").
* pass_rate_baseline = P(raw m6_kappa >= 0.6).
* pass_rate_augmented = P(CV-predicted m6_kappa >= 0.6).
* pass_rate_lift = pass_rate_augmented - pass_rate_baseline
  (the threshold-reaching recovery signal).

Honest framing per AGENTS.md §5.2:

* No GPU launch; CPU-only feature extraction + sklearn
  Ridge with ``cross_val_predict``.
* No retrain; downstream regressor is Ridge (CPU sklearn).
* MEASURED-vs-PROJECTED tags in the result JSON.
* ``delta = augmented_mean - baseline_mean`` is *expected*
  to be near 0 (unbiased cross-validated predictions
  centre around the target mean); the meaningful recovery
  signal is ``cv_r2`` + ``pass_rate_lift``.

Run on the metallo cohort by default::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/scripts/wf331r3_m6_kappa_multiph_pll.py \\
        --output-dir molmetal/reports/wf331r3_m6_kappa

Recovery vs. honest-negative verdict:

* If ``augmented_mean >= 0.6`` (full cohort) →
  ``HONEST_NEGATIVE_M6_KAPPA_RESOLVED``.
* Else → ``HONEST_NEGATIVE_PERSISTENT`` and the metric
  stays at its Wf-325D honest-negative state.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# gfx1101 env vars applied by __init__ if present
try:
    import molmetal  # noqa: F401
except Exception:
    pass

# Quiet RDKit warnings — the metallo cohort has many non-standard
# atom types (Sc2+3, Pt2+, Y_5 etc.) that RDKit logs UFFTYPER
# errors for.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]

#: ChemAxon PMC11387870 Shape Kappa Index druglikeness threshold.
#: Matches the Wf-252 / Wf-325D ship surface (literal 0.6).
M6_KAPPA_THRESHOLD: float = 0.6

#: Default 5-fold cross-validation for the Ridge recovery path.
DEFAULT_CV_FOLDS: int = 5

#: Default Ridge alpha — matches the Wf-329-W4 wire-in.
DEFAULT_RIDGE_ALPHA: float = 1.0

#: Wf-325D MEASURED baseline (449-row metallo_drugs_500_train cohort).
#: Mirrored here for the JSON report's "MEASURED-vs-PROJECTED" tags.
WF_325D_MEASURED_BASELINE: float = 0.4719079955824899


def _load_cohort(csv_path: Path, max_samples: int) -> List[str]:
    """Load SMILES from the metallo cohort CSV."""
    smiles: List[str] = []
    if not csv_path.exists():
        return smiles
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles.append(smi)
            if len(smiles) >= max_samples:
                break
    return smiles


def _safe_m6_kappa(smi: str) -> Optional[float]:
    """Compute ``m6_kappa_from_smiles`` defensively; return None on error."""
    try:
        from molmetal.molmetal_lam.sbdd_env.metrics_v2 import (
            m6_kappa_from_smiles,
        )
        v = m6_kappa_from_smiles(smi)
    except Exception:
        return None
    if v is None:
        return None
    return float(v)


def _compute_baseline_pass_rate(values: Sequence[float], threshold: float) -> float:
    """Fraction of ``values`` that meet ``threshold`` (NaN-safe)."""
    finite = [v for v in values if v == v]  # NaN-safe
    if not finite:
        return 0.0
    n_meet = sum(1 for v in finite if v >= threshold)
    return float(n_meet) / float(len(finite))


def _cv_r2_joint_predict(
    X: Any, y: Any, alpha: float, cv_folds: int,
) -> Dict[str, Any]:
    """Cross-validated Ridge prediction + R² for joint m6_kappa recovery.

    Returns
    -------
    Dict with keys:
        ``y_pred_cv`` — cross-validated predictions (len = n).
        ``r2_cv`` — R² of cross-validated predictions vs ``y``.
        ``alpha`` / ``cv_folds`` — echoed for traceability.
        ``n_samples`` — n.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score
    import numpy as np

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = X_arr.shape[0]
    if n < 4:
        return {
            "y_pred_cv": np.full(n, float("nan"), dtype=np.float64),
            "r2_cv": float("nan"),
            "alpha": float(alpha),
            "cv_folds": int(cv_folds),
            "n_samples": int(n),
        }
    cv = min(int(cv_folds), max(2, n - 1))
    model = Ridge(alpha=float(alpha))
    y_pred_cv = cross_val_predict(model, X_arr, y_arr, cv=cv)
    r2 = float(r2_score(y_arr, y_pred_cv))
    return {
        "y_pred_cv": y_pred_cv,
        "r2_cv": r2,
        "alpha": float(alpha),
        "cv_folds": int(cv),
        "n_samples": int(n),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=str(
            REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
        ),
        help="Path to metallo_drugs_500_train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT / "molmetal" / "reports" / "wf331r3_m6_kappa"
        ),
        help="Where to write result.json + per-axis csv",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=60,
        help="Cap cohort size for the smoke run (CPU-only); "
             "default 60 keeps wall < ~5 min on a 32-bin grid.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Cross-validation folds for the Ridge recovery path",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
        help="Ridge alpha for the joint design matrix",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260929,
        help="RNG seed for any randomised sub-sampling",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load cohort
    # ------------------------------------------------------------------
    print(f"[Wf-331-R3] Loading cohort from {csv_path} "
          f"(max {args.max_samples}) ...")
    t0 = time.time()
    smiles = _load_cohort(csv_path, args.max_samples)
    t_load = time.time() - t0
    print(f"[Wf-331-R3] Loaded {len(smiles)} SMILES in {t_load:.2f}s")

    if not smiles:
        # Fallback to a tiny synthetic cohort so the smoke doesn't
        # crash on an empty environment.
        smiles = ["CCO", "c1ccccc1", "[Pt](N)(N)(Cl)Cl"]
        print(f"[Wf-331-R3] CSV empty; using synthetic fallback "
              f"({len(smiles)} SMILES)")

    # ------------------------------------------------------------------
    # 2. Compute baseline m6_kappa on the cohort
    # ------------------------------------------------------------------
    print("[Wf-331-R3] Computing baseline m6_kappa on cohort ...")
    t0 = time.time()
    baseline_vals: List[Optional[float]] = [_safe_m6_kappa(s) for s in smiles]
    t_baseline = time.time() - t0
    baseline_clean = [v for v in baseline_vals if v is not None]
    n_parsed = len(baseline_clean)
    n_failed = len(baseline_vals) - n_parsed
    if not baseline_clean:
        print("[Wf-331-R3] No valid m6_kappa values; aborting")
        return 2
    baseline_mean = float(sum(baseline_clean) / n_parsed)
    baseline_pass_rate = _compute_baseline_pass_rate(
        baseline_clean, M6_KAPPA_THRESHOLD,
    )
    print(f"[Wf-331-R3] baseline m6_kappa: mean={baseline_mean:.4f} "
          f"(std populated below, n_parsed={n_parsed}, "
          f"n_failed={n_failed}, pass_rate@{M6_KAPPA_THRESHOLD:.2f}="
          f"{baseline_pass_rate:.4f}, {t_baseline:.2f}s)")

    # ------------------------------------------------------------------
    # 3. Compute joint (Fourier + Multi-PH + PLL) features
    # ------------------------------------------------------------------
    # Late import so the gfx1101 + RDKit env vars are applied first.
    from molmetal.molmetal_lam.reward.multiph_pll_feature import (
        MultiPHPLLFeatureExtractor,
        fourier_32bin_features,
    )

    print("[Wf-331-R3] Computing joint features (32-bin Fourier + "
          "Multi-PH + PLL) ...")
    t0 = time.time()
    extractor = MultiPHPLLFeatureExtractor()
    fourier_b, multi_ph_b, pll_b = extractor.compute_batch(smiles)
    t_features = time.time() - t0
    n_samples = fourier_b.shape[0]
    print(f"[Wf-331-R3] Extractor: n_bins={extractor.n_bins} "
          f"(Fourier={extractor.fourier_dim}, "
          f"Multi-PH={extractor.multi_ph_dim}, "
          f"PLL={extractor.pll_dim}, joint={extractor.joint_dim})")
    print(f"[Wf-331-R3] Batch shape: fourier={fourier_b.shape}, "
          f"multi_ph={multi_ph_b.shape}, pll={pll_b.shape} "
          f"({t_features:.2f}s)")

    # Truncate baseline values to the cohort that survived RDKit
    # conformer embedding (compute_batch zero-fills failures).
    # We use the clean baseline list and align on index.
    y_baseline = []
    kept_smiles: List[str] = []
    kept_index: List[int] = []
    for i, v in enumerate(baseline_vals):
        if v is None:
            continue
        y_baseline.append(float(v))
        kept_smiles.append(smiles[i])
        kept_index.append(i)
    if not y_baseline:
        print("[Wf-331-R3] No SMILES with valid m6_kappa baseline; aborting")
        return 2

    # Build the joint design matrix on the kept subset.
    import numpy as np
    fourier_kept = fourier_b[kept_index]
    multi_ph_kept = multi_ph_b[kept_index]
    pll_kept = pll_b[kept_index]
    X_joint = np.concatenate(
        [fourier_kept, multi_ph_kept.reshape(len(kept_index), -1), pll_kept],
        axis=-1,
    )
    X_baseline_only = np.concatenate([fourier_kept, pll_kept], axis=-1)
    y = np.asarray(y_baseline, dtype=np.float64)

    # ------------------------------------------------------------------
    # 4. Cross-validated Ridge recovery
    # ------------------------------------------------------------------
    print(f"[Wf-331-R3] Cross-validated Ridge recovery "
          f"(alpha={args.ridge_alpha}, cv_folds={args.cv_folds}) ...")
    t0 = time.time()
    cv_full = _cv_r2_joint_predict(
        X_joint, y, alpha=args.ridge_alpha, cv_folds=args.cv_folds,
    )
    cv_baseline = _cv_r2_joint_predict(
        X_baseline_only, y, alpha=args.ridge_alpha, cv_folds=args.cv_folds,
    )
    t_cv = time.time() - t0
    y_pred_cv = cv_full["y_pred_cv"]
    augmented_mean = float(np.mean(y_pred_cv))
    augmented_pass_rate = _compute_baseline_pass_rate(
        y_pred_cv, M6_KAPPA_THRESHOLD,
    )
    pass_rate_lift = augmented_pass_rate - baseline_pass_rate
    cv_r2 = cv_full["r2_cv"]
    cv_r2_baseline_only = cv_baseline["r2_cv"]

    # Standard deviation of baseline (informational).
    baseline_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0

    # Recovery verdict.
    recovery_verdict = (
        "HONEST_NEGATIVE_M6_KAPPA_RESOLVED"
        if augmented_mean >= M6_KAPPA_THRESHOLD
        else "HONEST_NEGATIVE_PERSISTENT"
    )

    # ------------------------------------------------------------------
    # 5. Per-feature breakdown — sanity check on each modality
    # ------------------------------------------------------------------
    cv_fourier_only = _cv_r2_joint_predict(
        fourier_kept, y, alpha=args.ridge_alpha, cv_folds=args.cv_folds,
    )
    cv_multi_ph_only = _cv_r2_joint_predict(
        multi_ph_kept.reshape(len(kept_index), -1), y,
        alpha=args.ridge_alpha, cv_folds=args.cv_folds,
    )
    cv_pll_only = _cv_r2_joint_predict(
        pll_kept, y, alpha=args.ridge_alpha, cv_folds=args.cv_folds,
    )

    print(f"[Wf-331-R3] baseline_mean={baseline_mean:.4f} "
          f"(Wf-325D MEASURED={WF_325D_MEASURED_BASELINE:.4f})")
    print(f"[Wf-331-R3] augmented_mean={augmented_mean:.4f} "
          f"(CV-Ridge joint)")
    print(f"[Wf-331-R3] delta={augmented_mean - baseline_mean:+.4f} "
          f"(expected ~0 for unbiased CV)")
    print(f"[Wf-331-R3] cv_r2 (joint)        = {cv_r2:+.4f}")
    print(f"[Wf-331-R3] cv_r2 (fourier only) = {cv_fourier_only['r2_cv']:+.4f}")
    print(f"[Wf-331-R3] cv_r2 (multi-ph only) = "
          f"{cv_multi_ph_only['r2_cv']:+.4f}")
    print(f"[Wf-331-R3] cv_r2 (pll only)     = {cv_pll_only['r2_cv']:+.4f}")
    print(f"[Wf-331-R3] cv_r2 (baseline f+p) = {cv_r2_baseline_only:+.4f}")
    print(f"[Wf-331-R3] pass_rate_baseline   = {baseline_pass_rate:.4f} "
          f"(raw m6_kappa >= {M6_KAPPA_THRESHOLD:.2f})")
    print(f"[Wf-331-R3] pass_rate_augmented  = {augmented_pass_rate:.4f} "
          f"(CV-predicted m6_kappa >= {M6_KAPPA_THRESHOLD:.2f})")
    print(f"[Wf-331-R3] pass_rate_lift       = {pass_rate_lift:+.4f}")
    print(f"[Wf-331-R3] verdict              = {recovery_verdict}")

    # ------------------------------------------------------------------
    # 6. JSON report (MEASURED-vs-PROJECTED tagged)
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "wf": "Wf-331-R3",
        "task": "m6_kappa full Multi-PH/PLL integration "
                "(32-bin Fourier + Multi-PH + PLL)",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "no_gpu_launch": True,
        "no_retrain": True,
        "cpu_only": True,
        "wall_seconds": round(
            t_load + t_baseline + t_features + t_cv, 4,
        ),
        "wf_325d_baseline_measured": WF_325D_MEASURED_BASELINE,
        "wf_325d_threshold": M6_KAPPA_THRESHOLD,
        "n_smiles_loaded": len(smiles),
        "n_m6_parsed": n_parsed,
        "n_m6_failed": n_failed,
        "n_kept_in_cv": len(kept_index),
        "baseline_mean": round(baseline_mean, 6),
        "baseline_std": round(baseline_std, 6),
        "baseline_pass_rate": round(baseline_pass_rate, 6),
        "augmented_mean": round(augmented_mean, 6),
        "augmented_pass_rate": round(augmented_pass_rate, 6),
        "delta_mean": round(augmented_mean - baseline_mean, 6),
        "pass_rate_lift": round(pass_rate_lift, 6),
        "cv_r2_joint": round(cv_r2, 6),
        "cv_r2_fourier_only": round(cv_fourier_only["r2_cv"], 6),
        "cv_r2_multi_ph_only": round(cv_multi_ph_only["r2_cv"], 6),
        "cv_r2_pll_only": round(cv_pll_only["r2_cv"], 6),
        "cv_r2_fourier_pll_baseline": round(cv_r2_baseline_only, 6),
        "ridge_alpha": args.ridge_alpha,
        "cv_folds": int(cv_full["cv_folds"]),
        "extractor": {
            "n_bins": extractor.n_bins,
            "k": extractor.k,
            "knn": extractor.knn,
            "max_radius": extractor.max_radius,
            "max_coord": extractor.max_coord,
            "fourier_dim": extractor.fourier_dim,
            "multi_ph_dim": extractor.multi_ph_dim,
            "pll_dim": extractor.pll_dim,
            "joint_dim": extractor.joint_dim,
        },
        "fourier_features": (
            "32-bin radially-binned magnitude spectrum of multi-PH "
            "feature map (Kim-Jung 2026 §4 acceleration)"
        ),
        "multi_ph_features": (
            f"TwoParameterFiltration n_bins={extractor.n_bins} "
            f"(W2.3 R18 P3 surface)"
        ),
        "pll_features": (
            f"PLL top-{extractor.k} eigenvalues (Liu-Feng 2026)"
        ),
        "recovery_verdict": recovery_verdict,
        "honest_framing": {
            "is_cpu_only": True,
            "is_no_retrain": True,
            "is_no_paper_touch": True,
            "baseline_mean_status": "MEASURED (Wf-325D, n=449 full cohort)",
            "smoke_baseline_status": (
                "MEASURED on this smoke cohort "
                f"(n={n_parsed}/{len(smiles)})"
            ),
            "cv_r2_status": "MEASURED (5-fold Ridge cross-validation)",
            "augmented_mean_status": (
                "MEASURED (mean of 5-fold CV-predicted m6_kappa); "
                "expected ~baseline_mean by construction"
            ),
            "pass_rate_lift_status": (
                "MEASURED (CV-predicted pass rate - raw pass rate)"
            ),
            "r2_lift_status": "MEASURED",
            "m6_kappa_lift_status": (
                "RESOLVED" if recovery_verdict == "HONEST_NEGATIVE_M6_KAPPA_RESOLVED"
                else "PERSISTENT"
            ),
            "interpretation_note": (
                "Wf-331-R3 is the Wf-329-W4 wire-in applied to the "
                "m6_kappa axis.  The CV-R² on the joint (Fourier + "
                "Multi-PH + PLL) design matrix is the structural "
                "recovery signal — positive R² means the joint "
                "topological features carry the m6_kappa signal.  "
                "The pass-rate lift (>= 0.6 threshold) is the "
                "threshold-reaching recovery signal.  The delta of "
                "means is ~0 by construction (unbiased CV).  "
                "augmented_mean reaches >= 0.6 only if the joint "
                "model can shift the cohort's kappa distribution."
            ),
        },
        "notes": {
            "m6_baseline": (
                f"Raw m6_kappa cohort mean = {baseline_mean:.4f} "
                f"(Wf-325D MEASURED full cohort = "
                f"{WF_325D_MEASURED_BASELINE:.4f}); "
                f"gap to threshold {M6_KAPPA_THRESHOLD:.2f} = "
                f"{baseline_mean - M6_KAPPA_THRESHOLD:+.4f}"
            ),
            "m6_augmented": (
                f"5-fold CV-Ridge joint prediction mean = "
                f"{augmented_mean:.4f}; "
                f"CV-R² = {cv_r2:+.4f}; "
                f"pass_rate_lift = {pass_rate_lift:+.4f}"
            ),
            "fourier_features": (
                f"32-bin radially-binned magnitude spectrum "
                f"(n_bins={extractor.n_bins}, fourier_dim="
                f"{extractor.fourier_dim})"
            ),
            "multi_ph_features": (
                f"TwoParameterFiltration with n_bins={extractor.n_bins}, "
                f"max_radius={extractor.max_radius}, "
                f"max_coord={extractor.max_coord}"
            ),
            "pll_features": (
                f"PLL top-{extractor.k} eigenvalues "
                f"(k_nn={extractor.knn}, Liu-Feng 2026)"
            ),
        },
    }

    out_path = output_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[Wf-331-R3] Wrote {out_path}")

    # ------------------------------------------------------------------
    # 7. Per-axis CSV
    # ------------------------------------------------------------------
    csv_path_out = output_dir / "per_axis.csv"
    with open(csv_path_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "axis", "baseline", "augmented", "delta",
                "cv_r2", "pass_rate_baseline", "pass_rate_augmented",
                "pass_rate_lift", "n_samples", "wall_seconds",
                "verdict", "tag",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "axis": "m6_kappa",
            "baseline": round(baseline_mean, 6),
            "augmented": round(augmented_mean, 6),
            "delta": round(augmented_mean - baseline_mean, 6),
            "cv_r2": round(cv_r2, 6),
            "pass_rate_baseline": round(baseline_pass_rate, 6),
            "pass_rate_augmented": round(augmented_pass_rate, 6),
            "pass_rate_lift": round(pass_rate_lift, 6),
            "n_samples": n_parsed,
            "wall_seconds": round(
                t_load + t_baseline + t_features + t_cv, 4,
            ),
            "verdict": recovery_verdict,
            "tag": "MEASURED",
        })
    print(f"[Wf-331-R3] Wrote {csv_path_out}")

    # ------------------------------------------------------------------
    # 8. Console summary (5-line plain-text per AGENTS.md §8.3)
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"Wf-331-R3 SUMMARY")
    print(f"  m6_kappa before/after : "
          f"{baseline_mean:.4f} -> {augmented_mean:.4f} "
          f"(delta={augmented_mean - baseline_mean:+.4f})")
    print(f"  multiph_features      : "
          f"F={extractor.fourier_dim}, M={extractor.multi_ph_dim}, "
          f"P={extractor.pll_dim}, joint={extractor.joint_dim}")
    print(f"  pll_signature         : "
          f"top-{extractor.k} eigenvalues (k_nn={extractor.knn})")
    print(f"  recovery_verdict      : {recovery_verdict}")
    print("=" * 72)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())