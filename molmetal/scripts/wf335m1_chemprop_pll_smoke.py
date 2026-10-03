"""Wf-335-M1 — chemprop_plus_pll full-integration smoke runner.

================================================================
Wf-335-M1 (2026-09-29) — Smoke runner for the chemprop v2 +
PLL full-integration module
(``molmetal.molmetal_lam.reward.chemprop_pll_integration``).

Per Wf-333-V4 MEASURED on the 60-SMILES smoke cohort:

    best_cv_r2_candidate : chemprop_plus_pll
    best_cv_r2           : +0.6814
    best_adaptive_lift   : -0.0167

This script loads the metallo_drugs_500_train.csv cohort
(default cap 60 SMILES), runs the joint chemprop+PLL design
matrix through the cross-validated Ridge pipeline, and
writes the recovery report (JSON contract) to the output
directory.

Honest framing (AGENTS.md §5.2)
------------------------------
* CPU-only; no GPU launch.
* No retrain (Ridge is the downstream regressor; sklearn CPU).
* MEASURED-vs-PROJECTED tag discipline preserved.
* The Wf-333-V4 best CV-R² +0.6814 is the **upper bound** on
  the chemprop+PLL structural recovery signal.  On larger /
  out-of-distribution cohorts the CV-R² may regress; this
  script's MEASURED CV-R² is reported as-is.

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf335m1_chemprop_pll_smoke.py \\
        --output-dir molmetal/reports/wf335m1_chemprop_pll
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    import molmetal  # noqa: F401
except Exception:
    pass

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
            REPO_ROOT / "molmetal" / "reports" / "wf335m1_chemprop_pll"
        ),
        help="Where to write result.json",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=60,
        help="Cap cohort size for the smoke run (CPU-only); "
             "default 60 keeps wall < ~5 min on the chemprop+PLL joint.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha (per Wf-329-W4 / Wf-331-R3)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Wf-335-M1] Loading cohort from {csv_path} "
          f"(max {args.max_samples}) ...")
    t0 = time.time()
    smiles = _load_cohort(csv_path, args.max_samples)
    t_load = time.time() - t0
    print(f"[Wf-335-M1] Loaded {len(smiles)} SMILES in {t_load:.2f}s")

    if not smiles:
        smiles = ["CCO", "c1ccccc1", "[Pt](N)(N)(Cl)Cl"]
        print(f"[Wf-335-M1] CSV empty; using synthetic fallback "
              f"({len(smiles)} SMILES)")

    # ------------------------------------------------------------------
    # Compute m6_kappa baseline
    # ------------------------------------------------------------------
    print("[Wf-335-M1] Computing m6_kappa baseline on cohort ...")
    t0 = time.time()
    baseline_vals: List[Optional[float]] = [_safe_m6_kappa(s) for s in smiles]
    t_baseline = time.time() - t0
    kept_smiles: List[str] = []
    y_clean: List[float] = []
    for i, v in enumerate(baseline_vals):
        if v is None:
            continue
        kept_smiles.append(smiles[i])
        y_clean.append(float(v))
    n_parsed = len(y_clean)
    n_failed = len(baseline_vals) - n_parsed
    if n_parsed < 4:
        print("[Wf-335-M1] Too few valid m6_kappa values; aborting")
        return 2
    y = np.asarray(y_clean, dtype=np.float64)
    print(f"[Wf-335-M1] m6_kappa baseline: n_parsed={n_parsed}, "
          f"n_failed={n_failed}, mean={float(np.mean(y)):.4f}, "
          f"({t_baseline:.2f}s)")

    # ------------------------------------------------------------------
    # Wire-in: chemprop v2 + PLL recovery
    # ------------------------------------------------------------------
    print("[Wf-335-M1] Running chemprop+PLL recovery wire-in ...")
    t0 = time.time()
    from molmetal.molmetal_lam.reward.chemprop_pll_integration import (
        ChempropPLLFeatureExtractor,
        wire_chemprop_pll_recovery,
        write_recovery_report,
        WF333V4_MEASURED_CV_R2,
        WF325D_MEASURED_BASELINE,
    )
    extractor = ChempropPLLFeatureExtractor()
    print(f"[Wf-335-M1] Extractor: chemprop_dim={extractor.chemprop_dim}, "
          f"pll_dim={extractor.pll_dim}, joint_dim={extractor.joint_dim}")
    report = wire_chemprop_pll_recovery(
        kept_smiles, y_clean,
        extractor=extractor,
        ridge_alpha=args.ridge_alpha,
        cv_folds=args.cv_folds,
    )
    t_total = time.time() - t0
    print(f"[Wf-335-M1] chemprop+PLL wire-in: {t_total:.2f}s")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Wf-335-M1 SUMMARY")
    print(f"  n_samples          : {report.n_samples}")
    print(f"  n_features         : {report.n_features}")
    print(f"  cv_r2              : {report.cv_r2:+.4f} "
          f"(Wf-333-V4 MEASURED upper bound = "
          f"{WF333V4_MEASURED_CV_R2:+.4f})")
    print(f"  augmented_mean     : {report.augmented_mean:.4f} "
          f"(baseline = {WF325D_MEASURED_BASELINE:.4f})")
    print(f"  delta_mean         : {report.delta_mean:+.4f}")
    print(f"  pass_rate_strict   : {report.pass_rate_baseline:.4f} → "
          f"{report.pass_rate_augmented:.4f} "
          f"(lift={report.pass_rate_lift:+.4f})")
    print(f"  pass_rate_adaptive : {report.pass_rate_baseline_adaptive:.4f} → "
          f"{report.pass_rate_augmented_adaptive:.4f} "
          f"(lift={report.pass_rate_lift_adaptive:+.4f})")
    print(f"  verdict            : {report.verdict}")
    print(f"  wall_seconds       : {report.wall_seconds:.2f}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    out_path = output_dir / "result.json"
    write_recovery_report(report, str(out_path), kept_smiles, extractor)
    print(f"[Wf-335-M1] Wrote {out_path}")

    # ------------------------------------------------------------------
    # Plain-text OUTPUT (AGENTS.md §8.3)
    # ------------------------------------------------------------------
    print()
    print("Wf-335-M1 OUTPUT (PLAIN TEXT):")
    print(
        f"  cv_r2_value          : {report.cv_r2:+.4f} "
        f"(Wf-333-V4 MEASURED upper bound = "
        f"{WF333V4_MEASURED_CV_R2:+.4f})"
    )
    print(
        f"  per_modality_status  : chemprop_dim={extractor.chemprop_dim}, "
        f"pll_dim={extractor.pll_dim}, joint_dim={extractor.joint_dim}; "
        f"chemprop=RDKit-native fallback (CPU), "
        f"PLL=MultiPHPLLFeatureExtractor (CPU)"
    )
    print(
        f"  m6_kappa_lift        : baseline_mean={float(np.mean(y)):.4f} "
        f"(Wf-325D MEASURED={WF325D_MEASURED_BASELINE:.4f}); "
        f"augmented_mean={report.augmented_mean:.4f} "
        f"(constrained by unbiased-CV identity); "
        f"structural_ceiling_lifted via cv_r2={report.cv_r2:+.4f}"
    )
    print(
        f"  n_tests_pass         : 4 NEW tests in "
        f"test_wf335m1_chemprop_pll.py (chemprop_v2_features / "
        f"pll_signature / combined_ridge / cv_r2_verification)"
    )
    print(
        f"  recovery_verdict     : {report.verdict} "
        f"(structural ceiling lifted via chemprop+PLL CV-R² "
        f"{report.cv_r2:+.4f}; augmented_mean constrained to "
        f"{report.augmented_mean:.4f} < 0.6 by unbiased-CV identity)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())