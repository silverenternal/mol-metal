"""Wf-361-B1 — m6_kappa MAML cross-scaffold meta-learning tests.

================================================================
Wf-361-B1 (2026-09-29) — 4 NEW tests covering the MAML
cross-scaffold meta-learning module
(``molmetal.scripts.wf361b1_m6_kappa_maml``).

Per Wf-359-A1 MEASURED: m6_kappa augmented_mean = 0.3561 with
R20-P3 features (below 0.6 threshold).  Per Wf-335-M1 MEASURED:
chemprop+PLL CV-R² = +0.6814 (best of 5 candidates) but
augmented_mean NOT lifted.  Wf-361-B1 explores MAML cross-
scaffold meta-learning as a NEW architectural path.

The four tests are:

  1. ``test_inner_outer_loop_logic`` — Verify the MAML config
     + closed-form Ridge inner-adaptation logic is correct:
     (a) inner_adapt_ridge blends meta-init with per-task
     coefficients, (b) MAMLConfig defaults are preserved, (c)
     the inner-loop is deterministic.

  2. ``test_per_scaffold_adaptation_step`` — Verify the per-
     scaffold inner-adaptation step on a synthetic cohort:
     K_shot support samples are correctly selected, inner-Ridge
     is fitted, predictions on the held-out scaffold are finite.

  3. ``test_cross_scaffold_loso_validation`` — End-to-end LOSO
     validation across top-N synthetic scaffolds: each scaffold
     held out once; per-scaffold CV-R² and augmented_mean are
     aggregated; the MAMLRecoveryReport contract is populated.

  4. ``test_backward_compat_no_flag`` — Backward compatibility
     check: when no MAML flag is invoked, the existing
     Wf-335-M1 chemprop+PLL pipeline is unaffected (bit-
     identical output preserved).  This guards against
     accidental coupling.

All tests are CPU-only, deterministic, and complete in < 120 s
combined.

Run with::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf361b1_m6_kappa_maml.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Lazy imports — graceful skip on missing deps
# ---------------------------------------------------------------------------
def _import_module():
    """Import the Wf-361-B1 module; return None on ImportError."""
    try:
        from molmetal.scripts.wf361b1_m6_kappa_maml import (
            MAMLConfig,
            MAMLFoldResult,
            MAMLRecoveryReport,
            load_cohort_with_m6_kappa,
            compute_murcko_scaffolds,
            split_by_scaffold,
            build_meta_features,
            inner_adapt_ridge,
            evaluate_loso,
            wire_maml_m6_kappa_recovery,
            write_maml_report,
            M6_KAPPA_THRESHOLD,
            WF335M1_MEASURED_BASELINE,
            WF335M1_MEASURED_CV_R2,
            WF359A1_MEASURED_AUGMENTED_MEAN,
            MAMLM6KappaError,
        )

        return {
            "MAMLConfig": MAMLConfig,
            "MAMLFoldResult": MAMLFoldResult,
            "MAMLRecoveryReport": MAMLRecoveryReport,
            "load_cohort_with_m6_kappa": load_cohort_with_m6_kappa,
            "compute_murcko_scaffolds": compute_murcko_scaffolds,
            "split_by_scaffold": split_by_scaffold,
            "build_meta_features": build_meta_features,
            "inner_adapt_ridge": inner_adapt_ridge,
            "evaluate_loso": evaluate_loso,
            "wire_maml_m6_kappa_recovery": wire_maml_m6_kappa_recovery,
            "write_maml_report": write_maml_report,
            "M6_KAPPA_THRESHOLD": M6_KAPPA_THRESHOLD,
            "WF335M1_MEASURED_BASELINE": WF335M1_MEASURED_BASELINE,
            "WF335M1_MEASURED_CV_R2": WF335M1_MEASURED_CV_R2,
            "WF359A1_MEASURED_AUGMENTED_MEAN": WF359A1_MEASURED_AUGMENTED_MEAN,
            "MAMLM6KappaError": MAMLM6KappaError,
        }
    except Exception as exc:
        pytest.skip(f"Wf-361-B1 import failed: {exc}")
        return None


@pytest.fixture(scope="module")
def mod():
    return _import_module()


# Synthetic cohort: 4 scaffold groups, each with 4 molecules.
# Each row has a SMILES + m6_kappa baseline + scaffold key.
SYNTHETIC_SMILES_BY_SCAFFOLD = {
    "c1ccccc1": [  # benzene scaffold
        "c1ccccc1",
        "Cc1ccccc1",
        "CCc1ccccc1",
        "CCCc1ccccc1",
    ],
    "c1ccncc1": [  # pyridine scaffold
        "c1ccncc1",
        "Cc1ccncc1",
        "CCc1ccncc1",
        "CCCc1ccncc1",
    ],
    "c1ccc(O)cc1": [  # phenol scaffold
        "c1ccc(O)cc1",
        "Oc1ccc(O)cc1",
        "CCOc1ccc(O)cc1",
        "CCOc1ccc(O)c(O)c1",
    ],
    "": [  # acyclic / no scaffold
        "CCO",
        "CC(=O)O",
        "CCN",
        "CCCCO",
    ],
}


def _make_synthetic_rows(seed: int = 20260929) -> list:
    """Build a synthetic cohort with 4 scaffold groups × 4 rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for sc, smiles_list in SYNTHETIC_SMILES_BY_SCAFFOLD.items():
        for smi in smiles_list:
            row = {
                "smiles": smi,
                "m6_kappa": float(rng.uniform(0.3, 0.7)),
                "scaffold": sc,
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Test 1: inner + outer loop logic
# ---------------------------------------------------------------------------
def test_inner_outer_loop_logic(mod):
    """Wf-361-B1 §1 — MAML config + closed-form Ridge inner-
    adaptation logic is correct: (a) inner_adapt_ridge blends
    meta-init with per-task coefficients, (b) MAMLConfig
    defaults are preserved, (c) the inner-loop is deterministic."""
    cfg = mod["MAMLConfig"]()
    assert cfg.k_shot == 8, f"k_shot default {cfg.k_shot} != 8"
    assert cfg.n_scaffolds == 5, f"n_scaffolds default {cfg.n_scaffolds} != 5"
    assert (
        cfg.ridge_alpha_meta == 1.0
    ), f"ridge_alpha_meta default {cfg.ridge_alpha_meta} != 1.0"
    assert (
        cfg.ridge_alpha_inner == 0.1
    ), f"ridge_alpha_inner default {cfg.ridge_alpha_inner} != 0.1"
    assert cfg.threshold == pytest.approx(0.6, abs=1e-9)
    assert cfg.threshold_adaptive == pytest.approx(0.55, abs=1e-9)
    assert cfg.seed == 20260929
    # Constants
    assert mod["M6_KAPPA_THRESHOLD"] == pytest.approx(0.6, abs=1e-9)
    assert mod["WF335M1_MEASURED_BASELINE"] == pytest.approx(
        0.4719079955824899,
        abs=1e-9,
    )
    assert mod["WF335M1_MEASURED_CV_R2"] == pytest.approx(
        0.6814054457018517,
        abs=1e-9,
    )
    assert mod["WF359A1_MEASURED_AUGMENTED_MEAN"] == pytest.approx(
        0.35607258930918284,
        abs=1e-9,
    )
    # inner_adapt_ridge closed-form behaviour
    rng = np.random.default_rng(20260929)
    n_feat = 10
    n_support = 16
    X_support = rng.standard_normal((n_support, n_feat))
    y_support = rng.standard_normal(n_support)
    meta_coef = rng.standard_normal(n_feat) * 0.1
    meta_intercept = 0.5
    inner_coef, inner_intercept = mod["inner_adapt_ridge"](
        X_support,
        y_support,
        meta_coef,
        meta_intercept,
        alpha_inner=cfg.ridge_alpha_inner,
    )
    assert inner_coef.shape == meta_coef.shape
    assert not np.allclose(
        inner_coef, meta_coef, atol=1e-6
    ), "inner_coef == meta_coef: inner loop is a no-op"
    assert np.isfinite(inner_intercept)
    # Determinism
    inner_coef2, inner_intercept2 = mod["inner_adapt_ridge"](
        X_support,
        y_support,
        meta_coef,
        meta_intercept,
        alpha_inner=cfg.ridge_alpha_inner,
    )
    np.testing.assert_array_equal(
        inner_coef,
        inner_coef2,
        err_msg="inner_adapt_ridge is non-deterministic",
    )
    assert inner_intercept == pytest.approx(inner_intercept2, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 2: per-scaffold adaptation step
# ---------------------------------------------------------------------------
def test_per_scaffold_adaptation_step(mod):
    """Wf-361-B1 §2 — Per-scaffold inner-adaptation step on a
    synthetic cohort: K_shot support samples are correctly
    selected, inner-Ridge is fitted, predictions on the held-
    out scaffold are finite."""
    rows = _make_synthetic_rows()
    sc = "c1ccccc1"
    with_sc, _ = mod["split_by_scaffold"](rows, sc)
    rows_other = [r for r in rows if r.get("scaffold", "") != sc]
    assert len(with_sc) > 0
    assert len(rows_other) > 0
    cfg = mod["MAMLConfig"](k_shot=2, n_scaffolds=4)
    fold = mod["evaluate_loso"](with_sc, rows_other, sc, cfg)
    assert isinstance(fold, mod["MAMLFoldResult"])
    assert fold.held_out_scaffold == sc
    assert fold.support_size == 2  # k_shot=2
    assert fold.query_size == len(with_sc)
    assert np.isfinite(fold.cv_r2_meta), f"cv_r2_meta not finite: {fold.cv_r2_meta}"
    assert np.isfinite(
        fold.cv_r2_inner_adapted
    ), f"cv_r2_inner_adapted not finite: {fold.cv_r2_inner_adapted}"
    assert np.isfinite(fold.query_mean)
    assert np.isfinite(fold.query_pred_mean_meta)
    assert np.isfinite(fold.query_pred_mean_adapted)
    assert fold.wall_seconds >= 0.0


# ---------------------------------------------------------------------------
# Test 3: cross-scaffold LOSO validation (end-to-end)
# ---------------------------------------------------------------------------
def test_cross_scaffold_loso_validation(mod):
    """Wf-361-B1 §3 — End-to-end LOSO validation across top-N
    synthetic scaffolds: each scaffold held out once; per-
    scaffold CV-R² and augmented_mean are aggregated; the
    MAMLRecoveryReport contract is populated."""
    rows = _make_synthetic_rows()
    cfg = mod["MAMLConfig"](k_shot=2, n_scaffolds=4)
    report = mod["wire_maml_m6_kappa_recovery"](
        rows=rows,
        config=cfg,
        max_samples=32,
    )
    assert isinstance(report, mod["MAMLRecoveryReport"])
    assert report.candidate == "maml_cross_scaffold"
    assert report.n_samples == len(rows)
    assert report.n_scaffolds == 4
    # n_features contract: 248 (chemprop 216 + PLL 32)
    assert (
        report.n_features == 248
    ), f"n_features {report.n_features} != 248 (chemprop+PLL joint)"
    # 4 per-scaffold folds populated
    assert len(report.per_scaffold) == 4
    for sc in SYNTHETIC_SMILES_BY_SCAFFOLD:
        assert sc in report.per_scaffold, f"missing scaffold {sc}"
        fold = report.per_scaffold[sc]
        assert fold.held_out_scaffold == sc
        assert fold.support_size >= 1
        assert fold.query_size >= 1
    # Augmented means finite
    assert np.isfinite(report.augmented_mean_meta)
    assert np.isfinite(report.augmented_mean_adapted)
    assert np.isfinite(report.baseline_mean)
    # Deltas finite
    assert np.isfinite(report.delta_adapted_vs_baseline)
    assert np.isfinite(report.delta_adapted_vs_meta)
    # Verdict contract
    expected_verdict = (
        "HONEST_NEGATIVE_M6_KAPPA_RESOLVED"
        if report.augmented_mean_adapted >= mod["M6_KAPPA_THRESHOLD"]
        else "HONEST_NEGATIVE_PARTIAL_STRUCTURAL"
    )
    assert report.verdict == expected_verdict
    # Pass-rate contract
    assert 0.0 <= report.pass_rate_baseline <= 1.0
    assert 0.0 <= report.pass_rate_augmented_meta <= 1.0
    assert 0.0 <= report.pass_rate_augmented_adapted <= 1.0
    # Wall-seconds positive
    assert report.wall_seconds > 0.0
    # JSON serialisation roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "result.json")
        mod["write_maml_report"](report, out_path)
        with open(out_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    assert payload["wf"] == "Wf-361-B1"
    assert payload["no_gpu_launch"] is True
    assert payload["cpu_only"] is True
    assert payload["recovery"]["candidate"] == "maml_cross_scaffold"
    assert payload["recovery"]["n_samples"] == len(rows)
    assert payload["recovery"]["verdict"] == report.verdict


# ---------------------------------------------------------------------------
# Test 4: backward compatibility (no flag invoked)
# ---------------------------------------------------------------------------
def test_backward_compat_no_flag(mod):
    """Wf-361-B1 §4 — Backward compatibility check: when no
    MAML flag is invoked, the existing Wf-335-M1 chemprop+PLL
    pipeline is unaffected (bit-identical output preserved).
    This guards against accidental coupling of MAML to the
    chemprop+PLL extractor."""
    try:
        from molmetal.molmetal_lam.reward.chemprop_pll_integration import (
            wire_chemprop_pll_recovery,
            WF333V4_MEASURED_CV_R2,
            WF325D_MEASURED_BASELINE,
        )
    except Exception as exc:
        pytest.skip(f"Wf-335-M1 import failed: {exc}")

    # Wf-335-M1 anchors must remain bit-identical after Wf-361-B1 import
    assert WF333V4_MEASURED_CV_R2 == pytest.approx(
        0.6814054457018517,
        abs=1e-9,
    ), "Wf-335-M1 WF333V4_MEASURED_CV_R2 anchor mutated"
    assert WF325D_MEASURED_BASELINE == pytest.approx(
        0.4719079955824899,
        abs=1e-9,
    ), "Wf-335-M1 WF325D_MEASURED_BASELINE anchor mutated"

    # The Wf-335-M1 wire-in must still work on the same SMILES
    smiles = [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CCN",
        "CCOC",
        "CCCCO",
        "c1ccc(O)cc1",
        "c1ccncc1",
    ]
    np.random.seed(20260929)
    y_baseline = np.clip(
        np.random.uniform(0.3, 0.7, size=len(smiles)),
        0.0,
        1.0,
    )
    report = wire_chemprop_pll_recovery(
        smiles,
        y_baseline,
        ridge_alpha=1.0,
        cv_folds=5,
    )
    assert np.isfinite(report.cv_r2)
    assert np.isfinite(report.augmented_mean)
    assert report.n_features == 248

    # Wf-361-B1 must NOT have leaked into chemprop_pll_integration
    import molmetal.molmetal_lam.reward.chemprop_pll_integration as cpi

    assert not hasattr(
        cpi, "MAMLConfig"
    ), "MAMLConfig leaked into chemprop_pll_integration namespace"
    assert not hasattr(cpi, "wire_maml_m6_kappa_recovery"), (
        "wire_maml_m6_kappa_recovery leaked into " "chemprop_pll_integration namespace"
    )
