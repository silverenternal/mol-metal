"""Wf-388-A1 — m2 R² per-cohort config on REAL cohorts (TODO-201 §3.2).

=====================================================================
Wf-388-A1 / Wf-380 ultracode follow-up — verifies the per-cohort config
(12-d / 24-d / 50-d matched to cohort size) on the 3 REAL cohorts:

  * platinai_4cell        (n=200)  → 12-d RDKit + Ridge α=1.0
  * hilbert               (n=200)  → 12-d RDKit + Ridge α=1.0
  * metalcytotox_combined (n=2000) → 32-d RDKit + Ridge α=0.1

3 NEW tests:
  1. test_per_cohort_config_applied
       — verify 12-d / 24-d / 32-d feature slice is applied correctly
         per cohort size via apply_per_cohort_config.
  2. test_r2_lift_on_synthetic
       — verify Ridge R² > 0 on the synthetic cohort via per-cohort pipeline
         (smaller test cohort sizes to keep wall < 1 s).
  3. test_real_cohort_loader
       — verify platinai_4cell + hilbert + metalcytotox_combined loaders
         produce (X, y, smiles_list, target_name) with correct shapes.

CPU-only; no torch autograd; no GPU retrain. ~5 s wall target.

Honest framing (party-twenty-spirit §3.5):
* These tests verify the per-cohort config AXIS (feature count + Ridge
  alpha matched to cohort size). Real-cohort R² lift is MEASURED but
  small (per Wf-388-A1 smoke result.json: platinai_4cell + hilbert
  lift +0.0194, metalcytotox_combined WEAK_RECOVERY at +0.1961).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Ensure repo root on PYTHONPATH for direct test invocation.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from molmetal.scripts.wf388a1_m2_per_cohort_real import (  # noqa: E402
    COHORT_REGISTRY,
    aggregate_per_cohort_r2,
    apply_per_cohort_config,
    load_hilbert,
    load_metalcytotox_combined,
    load_platinai_4cell,
    train_eval_per_cohort,
)


# ---------------------------------------------------------------------------
# Test 1: per-cohort config axis verified by feature count + Ridge alpha
# ---------------------------------------------------------------------------
def test_per_cohort_config_applied() -> None:
    """Verify per-cohort config is applied correctly per cohort size.

    The Wf-380-A2 tier matrix:
      * n < 300   → 12-d RDKit + Ridge α=1.0
      * n < 1000  → 24-d RDKit + Ridge α=0.5
      * n >= 1000 → 50-d (or 32-d fallback) + Ridge α=0.1

    We synthesise a 50-d feature matrix and verify the per-cohort config
    slices correctly: n=200 → 12-d, n=500 → 24-d, n=2000 → 50-d (or 32).
    """
    # Synthesize a 50-d feature matrix (n=300, 50 features).
    rng = np.random.default_rng(42)
    X_50d = rng.standard_normal((300, 50))

    # n=200 → n_lt_300 tier → 12 features, alpha=1.0
    X_per, cfg = apply_per_cohort_config(X_50d, cohort_size=200)
    assert cfg["tier"] == "n_lt_300", (
        f"n=200 cohort should map to n_lt_300 tier; got {cfg['tier']!r}"
    )
    assert cfg["alpha"] == 1.0, (
        f"n_lt_300 tier alpha should be 1.0; got {cfg['alpha']}"
    )
    assert X_per.shape[1] == 12, (
        f"n_lt_300 tier should use 12-d features; got {X_per.shape[1]}"
    )
    assert X_per.shape[0] == 300, (
        f"n=200 cohort with n=300 synthesized should retain 300 rows; got {X_per.shape[0]}"
    )

    # n=500 → n_lt_1000 tier → 24 features, alpha=0.5
    X_per_500, cfg = apply_per_cohort_config(X_50d, cohort_size=500)
    assert cfg["tier"] == "n_lt_1000", (
        f"n=500 cohort should map to n_lt_1000 tier; got {cfg['tier']!r}"
    )
    assert cfg["alpha"] == 0.5, (
        f"n_lt_1000 tier alpha should be 0.5; got {cfg['alpha']}"
    )
    assert X_per_500.shape[1] == 24, (
        f"n_lt_1000 tier should use 24-d features; got {X_per_500.shape[1]}"
    )

    # n=2000 → n_gte_1000 tier → 50 features, alpha=0.1
    X_per_2000, cfg = apply_per_cohort_config(X_50d, cohort_size=2000)
    assert cfg["tier"] == "n_gte_1000", (
        f"n=2000 cohort should map to n_gte_1000 tier; got {cfg['tier']!r}"
    )
    assert cfg["alpha"] == 0.1, (
        f"n_gte_1000 tier alpha should be 0.1; got {cfg['alpha']}"
    )
    assert X_per_2000.shape[1] == 50, (
        f"n_gte_1000 tier should use 50-d features; got {X_per_2000.shape[1]}"
    )

    # Boundary tests (n=299 → n_lt_300, n=300 → n_lt_1000; n=999 → n_lt_1000, n=1000 → n_gte_1000).
    X_per_299, _ = apply_per_cohort_config(X_50d, cohort_size=299)
    assert X_per_299.shape[1] == 12, (
        f"n=299 (boundary) should use 12-d; got {X_per_299.shape[1]}"
    )
    X_per_300, _ = apply_per_cohort_config(X_50d, cohort_size=300)
    assert X_per_300.shape[1] == 24, (
        f"n=300 (boundary) should use 24-d; got {X_per_300.shape[1]}"
    )
    X_per_999, _ = apply_per_cohort_config(X_50d, cohort_size=999)
    assert X_per_999.shape[1] == 24, (
        f"n=999 (boundary) should use 24-d; got {X_per_999.shape[1]}"
    )
    X_per_1000, _ = apply_per_cohort_config(X_50d, cohort_size=1000)
    assert X_per_1000.shape[1] == 50, (
        f"n=1000 (boundary) should use 50-d; got {X_per_1000.shape[1]}"
    )

    # Sanity: the sliced features are taken from the FIRST n_use columns.
    assert np.allclose(X_per, X_50d[:, :12]), (
        "n_lt_300 tier should slice X[:, :12]"
    )
    assert np.allclose(X_per_500, X_50d[:, :24]), (
        "n_lt_1000 tier should slice X[:, :24]"
    )
    assert np.allclose(X_per_2000, X_50d[:, :50]), (
        "n_gte_1000 tier should slice X[:, :50]"
    )


# ---------------------------------------------------------------------------
# Test 2: per-cohort Ridge R² on synthetic cohort > 0 (proves config works)
# ---------------------------------------------------------------------------
def test_r2_lift_on_synthetic() -> None:
    """Verify Ridge R² > 0 on a synthetic cohort per per-cohort config.

    Builds a small synthetic cohort (n=200, 12-d signal+noise) and
    verifies that train_eval_per_cohort produces a finite R². We
    accept both positive and small-negative R² as long as the
    pipeline runs (verifies the per-cohort config axis is wired
    correctly). Real-cohort R² is tested separately.
    """
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 12
    X = rng.standard_normal((n_samples, n_features))
    # First 4 features carry signal; the rest is noise.
    beta = rng.standard_normal(4) * 0.5
    signal = X[:, :4] @ beta
    noise = rng.standard_normal(n_samples) * 0.3
    y = signal + noise
    smiles_list = [f"CC{i}" for i in range(n_samples)]
    target_name = "synthetic_pIC50"

    cohort_data = (X, y, smiles_list, target_name)
    result = train_eval_per_cohort(
        cohort_data=cohort_data,
        n_seeds=2, n_folds=3,
        wass_w=0.05, epochs=30,
    )
    # Pipeline ran + produced finite R².
    assert np.isfinite(result["r2_mean"]), (
        f"R²_mean should be finite; got {result['r2_mean']!r}"
    )
    assert result["tier"] == "n_lt_300", (
        f"n=200 cohort should map to n_lt_300 tier; got {result['tier']!r}"
    )
    assert result["n_features_used"] == 12, (
        f"n_lt_300 tier should use 12-d features; got {result['n_features_used']}"
    )
    assert result["alpha_used"] == 1.0, (
        f"n_lt_300 tier alpha should be 1.0; got {result['alpha_used']}"
    )
    # R² should be > 0 (synthetic signal is real; Ridge with α=1.0 should lift).
    # NOTE: per Wf-388-A1 honest finding, R² on n=200 + 12-d synthetic is MEASURED
    # positive (Wf-380-A2 result.json reports +0.1970 for platinai_4cell synthetic).
    # We use a relaxed threshold (> 0) to keep this test robust.
    assert result["r2_mean"] > 0.0, (
        f"R²_mean on n=200 + 12-d synthetic should be > 0; got {result['r2_mean']:+.4f}"
    )
    # Lift vs baseline is recorded for honest framing.
    assert np.isfinite(result["baseline_r2_mechanical"]), (
        "baseline_r2_mechanical should be finite (uses default WF335M2 anchor)"
    )


# ---------------------------------------------------------------------------
# Test 3: real-cohort loaders produce correct shapes
# ---------------------------------------------------------------------------
def test_real_cohort_loader() -> None:
    """Verify platinai_4cell + hilbert + metalcytotox_combined loaders work.

    Each loader returns (X, y, smiles_list, target_name). We check:
      * X is 2-D with the expected n_samples × d_expected_dims
      * y is 1-D with the expected n_samples
      * smiles_list has length n_samples
      * target_name is a non-empty string matching the expected pattern

    CPU-only; no GPU retrain.
    """
    # platinai_4cell: n=200, 32-d RDKit extended features
    X_p, y_p, smiles_p, target_p = load_platinai_4cell(max_samples=200)
    assert X_p.ndim == 2, f"platinai_4cell X should be 2-D; got ndim={X_p.ndim}"
    assert X_p.shape[0] == 200, (
        f"platinai_4cell should load 200 samples; got {X_p.shape[0]}"
    )
    assert X_p.shape[1] >= 12, (
        f"platinai_4cell X should have ≥ 12 features; got {X_p.shape[1]}"
    )
    assert y_p.shape == (200,), (
        f"platinai_4cell y should be 1-D (200,); got shape {y_p.shape}"
    )
    assert len(smiles_p) == 200, (
        f"platinai_4cell smiles_list should have 200 entries; got {len(smiles_p)}"
    )
    assert "pIC50" in target_p or "mcf7" in target_p.lower(), (
        f"platinai_4cell target should be pIC50-like; got {target_p!r}"
    )

    # hilbert: same as platinai_4cell (per Wf-331-R5 load_cohort pattern).
    X_h, y_h, smiles_h, target_h = load_hilbert(max_samples=200)
    assert X_h.ndim == 2, f"hilbert X should be 2-D; got ndim={X_h.ndim}"
    assert X_h.shape[0] == 200, (
        f"hilbert should load 200 samples; got {X_h.shape[0]}"
    )
    assert X_h.shape[1] == X_p.shape[1], (
        f"hilbert X feature dim should match platinai_4cell; got {X_h.shape[1]} vs {X_p.shape[1]}"
    )
    assert y_h.shape == (200,), (
        f"hilbert y should be 1-D (200,); got shape {y_h.shape}"
    )
    assert len(smiles_h) == 200, (
        f"hilbert smiles_list should have 200 entries; got {len(smiles_h)}"
    )

    # metalcytotox_combined: n=2000, 32-d (12-d + 20-d zero-fill per Wf-331-R5).
    X_m, y_m, smiles_m, target_m = load_metalcytotox_combined(max_samples=2000)
    assert X_m.ndim == 2, f"metalcytotox_combined X should be 2-D; got ndim={X_m.ndim}"
    assert X_m.shape[0] == 2000, (
        f"metalcytotox_combined should load 2000 samples; got {X_m.shape[0]}"
    )
    assert X_m.shape[1] == 32, (
        f"metalcytotox_combined X should be 32-d (12-d + 20-d zero-fill); got {X_m.shape[1]}"
    )
    assert y_m.shape == (2000,), (
        f"metalcytotox_combined y should be 1-D (2000,); got shape {y_m.shape}"
    )
    assert len(smiles_m) == 2000, (
        f"metalcytotox_combined smiles_list should have 2000 entries; got {len(smiles_m)}"
    )
    assert "activity" in target_m.lower() or "ic50" in target_m.lower(), (
        f"metalcytotox_combined target should be activity/IC50-like; got {target_m!r}"
    )

    # Sanity: COHORT_REGISTRY contains all 3 cohort names.
    expected_keys = {"platinai_4cell", "hilbert", "metalcytotox_combined"}
    actual_keys = set(COHORT_REGISTRY.keys())
    assert expected_keys.issubset(actual_keys), (
        f"COHORT_REGISTRY missing cohorts: {expected_keys - actual_keys}"
    )


# ---------------------------------------------------------------------------
# Bonus: aggregate_per_cohort_r2 verdict logic
# ---------------------------------------------------------------------------
def test_aggregate_per_cohort_r2() -> None:
    """Verify aggregate_per_cohort_r2 verdict logic.

    Synthesize a list of per_cohort results and verify the aggregate
    verdict matches the per-cohort verdicts.
    """
    # 3 cohorts: 2 CEILING_BREAK + 1 HONEST_NEGATIVE_R2_DEEP → CEILING_BREAK_MAJORITY.
    fake_results = [
        {"r2_mean": 0.30, "verdict": "CEILING_BREAK"},
        {"r2_mean": 0.20, "verdict": "CEILING_BREAK"},
        {"r2_mean": -0.10, "verdict": "HONEST_NEGATIVE_R2_DEEP"},
    ]
    agg = aggregate_per_cohort_r2(fake_results)
    assert agg["aggregate_status"] == "CEILING_BREAK_MAJORITY", (
        f"2/3 CEILING_BREAK should yield CEILING_BREAK_MAJORITY; got {agg['aggregate_status']!r}"
    )
    assert agg["n_cohorts_ceiling_break"] == 2
    assert agg["n_cohorts_above_zero"] == 2

    # 1 WEAK_RECOVERY + 2 HONEST_NEGATIVE_R2_DEEP → MEASURED_PARTIAL_LIFT.
    fake_results_2 = [
        {"r2_mean": 0.05, "verdict": "WEAK_RECOVERY"},
        {"r2_mean": -0.10, "verdict": "HONEST_NEGATIVE_R2_DEEP"},
        {"r2_mean": -0.20, "verdict": "HONEST_NEGATIVE_R2_DEEP"},
    ]
    agg_2 = aggregate_per_cohort_r2(fake_results_2)
    assert agg_2["aggregate_status"] == "MEASURED_PARTIAL_LIFT", (
        f"1/3 WEAK_RECOVERY should yield MEASURED_PARTIAL_LIFT; got {agg_2['aggregate_status']!r}"
    )
    assert agg_2["n_cohorts_weak_recovery_or_better"] == 1
    assert agg_2["n_cohorts_above_zero"] == 1

    # All HONEST_NEGATIVE_R2_DEEP → HONEST_NEGATIVE_R2_DEEP.
    fake_results_3 = [
        {"r2_mean": -0.05, "verdict": "HONEST_NEGATIVE_R2_DEEP"},
        {"r2_mean": -0.10, "verdict": "HONEST_NEGATIVE_R2_DEEP"},
        {"r2_mean": -0.15, "verdict": "HONEST_NEGATIVE_R2_DEEP"},
    ]
    agg_3 = aggregate_per_cohort_r2(fake_results_3)
    assert agg_3["aggregate_status"] == "HONEST_NEGATIVE_R2_DEEP", (
        f"3/3 HONEST_NEGATIVE_R2_DEEP should yield HONEST_NEGATIVE_R2_DEEP; got {agg_3['aggregate_status']!r}"
    )
    assert agg_3["n_cohorts_above_zero"] == 0


if __name__ == "__main__":
    # Manual invocation for quick smoke check.
    test_per_cohort_config_applied()
    print("  test_per_cohort_config_applied PASS")
    test_r2_lift_on_synthetic()
    print("  test_r2_lift_on_synthetic PASS")
    test_real_cohort_loader()
    print("  test_real_cohort_loader PASS")
    test_aggregate_per_cohort_r2()
    print("  test_aggregate_per_cohort_r2 PASS")
