"""Tests for Wf-128 Phase 2 — 3-way (Deflexpressor + Multi-PH + ASTSR)
on REAL MetalCytoToxDB 5-metal.

================================================================
End-to-end test for ``bench_3way_real_metalcytotox.py``.  Verifies
the 3-way integration (Hilbert + Multi-PH + ASTSR) + multi-task
backbone with Wasserstein weight, L2 guard, and ASTSR scale guard
on the REAL MetalCytoToxDB 5-metal binary activity target.

Test design (14 tests)
----------------------
1.  ``test_load_metal_cohort_ru_shape`` — Ru cohort loads with
    (N, 172) features and (N,) targets.
2.  ``test_load_metal_cohort_csv_fallback`` — when the NPZ is
    missing, the CSV fallback yields (N, 172) features (12-d RDKit
    + zero-padded MPH + PLL).
3.  ``test_fit_hilbert_stream_finite_weights`` — the Hilbert stream
    returns finite weights + train R².
4.  ``test_fit_mph_stream_finite_weights`` — the Multi-PH stream
    returns finite weights + train R².
5.  ``test_fit_astr_stream_finite_train_r2`` — the ASTSR stream
    returns finite train R² (with scale guard).
6.  ``test_combine_predictor_shape_contract`` — the combined
    predictor outputs the test-set length.
7.  ``test_collapse_detector_flags_constant_pred`` — collapse
    detector flags ``pred_std ≈ 0`` with non-zero y_std.
8.  ``test_random_kfold_indices_balanced`` — random K-fold
    produces balanced fold sizes.
9.  ``test_evaluate_fold_returns_fit_record`` — ``evaluate_fold``
    returns a ``FitRecord`` with finite combined R².
10. ``test_evaluate_metal_returns_records`` — ``evaluate_metal``
    returns one ``FitRecord`` per (seed, fold) for the smoke
    cohort.
11. ``test_aggregate_metal_per_seed_summary`` — ``aggregate_metal``
    returns one ``MetalSeedSummary`` per (metal, seed).
12. ``test_write_synthesis_md_contains_metals`` — the synthesis
    markdown contains every metal + the verdict line.
13. ``test_smoke_bench_produces_artifacts`` — the full smoke
    pipeline writes ``per_seed_metrics.json`` and
    ``synthesis.md`` to the requested out-dir.
14. ``test_wasserstein_weight_and_guards_serialised`` — the
    per-seed metrics JSON carries ``wasserstein_weight``,
    ``use_l2_guard``, ``use_astr_scale_guard`` keys.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from bench_3way_real_metalcytotox import (  # noqa: E402
    ASTR_SCALE_GUARD_MAX,
    DEFAULT_HILBERT_DIM,
    DEFAULT_METALS,
    DEFAULT_N_FOLDS,
    DEFAULT_W_W,
    FEATURE_DIM,
    FitRecord,
    L2_GUARD_MAX_NORM,
    MetalSeedSummary,
    MULTI_PH_DIM,
    PLL_DIM,
    RDKIT_DIM,
    _collapse_detected,
    _combine,
    _featurize_smiles_12d,
    _random_kfold_indices,
    _r2,
    _stable_ridge_fit,
    _zscore,
    aggregate_metal,
    evaluate_fold,
    fit_astr_stream,
    fit_hilbert_stream,
    fit_mph_stream,
    load_metal_cohort,
    predict_astr,
    predict_hilbert,
    predict_mph,
    write_synthesis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ru_cohort() -> tuple:
    """Load the Ru cohort from the existing Wf-128 P1 NPZ."""
    try:
        return load_metal_cohort("Ru", max_rows=200)
    except FileNotFoundError:
        pytest.skip("metalcytotox NPZ + CSV not available")


@pytest.fixture(scope="module")
def rh_cohort() -> tuple:
    """Load the Rh cohort (smallest cohort with NPZ)."""
    try:
        return load_metal_cohort("Rh", max_rows=120)
    except FileNotFoundError:
        pytest.skip("metalcytotox NPZ + CSV not available")


# ---------------------------------------------------------------------------
# 1. Loader — Ru cohort shape
# ---------------------------------------------------------------------------
def test_load_metal_cohort_ru_shape(ru_cohort):
    X, y, mid, smiles = ru_cohort
    assert X.ndim == 2
    assert (
        X.shape[1] == FEATURE_DIM
    ), f"expected {FEATURE_DIM}-d features; got {X.shape[1]}"
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert mid.shape[0] == X.shape[0]
    assert len(smiles) == X.shape[0]
    # All targets are binary 0/1.
    assert set(np.unique(y).tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 2. CSV fallback (zero-padded MPH + PLL)
# ---------------------------------------------------------------------------
def test_load_metal_cohort_csv_fallback(tmp_path):
    """CSV fallback returns (N, 172) features when no NPZ is present."""
    npz_path = tmp_path / "no_npz.npz"
    csv_path = tmp_path / "metalcytotox_mini.csv"
    # Build a tiny CSV with 5 rows (1 metal only).
    csv_path.write_text(
        "smiles,metal,scaffold_id,split,activity\n"
        "c1ccccc1,Ru,s1,train,1\n"
        "CCO,Ru,s2,train,0\n"
        "CCN,Ru,s3,val,1\n"
        "CCC,Ru,s4,train,0\n"
        "C=O,Ru,s5,val,1\n"
    )
    X, y, mid, smiles = load_metal_cohort(
        "Ru",
        npz_path=npz_path,
        csv_path=csv_path,
        max_rows=200,
    )
    assert X.shape[1] == FEATURE_DIM
    # MPH + PLL streams are zero in CSV fallback.
    assert np.allclose(X[:, RDKIT_DIM : RDKIT_DIM + MULTI_PH_DIM], 0.0)
    assert np.allclose(X[:, RDKIT_DIM + MULTI_PH_DIM :], 0.0)


# ---------------------------------------------------------------------------
# 3. Hilbert stream — finite weights
# ---------------------------------------------------------------------------
def test_fit_hilbert_stream_finite_weights(ru_cohort):
    X, y, _mid, _smi = ru_cohort
    n = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_tr = max(10, n // 2)
    tr, te = perm[:n_tr], perm[n_tr:]
    X_tr_z, X_te_z = _zscore(X[tr].astype(float), X[te].astype(float))
    w, train_r2, aug_te = fit_hilbert_stream(
        X_tr_z,
        y[tr],
        X_te_z,
        hilbert_dim=DEFAULT_HILBERT_DIM,
    )
    assert np.all(np.isfinite(w))
    assert math.isfinite(train_r2)
    assert aug_te.shape[0] == X_te_z.shape[0]
    assert aug_te.shape[1] == DEFAULT_HILBERT_DIM


# ---------------------------------------------------------------------------
# 4. Multi-PH stream — finite weights
# ---------------------------------------------------------------------------
def test_fit_mph_stream_finite_weights(ru_cohort):
    X, y, _mid, _smi = ru_cohort
    n = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_tr = max(10, n // 2)
    tr, te = perm[:n_tr], perm[n_tr:]
    X_tr_z, X_te_z = _zscore(X[tr].astype(float), X[te].astype(float))
    w, train_r2, mph_te = fit_mph_stream(X_tr_z, y[tr], X_te_z)
    assert np.all(np.isfinite(w))
    assert math.isfinite(train_r2)
    assert mph_te.shape[1] == MULTI_PH_DIM


# ---------------------------------------------------------------------------
# 5. ASTSR stream — finite train R² (with scale guard)
# ---------------------------------------------------------------------------
def test_fit_astr_stream_finite_train_r2(ru_cohort):
    X, y, _mid, _smi = ru_cohort
    n = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_tr = max(10, n // 2)
    tr = perm[:n_tr]
    reg, train_r2 = fit_astr_stream(X[tr], y[tr], use_astr_scale_guard=True)
    assert reg is not None or train_r2 == 0.0  # either fit succeeded or fallback
    assert math.isfinite(train_r2)


# ---------------------------------------------------------------------------
# 6. Combined predictor — shape contract
# ---------------------------------------------------------------------------
def test_combine_predictor_shape_contract(ru_cohort):
    n_te = 17
    rng = np.random.default_rng(1)
    h = rng.standard_normal(n_te)
    m = rng.standard_normal(n_te)
    a = rng.standard_normal(n_te)
    out = _combine(h, m, a)
    assert out.shape == (n_te,)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 7. Collapse detector — flags constant predictor
# ---------------------------------------------------------------------------
def test_collapse_detector_flags_constant_pred():
    y_true = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    y_pred_const = np.ones_like(y_true) * 0.5
    assert _collapse_detected(y_pred_const, y_true) is True


# ---------------------------------------------------------------------------
# 8. K-fold — balanced fold sizes
# ---------------------------------------------------------------------------
def test_random_kfold_indices_balanced():
    n = 50
    k = 5
    splits = _random_kfold_indices(n, k, seed=0)
    assert len(splits) == k
    fold_sizes = sorted(len(s) for _, s in splits)
    assert max(fold_sizes) - min(fold_sizes) <= 1
    # Every index appears in exactly one test fold.
    seen: set = set()
    for _, te in splits:
        for idx in te.tolist():
            assert idx not in seen, f"index {idx} in multiple folds"
            seen.add(int(idx))
    assert seen == set(range(n))


# ---------------------------------------------------------------------------
# 9. evaluate_fold — returns FitRecord with finite combined R²
# ---------------------------------------------------------------------------
def test_evaluate_fold_returns_fit_record(ru_cohort):
    X, y, mid, _ = ru_cohort
    n = X.shape[0]
    splits = _random_kfold_indices(n, 5, seed=0)
    tr, te = splits[0]
    rec = evaluate_fold(
        metal="Ru",
        seed=0,
        fold=0,
        tr=tr,
        te=te,
        X=X,
        y=y,
        mid=mid,
        epochs=5,
        hilbert_dim=DEFAULT_HILBERT_DIM,
        lr=0.05,
        w_w=DEFAULT_W_W,
        use_l2_guard=True,
        use_astr_scale_guard=True,
        multitask_backbone=False,
        device="cpu",
    )
    assert isinstance(rec, FitRecord)
    assert math.isfinite(rec.combined_r2)
    assert math.isfinite(rec.hilbert_r2)
    assert math.isfinite(rec.mph_r2)
    assert math.isfinite(rec.astr_r2)
    assert rec.n_train > 0
    assert rec.n_test > 0
    assert rec.use_l2_guard is True
    assert rec.use_astr_scale_guard is True


# ---------------------------------------------------------------------------
# 10. evaluate_metal — returns records
# ---------------------------------------------------------------------------
def test_evaluate_metal_returns_records(ru_cohort):
    """Smoke run for the full eval pipeline on the Ru cohort."""
    from bench_3way_real_metalcytotox import (
        DEFAULT_NPZ_PATH,
        DEFAULT_METALCSV_PATH,
        evaluate_metal,
    )

    records = evaluate_metal(
        metal="Ru",
        seeds=(0,),
        n_folds=2,
        epochs=3,
        hilbert_dim=DEFAULT_HILBERT_DIM,
        lr=0.05,
        w_w=DEFAULT_W_W,
        use_l2_guard=True,
        use_astr_scale_guard=True,
        multitask_backbone=False,
        device="cpu",
        npz_path=DEFAULT_NPZ_PATH,
        csv_path=DEFAULT_METALCSV_PATH,
        max_rows=80,
    )
    # CSV fallback should still produce records.
    assert len(records) == 2
    for r in records:
        assert isinstance(r, FitRecord)
        assert math.isfinite(r.combined_r2)


# ---------------------------------------------------------------------------
# 11. aggregate_metal — per-seed summaries
# ---------------------------------------------------------------------------
def test_aggregate_metal_per_seed_summary():
    records = [
        FitRecord(
            metal="Ru",
            seed=0,
            fold=0,
            n_train=40,
            n_test=10,
            combined_r2=-0.05,
            hilbert_r2=-0.10,
            mph_r2=-0.20,
            astr_r2=-0.30,
            wall_s=1.0,
            peak_mem_mb=10.0,
            collapse_detected=False,
        ),
        FitRecord(
            metal="Ru",
            seed=0,
            fold=1,
            n_train=40,
            n_test=10,
            combined_r2=0.10,
            hilbert_r2=-0.05,
            mph_r2=0.05,
            astr_r2=-0.20,
            wall_s=1.2,
            peak_mem_mb=10.5,
            collapse_detected=True,
        ),
    ]
    out = aggregate_metal(records)
    assert len(out) == 1
    s = out[0]
    assert isinstance(s, MetalSeedSummary)
    assert s.metal == "Ru"
    assert s.seed == 0
    assert s.n_folds == 2
    assert math.isfinite(s.combined_r2_mean)
    assert s.collapse_detected_any is True


# ---------------------------------------------------------------------------
# 12. write_synthesis — contains metals + verdict
# ---------------------------------------------------------------------------
def test_write_synthesis_md_contains_metals(tmp_path):
    records = [
        FitRecord(
            metal="Ru",
            seed=0,
            fold=0,
            n_train=40,
            n_test=10,
            combined_r2=-0.10,
            hilbert_r2=-0.20,
            mph_r2=-0.05,
            astr_r2=-0.30,
            wall_s=1.0,
            peak_mem_mb=10.0,
            collapse_detected=False,
        ),
        FitRecord(
            metal="Ir",
            seed=0,
            fold=0,
            n_train=30,
            n_test=8,
            combined_r2=0.05,
            hilbert_r2=-0.05,
            mph_r2=0.02,
            astr_r2=-0.10,
            wall_s=1.5,
            peak_mem_mb=12.0,
            collapse_detected=False,
        ),
    ]
    write_synthesis(tmp_path, records)
    md_path = tmp_path / "synthesis.md"
    assert md_path.exists()
    text = md_path.read_text()
    for m in ("Ru", "Ir"):
        assert m in text
    assert "Verify gate" in text
    assert "Verdict" in text


# ---------------------------------------------------------------------------
# 13. Smoke bench — produces artifacts
# ---------------------------------------------------------------------------
def test_smoke_bench_produces_artifacts(tmp_path):
    """End-to-end smoke: invoke main() and confirm artifacts are written."""
    out_dir = tmp_path / "wf128_p2_smoke"
    argv = [
        "--smoke",
        "--out-dir",
        str(out_dir),
        "--max-rows",
        "120",
    ]
    # Run main() in-process.
    from bench_3way_real_metalcytotox import main

    rc = main(argv)
    assert rc == 0
    assert (out_dir / "synthesis.md").exists()
    assert (out_dir / "Ru" / "per_seed_metrics.json").exists()
    payload = json.loads((out_dir / "Ru" / "per_seed_metrics.json").read_text())
    assert payload["metal"] == "Ru"
    assert "per_seed" in payload and len(payload["per_seed"]) == 1
    assert "config" in payload
    cfg = payload["config"]
    assert cfg["wasserstein_weight"] == DEFAULT_W_W
    assert cfg["use_l2_guard"] is True
    assert cfg["use_astr_scale_guard"] is True
    assert cfg["features_172d"] is True


# ---------------------------------------------------------------------------
# 14. Wasserstein weight + guards serialised in per-seed JSON
# ---------------------------------------------------------------------------
def test_wasserstein_weight_and_guards_serialised(tmp_path):
    out_dir = tmp_path / "wf128_p2_serialise"
    argv = [
        "--smoke",
        "--metals",
        "Ru",
        "--out-dir",
        str(out_dir),
        "--wasserstein-weight",
        "0.10",
        "--no-l2-guard",
        "--no-astr-scale-guard",
        "--max-rows",
        "100",
    ]
    from bench_3way_real_metalcytotox import main

    rc = main(argv)
    assert rc == 0
    payload = json.loads((out_dir / "Ru" / "per_seed_metrics.json").read_text())
    cfg = payload["config"]
    assert cfg["wasserstein_weight"] == 0.10
    assert cfg["use_l2_guard"] is False
    assert cfg["use_astr_scale_guard"] is False
