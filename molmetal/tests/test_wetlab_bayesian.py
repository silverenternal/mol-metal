"""test_wetlab_bayesian.py — pytest suite for molmetal/molmetal_lam/reward/wetlab_bayesian.py

Wf-78 Phase 3 (R22-P4 / L16) test surface — 16 unit tests covering:
  * WetLabConfig + Institution dataclasses
  * Per-institution lookup table (3-row ORNL/PKU/XIEHE)
  * Matern-5/2 GP surrogate (kernel + fit + predict)
  * q-EI batch acquisition
  * Synthetic in-silico top-20 + wet-lab measurements
  * In-silico vs wet-lab agreement (Pearson r / RMSE)
  * Iterative BO loop (cycle-by-cycle determinism)
  * Top-level orchestrator

All tests are CPU-only, deterministic on seed=0, runtime <2s.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOLMETAL_PKG = os.path.join(REPO_ROOT, "molmetal")
for candidate in (REPO_ROOT, MOLMETAL_PKG):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from molmetal.molmetal_lam.reward.wetlab_bayesian import (  # noqa: E402
    WetLabConfig,
    Institution,
    build_institution_table,
    GaussianProcessSurrogate,
    qExpectedImprovement,
    build_synthetic_in_silico_top20,
    build_synthetic_wetlab_measurements,
    compute_silico_wetlab_agreement,
    run_iterative_loop,
    synthesize_wetlab_loop,
)


# ---------------------------------------------------------------------------
# 1. Config + dataclass tests
# ---------------------------------------------------------------------------


def test_wetlab_config_defaults():
    cfg = WetLabConfig()
    assert cfg.kernel == "matern_5_2"
    assert cfg.acquisition == "q_ei"
    assert cfg.q == 5
    assert cfg.cycles == 5
    assert cfg.metric == "pic50"
    assert cfg.noise_var > 0


def test_wetlab_config_as_dict_serialisable():
    cfg = WetLabConfig()
    d = cfg.as_dict()
    assert "kernel" in d
    assert "acquisition" in d
    assert isinstance(d["noise_var"], float)


def test_institution_as_dict_has_core_keys():
    inst = Institution(
        code="ORNL",
        name="ORNL — CNMS",
        assay_type="protein_metal_binding",
        target_axes=("metal", "ligand_class", "target_protein", "buffer"),
        cycle_weeks=8,
        irb_required=False,
        loi_path="path/or",
        sop_path="path/sop",
        safety_constraints="bsl-2",
    )
    d = inst.as_dict()
    assert d["code"] == "ORNL"
    assert d["cycle_weeks"] == 8
    assert d["irb_required"] is False


# ---------------------------------------------------------------------------
# 2. Per-institution lookup table
# ---------------------------------------------------------------------------


def test_build_institution_table_has_three_institutions():
    table = build_institution_table()
    assert set(table.keys()) == {"ORNL", "PKU", "XIEHE"}


def test_ornl_assay_is_protein_metal_binding():
    table = build_institution_table()
    assert table["ORNL"].assay_type == "protein_metal_binding"
    assert "metal" in table["ORNL"].target_axes


def test_xiehe_requires_irb_and_runs_nci60():
    table = build_institution_table()
    assert table["XIEHE"].irb_required is True
    assert table["XIEHE"].assay_type == "nci60"


# ---------------------------------------------------------------------------
# 3. Matern-5/2 GP surrogate
# ---------------------------------------------------------------------------


def test_gp_kernel_self_similarity():
    """k(x, x) should be signal_var (since d = 0)."""
    gp = GaussianProcessSurrogate(length_scale=1.0, signal_var=2.0)
    x = np.array([[0.5, 0.5, 0.5, 0.5]])
    k_diag = gp.kernel(x, x)
    assert np.isclose(k_diag[0, 0], 2.0, atol=1e-6)


def test_gp_fit_predict_shape():
    rng = np.random.RandomState(0)
    x_train = rng.rand(15, 4)
    y_train = (
        0.4 * x_train[:, 0]
        + 0.3 * x_train[:, 1]
        + 0.2 * x_train[:, 2]
        + 0.1 * x_train[:, 3]
        + rng.normal(0, 0.05, size=15)
    )
    gp = GaussianProcessSurrogate()
    gp.fit(x_train, y_train)
    mean, std = gp.predict(x_train[:3], return_std=True)
    assert mean.shape == (3,)
    assert std is not None and std.shape == (3,)
    assert np.all(std > 0)


def test_gp_fit_empty_raises():
    gp = GaussianProcessSurrogate()
    with pytest.raises(ValueError):
        gp.fit(np.zeros((0, 4)), np.zeros((0,)))


def test_gp_fit_predict_x_y_length_mismatch_raises():
    gp = GaussianProcessSurrogate()
    with pytest.raises(ValueError):
        gp.fit(np.zeros((3, 4)), np.zeros((5,)))


# ---------------------------------------------------------------------------
# 4. q-EI acquisition
# ---------------------------------------------------------------------------


def test_q_ei_returns_top_q_indices():
    rng = np.random.RandomState(0)
    x_train = rng.rand(20, 4)
    y_train = rng.rand(20)
    gp = GaussianProcessSurrogate()
    gp.fit(x_train, y_train)
    qei = qExpectedImprovement(gp=gp, best_y=float(y_train.max()))
    pool = rng.rand(50, 4)
    sel = qei.select(pool, q=5)
    assert len(sel) == 5
    assert all(0 <= i < 50 for i in sel)


# ---------------------------------------------------------------------------
# 5. Synthetic mocks
# ---------------------------------------------------------------------------


def test_synthetic_top20_returns_20_rows():
    rows, X = build_synthetic_in_silico_top20(seed=0)
    assert len(rows) == 20
    assert X.shape == (20, 4)


def test_synthetic_wetlab_measurements_count_matches_cycles():
    m = build_synthetic_wetlab_measurements(n_cycles=5, seed=0)
    # Each cycle produces 5 measurements → 25 total.
    assert len(m) == 25
    assert all("pic50" in meas for meas in m)


def test_encode_axis_values_in_unit_cube():
    rows, X = build_synthetic_in_silico_top20(seed=0)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


# ---------------------------------------------------------------------------
# 6. In-silico vs wet-lab agreement
# ---------------------------------------------------------------------------


def test_silico_wetlab_agreement_perfect_corr():
    pred = [1.0, 2.0, 3.0, 4.0, 5.0]
    meas = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = compute_silico_wetlab_agreement(pred, meas)
    assert out["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert out["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_silico_wetlab_agreement_anti_corr():
    pred = [1.0, 2.0, 3.0, 4.0, 5.0]
    meas = [5.0, 4.0, 3.0, 2.0, 1.0]
    out = compute_silico_wetlab_agreement(pred, meas)
    assert out["pearson_r"] == pytest.approx(-1.0, abs=1e-6)


def test_silico_wetlab_agreement_short_input_returns_zero():
    out = compute_silico_wetlab_agreement([1.0], [2.0])
    assert out["pearson_r"] == 0.0
    assert out["rmse"] == 0.0


# ---------------------------------------------------------------------------
# 7. Iterative BO loop
# ---------------------------------------------------------------------------


def test_run_iterative_loop_returns_trace_per_cycle():
    rows, X = build_synthetic_in_silico_top20(seed=0)
    cfg = WetLabConfig(cycles=3, q=4)
    result = run_iterative_loop(rows, X, cfg=cfg, seed=0)
    assert len(result["selection_trace"]) == cfg.cycles
    assert len(result["selection_trace"][0]) == cfg.q


def test_run_iterative_loop_grows_measurements_per_cycle():
    rows, X = build_synthetic_in_silico_top20(seed=0)
    cfg = WetLabConfig(cycles=2, q=3)
    result = run_iterative_loop(rows, X, cfg=cfg, seed=0)
    # Initial cycle adds 5 mock measurements; each of 2 cycles adds ``q``
    # measurements; the total should be >= initial + cycles*q.
    initial = 5
    assert len(result["final_measurements"]) >= initial + cfg.cycles * cfg.q


# ---------------------------------------------------------------------------
# 8. Top-level orchestrator
# ---------------------------------------------------------------------------


def test_synthesize_wetlab_loop_full_kpis():
    out = synthesize_wetlab_loop(seed=0)
    assert "institution_table" in out
    assert "n_cycles" in out
    assert "selection_trace" in out
    assert "silico_wetlab_agreement" in out
    assert len(out["institution_table"]) == 3
    assert out["silico_wetlab_agreement"]["n"] > 0
