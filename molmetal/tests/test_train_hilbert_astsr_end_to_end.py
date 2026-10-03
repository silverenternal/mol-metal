"""Tests for Wf-37 Phase 1 — Hilbert-ASTSR end-to-end training on real IC50
(TODO-83 §L5, 2026-09-26).

================================================================
Validates that ``molmetal/scripts/train_hilbert_astsr_end_to_end.py``
produces the r2_bench.json + comparison_table.md + run.log artefacts
the Wf-37 L5 contract requires and that the Hilbert-ASTSR composite
has fixed the Wf-25 numerical instability (joint R² -0.443 → ≥ 0.0).

Test inventory (~20 tests)
--------------------------
1. **Artefact existence** — r2_bench.json, comparison_table.md,
   run.log under ``molmetal/reports/wf37_hilbert_astsr/`` (3 tests).
2. **Schema correctness** — required keys, types, config payload
   round-trip (3 tests).
3. **Per-seed consistency** — len(per_seed) == seeds, per-seed dicts
   have all expected scalars, per_fold length matches (3 tests).
4. **Numerical stability (regression Wf-25)** — Hilbert R² mean is
   strictly positive (was -0.443 → now ≥ 0.05); speedup ≥ 10x
   (2 tests).
5. **Helper correctness** — loaders, featurizer, pad_to_hilbert_dim,
   z-score, fit_hilbert_stable, predict_hilbert, fit_evolutionary,
   predict_evolutionary, _compute_r2 (6 tests).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


REPORT_DIR = _REPO_ROOT / "molmetal/reports/wf37_hilbert_astsr"
BENCH_PATH = REPORT_DIR / "r2_bench.json"
TABLE_PATH = REPORT_DIR / "comparison_table.md"
LOG_PATH = REPORT_DIR / "run.log"


# ---------------------------------------------------------------------------
# Helpers — import the script module
# ---------------------------------------------------------------------------
def _import_script():
    """Import the training script as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_hilbert_astsr_end_to_end",
        str(_REPO_ROOT / "molmetal/scripts/train_hilbert_astsr_end_to_end.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Artefact existence
# ---------------------------------------------------------------------------
def test_bench_json_exists() -> None:
    assert BENCH_PATH.exists(), f"missing: {BENCH_PATH}"


def test_comparison_table_exists() -> None:
    assert TABLE_PATH.exists(), f"missing: {TABLE_PATH}"


def test_run_log_exists() -> None:
    assert LOG_PATH.exists(), f"missing: {LOG_PATH}"


# ---------------------------------------------------------------------------
# 2. Schema correctness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench() -> dict:
    if not BENCH_PATH.exists():
        pytest.skip(
            f"r2_bench.json not produced; run train_hilbert_astsr_end_to_end.py first"
        )
    with open(BENCH_PATH) as f:
        return json.load(f)


def test_bench_has_required_top_level_keys(bench: dict) -> None:
    expected = {
        "hilbert_r2_mean",
        "hilbert_r2_std",
        "hilbert_r2_per_seed",
        "evolutionary_r2_mean",
        "evolutionary_r2_std",
        "evolutionary_r2_per_seed",
        "hilbert_wall_mean_s",
        "evolutionary_wall_mean_s",
        "speedup",
        "n_samples",
        "n_seeds",
        "n_folds_per_seed",
        "parsed",
        "failed",
        "feature_columns",
        "population_dim",
        "config",
        "per_seed",
    }
    missing = expected - set(bench.keys())
    assert not missing, f"missing keys: {missing}"


def test_bench_types_correct(bench: dict) -> None:
    assert isinstance(bench["hilbert_r2_mean"], float)
    assert isinstance(bench["evolutionary_r2_mean"], float)
    assert isinstance(bench["hilbert_r2_per_seed"], list)
    assert isinstance(bench["per_seed"], list)
    assert isinstance(bench["config"], dict)
    assert isinstance(bench["feature_columns"], list)
    assert bench["population_dim"] == 64


def test_bench_config_round_trips(bench: dict) -> None:
    cfg = bench["config"]
    for k in (
        "dataset",
        "n_samples",
        "seeds",
        "epochs",
        "lr",
        "hilbert_dim",
        "n_folds",
    ):
        assert k in cfg, f"config.{k} missing"


# ---------------------------------------------------------------------------
# 3. Per-seed consistency
# ---------------------------------------------------------------------------
def test_per_seed_count_matches_seeds(bench: dict) -> None:
    assert (
        len(bench["per_seed"]) == bench["n_seeds"]
    ), f"per_seed count {len(bench['per_seed'])} != n_seeds {bench['n_seeds']}"


def test_per_seed_dict_has_expected_keys(bench: dict) -> None:
    expected = {
        "seed",
        "n_folds",
        "n_samples",
        "hilbert_r2_mean",
        "hilbert_r2_std",
        "evolutionary_r2_mean",
        "evolutionary_r2_std",
        "hilbert_wall_mean_s",
        "evol_wall_mean_s",
        "per_fold",
    }
    for entry in bench["per_seed"]:
        missing = expected - set(entry.keys())
        assert not missing, f"missing per_seed keys: {missing}"


def test_per_fold_count_matches(bench: dict) -> None:
    for entry in bench["per_seed"]:
        assert (
            len(entry["per_fold"]) == entry["n_folds"]
        ), f"per_fold count {len(entry['per_fold'])} != n_folds {entry['n_folds']}"


# ---------------------------------------------------------------------------
# 4. Numerical stability (regression Wf-25 bug)
# ---------------------------------------------------------------------------
def test_hilbert_R2_positive_after_fix(bench: dict) -> None:
    """Wf-25: hilbert R² = -0.443 (numerical instability). Wf-37 L5
    must fix the sign — Hilbert R² mean must be strictly positive on
    the real pIC50 target.
    """
    assert bench["hilbert_r2_mean"] > 0.0, (
        f"Wf-37 L5 regression: hilbert R²={bench['hilbert_r2_mean']:+.4f} "
        f"should be > 0.0 (was -0.443 in Wf-25)"
    )


def test_speedup_substantial(bench: dict) -> None:
    """Hilbert PD must be ≥ 10× faster than evolutionary ASTSR."""
    assert (
        bench["speedup"] >= 10.0
    ), f"speedup {bench['speedup']:.2f}x below 10x threshold"


# ---------------------------------------------------------------------------
# 5. Helper correctness — featurizer, padding, normalization, optimizers
# ---------------------------------------------------------------------------
def test_featurize_smiles_returns_8_to_12_dim() -> None:
    mod = _import_script()
    feats = mod._featurize_smiles("CCO")
    assert feats is not None
    assert len(feats) == len(mod.FEATURE_COLUMNS)
    assert all(math.isfinite(v) for v in feats)


def test_featurize_smiles_handles_invalid() -> None:
    mod = _import_script()
    # Truly invalid SMILES → None
    assert mod._featurize_smiles("not_a_smiles!!!") is None
    # Empty string → RDKit treats as empty mol (12 zeros), not None.
    # We only assert that the result is either None or a 12-d finite vector.
    out = mod._featurize_smiles("")
    if out is not None:
        assert len(out) == len(mod.FEATURE_COLUMNS)
        assert all(math.isfinite(v) for v in out)


def test_pad_to_hilbert_dim_consistent_train_test() -> None:
    """The pairwise-interaction augmentation must produce the same
    column layout for train and test (column-rank learned on training)."""
    mod = _import_script()
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((40, len(mod.FEATURE_COLUMNS)))
    X_te = rng.standard_normal((10, len(mod.FEATURE_COLUMNS)))
    out_tr, out_te = mod.pad_to_hilbert_dim(X_tr, mod.POPULATION_DIM, X_8d_test=X_te)
    assert out_tr.shape == (40, mod.POPULATION_DIM)
    assert out_te.shape == (10, mod.POPULATION_DIM)
    # Bias column present in both: look for a column that is constant 1.0
    # (its index depends on whether interactions were truncated).
    bias_col_tr = np.where(np.all(np.isclose(out_tr, 1.0), axis=0))[0]
    bias_col_te = np.where(np.all(np.isclose(out_te, 1.0), axis=0))[0]
    assert len(bias_col_tr) >= 1, "no bias column in train"
    assert np.array_equal(
        bias_col_tr, bias_col_te
    ), f"train/test bias columns differ: {bias_col_tr} vs {bias_col_te}"


def test_zscore_normalization_is_deterministic() -> None:
    mod = _import_script()
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((50, 8))
    X_te = rng.standard_normal((20, 8))
    a_tr, a_te, mu, sigma = mod.zscore_fit_transform(X_tr, X_te)
    b_tr, b_te, mu2, sigma2 = mod.zscore_fit_transform(X_tr, X_te)
    assert np.allclose(a_tr, b_tr)
    assert np.allclose(a_te, b_te)
    assert np.allclose(mu, mu2)
    assert np.allclose(sigma, sigma2)


def test_fit_hilbert_stable_returns_finite_r2() -> None:
    mod = _import_script()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, mod.POPULATION_DIM))
    w = rng.standard_normal(mod.POPULATION_DIM) * 0.1
    y = X @ w + 0.05 * rng.standard_normal(50)
    best_x, info = mod.fit_hilbert_stable(
        X,
        y,
        hilbert_dim=mod.POPULATION_DIM,
        lr=0.05,
        epochs=100,
        seed=0,
    )
    assert best_x.shape == (mod.POPULATION_DIM,)
    assert math.isfinite(info["train_r2"])
    assert (
        info["train_r2"] > 0.5
    ), f"linear-recovery R² should be > 0.5 on clean signal; got {info['train_r2']:.4f}"


def test_compute_r2_zero_on_constant_prediction() -> None:
    mod = _import_script()
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_hat = np.full_like(y, float(np.mean(y)))
    assert mod._compute_r2(y, y_hat) == 0.0


def test_compute_r2_perfect_on_identity() -> None:
    mod = _import_script()
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(mod._compute_r2(y, y), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 6. Loader smoke tests
# ---------------------------------------------------------------------------
def test_load_nci60_precious_filters_empty_pic50() -> None:
    mod = _import_script()
    csv_path = _REPO_ROOT / "molmetal/data/nci60_precious_9cell.csv"
    if not csv_path.exists():
        pytest.skip("NCI-60 CSV not present")
    rows = mod.load_nci60_precious(csv_path, n_max=20)
    assert len(rows) > 0
    for row in rows:
        assert "smiles" in row and "pic50" in row
        assert math.isfinite(row["pic50"])


def test_load_platinai_4cell_filters_empty_ic50() -> None:
    mod = _import_script()
    csv_path = _REPO_ROOT / "molmetal/data/platinai_4cell.csv"
    if not csv_path.exists():
        pytest.skip("PlatinAI 4cell CSV not present")
    rows = mod.load_platinai_4cell(csv_path, n_max=20)
    assert len(rows) > 0
    for row in rows:
        assert "smiles" in row and "pic50" in row
        assert math.isfinite(row["pic50"])


# ---------------------------------------------------------------------------
# 7. Wf-246-A wire-in — Wasserstein pIC50 alignment (TODO-179)
# ---------------------------------------------------------------------------
def test_wasserstein_pic50_loss_importable() -> None:
    """G1: wasserstein_pic50_loss must be importable from the M1 module."""
    mod = _import_script()
    try:
        wpl = mod._import_wasserstein_pic50_loss()
    except ImportError:
        pytest.skip("torch or wasserstein_loss module unavailable")
    assert callable(wpl), "wasserstein_pic50_loss must be callable"


def test_compute_w2_pic50_alignment_returns_per_cell_breakdown() -> None:
    """G5: per-cell breakdown with cap (TODO-179 §3 task spec)."""
    mod = _import_script()
    try:
        mod._import_wasserstein_pic50_loss()
    except ImportError:
        pytest.skip("torch or wasserstein_loss module unavailable")
    rng = np.random.default_rng(0)
    rows = [
        {"smiles": "CCO", "pic50": float(rng.uniform(4.0, 8.0)), "cell_line": "MCF7"}
        for _ in range(50)
    ] + [
        {"smiles": "CCN", "pic50": float(rng.uniform(4.0, 8.0)), "cell_line": "A549"}
        for _ in range(50)
    ]
    y_pred = np.array([r["pic50"] + rng.normal(0, 0.1) for r in rows])
    out = mod.compute_w2_pic50_alignment(
        y_pred,
        rows,
        ref_distribution="platinai_4cell",
        per_cell_cap=32,
        seed=0,
    )
    # Schema
    for k in (
        "w2_overall",
        "w2_per_cell",
        "n_per_cell",
        "ref_distribution",
        "per_cell_cap",
        "engine",
    ):
        assert k in out, f"missing key: {k}"
    assert out["engine"] == "wasserstein_reg"
    assert out["ref_distribution"] == "platinai_4cell"
    # Per-cell cap enforced at 32
    assert out["n_per_cell"]["MCF7"] == 32
    assert out["n_per_cell"]["A549"] == 32
    # Per-cell W₂ finite
    for cl, val in out["w2_per_cell"].items():
        assert math.isfinite(val), f"w2[{cl}] not finite: {val}"


def test_compute_w2_pic50_alignment_caps_at_32() -> None:
    """G6: cap 32 rows for O(N^2) bound per task spec."""
    mod = _import_script()
    try:
        mod._import_wasserstein_pic50_loss()
    except ImportError:
        pytest.skip("torch or wasserstein_loss module unavailable")
    rng = np.random.default_rng(0)
    # 100 rows in a single cell line → must be capped at 32.
    rows = [
        {"smiles": "CCO", "pic50": float(rng.uniform(4.0, 8.0)), "cell_line": "MCF7"}
        for _ in range(100)
    ]
    y_pred = np.array([r["pic50"] + rng.normal(0, 0.1) for r in rows])
    out = mod.compute_w2_pic50_alignment(
        y_pred,
        rows,
        ref_distribution="platinai_4cell",
        per_cell_cap=32,
        seed=0,
    )
    assert (
        out["n_per_cell"]["MCF7"] == 32
    ), f"cap not enforced: n_per_cell={out['n_per_cell']}"


def test_compute_w2_pic50_alignment_zero_on_identical() -> None:
    """W₂ must be 0 when predicted == reference distribution."""
    mod = _import_script()
    try:
        mod._import_wasserstein_pic50_loss()
    except ImportError:
        pytest.skip("torch or wasserstein_loss module unavailable")
    rng = np.random.default_rng(0)
    rows = [
        {"smiles": "CCO", "pic50": float(rng.uniform(4.0, 8.0)), "cell_line": "MCF7"}
        for _ in range(40)
    ]
    y_pred = np.array([r["pic50"] for r in rows])
    out = mod.compute_w2_pic50_alignment(
        y_pred,
        rows,
        ref_distribution="platinai_4cell",
        per_cell_cap=32,
        seed=0,
    )
    assert (
        out["w2_per_cell"]["MCF7"] < 1e-4
    ), f"W₂ should be ~0 for identical distributions; got {out['w2_per_cell']['MCF7']}"
    assert (
        out["w2_overall"] < 1e-4
    ), f"W₂ overall should be ~0 for identical distributions; got {out['w2_overall']}"


def test_run_single_seed_wass_weight_zero_disables_w2() -> None:
    """Backward compat: --wass-weight 0.0 → w2_aggregate is None."""
    mod = _import_script()
    rng = np.random.default_rng(0)
    # Use raw 12-d features (FEATURE_COLUMNS schema) — run_single_seed
    # passes them straight to predict_evolutionary which expects 12-d.
    X = rng.standard_normal((50, len(mod.FEATURE_COLUMNS)))
    w = rng.standard_normal(len(mod.FEATURE_COLUMNS)) * 0.1
    y = X @ w + 0.05 * rng.standard_normal(50)
    res = mod.run_single_seed(
        X,
        y,
        seed=0,
        hilbert_dim=mod.POPULATION_DIM,
        epochs=10,
        lr=0.05,
        n_folds=2,
        evolutionary_population=4,
        evolutionary_generations=2,
        wass_weight=0.0,
        rows_meta=None,
    )
    assert res["w2_aggregate"] is None
    assert res["w2_per_fold"] is None


def test_run_single_seed_wass_weight_positive_populates_w2() -> None:
    """G4: positive wass_weight + rows_meta → w2_aggregate populated."""
    mod = _import_script()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, len(mod.FEATURE_COLUMNS)))
    w = rng.standard_normal(len(mod.FEATURE_COLUMNS)) * 0.1
    y = X @ w + 0.05 * rng.standard_normal(50)
    rows_meta = [
        {
            "smiles": "CCO",
            "pic50": float(rng.uniform(4.0, 8.0)),
            "cell_line": "MCF7" if i % 2 == 0 else "A549",
        }
        for i in range(50)
    ]
    res = mod.run_single_seed(
        X,
        y,
        seed=0,
        hilbert_dim=mod.POPULATION_DIM,
        epochs=10,
        lr=0.05,
        n_folds=2,
        evolutionary_population=4,
        evolutionary_generations=2,
        wass_weight=0.05,
        ref_distribution="platinai_4cell",
        per_cell_cap=32,
        rows_meta=rows_meta,
    )
    assert res["w2_aggregate"] is not None
    assert "w2_overall" in res["w2_aggregate"]
    assert "w2_per_cell" in res["w2_aggregate"]
    assert res["w2_aggregate"]["engine"] == "wasserstein_reg"
    assert res["w2_aggregate"]["per_cell_cap"] == 32
    # Per-fold also populated
    assert res["w2_per_fold"] is not None
    assert len(res["w2_per_fold"]) == 2


def test_cli_flags_present() -> None:
    """G2/G3: --wass-weight and --ref-distribution flags must be defined."""
    mod = _import_script()
    # Re-run main() with --help, but cap on stdin — easier to inspect the
    # argparse parser directly.
    import argparse

    # Build the parser the same way main() does.
    # Simpler: invoke main with --help and check exit code 0 + argparse usage.
    import subprocess

    result = subprocess.run(
        [
            "/home/hugo/codes/mol-metal/.venv/bin/python",
            str(_REPO_ROOT / "molmetal/scripts/train_hilbert_astsr_end_to_end.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--wass-weight" in out, "missing --wass-weight flag"
    assert "--ref-distribution" in out, "missing --ref-distribution flag"
    assert "--per-cell-cap" in out, "missing --per-cell-cap flag"
    assert "platinai_4cell" in out, "missing default ref_distribution"
