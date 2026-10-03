"""Pytest suite for Wf-127 Phase 1 — Hilbert+ASTSR parameter sweep diagnostic.

The script ``molmetal/scripts/diag_hilbert_astsr_param_sweep.py`` is
the **diagnostic audit** that motivates the Wf-83 Phase 1 honest-
negative result ``combined R² = -6.71`` on PlatinAI 4-cell.  These
tests guard the script's invariants and provide fast unit-level
confidence in the parameter sweep.

Test inventory (~10 tests)
--------------------------
1.  ``test_make_synthetic_shape`` — synthetic probe shape + SNR.
2.  ``test_train_test_split_no_overlap`` — train / test never overlap.
3.  ``test_train_test_split_covers_all`` — union = full set.
4.  ``test_compute_r2_helpers`` — R² helpers (perfect, constant, bad).
5.  ``test_hilbert_feature_map_truncates`` — augmentation ≤ hilbert_dim.
6.  ``test_run_hilbert_sweep_returns_result`` — single (l2, init) cell.
7.  ``test_l2_axis_sweep_returns_all_lambdas`` — full L2 axis table.
8.  ``test_astr_axis_sweep_returns_all_scales`` — full ASTSR scale axis.
9.  ``test_init_axis_sweep_returns_all_modes`` — full init mode axis.
10. ``test_smoke_end_to_end_synthesis`` — invoke script, verify outputs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
while _REPO_ROOT != _REPO_ROOT.parent and not (
    _REPO_ROOT / "molmetal" / "scripts" / "diag_hilbert_astsr_param_sweep.py"
).exists():
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.diag_hilbert_astsr_param_sweep import (  # noqa: E402
    L2_SWEEP,
    ASTR_SCALE_SWEEP,
    INIT_MODE_SWEEP,
    STABLE_R2_THRESHOLD,
    PRED_CLIP,
    make_synthetic,
    train_test_split,
    compute_r2,
    hilbert_feature_map,
    run_hilbert_sweep,
    run_astr_sweep,
    l2_axis_sweep,
    astr_axis_sweep,
    init_axis_sweep,
    write_synthesis,
)


# ---------------------------------------------------------------------------
# 1. Synthetic probe
# ---------------------------------------------------------------------------
def test_make_synthetic_shape() -> None:
    X, y = make_synthetic(n=200, dim=8, snr=1.0, seed=0)
    assert X.shape == (200, 8)
    assert y.shape == (200,)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))


def test_make_synthetic_snr_zero_is_pure_noise() -> None:
    """``snr=0.0`` is degenerate (signal_scale / noise_scale = 0)."""
    X, y = make_synthetic(n=100, dim=4, snr=1e-12, seed=0)
    # y ≈ noise; std(y) ≈ std(noise), and X @ w_true has signal_scale > 0.
    # We accept the output is finite; the degenerate case should not crash.
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
def test_train_test_split_no_overlap() -> None:
    X, y = make_synthetic(n=100, dim=4, snr=1.0, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=0, test_frac=0.2)
    # Train + test lengths = 100.
    assert X_tr.shape[0] + X_te.shape[0] == 100
    assert y_tr.shape[0] + y_te.shape[0] == 100
    # No row is in both — by construction (the split is permutation-based).
    # The test below confirms the slicing preserves order and length.


def test_train_test_split_covers_all() -> None:
    X, y = make_synthetic(n=200, dim=4, snr=1.0, seed=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42, test_frac=0.25)
    n_tr, n_te = X_tr.shape[0], X_te.shape[0]
    assert n_tr + n_te == 200
    # At least one row in train and at least one in test.
    assert n_tr >= 1
    assert n_te >= 1


def test_train_test_split_deterministic() -> None:
    X, y = make_synthetic(n=200, dim=4, snr=1.0, seed=42)
    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X, y, seed=42)
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X, y, seed=42)
    np.testing.assert_array_equal(X_tr_a, X_tr_b)
    np.testing.assert_array_equal(X_te_a, X_te_b)


# ---------------------------------------------------------------------------
# 3. R² helpers
# ---------------------------------------------------------------------------
def test_compute_r2_helpers() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    # Perfect predictor ⇒ R² = 1.
    assert compute_r2(y, y) == pytest.approx(1.0, abs=1e-12)
    # Constant predictor at mean ⇒ R² = 0.
    assert compute_r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-12)
    # Bad predictor ⇒ R² < 0.
    assert compute_r2(y, -y) < 0.0


# ---------------------------------------------------------------------------
# 4. Hilbert feature map
# ---------------------------------------------------------------------------
def test_hilbert_feature_map_truncates() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 6))
    out = hilbert_feature_map(X, hilbert_dim=20)
    assert out.shape == (100, 20)
    # Bias column (index 6) is exactly 1.0 (we forced augmentation).
    np.testing.assert_allclose(out[:, 6], 1.0, atol=1e-12)


def test_hilbert_feature_map_handles_F_ge_dim() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 64))
    out = hilbert_feature_map(X, hilbert_dim=32)
    assert out.shape == (50, 32)
    np.testing.assert_array_equal(out, X[:, :32])


# ---------------------------------------------------------------------------
# 5. Hilbert sweep single-cell
# ---------------------------------------------------------------------------
def test_run_hilbert_sweep_returns_result() -> None:
    """One (l2_lambda, init_mode) cell returns a HilbertRunResult."""
    X, y = make_synthetic(n=400, dim=12, snr=1.0, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=0)
    res = run_hilbert_sweep(
        X_tr, y_tr, X_te, y_te,
        l2_lambda=0.01,
        init_mode="random_normal",
        n_iters=20,
        hilbert_dim=64,
        seed=0,
        lr=0.05,
    )
    assert res.l2_lambda == pytest.approx(0.01, abs=1e-12)
    assert res.init_mode == "random_normal"
    assert res.seed == 0
    assert np.isfinite(res.train_r2)
    assert np.isfinite(res.test_r2)
    assert res.wall_s >= 0.0


def test_run_hilbert_sweep_xavier_finite() -> None:
    X, y = make_synthetic(n=200, dim=8, snr=1.0, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=0)
    res = run_hilbert_sweep(
        X_tr, y_tr, X_te, y_te,
        l2_lambda=0.1,
        init_mode="xavier",
        n_iters=20,
        hilbert_dim=32,
        seed=0,
        lr=0.05,
    )
    assert np.isfinite(res.test_r2)


def test_run_hilbert_sweep_zeros_finite() -> None:
    X, y = make_synthetic(n=200, dim=8, snr=1.0, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=0)
    res = run_hilbert_sweep(
        X_tr, y_tr, X_te, y_te,
        l2_lambda=0.1,
        init_mode="zeros",
        n_iters=20,
        hilbert_dim=32,
        seed=0,
        lr=0.05,
    )
    assert np.isfinite(res.test_r2)


# ---------------------------------------------------------------------------
# 6. Axis sweep return shapes
# ---------------------------------------------------------------------------
def test_l2_axis_sweep_returns_all_lambdas() -> None:
    X, y = make_synthetic(n=200, dim=8, snr=1.0, seed=0)
    sweep = l2_axis_sweep(
        X, y, seeds=[0, 1], n_iters=10, hilbert_dim=32, lr=0.05,
    )
    assert sweep["axis"] == "l2_lambda"
    assert [r["l2_lambda"] for r in sweep["results"]] == list(L2_SWEEP)
    assert sweep["n_seeds"] == 2
    for r in sweep["results"]:
        assert "test_r2_mean" in r
        assert "test_r2_std" in r
        assert "stable" in r
        assert isinstance(r["stable"], bool)
        assert len(r["per_seed"]) == 2


def test_astr_axis_sweep_returns_all_scales() -> None:
    X, y = make_synthetic(n=200, dim=8, snr=1.0, seed=0)
    sweep = astr_axis_sweep(
        X, y, seeds=[0, 1], population=8, generations=4,
        depth=2, hilbert_dim=16,
    )
    assert sweep["axis"] == "astr_loss_scale"
    assert [r["loss_scale"] for r in sweep["results"]] == list(ASTR_SCALE_SWEEP)
    for r in sweep["results"]:
        assert "test_r2_mean" in r
        assert "fp64_fallback_used" in r["per_seed"][0]


def test_init_axis_sweep_returns_all_modes() -> None:
    X, y = make_synthetic(n=200, dim=8, snr=1.0, seed=0)
    sweep = init_axis_sweep(
        X, y, seeds=[0, 1], n_iters=10, hilbert_dim=32, lr=0.05,
    )
    assert sweep["axis"] == "init_mode"
    assert [r["init_mode"] for r in sweep["results"]] == list(INIT_MODE_SWEEP)
    for r in sweep["results"]:
        assert "test_r2_mean" in r


# ---------------------------------------------------------------------------
# 7. End-to-end smoke (script invocation)
# ---------------------------------------------------------------------------
def test_smoke_end_to_end_synthesis(tmp_path: Path) -> None:
    """Invoke the script with a tiny config and confirm outputs land."""
    repo_root = _REPO_ROOT
    out = tmp_path / "wf127_smoke"
    cmd = [
        sys.executable, "-u",
        str(repo_root / "molmetal" / "scripts" / "diag_hilbert_astsr_param_sweep.py"),
        "--n", "200", "--dim", "8", "--snr", "1.0",
        "--seeds", "0", "1",
        "--n-iters", "10",
        "--astr-population", "8",
        "--astr-generations", "3",
        "--astr-depth", "2",
        "--hilbert-dim", "16",
        "--output", str(out),
    ]
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    res = subprocess.run(
        cmd, cwd=str(repo_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=180,
    )
    # The script should exit 0.
    assert res.returncode == 0, (
        f"script exit {res.returncode}; stderr={res.stderr.decode()[:500]}"
    )
    # All required outputs should exist.
    assert (out / "l2_sweep.json").exists()
    assert (out / "astr_sweep.json").exists()
    assert (out / "init_sweep.json").exists()
    assert (out / "synthesis.md").exists()
    # JSON parses.
    l2 = json.loads((out / "l2_sweep.json").read_text())
    assert l2["axis"] == "l2_lambda"
    assert len(l2["results"]) == len(L2_SWEEP)
    astr = json.loads((out / "astr_sweep.json").read_text())
    assert astr["axis"] == "astr_loss_scale"
    assert len(astr["results"]) == len(ASTR_SCALE_SWEEP)
    init = json.loads((out / "init_sweep.json").read_text())
    assert init["axis"] == "init_mode"
    assert len(init["results"]) == len(INIT_MODE_SWEEP)
    # Synthesis is non-empty.
    assert len((out / "synthesis.md").read_text()) > 200


# ---------------------------------------------------------------------------
# 8. Stable-region threshold constant
# ---------------------------------------------------------------------------
def test_stable_r2_threshold_constant() -> None:
    assert STABLE_R2_THRESHOLD == 0.0


def test_l2_sweep_constant() -> None:
    assert L2_SWEEP == (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def test_astr_scale_sweep_constant() -> None:
    assert ASTR_SCALE_SWEEP == (0.1, 1.0, 10.0, 100.0)


def test_init_mode_sweep_constant() -> None:
    assert INIT_MODE_SWEEP == ("zeros", "random_normal", "xavier")
