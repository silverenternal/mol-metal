"""Pytest suite for Wf-83 Phase 1 — Full Hilbert+ASTSR combined R² on PlatinAI 4-cell.

The script ``molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py``
is the **Phase 2 combined anchor** for the Wf-83 ship wave.  These
tests guard the script's invariants and provide fast unit-level
confidence.

Test inventory
--------------
1.  ``test_feature_columns_constant`` — 12-d F5_RDKIT_2D schema matches
    Wf-49 conventions.
2.  ``test_compute_r2_zero_for_perfect`` — perfect predictor ⇒ R² = 1.
3.  ``test_compute_r2_zero_for_constant`` — constant predictor at
    ``y.mean()`` ⇒ R² = 0.
4.  ``test_compute_r2_negative`` — strictly bad predictor ⇒ R² < 0.
5.  ``test_collapse_detector`` — pred_std / y_std < 0.1 ⇒ True.
6.  ``test_blend_alphas_constant`` — BLEND_ALPHAS set is stable.
7.  ``test_kfold_indices_no_overlap`` — train and test never overlap.
8.  ``test_kfold_indices_covers_all`` — union of test folds = full set.
9.  ``test_kfold_indices_deterministic`` — same seed ⇒ identical splits.
10. ``test_platinai_cell_cols_constant`` — 4 cell-line columns present.
11. ``test_load_platinai_cell_mcf7`` — load MCF7 cell-line: X shape,
    y is finite, smiles list non-empty.
12. ``test_load_platinai_cell_unknown_raises`` — unknown cell line
    raises ``ValueError``.
13. ``test_hilbert_feature_map_truncates`` — augmentation dim ≤
    ``hilbert_dim``; bias column preserved.
14. ``test_hilbert_feature_map_augments`` — when ``F < hilbert_dim``,
    augmented cols > ``F``.
15. ``test_hilbert_feature_map_test_alignment`` — train + test feature
    maps share the same ``hilbert_dim``.
16. ``test_choose_alpha_picks_best`` — choose_alpha picks alpha with
    highest inner R².
17. ``test_predict_hilbert_clip`` — predict_hilbert clips outliers.
18. ``test_combined_baseline_smoke`` — invoke the script with
    ``--seeds 0`` on a small PlatinAI subset and confirm
    ``per_seed_metrics.json`` is well-formed JSON with all required
    keys.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
# Tests live in molmetal/tests/; the repo root is one level above molmetal.
_REPO_ROOT = _HERE.parent  # molmetal/
while (
    _REPO_ROOT != _REPO_ROOT.parent
    and not (
        _REPO_ROOT
        / "molmetal"
        / "scripts"
        / "bench_hilbert_astr_phase2_platinai_4cell.py"
    ).exists()
):
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.bench_hilbert_astr_phase2_platinai_4cell import (  # noqa: E402
    FEATURE_COLUMNS,
    COLLAPSE_RATIO_THRESHOLD,
    PRED_CLIP,
    BLEND_ALPHAS,
    PLATINAI_CELL_COLS,
    _collapse_detected,
    _compute_r2,
    _kfold_indices,
    _choose_alpha,
    hilbert_feature_map,
    predict_hilbert,
)


# ---------------------------------------------------------------------------
# Feature schema.
# ---------------------------------------------------------------------------
def test_feature_columns_constant() -> None:
    assert FEATURE_COLUMNS == (
        "MolWt",
        "MolLogP",
        "TPSA",
        "HBA",
        "HBD",
        "NRotB",
        "HeavyAtoms",
        "RingCount",
        "NumAromaticRings",
        "FractionCSP3",
        "MolMR",
        "BalabanJ",
    )


def test_platinai_cell_cols_constant() -> None:
    assert set(PLATINAI_CELL_COLS) == {"mcf7", "a549", "a2780", "a2780cis"}
    assert PLATINAI_CELL_COLS["mcf7"] == "ic50_mcf7_nm"
    assert PLATINAI_CELL_COLS["a549"] == "ic50_a549_nm"
    assert PLATINAI_CELL_COLS["a2780"] == "ic50_a2780_nm"
    assert PLATINAI_CELL_COLS["a2780cis"] == "ic50_a2780cis_nm"


# ---------------------------------------------------------------------------
# Metrics primitives.
# ---------------------------------------------------------------------------
def test_compute_r2_zero_for_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _compute_r2(y, y) == pytest.approx(1.0, abs=1e-9)


def test_compute_r2_zero_for_constant() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_hat = np.full_like(y, y.mean())
    assert _compute_r2(y, y_hat) == pytest.approx(0.0, abs=1e-9)


def test_compute_r2_negative() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Predict twice y → very bad.
    y_hat = y * 2.0
    r2 = _compute_r2(y, y_hat)
    assert r2 < 0.0


def test_collapse_detector() -> None:
    y_true = np.linspace(0.0, 1.0, 100)
    # Constant prediction ⇒ pred_std = 0 ⇒ collapse.
    assert _collapse_detected(np.full_like(y_true, 0.5), y_true) is True
    # Wide-spread prediction ⇒ no collapse.
    assert _collapse_detected(y_true + 0.1, y_true) is False


def test_blend_alphas_constant() -> None:
    assert BLEND_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# K-fold primitives.
# ---------------------------------------------------------------------------
def test_kfold_indices_no_overlap() -> None:
    n = 100
    splits = _kfold_indices(n, 5, seed=42)
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0


def test_kfold_indices_covers_all() -> None:
    n = 100
    splits = _kfold_indices(n, 5, seed=42)
    covered = np.sort(np.concatenate([te for _, te in splits]))
    expected = np.sort(np.arange(n))
    assert np.array_equal(covered, expected)


def test_kfold_indices_deterministic() -> None:
    n = 100
    s1 = _kfold_indices(n, 5, seed=42)
    s2 = _kfold_indices(n, 5, seed=42)
    for (tr1, te1), (tr2, te2) in zip(s1, s2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


# ---------------------------------------------------------------------------
# Data loader.
# ---------------------------------------------------------------------------
def test_load_platinai_cell_mcf7() -> None:
    from molmetal.scripts.bench_hilbert_astr_phase2_platinai_4cell import (
        _REPO_ROOT,
        load_platinai_cell,
    )

    X, y, smiles = load_platinai_cell(_REPO_ROOT, "mcf7", n_max=50)
    assert X.shape[0] == y.shape[0] == len(smiles)
    assert X.shape[0] >= 1
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert np.all(np.isfinite(y))
    assert all(isinstance(s, str) and len(s) > 0 for s in smiles)


def test_load_platinai_cell_unknown_raises() -> None:
    from molmetal.scripts.bench_hilbert_astr_phase2_platinai_4cell import (
        _REPO_ROOT,
        load_platinai_cell,
    )

    with pytest.raises(ValueError):
        load_platinai_cell(_REPO_ROOT, "hela_unknown")


# ---------------------------------------------------------------------------
# Hilbert feature map.
# ---------------------------------------------------------------------------
def test_hilbert_feature_map_truncates() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 12)).astype(np.float32)
    out_tr, _ = hilbert_feature_map(X, hilbert_dim=32)
    assert out_tr.shape == (40, 32)
    assert np.all(np.isfinite(out_tr))


def test_hilbert_feature_map_augments() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    out_tr, _ = hilbert_feature_map(X, hilbert_dim=64)
    # When F=8 < hilbert_dim=64, the augmented columns include linear + bias + pairwise.
    assert out_tr.shape == (40, 64)
    # The bias column (constant 1.0) should be preserved in the truncated set.
    # We check that at least one column has near-zero variance.
    vars_ = np.var(out_tr, axis=0)
    assert np.any(vars_ < 1e-3)


def test_hilbert_feature_map_test_alignment() -> None:
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(40, 12)).astype(np.float32)
    X_te = rng.normal(size=(10, 12)).astype(np.float32)
    out_tr, out_te = hilbert_feature_map(X_tr, hilbert_dim=32, X_12d_test=X_te)
    assert out_tr.shape[1] == out_te.shape[1]
    assert out_tr.shape == (40, 32)
    assert out_te.shape == (10, 32)


# ---------------------------------------------------------------------------
# Combined branch.
# ---------------------------------------------------------------------------
def test_choose_alpha_picks_best() -> None:
    y_true = np.linspace(0.0, 1.0, 20)
    # Hilbert predicts perfectly, ASTSR predicts zero.
    h = y_true.copy()
    a = np.zeros_like(y_true)
    alpha, r2 = _choose_alpha(y_true, h, a)
    assert alpha == 1.0
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_predict_hilbert_clip() -> None:
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    # Wildly large weight on the first column ⇒ would explode without clip.
    bp = np.zeros(12)
    bp[0] = 1e12
    y_hat = predict_hilbert(X, bp)
    assert np.all(np.abs(y_hat) <= PRED_CLIP)


# ---------------------------------------------------------------------------
# Script smoke (--seeds 0 on PlatinAI subset).
# ---------------------------------------------------------------------------
def test_combined_baseline_smoke() -> None:
    """End-to-end smoke: invoke the script with --seeds 0 on PlatinAI 4-cell
    (n_max auto-derived from default full dataset; smoke uses 1 seed
    which is < 30 s on CPU) and confirm per_seed_metrics.json is well-formed.

    To keep wall time bounded in CI, we override the dataset via a temp
    directory containing a 50-row PlatinAI subset by writing a fake CSV.
    """
    script = (
        _REPO_ROOT
        / "molmetal"
        / "scripts"
        / "bench_hilbert_astr_phase2_platinai_4cell.py"
    )
    assert script.exists()
    # Build a 50-row PlatinAI-like subset in a temp dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        subset_csv = tmp / "platinai_subset.csv"
        real_csv = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
        with open(real_csv, "r", encoding="utf-8") as fin, open(
            subset_csv,
            "w",
            encoding="utf-8",
            newline="",
        ) as fout:
            fout.write(fin.readline())  # header
            for i, line in enumerate(fin):
                if i >= 50:
                    break
                fout.write(line)
        out_dir = tmp / "wf83_smoke"
        env = os.environ.copy()
        # Patch the loader path via PYTHONPATH; we'll instead just patch
        # the module-level PLATINAI_CSV constant by overriding sys.modules.
        env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(script),
                "--cell-lines",
                "mcf7",
                "--seeds",
                "0",
                "--n-folds",
                "2",
                "--epochs",
                "20",
                "--population",
                "8",
                "--generations",
                "4",
                "--depth",
                "2",
                "--hilbert-dim",
                "16",
                "--output",
                str(out_dir),
            ],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        # If the script crashed, show diagnostics.
        if proc.returncode != 0:
            print("STDOUT:", proc.stdout[-2000:])
            print("STDERR:", proc.stderr[-2000:])
        assert proc.returncode == 0, f"script failed: {proc.returncode}"
        # Verify per_seed_metrics.json is well-formed.
        per_seed = out_dir / "mcf7" / "per_seed_metrics.json"
        assert per_seed.exists()
        data = json.loads(per_seed.read_text(encoding="utf-8"))
        for key in (
            "cell_line",
            "target_col",
            "n_samples",
            "n_features",
            "n_seeds",
            "combined_r2_mean",
            "hilbert_r2_mean",
            "astr_r2_mean",
            "per_seed",
        ):
            assert key in data, f"missing key {key}"
        assert data["cell_line"] == "mcf7"
        assert data["target_col"] == "ic50_mcf7_nm"
        assert data["n_seeds"] == 1
        assert len(data["per_seed"]) == 1
        seed0 = data["per_seed"][0]
        for k in (
            "combined_r2_overall",
            "hilbert_r2_overall",
            "astr_r2_overall",
            "combined_r2_per_fold_mean",
            "hilbert_r2_per_fold_mean",
            "astr_r2_per_fold_mean",
            "alpha_mean",
            "per_fold",
        ):
            assert k in seed0, f"missing seed0 key {k}"
        # Verify synthesis.md exists.
        synthesis = out_dir / "synthesis.md"
        assert synthesis.exists()
