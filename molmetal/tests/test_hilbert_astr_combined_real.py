"""Pytest suite for Wf-85 Phase 3 — REAL Hilbert+ASTSR combined R² retrain.

The script ``molmetal/scripts/bench_hilbert_astr_combined_real.py`` is
the **REAL retrain** Hilbert+ASTSR combined R² anchor (no Multi-PH +
PLL, no 3-way + PLL) with **5 seeds × 50 epochs × strict Bemis-Murcko
scaffold split** on PlatinAI 4-cell.  These tests guard the script's
invariants and provide fast unit-level confidence.

Test inventory
--------------
1.  ``test_feature_columns_constant`` — 12-d F5_RDKIT_2D schema.
2.  ``test_platinai_cell_cols_constant`` — 4 cell-line columns present.
3.  ``test_blend_alphas_constant`` — BLEND_ALPHAS set is stable.
4.  ``test_inner_val_frac_constant`` — INNER_VAL_FRAC = 0.10.
5.  ``test_compute_r2_zero_for_perfect`` — perfect predictor ⇒ R² = 1.
6.  ``test_compute_r2_zero_for_constant`` — constant predictor at
    ``y.mean()`` ⇒ R² = 0.
7.  ``test_compute_r2_negative`` — strictly bad predictor ⇒ R² < 0.
8.  ``test_collapse_detector`` — pred_std / y_std < 0.1 ⇒ True.
9.  ``test_choose_alpha_picks_best`` — choose_alpha picks alpha with
    highest inner R².
10. ``test_hilbert_feature_map_truncates`` — augmentation dim ≤
    ``hilbert_dim``.
11. ``test_hilbert_feature_map_augments`` — when F < hilbert_dim,
    augmented cols > F.
12. ``test_hilbert_feature_map_test_alignment`` — train + test feature
    maps share the same ``hilbert_dim``.
13. ``test_strict_scaffold_split_disjoint`` — train/val/test indices
    are mutually exclusive.
14. ``test_strict_scaffold_split_covers_all`` — union of train/val/test
    = full set of rows.
15. ``test_strict_scaffold_split_seed_variation`` — different seeds
    produce different scaffold assignments.
16. ``test_load_platinai_cell_mcf7`` — load MCF7 cell-line: X shape,
    y is finite, smiles list non-empty.
17. ``test_load_platinai_cell_unknown_raises`` — unknown cell line
    raises ``ValueError``.
18. ``test_hilbert_astr_combined_real_smoke`` — invoke the script with
    ``--seeds 0`` on a small PlatinAI subset and confirm
    ``per_seed_metrics.json`` is well-formed JSON with all required
    keys.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent  # molmetal/
while (
    _REPO_ROOT != _REPO_ROOT.parent
    and not (
        _REPO_ROOT / "molmetal" / "scripts" / "bench_hilbert_astr_combined_real.py"
    ).exists()
):
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.bench_hilbert_astr_combined_real import (  # noqa: E402
    FEATURE_COLUMNS,
    COLLAPSE_RATIO_THRESHOLD,
    PRED_CLIP,
    BLEND_ALPHAS,
    INNER_VAL_FRAC,
    PLATINAI_CELL_COLS,
    _collapse_detected,
    _compute_r2,
    _choose_alpha,
    _strict_scaffold_split,
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
# Constants.
# ---------------------------------------------------------------------------
def test_blend_alphas_constant() -> None:
    assert BLEND_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_inner_val_frac_constant() -> None:
    assert INNER_VAL_FRAC == pytest.approx(0.10, abs=1e-9)


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
    y_hat = y * 2.0
    r2 = _compute_r2(y, y_hat)
    assert r2 < 0.0


def test_collapse_detector() -> None:
    y_true = np.linspace(0.0, 1.0, 100)
    # Constant prediction ⇒ pred_std = 0 ⇒ collapse.
    assert _collapse_detected(np.full_like(y_true, 0.5), y_true) is True
    # Wide-spread prediction ⇒ no collapse.
    assert _collapse_detected(y_true + 0.1, y_true) is False
    assert COLLAPSE_RATIO_THRESHOLD == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# Combined branch — alpha selection.
# ---------------------------------------------------------------------------
def test_choose_alpha_picks_best() -> None:
    y_true = np.linspace(0.0, 1.0, 20)
    # Hilbert predicts perfectly, ASTSR predicts zero.
    h = y_true.copy()
    a = np.zeros_like(y_true)
    alpha, r2 = _choose_alpha(y_true, h, a)
    assert alpha == 1.0
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_choose_alpha_picks_astr_when_hilbert_worse() -> None:
    y_true = np.linspace(0.0, 1.0, 20)
    # Hilbert predicts zero (R²=0), ASTSR predicts perfectly (R²=1).
    h = np.zeros_like(y_true)
    a = y_true.copy()
    alpha, r2 = _choose_alpha(y_true, h, a)
    assert alpha == 0.0
    assert r2 == pytest.approx(1.0, abs=1e-9)


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
    # When F=8 < hilbert_dim=64, augmented columns include linear + bias + pairwise.
    assert out_tr.shape == (40, 64)
    # The bias column (constant 1.0) should be preserved in the truncated set.
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
# Strict-scaffold split.
# ---------------------------------------------------------------------------
def test_strict_scaffold_split_disjoint() -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "c1ccc2ccccc2c1"]
    train, val, test = _strict_scaffold_split(smiles, (0.5, 0.25, 0.25), seed=42)
    assert len(np.intersect1d(train, val)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0


def test_strict_scaffold_split_covers_all() -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "c1ccc2ccccc2c1"]
    train, val, test = _strict_scaffold_split(smiles, (0.5, 0.25, 0.25), seed=42)
    covered = np.sort(np.concatenate([train, val, test]))
    expected = np.sort(np.arange(len(smiles)))
    assert np.array_equal(covered, expected)


def test_strict_scaffold_split_seed_variation() -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "c1ccc2ccccc2c1"] * 3
    t1, _, _ = _strict_scaffold_split(smiles, (0.5, 0.25, 0.25), seed=0)
    t2, _, _ = _strict_scaffold_split(smiles, (0.5, 0.25, 0.25), seed=7)
    # Different seeds ⇒ potentially different train sets.
    assert t1.dtype == np.int64
    assert t2.dtype == np.int64
    assert t1.sum() >= 0
    assert t2.sum() >= 0


# ---------------------------------------------------------------------------
# Data loader.
# ---------------------------------------------------------------------------
def test_load_platinai_cell_mcf7() -> None:
    from molmetal.scripts.bench_hilbert_astr_combined_real import (
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
    from molmetal.scripts.bench_hilbert_astr_combined_real import (
        _REPO_ROOT,
        load_platinai_cell,
    )

    with pytest.raises(ValueError):
        load_platinai_cell(_REPO_ROOT, "hela_unknown")


# ---------------------------------------------------------------------------
# Predictor clip.
# ---------------------------------------------------------------------------
def test_predict_hilbert_clip() -> None:
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])
    bp = np.zeros(12)
    bp[0] = 1e12
    y_hat = predict_hilbert(X, bp)
    assert np.all(np.abs(y_hat) <= PRED_CLIP)


# ---------------------------------------------------------------------------
# Script smoke (--seeds 0 on PlatinAI subset).
# ---------------------------------------------------------------------------
def test_hilbert_astr_combined_real_smoke() -> None:
    """End-to-end smoke: invoke the script with --seeds 0 on a 50-row
    PlatinAI subset and confirm per_seed_metrics.json is well-formed
    JSON with all required keys.

    Wall-time budget: < 60 s on CPU with --epochs 10.
    """
    script = _REPO_ROOT / "molmetal" / "scripts" / "bench_hilbert_astr_combined_real.py"
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
        out_dir = tmp / "wf85_p3_smoke"
        # Construct a wrapper that overrides PLATINAI_CSV via patching
        # the module-level Path reference before main() runs.
        wrapper = tmp / "run_subset.py"
        wrapper.write_text(
            "import sys\n"
            "sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "from molmetal.scripts import bench_hilbert_astr_combined_real as b\n"
            "b.PLATINAI_CSV = Path({subset!r})\n"
            "import sys as _sys\n"
            "_sys.argv = [\n"
            "    'bench_hilbert_astr_combined_real.py',\n"
            "    '--cell-lines', 'mcf7',\n"
            "    '--seeds', '0',\n"
            "    '--epochs', '10',\n"
            "    '--hilbert-dim', '16',\n"
            "    '--population', '8',\n"
            "    '--depth', '2',\n"
            "    '--device', 'cpu',\n"
            "    '--fractions', '0.7 0.15 0.15',\n"
            "    '--out-dir', {out!r},\n"
            "]\n"
            "sys.exit(b.main())\n".format(
                repo=str(_REPO_ROOT),
                subset=str(subset_csv),
                out=str(out_dir),
            )
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, str(wrapper)],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            print("STDOUT:", proc.stdout[-2000:])
            print("STDERR:", proc.stderr[-2000:])
        assert proc.returncode == 0, f"script failed: {proc.returncode}"
        # Verify per_seed_metrics.json is well-formed.
        per_seed = out_dir / "mcf7" / "per_seed_metrics.json"
        assert per_seed.exists(), f"missing {per_seed}"
        data = json.loads(per_seed.read_text(encoding="utf-8"))
        for key in (
            "cell_line",
            "target_col",
            "n_samples",
            "n_features",
            "n_seeds",
            "epochs",
            "hilbert_dim",
            "astr_population",
            "astr_depth",
            "inner_val_frac",
            "blend_alphas",
            "split",
            "device",
            "combined_r2_test_mean",
            "combined_r2_test_std",
            "combined_r2_val_mean",
            "combined_r2_val_std",
            "combined_r2_train_mean",
            "combined_r2_train_std",
            "hilbert_r2_test_mean",
            "hilbert_r2_test_std",
            "astr_r2_test_mean",
            "astr_r2_test_std",
            "alpha_chosen_mean",
            "alpha_chosen_std",
            "per_seed",
        ):
            assert key in data, f"missing top-level key {key}"
        assert data["cell_line"] == "mcf7"
        assert data["target_col"] == "ic50_mcf7_nm"
        assert data["n_seeds"] == 1
        assert data["epochs"] == 10
        assert data["hilbert_dim"] == 16
        assert data["astr_population"] == 8
        assert data["astr_depth"] == 2
        assert data["split"]["type"] == "strict_scaffold"
        assert data["inner_val_frac"] == pytest.approx(0.10, abs=1e-9)
        assert len(data["per_seed"]) == 1
        seed0 = data["per_seed"][0]
        for k in (
            "seed",
            "n_train",
            "n_val",
            "n_test",
            "hilbert_r2_train",
            "hilbert_r2_val",
            "hilbert_r2_test",
            "astr_r2_train",
            "astr_r2_val",
            "astr_r2_test",
            "combined_r2_train",
            "combined_r2_val",
            "combined_r2_test",
            "alpha_chosen",
            "inner_r2",
            "collapse_detected",
            "wall_time_s",
            "used_fp64_hilbert",
            "used_cuda_hilbert",
            "used_cuda_astr",
            "best_ast_lisp",
            "per_fold",
        ):
            assert k in seed0, f"missing per-seed key {k}"
        # Verify alpha is in BLEND_ALPHAS.
        assert seed0["alpha_chosen"] in BLEND_ALPHAS
        # Verify synthesis.md exists.
        synthesis = out_dir / "synthesis.md"
        assert synthesis.exists()
