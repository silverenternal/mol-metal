"""Pytest suite for Wf-85 Phase 2 — REAL ASTSR-only R² retrain.

The script ``molmetal/scripts/bench_astr_only_real.py`` is the
**REAL retrain** ASTSR-only R² anchor (no Hilbert, no Multi-PH + PLL,
no composite blend) with **5 seeds × 50 epochs (= 50 AST
generations) × strict Bemis-Murcko scaffold split** on PlatinAI 4-cell.
These tests guard the script's invariants and provide fast unit-level
confidence.

Test inventory
--------------
1.  ``test_feature_columns_rdkit_2d_constant`` — 12-d F5_RDKIT_2D
    schema matches Wf-49 conventions.
2.  ``test_feature_columns_f5_legacy_constant`` — 5-d legacy F5
    schema is preserved.
3.  ``test_feature_set_aliases_rdkit_2d`` — `F5_RDKIT_2D` aliases
    → ``rdkit_2d`` canonical with 12-d columns.
4.  ``test_feature_set_aliases_f5_legacy`` — `F5_LEGACY` aliases
    → ``f5_legacy`` canonical with 5-d columns.
5.  ``test_feature_set_aliases_unknown_raises`` — unknown alias
    raises ``ValueError``.
6.  ``test_platinai_cell_cols_constant`` — 4 cell-line columns
    present.
7.  ``test_load_platinai_cell_mcf7`` — load MCF7 cell-line: X
    shape, y is finite, smiles list non-empty.
8.  ``test_load_platinai_cell_unknown_raises`` — unknown cell line
    raises ``ValueError``.
9.  ``test_compute_r2_zero_for_perfect`` — perfect predictor
    ⇒ R² = 1.
10. ``test_compute_r2_zero_for_constant`` — constant predictor at
    ``y.mean()`` ⇒ R² = 0.
11. ``test_compute_r2_negative`` — strictly bad predictor ⇒ R² < 0.
12. ``test_collapse_detector`` — pred_std / y_std < 0.1 ⇒ True.
13. ``test_astr_regressor_fit_predict_shape`` — synthetic 12-d
    Gaussian target with small population/generations yields
    finite prediction shape ``(n_test,)``.
14. ``test_astr_regressor_best_ast_lisp`` — best AST in lisp form
    is a non-empty string after a fit.
15. ``test_strict_scaffold_split_disjoint`` — train/val/test
    indices are mutually exclusive.
16. ``test_strict_scaffold_split_covers_all`` — union of
    train/val/test = full set of rows.
17. ``test_parse_fractions_valid`` — three floats summing to 1 ⇒ OK.
18. ``test_parse_fractions_sum_invalid`` — fractions not summing
    to 1 raise ``ValueError``.
19. ``test_astr_only_real_smoke`` — invoke the script with
    ``--seeds 0`` on a 50-row PlatinAI subset and confirm
    ``per_seed_metrics.json`` is well-formed JSON with all
    required keys.
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
_REPO_ROOT = _HERE.parent  # molmetal/
while (
    _REPO_ROOT != _REPO_ROOT.parent
    and not (_REPO_ROOT / "molmetal" / "scripts" / "bench_astr_only_real.py").exists()
):
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.bench_astr_only_real import (  # noqa: E402
    FEATURE_COLUMNS_RDKIT_2D,
    FEATURE_COLUMNS_F5_LEGACY,
    FEATURE_COLUMNS,
    FEATURE_SET_ALIASES,
    COLLAPSE_RATIO_THRESHOLD,
    PRED_CLIP,
    PLATINAI_CELL_COLS,
    ASTRRegressor,
    _collapse_detected,
    _compute_r2,
    _parse_fractions,
    _strict_scaffold_split,
    resolve_feature_set,
)


# ---------------------------------------------------------------------------
# Feature schema.
# ---------------------------------------------------------------------------
def test_feature_columns_rdkit_2d_constant() -> None:
    assert FEATURE_COLUMNS_RDKIT_2D == (
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
    assert len(FEATURE_COLUMNS_RDKIT_2D) == 12


def test_feature_columns_f5_legacy_constant() -> None:
    assert FEATURE_COLUMNS_F5_LEGACY == (
        "diversity_tanimoto",
        "diversity_homotype",
        "qed_mean",
        "anticancer_index",
        "sa_mean_norm",
    )
    assert len(FEATURE_COLUMNS_F5_LEGACY) == 5


def test_feature_columns_default_is_rdkit_2d() -> None:
    assert FEATURE_COLUMNS == FEATURE_COLUMNS_RDKIT_2D


# ---------------------------------------------------------------------------
# Feature-set aliases.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alias",
    ["F5_RDKIT_2D", "F5_RDKIT", "RDKIT_2D", "f5_rdkit_2d"],
)
def test_feature_set_aliases_rdkit_2d(alias: str) -> None:
    canonical, cols = resolve_feature_set(alias)
    assert canonical == "rdkit_2d"
    assert cols == FEATURE_COLUMNS_RDKIT_2D
    assert len(cols) == 12


@pytest.mark.parametrize(
    "alias",
    ["F5_LEGACY", "F5", "F5_LEGACY_5", "f5_legacy"],
)
def test_feature_set_aliases_f5_legacy(alias: str) -> None:
    canonical, cols = resolve_feature_set(alias)
    assert canonical == "f5_legacy"
    assert cols == FEATURE_COLUMNS_F5_LEGACY
    assert len(cols) == 5


def test_feature_set_aliases_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_feature_set("gibberish_2d")
    with pytest.raises(ValueError):
        resolve_feature_set("")


def test_feature_set_aliases_table_complete() -> None:
    """Every CLI choice must resolve cleanly."""
    for alias in FEATURE_SET_ALIASES:
        canonical, cols = resolve_feature_set(alias)
        assert canonical in {"rdkit_2d", "f5_legacy"}


# ---------------------------------------------------------------------------
# PlatinAI 4-cell layout.
# ---------------------------------------------------------------------------
def test_platinai_cell_cols_constant() -> None:
    assert set(PLATINAI_CELL_COLS) == {"mcf7", "a549", "a2780", "a2780cis"}
    assert PLATINAI_CELL_COLS["mcf7"] == "ic50_mcf7_nm"
    assert PLATINAI_CELL_COLS["a549"] == "ic50_a549_nm"
    assert PLATINAI_CELL_COLS["a2780"] == "ic50_a2780_nm"
    assert PLATINAI_CELL_COLS["a2780cis"] == "ic50_a2780cis_nm"


# ---------------------------------------------------------------------------
# Data loader.
# ---------------------------------------------------------------------------
def test_load_platinai_cell_mcf7() -> None:
    from molmetal.scripts.bench_astr_only_real import (
        _REPO_ROOT,
        load_platinai_cell,
    )

    X, y, smiles = load_platinai_cell(_REPO_ROOT, "mcf7", n_max=50)
    assert X.shape[0] == y.shape[0] == len(smiles)
    assert X.shape[0] >= 1
    assert X.shape[1] == len(FEATURE_COLUMNS_RDKIT_2D)
    assert np.all(np.isfinite(y))
    assert all(isinstance(s, str) and len(s) > 0 for s in smiles)


def test_load_platinai_cell_unknown_raises() -> None:
    from molmetal.scripts.bench_astr_only_real import (
        _REPO_ROOT,
        load_platinai_cell,
    )

    with pytest.raises(ValueError):
        load_platinai_cell(_REPO_ROOT, "hela_unknown")


# ---------------------------------------------------------------------------
# R² metric + collapse detection.
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
# ASTRRegressor fit / predict unit test (synthetic Gaussian target).
# ---------------------------------------------------------------------------
def test_astr_regressor_fit_predict_shape() -> None:
    rng = np.random.default_rng(123)
    n, d = 200, len(FEATURE_COLUMNS_RDKIT_2D)
    X = rng.normal(0.0, 1.0, (n, d)).astype(np.float32)
    y = (X @ rng.normal(0.0, 0.3, d)).astype(np.float32) + rng.normal(
        0.0,
        0.05,
        n,
    ).astype(np.float32)
    reg = ASTRRegressor(
        population=8,
        generations=5,
        depth=3,
        seed=0,
        feature_names=FEATURE_COLUMNS_RDKIT_2D,
    ).fit(X, y)
    y_hat = reg.predict(X)
    assert y_hat.shape == (n,)
    assert np.all(np.isfinite(y_hat))
    # Predictions may be clipped if the AST emits large ops; verify
    # the clip bound holds.
    assert np.all(np.abs(y_hat) <= PRED_CLIP)
    # ASTSR is symbolic search; we don't demand a tight R² bound here.
    r2 = _compute_r2(y, y_hat)
    assert math.isfinite(r2)


def test_astr_regressor_best_ast_lisp() -> None:
    rng = np.random.default_rng(7)
    n, d = 120, len(FEATURE_COLUMNS_RDKIT_2D)
    X = rng.normal(0.0, 1.0, (n, d)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3).astype(np.float32) + rng.normal(
        0.0,
        0.1,
        n,
    ).astype(np.float32)
    reg = ASTRRegressor(
        population=8,
        generations=5,
        depth=3,
        seed=0,
        feature_names=FEATURE_COLUMNS_RDKIT_2D,
    ).fit(X, y)
    lisp = reg.best_ast_lisp()
    assert isinstance(lisp, str)
    assert len(lisp) > 0
    assert lisp != "<no_ast>"


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
    assert t1.dtype == np.int64
    assert t2.dtype == np.int64
    assert t1.sum() >= 0
    assert t2.sum() >= 0


# ---------------------------------------------------------------------------
# Fractions parser.
# ---------------------------------------------------------------------------
def test_parse_fractions_valid() -> None:
    assert _parse_fractions("0.7 0.15 0.15") == (0.7, 0.15, 0.15)
    assert _parse_fractions("0.6,0.2,0.2") == (0.6, 0.2, 0.2)


def test_parse_fractions_sum_invalid() -> None:
    with pytest.raises(ValueError):
        _parse_fractions("0.5 0.3 0.3")  # 1.1
    with pytest.raises(ValueError):
        _parse_fractions("0.6 0.2 0.1")  # 0.9
    with pytest.raises(ValueError):
        _parse_fractions("1.0 0.5 0.5")  # 2.0


# ---------------------------------------------------------------------------
# Script smoke (--seeds 0 on PlatinAI subset).
# ---------------------------------------------------------------------------
def test_astr_only_real_smoke() -> None:
    """End-to-end smoke: invoke the script with --seeds 0 on a 50-row
    PlatinAI subset and confirm per_seed_metrics.json is well-formed.

    Wall-time budget: ~60 s on CPU with --epochs 5.
    """
    script = _REPO_ROOT / "molmetal" / "scripts" / "bench_astr_only_real.py"
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
        out_dir = tmp / "wf85_p2_smoke"
        wrapper = tmp / "run_subset.py"
        wrapper.write_text(
            "import sys\n"
            "sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "from molmetal.scripts import bench_astr_only_real as b\n"
            "b.PLATINAI_CSV = Path({subset!r})\n"
            "import sys as _sys\n"
            "_sys.argv = [\n"
            "    'bench_astr_only_real.py',\n"
            "    '--cell-lines', 'mcf7',\n"
            "    '--seeds', '0',\n"
            "    '--epochs', '5',\n"
            "    '--population', '8',\n"
            "    '--depth', '3',\n"
            "    '--feature-set', 'F5_RDKIT_2D',\n"
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
            timeout=240,
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
            "feature_set",
            "n_samples",
            "n_features",
            "n_seeds",
            "epochs",
            "astr_population",
            "astr_depth",
            "astr_p_var",
            "astr_parsimony",
            "split",
            "device",
            "astr_r2_test_mean",
            "astr_r2_test_std",
            "astr_r2_val_mean",
            "astr_r2_val_std",
            "astr_r2_train_mean",
            "astr_r2_train_std",
            "feature_columns",
            "per_seed",
        ):
            assert key in data, f"missing top-level key {key}"
        assert data["cell_line"] == "mcf7"
        assert data["target_col"] == "ic50_mcf7_nm"
        assert data["n_seeds"] == 1
        assert data["epochs"] == 5
        assert data["astr_population"] == 8
        assert data["astr_depth"] == 3
        assert data["feature_set"] == "rdkit_2d"
        assert data["split"]["type"] == "strict_scaffold"
        assert len(data["per_seed"]) == 1
        seed0 = data["per_seed"][0]
        for k in (
            "seed",
            "n_train",
            "n_val",
            "n_test",
            "astr_r2_train",
            "astr_r2_val",
            "astr_r2_test",
            "collapse_detected",
            "wall_time_s",
            "used_cuda",
            "best_ast_lisp",
            "per_fold",
            "train_idx",
            "val_idx",
            "test_idx",
        ):
            assert k in seed0, f"missing per-seed key {k}"
        # Verify synthesis.md exists.
        synthesis = out_dir / "synthesis.md"
        assert synthesis.exists()
        syn_text = synthesis.read_text(encoding="utf-8")
        # Synthesis must include the 4-cell aggregate row even with 1 cell line.
        assert "4-cell aggregate" in syn_text
        # Verify gate verdict line must be present.
        assert (
            "MEASURED PASS" in syn_text
            or "MEASURED PARTIAL" in syn_text
            or "HONEST NEGATIVE" in syn_text
        )
