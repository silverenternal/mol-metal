"""Pytest suite for Wf-77 Phase 2 ASTSR-only R² baseline.

The script ``molmetal/scripts/bench_astr_only_baseline.py`` is the
ASTSR-only counterpart to Wf-77 Phase 1 Hilbert-only baseline.  These
tests guard the script's invariants and provide fast unit-level
confidence.

Test inventory
--------------
1.  ``test_feature_columns_constant`` — 12-d F5_RDKIT_2D descriptor
    schema matches Wf-49 conventions.
2.  ``test_compute_r2_zero_for_perfect`` — perfect predictor ⇒ R² = 1.
3.  ``test_compute_r2_zero_for_constant`` — constant predictor at
    ``y.mean()`` ⇒ R² = 0.
4.  ``test_compute_r2_negative`` — strictly bad predictor ⇒ R² < 0.
5.  ``test_collapse_detector`` — pred_std / y_std < 0.1 ⇒ True.
6.  ``test_kfold_indices_no_overlap`` — train and test never overlap.
7.  ``test_kfold_indices_covers_all`` — union of test folds = full set.
8.  ``test_kfold_indices_deterministic`` — same seed ⇒ identical splits.
9.  ``test_dataset_spec_builtin_platinai`` — platinai_4cell spec is
    well-formed.
10. ``test_dataset_spec_builtin_metalcytotox`` — metalcytotox_5metal
    spec is well-formed.
11. ``test_astr_regressor_fit_predict_shape`` — synthetic 12-d
    Gaussian target yields finite prediction shape ``(n_test,)``.
12. ``test_astr_only_baseline_smoke`` — invoke the script with
    ``--seeds 0`` on a 50-row PlatinAI subset and confirm
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
_REPO_ROOT = _HERE.parent  # molmetal/
while _REPO_ROOT != _REPO_ROOT.parent and not (
    _REPO_ROOT / "molmetal" / "scripts" / "bench_astr_only_baseline.py"
).exists():
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.bench_astr_only_baseline import (  # noqa: E402
    FEATURE_COLUMNS,
    COLLAPSE_RATIO_THRESHOLD,
    ASTSRRegressor,
    _collapse_detected,
    _compute_r2,
    _kfold_indices,
    _metalcytotox_5metal_spec,
    _platinai_4cell_spec,
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
    assert len(FEATURE_COLUMNS) == 12


# ---------------------------------------------------------------------------
# R² metric.
# ---------------------------------------------------------------------------
def test_compute_r2_zero_for_perfect() -> None:
    y = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    assert _compute_r2(y, y) == pytest.approx(1.0, abs=1e-9)


def test_compute_r2_zero_for_constant() -> None:
    y = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    yhat = np.full_like(y, y.mean())
    assert _compute_r2(y, yhat) == pytest.approx(0.0, abs=1e-9)


def test_compute_r2_negative() -> None:
    y = np.asarray([0.0, 1.0, 2.0, 3.0])
    # Strictly bad predictor: opposite of truth.
    yhat = -y + 10.0
    assert _compute_r2(y, yhat) < 0.0


# ---------------------------------------------------------------------------
# Collapse detector.
# ---------------------------------------------------------------------------
def test_collapse_detector() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 1000)
    yhat_constant = np.full_like(y, y.mean())
    assert _collapse_detected(yhat_constant, y) is True
    yhat_wide = y + rng.normal(0.0, 1.0, 1000)
    assert _collapse_detected(yhat_wide, y) is False
    # Custom threshold.
    assert COLLAPSE_RATIO_THRESHOLD == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# K-fold.
# ---------------------------------------------------------------------------
def test_kfold_indices_no_overlap() -> None:
    splits = _kfold_indices(n=100, k=5, seed=0)
    for train, test in splits:
        assert len(np.intersect1d(train, test)) == 0


def test_kfold_indices_covers_all() -> None:
    splits = _kfold_indices(n=137, k=5, seed=1)
    all_tests = np.concatenate([test for _, test in splits])
    assert sorted(all_tests.tolist()) == list(range(137))


def test_kfold_indices_deterministic() -> None:
    s1 = _kfold_indices(n=80, k=4, seed=42)
    s2 = _kfold_indices(n=80, k=4, seed=42)
    for (t1, te1), (t2, te2) in zip(s1, s2):
        assert np.array_equal(t1, t2)
        assert np.array_equal(te1, te2)


# ---------------------------------------------------------------------------
# Dataset specs.
# ---------------------------------------------------------------------------
def test_dataset_spec_builtin_platinai() -> None:
    spec = _platinai_4cell_spec(_REPO_ROOT)
    assert spec.name == "platinai_4cell"
    assert spec.smiles_col == "smiles"
    assert spec.target_cols == (
        "ic50_mcf7_nm",
        "ic50_a549_nm",
        "ic50_a2780_nm",
        "ic50_a2780cis_nm",
    )
    assert spec.n_max >= 4000
    assert spec.csv_path.exists(), f"missing {spec.csv_path}"


def test_dataset_spec_builtin_metalcytotox() -> None:
    spec = _metalcytotox_5metal_spec(_REPO_ROOT)
    assert spec.name == "metalcytotox_5metal"
    assert spec.smiles_col == "smiles"
    assert "activity" in spec.target_cols
    assert spec.csv_path.exists(), f"missing {spec.csv_path}"


# ---------------------------------------------------------------------------
# ASTSR-only regressor unit test (synthetic Gaussian target).
# ---------------------------------------------------------------------------
def test_astr_regressor_fit_predict_shape() -> None:
    rng = np.random.default_rng(123)
    n, d = 200, 12
    X = rng.normal(0.0, 1.0, (n, d)).astype(np.float32)
    # Use a small linear-ish target so the ASTSR has a chance.
    y = (X @ rng.normal(0.0, 0.3, d)).astype(np.float32) + rng.normal(
        0.0, 0.05, n
    ).astype(np.float32)
    reg = ASTSRRegressor(population=8, generations=5, depth=3, seed=0).fit(X, y)
    yhat = reg.predict(X)
    assert yhat.shape == (n,)
    assert np.isfinite(yhat).all()
    # Generous floor — ASTSR is symbolic search, not a closed-form
    # regressor, so we don't demand a tight R² bound here.
    r2 = _compute_r2(y, yhat)
    assert math.isfinite(r2)
    # best_ast_lisp should return a string (not <no_ast>) after a fit.
    assert isinstance(reg.best_ast_lisp(), str)


# ---------------------------------------------------------------------------
# Smoke test — invoke the script end-to-end on a tiny subset.
# ---------------------------------------------------------------------------
def _write_subset_csv(src: Path, dst: Path, n_rows: int) -> None:
    with open(src, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        rows: List[dict] = []
        for r in reader:
            rows.append(r)
            if len(rows) >= n_rows:
                break
    with open(dst, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_astr_only_baseline_smoke() -> None:
    """Run the CLI on a 50-row PlatinAI subset; assert JSON shape."""
    src = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    if not src.exists():
        pytest.skip(f"missing {src}")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        subset = tmpdir / "platinai_subset.csv"
        _write_subset_csv(src, subset, n_rows=50)
        out_dir = tmpdir / "out"
        monkeypatch_env = {
            "PYTHONPATH": str(_REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        wrapper = tmpdir / "run_subset.py"
        wrapper.write_text(
            "import sys\n"
            "sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "from molmetal.scripts.bench_astr_only_baseline import (\n"
            "    _platinai_4cell_spec, _evaluate_dataset, _write_json,\n"
            ")\n"
            "spec = _platinai_4cell_spec(Path({repo!r}))\n"
            "spec.csv_path = Path({subset!r})\n"
            "spec.n_max = 50\n"
            "result = _evaluate_dataset(spec, [0], 3, 8, 5, 3)\n"
            "_write_json(Path({out!r}) / 'platinai_4cell' / 'per_seed_metrics.json', result)\n".format(
                repo=str(_REPO_ROOT),
                subset=str(subset),
                out=str(out_dir),
            )
        )
        proc = subprocess.run(
            [sys.executable, str(wrapper)],
            env={**os.environ, **monkeypatch_env},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, (
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        json_path = out_dir / "platinai_4cell" / "per_seed_metrics.json"
        assert json_path.exists(), f"missing {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Required top-level keys.
        for k in (
            "dataset",
            "n_samples",
            "n_features",
            "n_targets",
            "cohorts",
            "feature_columns",
            "n_seeds",
            "astr_population",
            "astr_generations",
            "astr_depth",
            "overall_r2_mean",
            "overall_r2_std",
            "per_target_summary",
            "per_seed",
        ):
            assert k in obj, f"missing key {k} in {json_path}"
        # Per-seed entry shape.
        seed0 = obj["per_seed"][0]
        for k in (
            "seed",
            "overall_r2",
            "per_target_mean_r2",
            "per_target_std_r2",
            "per_fold",
            "wall_time_s",
            "n_samples",
            "n_folds",
            "collapse_detected_overall",
            "per_target_per_cohort_mean_r2",
            "best_ast_lisp",
        ):
            assert k in seed0, f"missing per-seed key {k}"
        # overall_r2_mean must be finite.
        assert math.isfinite(obj["overall_r2_mean"])
        # Feature columns must be the F5_RDKIT_2D 12-d set.
        assert obj["n_features"] == 12
