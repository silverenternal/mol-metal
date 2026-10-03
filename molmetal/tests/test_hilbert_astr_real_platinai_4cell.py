"""Pytest suite for Wf-86 Phase 3 — REAL Hilbert+ASTSR R² on PlatinAI 4-cell.

The script ``molmetal/scripts/bench_hilbert_astr_real_platinai_4cell.py``
is the **Wf-49 P4 cross-dataset R² close-out** for PlatinAI 4-cell
(MCF7 + A549 + A2780 + A2780cis on the Rusanov 2025 cohort shipped as
``molmetal/data/platinai_4cell.csv``).  These tests guard the script's
invariants and provide fast unit-level confidence.

Test inventory
--------------
1.  ``test_feature_columns_constant`` — 12-d F5_RDKIT_2D schema matches
    Wf-49 conventions.
2.  ``test_default_cell_lines_constant`` — default 4-cell-line cohort
    is ``("mcf7", "a549", "a2780", "a2780cis")`` per Rusanov 2025
    release.
3.  ``test_platinai_cell_cols_constant`` — 4 cell-line columns present.
4.  ``test_platinai_csv_path_exists`` — PlatinAI 4-cell CSV is present.
5.  ``test_blend_alphas_constant`` — BLEND_ALPHAS set is stable.
6.  ``test_collapse_ratio_threshold_constant`` — COLLAPSE_RATIO_THRESHOLD = 0.10.
7.  ``test_compute_r2_zero_for_perfect`` — perfect predictor ⇒ R² = 1.
8.  ``test_compute_r2_zero_for_constant`` — constant predictor at
    ``y.mean()`` ⇒ R² = 0.
9.  ``test_compute_r2_negative`` — strictly bad predictor ⇒ R² < 0.
10. ``test_collapse_detector`` — pred_std / y_std < 0.1 ⇒ True.
11. ``test_choose_alpha_picks_best`` — choose_alpha picks alpha with
    highest inner R².
12. ``test_hilbert_feature_map_truncates`` — augmentation dim ≤
    ``hilbert_dim``.
13. ``test_hilbert_feature_map_augments`` — when F < hilbert_dim,
    augmented cols > F.
14. ``test_strict_scaffold_disjoint`` — strict scaffold split
    guarantees no scaffold appears in both train and test.
15. ``test_strict_scaffold_covers_all`` — union of test folds = full set.
16. ``test_random_kfold_no_overlap`` — random fallback: train / test
    never overlap.
17. ``test_load_platinai_cell_mcf7_smoke`` — load MCF7 cell line: X
    shape (n, 12), y is finite, scaffolds non-empty, smiles list
    non-empty.
18. ``test_load_platinai_cell_unknown_raises`` — unknown cell line
    raises ``ValueError``.
19. ``test_bemis_murcko_scaffold_fallback`` — Bemis-Murcko returns
    non-empty scaffold for open-chain Pt complexes (tier-2 fallback).
20. ``test_write_synthesis_creates_file`` — write_synthesis writes a
    well-formed ``synthesis.md`` with all required keys.
21. ``test_combined_baseline_smoke`` — invoke the script with
    ``--cell-lines mcf7 --seeds 0 --epochs 5 --n-folds 2`` and confirm
    ``per_seed_metrics.json`` is well-formed JSON with all required
    keys + a non-zero ``n_samples``.
"""
from __future__ import annotations

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
while _REPO_ROOT != _REPO_ROOT.parent and not (
    _REPO_ROOT / "molmetal" / "scripts" / "bench_hilbert_astr_real_platinai_4cell.py"
).exists():
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.bench_hilbert_astr_real_platinai_4cell import (  # noqa: E402
    FEATURE_COLUMNS,
    COLLAPSE_RATIO_THRESHOLD,
    PRED_CLIP,
    BLEND_ALPHAS,
    DEFAULT_CELL_LINES,
    PLATINAI_CELL_COLS,
    PLATINAI_CSV,
    _bemis_murcko_scaffold,
    _collapse_detected,
    _compute_r2,
    _choose_alpha,
    _random_kfold_indices,
    _strict_scaffold_kfold_indices,
    _kfold_indices,
    hilbert_feature_map,
    predict_hilbert,
    load_platinai_cell,
    write_synthesis,
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


def test_default_cell_lines_constant() -> None:
    assert DEFAULT_CELL_LINES == ("mcf7", "a549", "a2780", "a2780cis")


def test_platinai_cell_cols_constant() -> None:
    assert set(PLATINAI_CELL_COLS) == {"mcf7", "a549", "a2780", "a2780cis"}
    assert PLATINAI_CELL_COLS["mcf7"] == "ic50_mcf7_nm"
    assert PLATINAI_CELL_COLS["a549"] == "ic50_a549_nm"
    assert PLATINAI_CELL_COLS["a2780"] == "ic50_a2780_nm"
    assert PLATINAI_CELL_COLS["a2780cis"] == "ic50_a2780cis_nm"


def test_platinai_csv_path_exists() -> None:
    csv_path = _REPO_ROOT / PLATINAI_CSV
    assert csv_path.exists(), f"PlatinAI CSV not found at {csv_path}"


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------
def test_blend_alphas_constant() -> None:
    assert BLEND_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_collapse_ratio_threshold_constant() -> None:
    assert COLLAPSE_RATIO_THRESHOLD == pytest.approx(0.10, abs=1e-9)


# ---------------------------------------------------------------------------
# R² + collapse detector.
# ---------------------------------------------------------------------------
def test_compute_r2_zero_for_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = y.copy()
    r2 = _compute_r2(y, y_hat)
    assert math.isclose(r2, 1.0, abs_tol=1e-9)


def test_compute_r2_zero_for_constant() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = np.full_like(y, y.mean())
    r2 = _compute_r2(y, y_hat)
    assert math.isclose(r2, 0.0, abs_tol=1e-9)


def test_compute_r2_negative() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = -y  # strictly worse than predicting the mean.
    r2 = _compute_r2(y, y_hat)
    assert r2 < -2.0  # Should be very negative.


def test_collapse_detector() -> None:
    y = np.linspace(-1.0, 1.0, 100)
    y_pred_collapsed = np.full_like(y, 0.5)  # std ≈ 0.
    assert _collapse_detected(y_pred_collapsed, y) is True
    y_pred_good = y + np.random.default_rng(0).normal(0, 0.1, size=y.shape)
    assert _collapse_detected(y_pred_good, y) is False


def test_choose_alpha_picks_best() -> None:
    y_true = np.linspace(0.0, 1.0, 20)
    # Hilbert predicts perfectly, ASTSR predicts zero.
    h = y_true.copy()
    a = np.zeros_like(y_true)
    alpha, r2 = _choose_alpha(y_true, h, a)
    assert alpha == 1.0
    assert r2 == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Hilbert feature map.
# ---------------------------------------------------------------------------
def test_hilbert_feature_map_truncates() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 12)).astype(np.float32)
    out_tr, _ = hilbert_feature_map(X, hilbert_dim=32)
    assert out_tr.shape == (50, 32)


def test_hilbert_feature_map_augments() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8)).astype(np.float32)
    out_tr, _ = hilbert_feature_map(X, hilbert_dim=64)
    assert out_tr.shape == (40, 64)


def test_hilbert_feature_map_test_alignment() -> None:
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((40, 12)).astype(np.float32)
    X_te = rng.standard_normal((10, 12)).astype(np.float32)
    out_tr, out_te = hilbert_feature_map(X_tr, hilbert_dim=32, X_12d_test=X_te)
    assert out_tr.shape[1] == out_te.shape[1]
    assert out_tr.shape == (40, 32)
    assert out_te.shape == (10, 32)


def test_predict_hilbert_clips() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    bp = np.full(4, 1e9, dtype=np.float64)
    y_hat = predict_hilbert(X, bp)
    # PRED_CLIP = 1e6, so each y_hat element must be ≤ 1e6.
    assert np.all(np.abs(y_hat) <= PRED_CLIP + 1e-3)


# ---------------------------------------------------------------------------
# Scaffold splitting.
# ---------------------------------------------------------------------------
def test_strict_scaffold_disjoint() -> None:
    scaffolds = ["A", "B", "C", "A", "B", "D", "C", "D"]
    splits = _strict_scaffold_kfold_indices(scaffolds, k=2, seed=42)
    # Each scaffold must appear in the test set of **at most one** fold
    # (the strict-scaffold invariant).  Multiple rows of the same
    # scaffold landing in the same fold's test is fine.
    seen_in_test_per_scaffold: dict = {}
    for fold_idx, (_, test) in enumerate(splits):
        # Dedupe scaffolds within this fold's test set first.
        unique_scaffolds_in_fold = {scaffolds[idx] for idx in test}
        for sc in unique_scaffolds_in_fold:
            assert sc not in seen_in_test_per_scaffold, (
                f"Scaffold {sc!r} appears in test of multiple folds"
            )
            seen_in_test_per_scaffold[sc] = fold_idx


def test_strict_scaffold_covers_all() -> None:
    n = 50
    scaffolds = [f"sc_{i // 3}" for i in range(n)]  # ~17 unique scaffolds.
    splits = _strict_scaffold_kfold_indices(scaffolds, k=5, seed=42)
    all_test = sorted(np.concatenate([t for _, t in splits]).tolist())
    assert all_test == list(range(n))


def test_strict_scaffold_deterministic() -> None:
    scaffolds = [f"sc_{i // 3}" for i in range(30)]
    a = _strict_scaffold_kfold_indices(scaffolds, k=3, seed=7)
    b = _strict_scaffold_kfold_indices(scaffolds, k=3, seed=7)
    for (ta, tea), (tb, teb) in zip(a, b):
        assert np.array_equal(ta, tb)
        assert np.array_equal(tea, teb)


def test_random_kfold_no_overlap() -> None:
    n = 100
    splits = _random_kfold_indices(n, k=4, seed=0)
    seen = set()
    for train, test in splits:
        for idx in test:
            assert idx not in seen, f"row {idx} appears in multiple test folds"
            seen.add(int(idx))
        # Train and test disjoint.
        assert set(int(x) for x in train) & set(int(x) for x in test) == set()


def test_kfold_indices_dispatches_to_strict_when_enabled() -> None:
    scaffolds = ["A", "B", "A", "B", "C", "C"]
    splits = _kfold_indices(
        len(scaffolds), k=2, seed=0,
        scaffold_ids=scaffolds, strict_scaffold=True,
    )
    # Strict path yields disjoint scaffolds per fold.
    seen: dict = {}
    for fold_idx, (_, test) in enumerate(splits):
        unique_scaffolds_in_fold = {scaffolds[idx] for idx in test}
        for sc in unique_scaffolds_in_fold:
            assert sc not in seen, (
                f"Scaffold {sc!r} appears in test of multiple folds"
            )
            seen[sc] = fold_idx


# ---------------------------------------------------------------------------
# Bemis-Murcko fallback.
# ---------------------------------------------------------------------------
def test_bemis_murcko_scaffold_fallback() -> None:
    """Open-chain Pt complexes (cisplatin-like) should get a non-empty
    scaffold key from the tier-2 fallback (canonicalised ligand)."""
    # Cisplatin: cis-[Pt(NH3)2Cl2]
    smi = "[NH3]->[Pt+2](<-[NH3])(<-[Cl-])<-[Cl-]"
    scaffold = _bemis_murcko_scaffold(smi)
    # Bemis-Murcko will return empty for this (no rings), but the
    # tier-2 fallback should yield a non-empty canonicalised ligand.
    assert scaffold != "", (
        f"Tier-2 fallback failed: scaffold={scaffold!r}"
    )


# ---------------------------------------------------------------------------
# PlatinAI 4-cell loader.
# ---------------------------------------------------------------------------
def test_load_platinai_cell_mcf7_smoke() -> None:
    X, y, smiles, scaffolds = load_platinai_cell(_REPO_ROOT, "mcf7", n_max=50)
    assert X.ndim == 2
    assert X.shape[1] == 12
    assert y.shape[0] == X.shape[0]
    assert all(math.isfinite(v) for v in y.tolist())
    assert all(math.isfinite(v) for v in X.flatten().tolist())
    assert len(smiles) == X.shape[0]
    # Strict-scaffold scaffolds must be non-empty for the script to
    # work (the two-tier fallback ensures this even for open-chain
    # Pt-coordination complexes where Bemis-Murcko returns "").
    assert all(s for s in scaffolds), (
        "Some rows have empty scaffold (would collapse strict-scaffold split)"
    )


def test_load_platinai_cell_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown cell line"):
        load_platinai_cell(_REPO_ROOT, "hela_unknown")


# ---------------------------------------------------------------------------
# Synthesis writer.
# ---------------------------------------------------------------------------
def test_write_synthesis_creates_file(tmp_path: Path) -> None:
    fake_results = [
        {
            "cell_line": "mcf7",
            "target_col": "ic50_mcf7_nm",
            "n_samples": 4078,
            "n_features": 12,
            "feature_columns": list(FEATURE_COLUMNS),
            "n_seeds": 2,
            "strict_scaffold": True,
            "combined_r2_mean": -0.01,
            "combined_r2_std": 0.005,
            "hilbert_r2_mean": -23.0,
            "hilbert_r2_std": 1.5,
            "astr_r2_mean": -0.01,
            "astr_r2_std": 0.005,
            "per_seed": [
                {
                    "seed": 0,
                    "combined_r2_overall": -0.01,
                    "hilbert_r2_overall": -23.0,
                    "astr_r2_overall": -0.01,
                    "combined_r2_per_fold_mean": -0.01,
                    "hilbert_r2_per_fold_mean": -23.0,
                    "astr_r2_per_fold_mean": -0.01,
                    "alpha_mean": 0.1,
                    "alpha_std": 0.0,
                    "per_fold": [],
                    "wall_time_s": 20.0,
                    "n_samples": 4078,
                    "n_folds": 5,
                    "collapse_detected_combined": False,
                },
                {
                    "seed": 1,
                    "combined_r2_overall": -0.012,
                    "hilbert_r2_overall": -24.0,
                    "astr_r2_overall": -0.012,
                    "combined_r2_per_fold_mean": -0.012,
                    "hilbert_r2_per_fold_mean": -24.0,
                    "astr_r2_per_fold_mean": -0.012,
                    "alpha_mean": 0.0,
                    "alpha_std": 0.0,
                    "per_fold": [],
                    "wall_time_s": 19.0,
                    "n_samples": 4078,
                    "n_folds": 5,
                    "collapse_detected_combined": False,
                },
            ],
        }
    ]
    write_synthesis(tmp_path, fake_results)
    synthesis_path = tmp_path / "synthesis.md"
    assert synthesis_path.exists()
    text = synthesis_path.read_text(encoding="utf-8")
    assert "Wf-86 Phase 3" in text
    assert "mcf7" in text
    assert "FAIL" in text  # aggregate < 0.0.


# ---------------------------------------------------------------------------
# Transformed Wf-85 P3 → Wf-86 P3 outputs.
# ---------------------------------------------------------------------------
def test_transformed_wf86_p3_outputs_exist() -> None:
    """Verify the Wf-86 P3 outputs are on disk for all 4 cell lines
    (transformed from the Wf-85 P3 MEASURED data via
    ``transform_wf85_p3_to_wf86_p3.py``).
    """
    wf86_dir = _REPO_ROOT / "molmetal" / "reports" / "wf86_p3_platinai_4cell_real"
    assert wf86_dir.exists(), f"Wf-86 P3 report dir missing: {wf86_dir}"
    # All 4 cell-line per_seed_metrics.json files exist.
    for cl in DEFAULT_CELL_LINES:
        cl_path = wf86_dir / cl / "per_seed_metrics.json"
        assert cl_path.exists(), f"missing Wf-86 P3 per_seed_metrics.json: {cl_path}"
        data = json.loads(cl_path.read_text(encoding="utf-8"))
        # Required keys per Wf-86 P3 schema.
        for key in (
            "cell_line", "target_col", "n_samples",
            "n_features", "feature_columns", "n_seeds",
            "strict_scaffold", "combined_r2_mean", "combined_r2_std",
            "hilbert_r2_mean", "hilbert_r2_std",
            "astr_r2_mean", "astr_r2_std", "per_seed",
        ):
            assert key in data, f"missing Wf-86 P3 key {key!r} for {cl}"
        assert data["strict_scaffold"] is True
        assert data["n_seeds"] == 5
        assert data["n_samples"] == 4078
        # Each per-seed entry has the Wf-86 P3 schema fields.
        for ps in data["per_seed"]:
            for k in (
                "seed", "combined_r2_overall", "hilbert_r2_overall",
                "astr_r2_overall", "combined_r2_per_fold_mean",
                "alpha_mean", "per_fold", "wall_time_s",
                "collapse_detected_combined", "source_wf85_p3",
            ):
                assert k in ps, f"missing Wf-86 P3 per-seed key {k!r} for {cl}"
    # Synthesis.md exists.
    assert (wf86_dir / "synthesis.md").exists()


# ---------------------------------------------------------------------------
# End-to-end CLI smoke.
# ---------------------------------------------------------------------------
def test_combined_baseline_smoke(tmp_path: Path) -> None:
    """Run the script on mcf7 with 1 seed, 5 epochs, 2 folds — fast
    smoke that exercises the full CLI path.  Confirms JSON shape
    and non-zero ``n_samples``.
    """
    repo_root = _REPO_ROOT
    if repo_root.name != "mol-metal":
        # Fallback: walk up to find ``mol-metal``.
        for cand in [_REPO_ROOT, _REPO_ROOT.parent]:
            if (cand / "molmetal" / "scripts" / "bench_hilbert_astr_real_platinai_4cell.py").exists():
                repo_root = cand
                break
    cmd = [
        "molmetal/.venvs/mol-metal-venv/bin/python",
        "-u",
        "molmetal/scripts/bench_hilbert_astr_real_platinai_4cell.py",
        "--cell-lines",
        "mcf7",
        "--seeds",
        "0",
        "--epochs",
        "5",
        "--n-folds",
        "2",
        "--out-dir",
        str(tmp_path),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"{repo_root}:{env.get('PYTHONPATH', '')}")
    proc = subprocess.run(
        cmd, cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"script failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    json_path = tmp_path / "mcf7" / "per_seed_metrics.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # All required keys present.
    for key in (
        "cell_line", "target_col", "n_samples", "n_features",
        "feature_columns", "n_seeds", "strict_scaffold",
        "combined_r2_mean", "combined_r2_std",
        "hilbert_r2_mean", "hilbert_r2_std",
        "astr_r2_mean", "astr_r2_std",
        "per_seed",
    ):
        assert key in data, f"missing key {key!r} in per_seed_metrics.json"
    assert data["n_samples"] > 0
    assert data["n_seeds"] == 1
    assert data["strict_scaffold"] is True
    assert len(data["per_seed"]) == 1
    # per_seed entries have per_fold breakdown.
    ps0 = data["per_seed"][0]
    assert "per_fold" in ps0
    assert "combined_r2_per_fold_mean" in ps0
    # Synthesis was written.
    assert (tmp_path / "synthesis.md").exists()
