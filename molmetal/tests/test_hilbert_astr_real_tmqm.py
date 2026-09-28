"""Pytest suite for Wf-86 Phase 2 — REAL Hilbert+ASTSR R² on tmQM d-block.

The script ``molmetal/scripts/bench_hilbert_astr_real_tmqm.py``
is the **Wf-49 P4 cross-dataset R² close-out** for tmQM d-block
5-element (Cu / Fe / Ni / Zn / Co on the Balcells & Skjelstad 2020
corpus subset shipped as ``molmetal/data/tmqm_dblock_subset.csv``).
These tests guard the script's invariants and provide fast unit-level
confidence.

Test inventory
--------------
1.  ``test_feature_columns_constant`` — 12-d F5_RDKIT_2D schema matches
    Wf-49 conventions.
2.  ``test_default_elements_constant`` — default 5-element cohort is
    ``("Cu", "Fe", "Ni", "Zn", "Co")`` per first-row d-block metals.
3.  ``test_tmqmcsv_candidates_constant`` — CSV path discovery has
    at least one candidate path.
4.  ``test_counterion_fragments_constant`` — counter-ion fragment
    list is stable (used to drop [Cl-], [PF6-], etc.).
5.  ``test_extract_ligand_smiles_drops_counterions`` — multi-component
    SMILES ``L1.[Cl-].L2`` → ``L1.L2`` (largest non-counter-ion).
6.  ``test_extract_ligand_smiles_keeps_largest`` — when 3 fragments
    are present, the largest non-counter-ion is kept.
7.  ``test_extract_ligand_smiles_returns_none_for_empty`` — empty
    SMILES returns ``None``.
8.  ``test_compute_r2_zero_for_perfect`` — perfect predictor ⇒ R² = 1.
9.  ``test_compute_r2_zero_for_constant`` — constant predictor at
    ``y.mean()`` ⇒ R² = 0.
10. ``test_compute_r2_negative`` — strictly bad predictor ⇒ R² < 0.
11. ``test_collapse_detector`` — pred_std / y_std < 0.1 ⇒ True.
12. ``test_blend_alphas_constant`` — BLEND_ALPHAS set is stable.
13. ``test_strict_scaffold_disjoint`` — strict scaffold split
    guarantees no scaffold appears in both train and test.
14. ``test_strict_scaffold_covers_all`` — union of test folds = full set.
15. ``test_strict_scaffold_deterministic`` — same seed ⇒ identical splits.
16. ``test_random_kfold_no_overlap`` — random fallback: train / test
    never overlap.
17. ``test_load_element_cohort_cu_smoke`` — load Cu cohort: X shape
    (n, 12), y is finite, scaffolds non-empty.
18. ``test_combined_baseline_smoke`` — invoke the script with
    ``--elements Cu --seeds 0 --epochs 5`` and confirm
    ``per_seed_metrics.json`` is well-formed JSON with all required
    keys + a non-zero ``n_samples``.
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
while _REPO_ROOT != _REPO_ROOT.parent and not (
    _REPO_ROOT / "molmetal" / "scripts" / "bench_hilbert_astr_real_tmqm.py"
).exists():
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.bench_hilbert_astr_real_tmqm import (  # noqa: E402
    FEATURE_COLUMNS,
    COLLAPSE_RATIO_THRESHOLD,
    PRED_CLIP,
    BLEND_ALPHAS,
    DEFAULT_ELEMENTS,
    SUPPORTED_TARGETS,
    TMQMCSV_CANDIDATES,
    COORD_COL,
    _COUNTERION_FRAGMENTS,
    _extract_ligand_smiles,
    _collapse_detected,
    _compute_r2,
    _random_kfold_indices,
    _strict_scaffold_kfold_indices,
    _kfold_indices,
    hilbert_feature_map,
    predict_hilbert,
    load_element_cohort,
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


def test_default_elements_constant() -> None:
    assert DEFAULT_ELEMENTS == ("Cu", "Fe", "Ni", "Zn", "Co")


def test_supported_targets_constant() -> None:
    assert "coord_number" in SUPPORTED_TARGETS
    assert "logp" in SUPPORTED_TARGETS
    assert "qed_score" in SUPPORTED_TARGETS
    assert "sa_score" in SUPPORTED_TARGETS


def test_tmqmcsv_candidates_constant() -> None:
    assert len(TMQMCSV_CANDIDATES) >= 1
    # At least one of the candidate paths must exist on disk (the
    # ``tmqm_dblock_subset.csv`` shipped with the repo).
    assert any(c.exists() for c in TMQMCSV_CANDIDATES), (
        f"None of TMQMCSV_CANDIDATES exist: {[str(c) for c in TMQMCSV_CANDIDATES]}"
    )


def test_counterion_fragments_constant() -> None:
    # Counter-ion fragments are stable for backwards-compatibility with
    # MetalCytoTox / tmQM d-block conventions.
    assert "[Cl-]" in _COUNTERION_FRAGMENTS
    assert "[PF6-]" in _COUNTERION_FRAGMENTS
    assert "[OH2]" in _COUNTERION_FRAGMENTS


# ---------------------------------------------------------------------------
# Ligand extraction.
# ---------------------------------------------------------------------------
def test_extract_ligand_smiles_drops_counterions() -> None:
    smi = "CCO.[Cl-].c1ccccc1"
    out = _extract_ligand_smiles(smi)
    # Largest non-counter-ion fragment is ``c1ccccc1`` (10 chars vs 3).
    assert out == "c1ccccc1"


def test_extract_ligand_smiles_keeps_largest() -> None:
    smi = "O.N.S.c1ccccc1c1ccccc1"
    out = _extract_ligand_smiles(smi)
    # Largest by SMILES length (excluding counterions like O/N/S).
    assert out == "c1ccccc1c1ccccc1"


def test_extract_ligand_smiles_returns_none_for_empty() -> None:
    assert _extract_ligand_smiles("") is None
    assert _extract_ligand_smiles("   ") is None


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


def test_blend_alphas_constant() -> None:
    assert BLEND_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)


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
# Hilbert feature map.
# ---------------------------------------------------------------------------
def test_hilbert_feature_map_shape() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 12)).astype(np.float32)
    Xt = rng.standard_normal((20, 12)).astype(np.float32)
    out_tr, out_te = hilbert_feature_map(X, hilbert_dim=64, X_12d_test=Xt)
    assert out_tr.shape == (50, 64)
    assert out_te is not None and out_te.shape == (20, 64)


def test_hilbert_feature_map_no_test() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 12)).astype(np.float32)
    out_tr, out_te = hilbert_feature_map(X, hilbert_dim=64)
    assert out_tr.shape == (30, 64)
    assert out_te is None


def test_predict_hilbert_clips() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    bp = np.full(4, 1e9, dtype=np.float64)
    y_hat = predict_hilbert(X, bp)
    # PRED_CLIP = 1e6, so each y_hat element must be ≤ 1e6.
    assert np.all(np.abs(y_hat) <= PRED_CLIP + 1e-3)


# ---------------------------------------------------------------------------
# Element cohort loader.
# ---------------------------------------------------------------------------
def test_load_element_cohort_cu_smoke() -> None:
    X, y, smiles, scaffolds = load_element_cohort("Cu", target_col=COORD_COL)
    assert X.ndim == 2
    assert X.shape[1] == 12
    assert y.shape[0] == X.shape[0]
    assert all(math.isfinite(v) for v in y.tolist())
    assert all(math.isfinite(v) for v in X.flatten().tolist())
    assert len(smiles) == X.shape[0]
    # Strict-scaffold scaffolds must be non-empty for the script to
    # work (the two-tier fallback ensures this even for open-chain
    # ligands where Bemis-Murcko returns "").
    assert all(s for s in scaffolds), (
        "Some rows have empty scaffold (would collapse strict-scaffold split)"
    )


def test_load_element_cohort_unsupported_target() -> None:
    # Should not raise — we just feed a non-existent column; the loader
    # falls back to all-NaN rows and raises a clear error.
    with pytest.raises(RuntimeError, match="No usable rows"):
        load_element_cohort("Cu", target_col="__nonexistent__")


# ---------------------------------------------------------------------------
# Synthesis writer.
# ---------------------------------------------------------------------------
def test_write_synthesis_creates_file(tmp_path: Path) -> None:
    fake_results = [
        {
            "element": "Cu",
            "target_col": "coord_number",
            "n_samples": 174,
            "n_features": 12,
            "feature_columns": list(FEATURE_COLUMNS),
            "n_seeds": 2,
            "strict_scaffold": True,
            "combined_r2_mean": -0.10,
            "combined_r2_std": 0.02,
            "hilbert_r2_mean": -1.5,
            "hilbert_r2_std": 0.1,
            "astr_r2_mean": -0.08,
            "astr_r2_std": 0.05,
            "per_seed": [
                {
                    "seed": 0,
                    "combined_r2_overall": -0.10,
                    "hilbert_r2_overall": -1.5,
                    "astr_r2_overall": -0.08,
                    "combined_r2_per_fold_mean": -0.10,
                    "hilbert_r2_per_fold_mean": -1.5,
                    "astr_r2_per_fold_mean": -0.08,
                    "alpha_mean": 0.1,
                    "alpha_std": 0.0,
                    "per_fold": [],
                    "wall_time_s": 1.5,
                    "n_samples": 174,
                    "n_folds": 5,
                    "collapse_detected_combined": False,
                },
                {
                    "seed": 1,
                    "combined_r2_overall": -0.11,
                    "hilbert_r2_overall": -1.6,
                    "astr_r2_overall": -0.09,
                    "combined_r2_per_fold_mean": -0.11,
                    "hilbert_r2_per_fold_mean": -1.6,
                    "astr_r2_per_fold_mean": -0.09,
                    "alpha_mean": 0.0,
                    "alpha_std": 0.0,
                    "per_fold": [],
                    "wall_time_s": 1.4,
                    "n_samples": 174,
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
    assert "Wf-86 Phase 2" in text
    assert "Cu" in text
    assert "FAIL" in text  # aggregate < 0.0.


# ---------------------------------------------------------------------------
# End-to-end CLI smoke.
# ---------------------------------------------------------------------------
def test_combined_baseline_smoke(tmp_path: Path) -> None:
    """Run the script on Cu with 1 seed, 5 epochs, 2 folds — fast
    smoke that exercises the full CLI path.  Confirms JSON shape
    and non-zero ``n_samples``.
    """
    repo_root = _REPO_ROOT
    if repo_root.name != "mol-metal":
        # Fallback: walk up to find ``mol-metal``.
        for cand in [_REPO_ROOT, _REPO_ROOT.parent]:
            if (cand / "molmetal" / "scripts" / "bench_hilbert_astr_real_tmqm.py").exists():
                repo_root = cand
                break
    cmd = [
        "molmetal/.venvs/mol-metal-venv/bin/python",
        "-u",
        "molmetal/scripts/bench_hilbert_astr_real_tmqm.py",
        "--elements",
        "Cu",
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
    json_path = tmp_path / "Cu" / "per_seed_metrics.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # All required keys present.
    for key in (
        "element", "target_col", "n_samples", "n_features",
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
