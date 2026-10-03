"""Pytest suite for Wf-86 Phase 1 — REAL Hilbert+ASTSR R² on MetalCytoToxDB 5-metal.

The script ``molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py``
is the **Wf-49 P4 cross-dataset R² close-out** for MetalCytoToxDB
5-metal (Ru / Ir / Os / Rh / Re).  These tests guard the script's
invariants and provide fast unit-level confidence.

Test inventory
--------------
1.  ``test_feature_columns_constant`` — 12-d F5_RDKIT_2D schema matches
    Wf-49 conventions.
2.  ``test_default_metals_constant`` — default 5-metal cohort is
    ``("Ru", "Ir", "Os", "Rh", "Re")`` per Wf-49 PARTIAL SHIP.
3.  ``test_metalcsv_candidates_constant`` — CSV path discovery has
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
17. ``test_load_metal_cohort_ru_smoke`` — load Ru cohort: X shape
    (n, 12), y is finite 0/1, scaffolds non-empty.
18. ``test_combined_baseline_smoke`` — invoke the script with
    ``--metals Re --seeds 0 --epochs 5`` and confirm
    ``per_seed_metrics.json`` is well-formed JSON with all required
    keys + a non-zero ``n_samples``.
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
# Tests live in molmetal/tests/; the repo root is one level above molmetal.
_REPO_ROOT = _HERE.parent  # molmetal/
while (
    _REPO_ROOT != _REPO_ROOT.parent
    and not (
        _REPO_ROOT / "molmetal" / "scripts" / "bench_hilbert_astr_real_metalcytotox.py"
    ).exists()
):
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.bench_hilbert_astr_real_metalcytotox import (  # noqa: E402
    FEATURE_COLUMNS,
    PRED_CLIP,
    BLEND_ALPHAS,
    DEFAULT_METALS,
    METALCSV_CANDIDATES,
    _COUNTERION_FRAGMENTS,
    _extract_ligand_smiles,
    _collapse_detected,
    _compute_r2,
    _random_kfold_indices,
    _strict_scaffold_kfold_indices,
    _kfold_indices,
    hilbert_feature_map,
    predict_hilbert,
    load_metal_cohort,
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


def test_default_metals_constant() -> None:
    assert DEFAULT_METALS == ("Ru", "Ir", "Os", "Rh", "Re")
    assert len(DEFAULT_METALS) == 5


def test_metalcsv_candidates_constant() -> None:
    assert len(METALCSV_CANDIDATES) >= 1
    assert all(isinstance(p, Path) for p in METALCSV_CANDIDATES)


def test_counterion_fragments_constant() -> None:
    # Sanity: at least [Cl-] must be in the counter-ion list.
    assert "[Cl-]" in _COUNTERION_FRAGMENTS
    assert "[PF6-]" in _COUNTERION_FRAGMENTS


# ---------------------------------------------------------------------------
# Ligand extraction.
# ---------------------------------------------------------------------------
def test_extract_ligand_smiles_drops_counterions() -> None:
    # Multi-component SMILES with explicit [Cl-] counter-ion.
    smi = "c1ccccc1.[Cl-].c1ccncc1"
    out = _extract_ligand_smiles(smi)
    assert out is not None
    assert "[Cl-]" not in out
    # Both organic fragments should be present (largest first).
    assert "c1ccccc1" in out or "c1ccncc1" in out


def test_extract_ligand_smiles_keeps_largest() -> None:
    # When two non-counter-ion fragments differ in size, keep the largest.
    smi = "C.C1CCCCC1.[Cl-]"
    out = _extract_ligand_smiles(smi)
    assert out is not None
    assert "C1CCCCC1" in out  # larger fragment survives


def test_extract_ligand_smiles_returns_none_for_empty() -> None:
    assert _extract_ligand_smiles("") is None
    assert _extract_ligand_smiles("...") is None
    assert _extract_ligand_smiles("   ") is None


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
    # Construct a clearly-collapsed prediction: tiny std (0.05) << y std (1.0).
    # pred_std/y_std ≈ 0.05 < COLLAPSE_RATIO_THRESHOLD (0.10).
    yhat_collapsed = 0.05 * rng.normal(0.0, 1.0, 1000)
    assert _collapse_detected(yhat_collapsed, y) is True
    # Construct a clearly-wide prediction: pred_std ≈ y_std.
    yhat_wide_again = y + rng.normal(0.0, 0.20, 1000)
    assert _collapse_detected(yhat_wide_again, y) is False


# ---------------------------------------------------------------------------
# Blend alphas.
# ---------------------------------------------------------------------------
def test_blend_alphas_constant() -> None:
    assert BLEND_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(BLEND_ALPHAS) == 5


# ---------------------------------------------------------------------------
# K-fold splitting.
# ---------------------------------------------------------------------------
def test_strict_scaffold_disjoint() -> None:
    # 30 rows × 5 scaffolds, each scaffold should be entirely in one fold.
    rng = np.random.default_rng(0)
    scaffolds = [f"sc_{i // 6}" for i in range(30)]
    splits = _strict_scaffold_kfold_indices(scaffolds, k=5, seed=0)
    for tr, te in splits:
        if len(te) == 0:
            continue
        # Build scaffold set of test rows.
        te_sc = set(scaffolds[i] for i in te)
        tr_sc = set(scaffolds[i] for i in tr)
        # No scaffold shared between train and test.
        assert te_sc.isdisjoint(tr_sc)


def test_strict_scaffold_covers_all() -> None:
    n = 50
    rng = np.random.default_rng(0)
    scaffolds = [f"sc_{i // 10}" for i in range(n)]
    splits = _strict_scaffold_kfold_indices(scaffolds, k=5, seed=0)
    union = np.sort(np.concatenate([te for _, te in splits if len(te) > 0]))
    assert len(union) == n
    assert np.array_equal(union, np.arange(n))


def test_strict_scaffold_deterministic() -> None:
    n = 40
    scaffolds = [f"sc_{i // 8}" for i in range(n)]
    splits_a = _strict_scaffold_kfold_indices(scaffolds, k=5, seed=42)
    splits_b = _strict_scaffold_kfold_indices(scaffolds, k=5, seed=42)
    for (tr_a, te_a), (tr_b, te_b) in zip(splits_a, splits_b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)


def test_random_kfold_no_overlap() -> None:
    n = 100
    splits = _random_kfold_indices(n, k=5, seed=0)
    for tr, te in splits:
        # No overlap between train and test.
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


def test_kfold_dispatcher_strict_scaffold() -> None:
    # When strict_scaffold=True and scaffold_ids provided, use strict split.
    scaffolds = [f"sc_{i // 10}" for i in range(50)]
    splits = _kfold_indices(
        n=50,
        k=5,
        seed=0,
        scaffold_ids=scaffolds,
        strict_scaffold=True,
    )
    for tr, te in splits:
        te_sc = set(scaffolds[i] for i in te)
        tr_sc = set(scaffolds[i] for i in tr)
        assert te_sc.isdisjoint(tr_sc)


def test_kfold_dispatcher_random_fallback() -> None:
    # When strict_scaffold=False, use random split.
    scaffolds = [f"sc_{i // 10}" for i in range(50)]
    splits = _kfold_indices(
        n=50,
        k=5,
        seed=0,
        scaffold_ids=scaffolds,
        strict_scaffold=False,
    )
    for tr, te in splits:
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


# ---------------------------------------------------------------------------
# Hilbert feature map (reused from phase2 4cell suite).
# ---------------------------------------------------------------------------
def test_hilbert_feature_map_truncates() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (50, 12)).astype(np.float32)
    Xt = rng.normal(0.0, 1.0, (10, 12)).astype(np.float32)
    X_aug, Xt_aug = hilbert_feature_map(X, hilbert_dim=8, X_12d_test=Xt)
    assert X_aug.shape == (50, 8)
    assert Xt_aug is not None and Xt_aug.shape == (10, 8)


def test_predict_hilbert_clip() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (10, 4)).astype(np.float32)
    # Huge weights to trigger clipping.
    bp = np.asarray([1e9] * 4, dtype=np.float32)
    y = predict_hilbert(X, bp)
    assert y.max() <= PRED_CLIP
    assert y.min() >= -PRED_CLIP


# ---------------------------------------------------------------------------
# Dataset loader smoke.
# ---------------------------------------------------------------------------
def test_load_metal_cohort_ru_smoke() -> None:
    try:
        X, y, smiles, scaffolds = load_metal_cohort("Ru", n_max=50)
    except FileNotFoundError:
        pytest.skip("metalcytotox_5metal_multitask_v4.csv not present")
    assert X.shape[1] == 12
    assert y.size == X.shape[0]
    # Binary activity.
    assert set(np.unique(y).tolist()).issubset({0.0, 1.0})
    assert len(smiles) == X.shape[0]
    assert len(scaffolds) == X.shape[0]


# ---------------------------------------------------------------------------
# End-to-end script smoke.
# ---------------------------------------------------------------------------
def test_combined_baseline_smoke() -> None:
    """Invoke the script with `--epochs 5` on Re (smallest cohort) and
    confirm `per_seed_metrics.json` is well-formed JSON with all
    required keys + non-zero n_samples."""
    csv_path = None
    for c in METALCSV_CANDIDATES:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        pytest.skip("metalcytotox_5metal_multitask_v4.csv not present")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "wf86_smoke"
        cmd = [
            sys.executable,
            "molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py",
            "--metals",
            "Re",
            "--seeds",
            "0",
            "--epochs",
            "5",
            "--strict-scaffold",
            "true",
            "--out-dir",
            str(out_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_HERE.parent) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert (
            proc.returncode == 0
        ), f"Script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        per_seed_path = out_dir / "Re" / "per_seed_metrics.json"
        assert per_seed_path.exists()
        with open(per_seed_path) as f:
            d = json.load(f)
        for key in (
            "metal",
            "target_col",
            "n_samples",
            "n_features",
            "feature_columns",
            "n_seeds",
            "strict_scaffold",
            "combined_r2_mean",
            "combined_r2_std",
            "hilbert_r2_mean",
            "hilbert_r2_std",
            "astr_r2_mean",
            "astr_r2_std",
            "per_seed",
        ):
            assert key in d, f"Missing key {key!r}"
        assert d["metal"] == "Re"
        assert d["n_samples"] > 0
        assert len(d["per_seed"]) == 1
        per_seed0 = d["per_seed"][0]
        for k in (
            "seed",
            "combined_r2_overall",
            "hilbert_r2_overall",
            "astr_r2_overall",
            "combined_r2_per_fold_mean",
            "alpha_mean",
            "wall_time_s",
            "n_samples",
            "n_folds",
            "per_fold",
        ):
            assert k in per_seed0, f"Missing per_seed key {k!r}"
        assert per_seed0["n_folds"] == 5
        assert len(per_seed0["per_fold"]) >= 1
        # Synthesis.md exists too.
        assert (out_dir / "synthesis.md").exists()
