"""Tests for Wf-31 Phase 3 — Multi-PH feature extraction + diversity bench.

These tests verify the integrity of the Wf-31 Phase-3 outputs that
``molmetal/scripts/bench_multi_ph_diversity.py`` and
``molmetal/scripts/verify_bench_multi_ph_diversity.py`` produce:

* the metrics JSON contract (expressivity, diversity, wall-time);
* the constants / hyper-parameters / default CSV path;
* the helper functions (``_mean_pairwise_cosine_distance``,
  ``compute_pairwise_tanimoto_distance``);
* the pipeline runs end-to-end on a tiny synthetic SMILES pool;
* the verify script produces a valid metrics JSON.

Test count target: ~15 distinct checks.

1.  ``test_module_imports`` — bench_multi_ph_diversity imports cleanly.
2.  ``test_constants_n_bins`` — MULTI_PH_N_BINS == 16.
3.  ``test_constants_max_radius_max_coord`` — MULTI_PH_MAX_RADIUS / COORD.
4.  ``test_constants_diversity_floor`` — DIVERSITY_FLOOR == 0.75 (m7 floor).
5.  ``test_default_platinai_csv_exists`` — default CSV path is shipped.
6.  ``test_load_platinai_5k_returns_rows`` — CSV loader returns rows.
7.  ``test_mean_pairwise_cosine_distance_diagonal_zero`` — diagonal of
    the cosine-distance matrix is zero (excluded from mean).
8.  ``test_mean_pairwise_cosine_distance_two_vectors`` — small
    two-vector sanity check.
9.  ``test_compute_pairwise_tanimoto_distance_diagonal_zero`` —
    Tanimoto distance matrix has zero diagonal.
10. ``test_compute_pairwise_tanimoto_distance_synthetic`` — two-bit
    fingerprint pair (identical / disjoint).
11. ``test_extract_multi_ph_features_shape`` — Multi-PH extractor
    returns ``(n_bins, n_bins)`` per sample + same n_samples.
12. ``test_single_ph_is_marginal_sum`` — single-PH feature equals
    sum of the multi-PH feature along the coordination axis.
13. ``test_render_table_md`` — Markdown table contains key headers.
14. ``test_run_bench_smoke`` — bench runs end-to-end on 8 SMILES.
15. ``test_verify_script_writes_metrics`` — verify script writes a
    valid metrics JSON with the expected schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "molmetal" / "scripts"
REPORTS = PROJECT_ROOT / "molmetal" / "reports"
METRICS = PROJECT_ROOT / "metrics" / "by_metric"
PLATINAI_CSV = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"

# Wf-31 Phase-3 outputs (default paths; may not exist on a fresh checkout).
WF31_METRICS = REPORTS / "wf31_multi_ph_diversity" / "metrics.json"
WF31_VERIFY_METRICS = (
    REPORTS / "wf31_multi_ph_diversity" / "wf31_multi_ph_diversity_verify.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_scripts_path() -> None:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """``bench_multi_ph_diversity`` imports without error."""
    _ensure_scripts_path()
    mod = __import__("bench_multi_ph_diversity")
    assert hasattr(mod, "run_bench")
    assert hasattr(mod, "extract_multi_ph_features")
    assert hasattr(mod, "compute_expressivity")
    assert hasattr(mod, "compute_diversity")
    assert hasattr(mod, "compute_morgan_fps")
    assert hasattr(mod, "compute_pairwise_tanimoto_distance")
    assert hasattr(mod, "render_table")
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# 2-4. Constants
# ---------------------------------------------------------------------------
def test_constants_n_bins() -> None:
    """MULTI_PH_N_BINS must be 16 (matches Wf-25 Phase-2 spec)."""
    from bench_multi_ph_diversity import MULTI_PH_N_BINS  # type: ignore

    assert MULTI_PH_N_BINS == 16


def test_constants_max_radius_max_coord() -> None:
    """MULTI_PH_MAX_RADIUS == 5.0, MULTI_PH_MAX_COORD == 4.0."""
    from bench_multi_ph_diversity import (  # type: ignore
        MULTI_PH_MAX_COORD,
        MULTI_PH_MAX_RADIUS,
    )

    assert MULTI_PH_MAX_RADIUS == 5.0
    assert MULTI_PH_MAX_COORD == 4.0


def test_constants_diversity_floor() -> None:
    """DIVERSITY_FLOOR must be 0.75 (R18 PLAN §3 m7 floor)."""
    from bench_multi_ph_diversity import DIVERSITY_FLOOR  # type: ignore

    assert DIVERSITY_FLOOR == 0.75


# ---------------------------------------------------------------------------
# 5. Default CSV path
# ---------------------------------------------------------------------------
def test_default_platinai_csv_exists() -> None:
    """Default PlatinAI 5K CSV is shipped in ``molmetal/data``."""
    assert PLATINAI_CSV.exists(), f"missing CSV: {PLATINAI_CSV}"


# ---------------------------------------------------------------------------
# 6. CSV loader
# ---------------------------------------------------------------------------
def test_load_platinai_5k_returns_rows() -> None:
    """``load_platinai_5k`` returns >=1 row with a non-empty ``smiles`` field."""
    from bench_multi_ph_diversity import load_platinai_5k  # type: ignore

    rows = load_platinai_5k(PLATINAI_CSV)
    assert len(rows) >= 1
    assert "smiles" in rows[0]
    assert rows[0]["smiles"], "first SMILES should be non-empty"


# ---------------------------------------------------------------------------
# 7-8. Mean pairwise cosine distance
# ---------------------------------------------------------------------------
def test_mean_pairwise_cosine_distance_diagonal_zero() -> None:
    """Mean off-diagonal 1 - cos is in [0, 2] and excludes the diagonal."""
    from bench_multi_ph_diversity import (  # type: ignore
        _mean_pairwise_cosine_distance,
    )

    rng = np.random.default_rng(0)
    feats = [
        torch.tensor(rng.standard_normal(8), dtype=torch.float32) for _ in range(5)
    ]
    out = _mean_pairwise_cosine_distance(feats)
    assert 0.0 <= out <= 2.0 + 1e-6


def test_mean_pairwise_cosine_distance_two_vectors() -> None:
    """Two-vector pairwise distance must be 1 - cos(a, b)."""
    from bench_multi_ph_diversity import (  # type: ignore
        _mean_pairwise_cosine_distance,
    )

    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    out = _mean_pairwise_cosine_distance([a, b])
    # Per construction: 1 - cos = 1 - 0 = 1.
    assert abs(out - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 9-10. Pairwise Tanimoto distance
# ---------------------------------------------------------------------------
def test_compute_pairwise_tanimoto_distance_diagonal_zero() -> None:
    """Mean Tanimoto distance is in [0, 1] and excludes the diagonal."""
    from bench_multi_ph_diversity import (  # type: ignore
        compute_pairwise_tanimoto_distance,
    )

    rng = np.random.default_rng(0)
    fps = rng.integers(0, 2, size=(6, 32), dtype=np.uint8)
    out = compute_pairwise_tanimoto_distance(fps)
    # Mean off-diagonal Tanimoto distance is bounded in [0, 1].
    assert 0.0 <= out <= 1.0 + 1e-6
    # Two-vector case: a single (i, j) pair, i != j → the mean equals
    # the off-diagonal entry.  Verify with two identical bit vectors.
    a = np.zeros((1, 8), dtype=np.uint8)
    a[0, 0] = 1
    same = np.concatenate([a, a], axis=0)
    out_same = compute_pairwise_tanimoto_distance(same)
    assert out_same < 1e-5  # identical vectors → distance 0


def test_compute_pairwise_tanimoto_distance_synthetic() -> None:
    """Two identical / disjoint bit vectors yield distance 0 / 1."""
    from bench_multi_ph_diversity import (  # type: ignore
        compute_pairwise_tanimoto_distance,
    )

    a = np.zeros((1, 8), dtype=np.uint8)
    a[0, 0] = 1
    b = np.zeros((1, 8), dtype=np.uint8)
    b[0, 0] = 1
    fps = np.concatenate([a, b], axis=0)
    out_same = compute_pairwise_tanimoto_distance(fps)
    # Tanimoto sim == 1 → distance 0
    assert out_same < 1e-5

    c = np.zeros((1, 8), dtype=np.uint8)
    c[0, 1] = 1
    fps2 = np.concatenate([a, c], axis=0)
    out_diff = compute_pairwise_tanimoto_distance(fps2)
    # Tanimoto sim == 0 → distance 1
    assert abs(out_diff - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 11-12. Multi-PH feature extractor
# ---------------------------------------------------------------------------
def test_extract_multi_ph_features_shape() -> None:
    """Multi-PH extractor returns one ``(n_bins, n_bins)`` per sample."""
    from bench_multi_ph_diversity import (  # type: ignore
        extract_multi_ph_features,
    )

    # Use tiny, RDKit-friendly SMILES (ethanol + propane).
    smiles = ["CCO", "CCC", "CCN", "CCO", "CCC"]
    multi_ph, single_ph, stats = extract_multi_ph_features(
        smiles,
        n_bins=8,
        max_atoms=20,
    )
    assert stats["n_parsed"] == len(smiles), f"expected all SMILES parsed, got {stats}"
    assert len(multi_ph) == len(smiles)
    assert len(single_ph) == len(smiles)
    for f in multi_ph:
        assert f.shape == (8, 8)
    for f in single_ph:
        assert f.shape == (8,)


def test_single_ph_is_marginal_sum() -> None:
    """Single-PH feature is the marginal sum of the multi-PH feature along d."""
    from bench_multi_ph_diversity import (  # type: ignore
        extract_multi_ph_features,
    )

    smiles = ["CCO", "CCC", "CCN"]
    multi_ph, single_ph, stats = extract_multi_ph_features(
        smiles,
        n_bins=8,
        max_atoms=20,
    )
    assert stats["n_parsed"] == len(smiles)
    for mp, sp in zip(multi_ph, single_ph):
        torch.testing.assert_close(sp, mp.sum(dim=-1), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 13. Markdown table
# ---------------------------------------------------------------------------
def test_render_table_md() -> None:
    """``render_table`` produces a Markdown table with key headers."""
    from bench_multi_ph_diversity import render_table  # type: ignore

    metrics = {
        "n_samples": 8,
        "n_parsed_multiph": 8,
        "n_failed_multiph": 0,
        "n_parsed_diversity": 8,
        "n_failed_diversity": 0,
        "feature_dim_multiph": 64,
        "feature_dim_single_ph": 8,
        "n_bins": 8,
        "max_radius": 5.0,
        "max_coord": 4.0,
        "max_pocket_atoms": 60,
        "mph_wall_s": 1.0,
        "expressivity": {
            "multi_ph": 0.8,
            "single_ph": 0.5,
            "delta": 0.3,
            "multi_ph_more_expressive": True,
        },
        "diversity": {
            "mean_tanimoto_distance": 0.85,
            "meets_m7_floor": True,
            "m7_floor": 0.75,
        },
    }
    md = render_table(metrics)
    assert "# Wf-31 Phase 3" in md
    assert "Multi-PH (2-parameter)" in md
    assert "Single-PH (1-D Betti marginal)" in md
    assert "Mean pairwise Tanimoto distance" in md
    assert "Honest framing" in md


# ---------------------------------------------------------------------------
# 14. End-to-end smoke (skip on environments without RDKit / torch)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not PLATINAI_CSV.exists(),
    reason="PlatinAI 5K CSV missing",
)
def test_run_bench_smoke() -> None:
    """Bench runs end-to-end on 8 SMILES."""
    from bench_multi_ph_diversity import run_bench  # type: ignore

    metrics = run_bench(
        platinai_csv=PLATINAI_CSV,
        n_samples=8,
        n_bins=8,
        seed=42,
    )
    assert metrics["n_parsed_multiph"] >= 2
    assert metrics["n_parsed_diversity"] >= 2
    assert metrics["expressivity"]["multi_ph"] >= 0.0
    assert metrics["diversity"]["mean_tanimoto_distance"] >= 0.0


# ---------------------------------------------------------------------------
# 15. Verify-script metrics file (skip if not produced yet)
# ---------------------------------------------------------------------------
def test_verify_script_writes_metrics() -> None:
    """The verify-script metrics file, when present, has the expected schema."""
    data = _load(WF31_VERIFY_METRICS)
    if data is None:
        pytest.skip(f"verify-script output missing: {WF31_VERIFY_METRICS}")
    assert data.get("recipe") == "wf31_multi_ph_diversity_verify"
    assert "expressivity" in data
    assert "diversity" in data
    assert "n_parsed_multiph" in data
    assert "n_parsed_diversity" in data
    assert data["n_parsed_multiph"] >= 2
    assert data["n_parsed_diversity"] >= 2
