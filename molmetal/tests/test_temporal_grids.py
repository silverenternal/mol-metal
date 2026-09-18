"""Unit tests for ``molmetal.data.temporal_grids``.

These tests intentionally build a *small synthetic dataframe* (rather
than reading the full MetalCytoToxDB.csv) so they run in milliseconds
and don't depend on the data-root path.

Run with::

    source .venv/bin/activate
    pytest molmetal/tests/test_temporal_grids.py -v
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from molmetal.data.temporal_grids import (
    TEMPORAL_GRIDS,
    hit_rate_by_year,
    resolve_grid,
    rolling_split,
    temporal_split_at_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synth_df() -> pd.DataFrame:
    """Synthetic dataframe with deterministic years + active labels.

    Counts:
      2000-2023 → 1200 rows  (mix of active / inactive)
      2024      → 150 rows
      2025      → 80 rows
      2026      → 20 rows
      Total     → 1450 rows

    Active fractions:
      <2024 : 80 %   (240 / 300 sample, plus deterministic rule below)
      2024  : 50 %
      2025  : 30 %
      2026  : 20 %
    """
    rows: list = []
    rng = np.random.default_rng(0)

    # 2000-2023 — most rows active (80%)
    for y in range(2000, 2024):
        n_year = 50
        active = rng.choice([True, False], size=n_year, p=[0.8, 0.2])
        for a in active:
            rows.append({"Year": y, "active": bool(a), "smiles": "C"})
    # 2024 — 150 rows, 50 % active
    active = rng.choice([True, False], size=150, p=[0.5, 0.5])
    for a in active:
        rows.append({"Year": 2024, "active": bool(a), "smiles": "C"})
    # 2025 — 80 rows, 30 % active
    active = rng.choice([True, False], size=80, p=[0.3, 0.7])
    for a in active:
        rows.append({"Year": 2025, "active": bool(a), "smiles": "C"})
    # 2026 — 20 rows, 20 % active
    active = rng.choice([True, False], size=20, p=[0.2, 0.8])
    for a in active:
        rows.append({"Year": 2026, "active": bool(a), "smiles": "C"})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. test_threshold_2024_correct_n
# ---------------------------------------------------------------------------
def test_threshold_2024_correct_n(synth_df: pd.DataFrame) -> None:
    """Pre-2024 vs ≥2024 split must yield correct row counts.

    Setup:
      2000-2023 : 50 × 24 = 1200 rows
      2024      : 150 rows
      2025      : 80 rows
      2026      : 20 rows
      Total     : 1450 rows

    With ``train_ratio=1.0`` (no trimming):
      n_train = 1200
      n_test  = 150 + 80 + 20 = 250
    """
    train_df, test_df, stats = temporal_split_at_threshold(
        synth_df, threshold=2024, train_ratio=1.0
    )
    assert len(train_df) == 1200, f"expected 1200 pre-2024 rows, got {len(train_df)}"
    assert len(test_df) == 250, f"expected 250 >=2024 rows, got {len(test_df)}"
    assert stats["n_train"] == 1200
    assert stats["n_test"] == 250
    # Sanity: every pre-2024 row should actually be < 2024
    assert (train_df["Year"] < 2024).all()
    # Every >=2024 row should be ≥ 2024
    assert (test_df["Year"] >= 2024).all()


# ---------------------------------------------------------------------------
# 2. test_rolling_split_consistent
# ---------------------------------------------------------------------------
def test_rolling_split_consistent(synth_df: pd.DataFrame) -> None:
    """Rolling + static split row-counts must sum to N.

    For any cutoff year ``y``, the static split gives
    ``n_train(y) + n_test(y) = N - dropped_pre``, where ``dropped_pre``
    is the number of pre-cutoff rows dropped when ``train_ratio<1.0``.

    With ``train_ratio=1.0`` the rolling sum must equal the total for
    each cutoff.
    """
    rolling = rolling_split(
        synth_df, cutoff_years=(2018, 2020, 2022, 2024), train_ratio=1.0
    )
    for y, (tr, te, stats) in rolling.items():
        n_train = stats["n_train"]
        n_test = stats["n_test"]
        n_total = n_train + n_test
        # All rows should be assigned to exactly one of train/test
        assert n_total == len(synth_df), (
            f"cutoff={y}: train({n_train}) + test({n_test}) = {n_total} != "
            f"{len(synth_df)}"
        )
        # No overlap (train < cutoff, test >= cutoff)
        assert (tr["Year"] < y).all()
        assert (te["Year"] >= y).all()
        # Order: cutoffs farther in the past have more train rows
        # (since more rows qualify as < cutoff).


# ---------------------------------------------------------------------------
# 3. test_stats_have_hit_rate
# ---------------------------------------------------------------------------
def test_stats_have_hit_rate(synth_df: pd.DataFrame) -> None:
    """``stats['hit_rate_test']`` must be a float in [0, 1].

    The full pipeline is exercised for the canonical 2024 cutoff:
      * train_df, test_df are non-empty
      * hit_rate_train ≈ 0.8 (80 % active pre-2024)
      * hit_rate_test  ∈ (0, 1) and should be markedly lower than
        hit_rate_train (since 2024-2026 rows are only 20-50 % active)
    """
    train_df, test_df, stats = temporal_split_at_threshold(
        synth_df, threshold=2024, train_ratio=1.0
    )
    assert "hit_rate_train" in stats
    assert "hit_rate_test" in stats
    hr_t = stats["hit_rate_test"]
    hr_tr = stats["hit_rate_train"]
    assert isinstance(hr_t, float)
    assert isinstance(hr_tr, float)
    assert 0.0 <= hr_t <= 1.0, f"hit_rate_test={hr_t} not in [0,1]"
    assert 0.0 <= hr_tr <= 1.0, f"hit_rate_train={hr_tr} not in [0,1]"
    # Mean years should bracket the cutoff
    assert stats["mean_year_train"] < 2024
    assert stats["mean_year_test"] >= 2024
    # Hit-rate decay is the whole point of A3
    assert hr_t < hr_tr, (
        f"hit_rate_test ({hr_t:.3f}) should be < hit_rate_train "
        f"({hr_tr:.3f}) for our synthetic dataset"
    )


# ---------------------------------------------------------------------------
# 4. (bonus) test_grid_registry_consistent — registry sanity
# ---------------------------------------------------------------------------
def test_grid_registry_consistent() -> None:
    """Every non-rolling grid entry should resolve to ``(int, bool)``."""
    for name, val in TEMPORAL_GRIDS.items():
        if val == "rolling":
            assert "rolling" in name
            continue
        threshold, include = resolve_grid(name)
        assert isinstance(threshold, int)
        assert isinstance(include, bool)
        assert 2010 <= threshold <= 2026, f"{name}: unreasonable threshold {threshold}"


# ---------------------------------------------------------------------------
# 5. (bonus) test_hit_rate_by_year_returns_dict
# ---------------------------------------------------------------------------
def test_hit_rate_by_year_returns_dict(synth_df: pd.DataFrame) -> None:
    """``hit_rate_by_year`` must return a sorted dict with values in [0, 1]."""
    out = hit_rate_by_year(synth_df)
    assert isinstance(out, dict)
    # Keys sorted ascending
    keys = list(out.keys())
    assert keys == sorted(keys)
    for y, hr in out.items():
        assert isinstance(y, int)
        assert 0.0 <= hr <= 1.0
    # 2000 should be present (first year)
    assert min(keys) == 2000
    # 2026 should be present (last year)
    assert max(keys) == 2026