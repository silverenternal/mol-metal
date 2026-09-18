"""Temporal-grid utilities — multi-granularity temporal splits.

This module powers open-question **A3** in
``TODO/07_risks/open_questions.md``: *how rapidly does hit-rate decay over
time?*  The honest OOD temporal evaluation of a cytotoxicity model on
MetalCytoToxDB depends entirely on the choice of cutoff year.  We expose
a small registry of canonical grids and a single helper function that
returns ``(train_df, test_df, stats)`` for any year threshold.

Grids
-----
* ``'pre_2020_vs_2020+'``  — anything < 2020 in train, ≥ 2020 in test.
* ``'pre_2022_vs_2022+'``  — anything < 2022 in train, ≥ 2022 in test.
* ``'pre_2024_vs_2024+'``  — anything < 2024 in train, ≥ 2024 in test.
* ``'pre_2024_vs_2025+'``  — anything < 2024 in train, ≥ 2025 in test.
* ``'rolling_2018_2020_2022_2024'`` — produces multiple sub-splits
  rolling one year at a time.  See :func:`rolling_split` for the per-year
  breakdown.

Function
--------
* :func:`temporal_split_at_threshold` — split a dataframe at ``threshold``
  and return ``(train_df, test_df, stats)`` where ``stats`` contains
  ``n_train``, ``n_test``, ``mean_year_train``, ``mean_year_test``,
  ``hit_rate_train``, ``hit_rate_test`` (hit rate := fraction of rows
  with ``active == True``).

The ``hit_rate`` field is the empirical fraction of *active* ligands
(IC50_Dark_value < 10 µM) in each split.  A large drop in test hit rate
relative to train is the canonical signature of **publication-bias
induced data drift** and is the reason A3 exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registry of canonical grids
# ---------------------------------------------------------------------------
# Each non-rolling value is ``(threshold_year, include_threshold_in_test)``.
# ``include_threshold_in_test=True`` means rows whose Year == threshold
# belong to the *test* split (i.e. ``>= threshold``).
TEMPORAL_GRIDS: Dict[str, object] = {
    "pre_2020_vs_2020+": (2020, False),  # split at 2020 (test = Year >= 2020)
    "pre_2022_vs_2022+": (2022, False),
    "pre_2024_vs_2024+": (2024, False),
    "pre_2024_vs_2025+": (2025, False),
    "rolling_2018_2020_2022_2024": "rolling",
}


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalSplitStats:
    """Summary statistics for a temporal split."""

    n_train: int
    n_test: int
    mean_year_train: float
    mean_year_test: float
    hit_rate_train: float
    hit_rate_test: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "mean_year_train": float(self.mean_year_train),
            "mean_year_test": float(self.mean_year_test),
            "hit_rate_train": float(self.hit_rate_train),
            "hit_rate_test": float(self.hit_rate_test),
        }


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------
def temporal_split_at_threshold(
    df: pd.DataFrame,
    year_col: str = "Year",
    threshold: int = 2024,
    train_ratio: float = 0.8,
    active_col: str = "active",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Split ``df`` at a single year threshold.

    Rows whose ``year_col`` is strictly less than ``threshold`` go to
    *train*; rows at or after ``threshold`` go to *test*.  If
    ``train_ratio`` is < 1.0 and there are more than ``train_ratio * N``
    pre-threshold rows, only the **most-recent** ``train_ratio * N``
    pre-threshold rows are kept (the older rows are dropped to simulate
    the model's actual training window — older ligands are usually
    out-of-fashion).

    Returns
    -------
    (train_df, test_df, stats) : :class:`tuple`
        ``stats`` is a dict with keys ``n_train``, ``n_test``,
        ``mean_year_train``, ``mean_year_test``, ``hit_rate_train``,
        ``hit_rate_test``.  Hit rate is the fraction of rows with
        ``active_col == True``.
    """
    if year_col not in df.columns:
        raise KeyError(f"year_col={year_col!r} not in dataframe (have {list(df.columns)})")
    years = df[year_col].to_numpy()
    # Boolean masks
    pre_mask = years < threshold
    post_mask = ~pre_mask
    pre_df = df.loc[pre_mask].copy()
    post_df = df.loc[post_mask].copy()
    # Trim pre_df to train_ratio * N if requested
    if train_ratio < 1.0 and len(pre_df) > 0:
        target = int(round(train_ratio * len(pre_df)))
        if target < len(pre_df):
            # Keep the *most-recent* rows (highest year)
            pre_df = pre_df.sort_values(year_col, ascending=False).head(target).copy()
    stats = _compute_stats(pre_df, post_df, active_col=active_col)
    return pre_df, post_df, stats


# ---------------------------------------------------------------------------
# Rolling-window grid
# ---------------------------------------------------------------------------
def rolling_split(
    df: pd.DataFrame,
    year_col: str = "Year",
    cutoff_years: Tuple[int, ...] = (2018, 2020, 2022, 2024),
    train_ratio: float = 0.8,
    active_col: str = "active",
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]]:
    """Apply :func:`temporal_split_at_threshold` at multiple cutoffs.

    Returns ``{cutoff_year: (train_df, test_df, stats)}``.
    """
    out: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]] = {}
    for y in cutoff_years:
        out[int(y)] = temporal_split_at_threshold(
            df,
            year_col=year_col,
            threshold=int(y),
            train_ratio=train_ratio,
            active_col=active_col,
        )
    return out


# ---------------------------------------------------------------------------
# Year-bucket hit-rate computation (for the ASCII plot)
# ---------------------------------------------------------------------------
def hit_rate_by_year(
    df: pd.DataFrame,
    year_col: str = "Year",
    active_col: str = "active",
) -> Dict[int, float]:
    """Per-year active fraction.  Useful for the text-based plot."""
    if year_col not in df.columns or active_col not in df.columns:
        return {}
    out: Dict[int, float] = {}
    for y, sub in df.groupby(year_col):
        n = len(sub)
        if n == 0:
            continue
        out[int(y)] = float(sub[active_col].astype(bool).mean())
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    active_col: str = "active",
) -> Dict[str, float]:
    """Compute :class:`TemporalSplitStats` for the given splits."""
    n_train = int(len(train_df))
    n_test = int(len(test_df))
    mean_year_train = float(np.mean(train_df["Year"].to_numpy())) if n_train > 0 else 0.0
    mean_year_test = float(np.mean(test_df["Year"].to_numpy())) if n_test > 0 else 0.0
    if active_col in train_df.columns and n_train > 0:
        hit_rate_train = float(train_df[active_col].astype(bool).mean())
    else:
        hit_rate_train = 0.0
    if active_col in test_df.columns and n_test > 0:
        hit_rate_test = float(test_df[active_col].astype(bool).mean())
    else:
        hit_rate_test = 0.0
    return {
        "n_train": n_train,
        "n_test": n_test,
        "mean_year_train": mean_year_train,
        "mean_year_test": mean_year_test,
        "hit_rate_train": hit_rate_train,
        "hit_rate_test": hit_rate_test,
    }


# ---------------------------------------------------------------------------
# Convenience: parse a grid name into a callable
# ---------------------------------------------------------------------------
def resolve_grid(
    name: str,
) -> Tuple[int, bool]:
    """Resolve a non-rolling grid name to ``(threshold_year, include_threshold_in_test)``.

    Raises :class:`KeyError` for unknown names or the special ``'rolling_...'`` value.
    """
    if name not in TEMPORAL_GRIDS:
        raise KeyError(
            f"Unknown grid {name!r}.  Known grids: {sorted(TEMPORAL_GRIDS)}"
        )
    val = TEMPORAL_GRIDS[name]
    if val == "rolling":
        raise ValueError(
            f"Grid {name!r} is a rolling grid — use :func:`rolling_split` instead."
        )
    threshold, include_in_test = val  # type: ignore[misc]
    return int(threshold), bool(include_in_test)


__all__ = [
    "TEMPORAL_GRIDS",
    "TemporalSplitStats",
    "temporal_split_at_threshold",
    "rolling_split",
    "hit_rate_by_year",
    "resolve_grid",
]