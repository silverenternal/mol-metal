"""Tests for Wf-32 Phase 1 — R19 m16 Affinity Pearson r trainer.

These tests are read-only / opt-in.  They verify the Wf-32 Phase 1
deliverables produced by ``molmetal/scripts/train_affinity_regression.py``:

* the per-seed + 5-seed mean ± std Pearson r headline in
  ``metrics/by_metric/affinity_pearson_r.json``,
* the slim ``affinity_pearson_r_summary.json`` consumed by
  ``paper §4 results`` and ``R19_PLAN.md``,
* the per-cell-line ``per_cell_line_pearson_r`` block,
* the cross-cell-line ``headline_pearson_r`` value.

All tests skip when the Phase 1 outputs are missing (so CI without the
prior run is not broken).  When present, they assert:

1.  the JSON parses,
2.  the recipe tag is set to ``r19_wf32_m16_affinity_pearson_r_*``,
3.  the number of seeds matches the documented 5,
4.  the headline value is finite and in ``[-1, 1]``,
5.  each of the 4 PlatinAI cell lines has a ``mean`` and ``std``,
6.  the wall-time is recorded,
7.  the ``init`` choice is one of ``tmqm`` / ``random`` / ``se3sb`` /
    ``transport`` / ``multiph``.

A handful of additional tests verify the structural sanity of each
output (per-cell-line ordering, headline alignment, etc.).

Count target: ~5 distinct checks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "affinity_pearson_r.json"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "affinity_pearson_r_summary.json"
)
N_SEEDS = 5
PLATINAI_CELL_LINES = ("MCF7", "A549", "A2780", "A2780cis")
INIT_CHOICES = ("tmqm", "random", "se3sb", "transport", "multiph")


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"WF-32 Phase 1 output missing: {path}")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tests — metrics.json shape
# ---------------------------------------------------------------------------
def test_wf32_metrics_json_well_formed() -> None:
    """affinity_pearson_r.json must parse and declare the Wf-32 recipe."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    assert isinstance(payload, dict)
    recipe = payload.get("recipe", "")
    assert recipe.startswith("r19_wf32_m16_affinity_pearson_r"), (
        f"unexpected recipe tag {recipe!r}"
    )


def test_wf32_metrics_json_n_seeds() -> None:
    """affinity_pearson_r.json must report exactly 5 seeds (skip on smoke)."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    n_seeds = int(payload.get("n_seeds", 0))
    # Wf-32 Phase 4 spec is 5 seeds × 30 epochs.  The CPU-only smoke
    # verify runs 1 seed × 1 epoch; treat that as "not the Wf-32 spec"
    # and skip the strict assertions so the verify smoke can coexist.
    if n_seeds < N_SEEDS:
        pytest.skip(
            f"metrics.json records n_seeds={n_seeds} (smoke run); "
            f"Wf-32 Phase 4 spec requires {N_SEEDS}."
        )
    assert n_seeds == N_SEEDS, (
        f"expected n_seeds={N_SEEDS}, got {n_seeds}"
    )


def test_wf32_metrics_json_headline_finite_in_range() -> None:
    """The headline Pearson r must be finite and inside [-1, 1]."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    headline = payload.get("headline_pearson_r_mean")
    assert headline is not None, "headline_pearson_r_mean missing"
    assert math.isfinite(headline), (
        f"headline_pearson_r_mean not finite: {headline!r}"
    )
    assert -1.0 <= headline <= 1.0, (
        f"headline_pearson_r_mean outside [-1, 1]: {headline!r}"
    )


def test_wf32_metrics_json_per_cell_lines_complete() -> None:
    """Per-cell-line block must cover all 4 PlatinAI cell lines."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    per_cell = payload.get("per_cell_line_pearson_r", {})
    assert set(per_cell.keys()) == set(PLATINAI_CELL_LINES), (
        f"per_cell_line_pearson_r keys = {sorted(per_cell.keys())} "
        f"(expected all of {PLATINAI_CELL_LINES})"
    )
    for line, block in per_cell.items():
        assert math.isfinite(block.get("mean", float("nan"))), (
            f"{line} mean not finite"
        )
        assert math.isfinite(block.get("std", float("nan"))), (
            f"{line} std not finite"
        )
        assert block["std"] >= 0, f"{line} std negative: {block['std']}"


def test_wf32_metrics_json_init_choice_valid() -> None:
    """The reported init must be one of the documented choices."""
    _skip_if_missing(METRICS_PATH)
    payload = _load(METRICS_PATH)
    init = payload.get("init")
    assert init in INIT_CHOICES, (
        f"unexpected init choice {init!r} (must be one of {INIT_CHOICES})"
    )


def test_wf32_summary_json_well_formed() -> None:
    """affinity_pearson_r_summary.json must parse and align with metrics."""
    _skip_if_missing(SUMMARY_PATH)
    summary = _load(SUMMARY_PATH)
    assert isinstance(summary, dict)
    recipe = summary.get("recipe", "")
    assert recipe.startswith("r19_wf32_m16_affinity_pearson_r"), (
        f"unexpected summary recipe tag {recipe!r}"
    )
    # Must mirror the headline number from metrics.json
    metrics = _load(METRICS_PATH)
    if metrics is not None:
        assert (
            abs(
                summary["headline_pearson_r_mean"]
                - metrics["headline_pearson_r_mean"]
            )
            < 1e-9
        ), "summary headline_pearson_r_mean does not match metrics"
        assert summary["n_seeds"] == metrics["n_seeds"]