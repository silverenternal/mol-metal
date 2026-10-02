"""Tests for WF-30 Phase 3 — 10-pocket × 5-candidate × 5-seed docking sweep.

These tests are read-only / opt-in. They verify the integrity of the
WF-30 Phase-3 docking outputs that ``molmetal/scripts/dock_10pockets_se3sb.py``
+ ``molmetal/scripts/wf30_dock_summarize.py`` produce:

* the 250-evaluation ``se3sb/metrics.json`` and
  ``baseline/metrics.json`` files,
* the merged ``per_pocket_metrics.json`` (per pocket × per method × per seed
  Vina kcal/mol + summary stats),
* the ``comparison.md`` (side-by-side se3sb vs baseline per pocket),
* the ``best_m3_summary.json`` (the headline m3 Vina Dock median across the
  10 pockets).

The tests cover both the **stub-scorer** path (default) and the **real-Vina**
path (which only works on metal-free candidates — the demo metal-drug cohort
fails PDBQT conversion in meeko). Tests skip with ``pytest.skip`` when the
Phase-3 outputs are missing (so CI without the prior sweep is not broken).

The smoke test count targets ~10 distinct checks:

1. ``test_se3sb_metrics_exists`` — se3sb/metrics.json is non-empty.
2. ``test_baseline_metrics_exists`` — baseline/metrics.json is non-empty.
3. ``test_se3sb_metrics_n_evaluations`` — 250 evaluations recorded.
4. ``test_baseline_metrics_n_evaluations`` — 250 evaluations recorded.
5. ``test_per_pocket_metrics_exists`` — per_pocket_metrics.json written.
6. ``test_per_pocket_metrics_covers_10_pockets`` — all 10 pockets present.
7. ``test_per_pocket_metrics_summary_per_pocket_per_method`` — both
   ``se3sb`` and ``baseline`` aggregates present for each pocket.
8. ``test_comparison_md_exists`` — comparison.md exists.
9. ``test_comparison_md_has_10_pocket_rows`` — markdown has 10 data rows.
10. ``test_best_m3_summary_json_well_formed`` — best_m3_summary.json has the
    expected keys, per-method m3 medians are finite, and the se3sb vs baseline
    delta is recorded.

A handful of additional tests verify the structural sanity of each output
(``aggregate`` keys, ``scorer`` field, ``pockets`` ordering, …).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCK_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_dock"

# Mirrors the canonical list in molmetal/scripts/check_10_pockets.py +
# molmetal/scripts/dock_10pockets_se3sb.py.
WF30_POCKETS = (
    "830c",
    "1h36",
    "3eml",
    "3ny8",
    "4aaw",
    "4rlu",
    "1g9v",
    "2hnx",
    "4rn0",
    "1fmc",
)

SE3SB_METRICS = DOCK_DIR / "se3sb" / "metrics.json"
BASELINE_METRICS = DOCK_DIR / "baseline" / "metrics.json"
PER_POCKET = DOCK_DIR / "per_pocket_metrics.json"
COMPARISON_MD = DOCK_DIR / "comparison.md"
BEST_M3 = DOCK_DIR / "best_m3_summary.json"

N_EVALS = 250  # 10 pockets × 5 candidates × 5 seeds × 1 method
N_POCKETS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.fail(f"WF-30 Phase-3 output missing: {path}")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tests — sweep outputs
# ---------------------------------------------------------------------------
def test_se3sb_metrics_exists() -> None:
    """se3sb/metrics.json must be present and parseable."""
    _skip_if_missing(SE3SB_METRICS)
    payload = _load(SE3SB_METRICS)
    assert isinstance(payload, dict)
    assert payload.get("recipe") == "wf30_dock_10pockets_se3sb"


def test_baseline_metrics_exists() -> None:
    """baseline/metrics.json must be present and parseable."""
    _skip_if_missing(BASELINE_METRICS)
    payload = _load(BASELINE_METRICS)
    assert isinstance(payload, dict)
    assert payload.get("recipe") == "wf30_dock_10pockets_se3sb"


def test_se3sb_metrics_n_evaluations() -> None:
    """se3sb/metrics.json must record exactly 250 evaluations."""
    _skip_if_missing(SE3SB_METRICS)
    payload = _load(SE3SB_METRICS)
    assert payload["n_evaluations"] == N_EVALS
    assert len(payload["per_evaluation"]) == N_EVALS


def test_baseline_metrics_n_evaluations() -> None:
    """baseline/metrics.json must record exactly 250 evaluations."""
    _skip_if_missing(BASELINE_METRICS)
    payload = _load(BASELINE_METRICS)
    assert payload["n_evaluations"] == N_EVALS
    assert len(payload["per_evaluation"]) == N_EVALS


# ---------------------------------------------------------------------------
# Tests — per-pocket metrics
# ---------------------------------------------------------------------------
def test_per_pocket_metrics_exists() -> None:
    """per_pocket_metrics.json must be written by the summarizer."""
    _skip_if_missing(PER_POCKET)
    payload = _load(PER_POCKET)
    assert isinstance(payload, dict)
    assert payload.get("recipe") == "wf30_dock_per_pocket"


def test_per_pocket_metrics_covers_10_pockets() -> None:
    """per_pocket_metrics.json must include all 10 WF-30 pockets."""
    _skip_if_missing(PER_POCKET)
    payload = _load(PER_POCKET)
    assert set(payload["pockets"]) == set(WF30_POCKETS)
    assert len(payload["pockets"]) == N_POCKETS


def test_per_pocket_metrics_summary_per_pocket_per_method() -> None:
    """Both ``se3sb`` and ``baseline`` aggregates must be present per pocket."""
    _skip_if_missing(PER_POCKET)
    payload = _load(PER_POCKET)
    summary = payload.get("summary", {})
    for pocket_id in WF30_POCKETS:
        block = summary.get(pocket_id, {})
        for method in ("se3sb", "baseline"):
            assert method in block, (
                f"{pocket_id} missing method={method} in per-pocket summary"
            )
            stats = block[method]
            # Stub path always populates these; real-vina path may leave them
            # empty for the metal-drug cohort.  Accept either.
            if stats:
                for key in ("mean", "std", "min", "max", "median", "n"):
                    assert key in stats, (
                        f"{pocket_id} {method}: missing stat {key!r}"
                    )
                assert stats["n"] >= 1
                assert math.isfinite(stats["mean"])


# ---------------------------------------------------------------------------
# Tests — comparison.md
# ---------------------------------------------------------------------------
def test_comparison_md_exists() -> None:
    """comparison.md must exist and reference the se3sb / baseline methods."""
    _skip_if_missing(COMPARISON_MD)
    body = COMPARISON_MD.read_text()
    assert "se3sb" in body.lower()
    assert "baseline" in body.lower()
    assert "kcal/mol" in body


def test_comparison_md_has_10_pocket_rows() -> None:
    """comparison.md must have 10 pocket rows in its main table."""
    _skip_if_missing(COMPARISON_MD)
    body = COMPARISON_MD.read_text()
    rows = [line for line in body.splitlines() if line.startswith("|")]
    # Header + separator + 10 pocket rows + bonus rows = at least 12
    assert len(rows) >= N_POCKETS + 2, (
        f"expected >= {N_POCKETS + 2} table rows in comparison.md, got {len(rows)}"
    )
    for pocket_id in WF30_POCKETS:
        # Pocket IDs appear as the first cell of their row
        assert re.search(rf"\|\s*{pocket_id}\s*\|", body), (
            f"comparison.md missing row for {pocket_id}"
        )


# ---------------------------------------------------------------------------
# Tests — best_m3 summary
# ---------------------------------------------------------------------------
def test_best_m3_summary_json_well_formed() -> None:
    """best_m3_summary.json must record finite per-method m3 medians."""
    _skip_if_missing(BEST_M3)
    payload = _load(BEST_M3)
    assert payload.get("recipe") == "wf30_best_m3_summary"
    per_method = payload.get("per_method", {})
    assert "se3sb" in per_method
    assert "baseline" in per_method
    for method, info in per_method.items():
        assert info["n_pockets"] == N_POCKETS, (
            f"{method}: expected {N_POCKETS} pockets, got {info['n_pockets']}"
        )
        for key in ("median_kcal_mol", "mean_kcal_mol", "min_kcal_mol", "max_kcal_mol"):
            assert math.isfinite(info[key]), (
                f"{method}: non-finite {key}={info[key]!r}"
            )
        # Per-pocket best table must include all 10 pockets
        per_pocket_best = info.get("per_pocket_best", {})
        assert set(per_pocket_best.keys()) == set(WF30_POCKETS)
        for p, v in per_pocket_best.items():
            assert math.isfinite(v), f"{method} {p}: non-finite best kcal/mol"
    delta = payload.get("delta_se3sb_minus_baseline")
    assert delta is None or math.isfinite(float(delta))


def test_best_m3_summary_scorer_field_is_set() -> None:
    """best_m3_summary.json must declare which scorer produced the numbers.

    The default Phase-3 run uses the deterministic stub scorer; the optional
    real-Vina organic-cohort run writes a separate ``best_m3_summary.json``
    that says ``scorer=vina``. Either is acceptable as long as the field is
    present and is one of ``stub`` / ``vina``.
    """
    _skip_if_missing(BEST_M3)
    payload = _load(BEST_M3)
    scorer = payload.get("scorer")
    assert scorer in ("stub", "vina"), (
        f"unexpected scorer field {scorer!r} in best_m3_summary.json"
    )


def test_per_pocket_metrics_scorer_field_is_set() -> None:
    """per_pocket_metrics.json must declare the scorer too."""
    _skip_if_missing(PER_POCKET)
    payload = _load(PER_POCKET)
    scorer = payload.get("scorer")
    assert scorer in ("stub", "vina"), (
        f"unexpected scorer field {scorer!r} in per_pocket_metrics.json"
    )