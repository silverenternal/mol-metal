"""Tests for molmetal.replay.curriculum.drift.

================================================================
Drift guard contract tests (TODO-150 §P1.7)
================================================================
Covers the per-sample retirement decision based on the divergence
threshold.

* :class:`DriftReport`  — 2 tests (to_dict shape, retire_fraction).
* :class:`DriftGuard`   — 5 tests (threshold validation, evaluate,
  is_retired, retire_fraction reporting, kept / retired separation).

Total: **7 tests**, CPU-only.  All under 1 s on any CI container.
"""

from __future__ import annotations

import pytest

try:
    from molmetal.replay.curriculum.drift import (
        DRIFT_DIVERGENCE_THRESHOLD,
        DriftGuard,
        DriftReport,
    )
except Exception:
    from molmetal.replay.curriculum import (  # type: ignore
        DRIFT_DIVERGENCE_THRESHOLD,
        DriftGuard,
        DriftReport,
    )


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------


class TestDriftReport:
    def test_to_dict_shape(self):
        r = DriftReport(
            metal="Rh",
            epoch=0,
            n_total=10,
            n_retired=3,
            n_kept=7,
            threshold=0.5,
            retired_row_ids=[0, 1, 2],
            kept_row_ids=list(range(3, 10)),
        )
        d = r.to_dict()
        assert d["metal"] == "Rh"
        assert d["n_total"] == 10
        assert d["n_retired"] == 3
        assert d["retired_row_ids"] == [0, 1, 2]
        assert d["retire_fraction"] == pytest.approx(0.3)

    def test_empty(self):
        r = DriftReport(metal="Ru", epoch=0, n_total=0, n_retired=0, n_kept=0)
        d = r.to_dict()
        assert d["retire_fraction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DriftGuard
# ---------------------------------------------------------------------------


class TestDriftGuard:
    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            DriftGuard(threshold=-0.1)

    def test_evaluate_basic(self):
        g = DriftGuard(threshold=0.5)
        samples = [
            {"row_id": 0, "predicted": 0.5, "on_policy": 0.5},  # keep
            {"row_id": 1, "predicted": 0.5, "on_policy": 1.5},  # retire (|diff|=1.0)
            {"row_id": 2, "predicted": 0.5, "on_policy": 0.9},  # keep (|diff|=0.4)
            {"row_id": 3, "predicted": 0.5, "on_policy": 0.0},  # keep (|diff|=0.5)
            {"row_id": 4, "predicted": 0.5, "on_policy": -0.1},  # retire (|diff|=0.6)
        ]
        r = g.evaluate("Rh", epoch=7, samples=samples)
        assert r.n_total == 5
        assert r.n_retired == 2
        assert r.n_kept == 3
        assert sorted(r.retired_row_ids) == [1, 4]
        assert sorted(r.kept_row_ids) == [0, 2, 3]
        assert r.epoch == 7
        assert r.metal == "Rh"

    def test_is_retired(self):
        g = DriftGuard(threshold=0.5)
        # Exactly at the boundary — NOT retired.
        assert not g.is_retired(0.5, 1.0)
        assert not g.is_retired(1.0, 0.5)
        # Strictly beyond — retired.
        assert g.is_retired(0.5, 1.0001)
        assert g.is_retired(1.0, 0.4999)

    def test_retire_fraction_reporting(self):
        g = DriftGuard(threshold=0.1)
        # Half the samples diverge.
        samples = [
            {
                "row_id": i,
                "predicted": 0.5,
                "on_policy": 0.5 + (1.0 if i % 2 == 0 else 0.0),
            }
            for i in range(10)
        ]
        r = g.evaluate("Ru", epoch=0, samples=samples)
        assert r.retire_fraction == pytest.approx(0.5)

    def test_empty_samples(self):
        g = DriftGuard()
        r = g.evaluate("Ru", epoch=0, samples=[])
        assert r.n_total == 0
        assert r.n_retired == 0
        assert r.n_kept == 0
        assert r.retire_fraction == pytest.approx(0.0)

    def test_threshold_constant_matches_default(self):
        g = DriftGuard()
        assert g.threshold == DRIFT_DIVERGENCE_THRESHOLD
