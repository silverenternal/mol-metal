"""Tests for Wf-R20-A1 bench harness — degeneracy guard + aggregation.

Per AGENTS.md §5.7 trainer-flag wire-in smoke-required rule + the degeneracy
guard principle (the Wf-40 / Wf-R20-A failure mode this harness is built
to prevent): every wire-in must produce a non-degenerate baseline before
any other fix is claimed SHIPPED.

These tests validate:
  - _aggregate_metrics computes the expected fields from a TargetRecord list.
  - _check_degeneracy refuses to emit JSON when n_success == 0.
  - _check_degeneracy refuses to emit JSON when mean_depth_reached < 4.
  - _check_degeneracy permits emission when both conditions clear.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest

from molmetal.scripts.bench_mcts_search_quality import (  # noqa: E402
    TargetRecord,
    _aggregate_metrics,
    _check_degeneracy,
)


def _rec(*, n_cand: int, n_dist: int, depth: int, wall: float = 0.1, error: str | None = None) -> TargetRecord:
    return TargetRecord(
        target_idx=0,
        smiles_in="C",
        seed=0,
        wall_time_s=wall,
        n_candidates=n_cand,
        n_distinct=n_dist,
        max_depth_reached=depth,
        n_simulations_requested=200,
        n_simulations_executed=200,
        error=error,
    )


class TestDegeneracyGuard(unittest.TestCase):
    def test_n_success_zero_blocks_emission(self) -> None:
        """All-zero candidates ⇒ degenerate, refuse to emit JSON."""
        records = [
            _rec(n_cand=0, n_dist=0, depth=3, wall=0.2) for _ in range(5)
        ]
        agg = _aggregate_metrics(records, n_simulations=200)
        self.assertEqual(agg["n_success"], 0)
        self.assertEqual(agg["success_rate"], 0.0)
        msg = _check_degeneracy(agg)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertIn("n_success=0", msg)
        self.assertIn("refusing to emit JSON", msg)

    def test_low_depth_blocks_emission(self) -> None:
        """Some success but mean_depth < 4 ⇒ still degenerate."""
        records = [
            _rec(n_cand=2, n_dist=1, depth=3, wall=0.3) for _ in range(5)
        ]
        agg = _aggregate_metrics(records, n_simulations=200)
        self.assertGreater(agg["n_success"], 0)
        self.assertEqual(agg["mean_depth_reached"], 3.0)
        msg = _check_degeneracy(agg)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertIn("mean_depth_reached", msg)

    def test_non_degenerate_permits_emission(self) -> None:
        """n_success > 0 AND mean_depth >= 4 ⇒ emit JSON."""
        records = [
            _rec(n_cand=3, n_dist=2, depth=5, wall=0.4) for _ in range(5)
        ]
        agg = _aggregate_metrics(records, n_simulations=200)
        self.assertEqual(agg["n_success"], 5)
        self.assertEqual(agg["success_rate"], 1.0)
        self.assertEqual(agg["mean_depth_reached"], 5.0)
        self.assertGreaterEqual(agg["mean_n_distinct"], 2.0)
        msg = _check_degeneracy(agg)
        self.assertIsNone(msg)

    def test_aggregation_includes_required_fields(self) -> None:
        """All 7 R20 metric buckets surface in the aggregation dict."""
        records = [
            _rec(n_cand=4, n_dist=3, depth=5, wall=0.5),
            _rec(n_cand=0, n_dist=0, depth=3, wall=0.2, error="search_error: foo"),
            _rec(n_cand=2, n_dist=1, depth=4, wall=0.3),
        ]
        agg = _aggregate_metrics(records, n_simulations=200)
        for k in (
            "n_total_runs",
            "n_errors",
            "n_success",
            "success_rate",
            "mean_n_distinct",
            "median_n_distinct",
            "mean_depth_reached",
            "median_depth_reached",
            "max_depth_reached_observed",
            "mean_wall_time_s",
            "median_wall_time_s",
            "max_wall_time_s",
            "total_wall_time_s",
            "n_simulations_per_run",
        ):
            self.assertIn(k, agg, msg=f"missing field {k}")
        self.assertEqual(agg["n_total_runs"], 3)
        self.assertEqual(agg["n_errors"], 1)
        self.assertEqual(agg["n_success"], 2)
        # success_rate = n_success / n_total (incl. errors) = 2/3 ≈ 0.667
        self.assertAlmostEqual(agg["success_rate"], 2 / 3, places=4)
        # mean_depth excludes the error record (it's not in ok_records).
        # depths = [5, 4] → mean = 4.5
        self.assertEqual(agg["mean_depth_reached"], 4.5)
        self.assertEqual(agg["max_depth_reached_observed"], 5)


if __name__ == "__main__":
    unittest.main()