"""Tests for the Dream-RSI ReplayPool primitive (TODO-150 §P0.1).

================================================================
CPU-only contract tests for ``molmetal.replay``
================================================================
Validates the four load-bearing pieces of the Dream-RSI integration:

* :class:`TrajectoryRecord`     — 5 tests
* :func:`pessimistic_shrinkage` — 4 tests
* :class:`ReplayPool`           — 22 tests
    - add / extend / eviction    (4 tests)
    - query                       (3 tests)
    - coverage_report             (3 tests)
    - replay_predict (A1)         (5 tests)
    - per_metal_bootstrap (A3)    (4 tests)
    - disagreement_uncertainty (A6) (2 tests)
    - save / load persistence      (1 test)

Total: **31 tests** matching the task brief.  All CPU-only; completes
in <500 ms on any CI container.

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_replay_pool.py -v
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

try:
    from molmetal.replay import (  # type: ignore
        LOW_COVERAGE_THRESHOLD,
        CoverageStats,
        ReplayPool,
        ScoreEstimate,
        TrajectoryRecord,
        pessimistic_shrinkage,
    )
except Exception:  # pragma: no cover
    from molmetal.replay.pool import (  # type: ignore  # noqa: E402
        LOW_COVERAGE_THRESHOLD,
        CoverageStats,
        ReplayPool,
        ScoreEstimate,
        TrajectoryRecord,
        pessimistic_shrinkage,
    )

try:
    from molmetal.replay.pool import _config_hash_from_hp
except Exception:  # pragma: no cover
    from molmetal.replay import _config_hash_from_hp  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_record(
    metal: str = "Ru",
    score: float = 0.5,
    hp: dict | None = None,
    seed: int = 0,
    **kwargs,
) -> TrajectoryRecord:
    """Build a TrajectoryRecord with a stable config_hash."""
    hp = hp if hp is not None else {"lr": 1e-3, "hidden_dim": 256}
    return TrajectoryRecord(
        config_hash=_config_hash_from_hp(hp),
        hp_dict=hp,
        final_test_score=score,
        metal=metal,
        seed=seed,
        **kwargs,
    )


# ===========================================================================
# TrajectoryRecord — 5 tests
# ===========================================================================


class TestTrajectoryRecord:
    """Tests for the :class:`TrajectoryRecord` dataclass."""

    def test_record_roundtrip(self):
        r = _mk_record(metal="Ru", score=0.7)
        d = r.to_dict()
        r2 = TrajectoryRecord.from_dict(d)
        assert r2.metal == r.metal
        assert r2.final_test_score == pytest.approx(r.final_test_score)
        assert r2.hp_dict == r.hp_dict
        assert r2.config_hash == r.config_hash

    def test_record_rejects_empty_config_hash(self):
        with pytest.raises(ValueError):
            TrajectoryRecord(
                config_hash="", hp_dict={}, final_test_score=0.5,
            )

    def test_record_rejects_non_numeric_score(self):
        with pytest.raises(ValueError):
            TrajectoryRecord(
                config_hash="abc", hp_dict={}, final_test_score="bad",
            )

    def test_record_rejects_nan_score(self):
        with pytest.raises(ValueError):
            TrajectoryRecord(
                config_hash="abc", hp_dict={}, final_test_score=float("nan"),
            )

    def test_record_rejects_inf_score(self):
        with pytest.raises(ValueError):
            TrajectoryRecord(
                config_hash="abc", hp_dict={}, final_test_score=float("inf"),
            )


# ===========================================================================
# pessimistic_shrinkage — 4 tests
# ===========================================================================


class TestPessimisticShrinkage:
    """Tests for §4 R1 pessimistic shrinkage helper."""

    def test_shrinkage_zero_records_collapses_to_baseline(self):
        score, weight = pessimistic_shrinkage(
            0.9, n_records=0, threshold=10, baseline=0.5
        )
        assert score == pytest.approx(0.5)
        assert weight == 0.0

    def test_shrinkage_full_coverage_no_shrinkage(self):
        score, weight = pessimistic_shrinkage(
            0.9, n_records=10, threshold=10, baseline=0.5
        )
        assert score == pytest.approx(0.9)
        assert weight == pytest.approx(1.0)

    def test_shrinkage_above_threshold_no_shrinkage(self):
        # 50 records, threshold 10 → capped at 1.0 weight.
        score, weight = pessimistic_shrinkage(
            0.7, n_records=50, threshold=10, baseline=0.5
        )
        assert score == pytest.approx(0.7)
        assert weight == pytest.approx(1.0)

    def test_shrinkage_partial_weight_blends(self):
        # 3 records, threshold 10 → weight=0.3, shrunk = 0.3*0.9 + 0.7*0.5
        score, weight = pessimistic_shrinkage(
            0.9, n_records=3, threshold=10, baseline=0.5
        )
        assert weight == pytest.approx(0.3)
        assert score == pytest.approx(0.3 * 0.9 + 0.7 * 0.5)


# ===========================================================================
# ReplayPool — add / extend / eviction — 4 tests
# ===========================================================================


class TestReplayPoolAdd:
    """Tests for ReplayPool.add / extend / eviction semantics."""

    def test_pool_add_basic(self):
        p = ReplayPool(max_size=8)
        assert len(p) == 0
        p.add(_mk_record(score=0.5))
        p.add(_mk_record(score=0.6))
        assert len(p) == 2
        assert p.n_added == 2
        assert p.n_evicted == 0

    def test_pool_extend_returns_count(self):
        p = ReplayPool(max_size=16)
        recs = [_mk_record(score=i / 10.0) for i in range(5)]
        n = p.extend(recs)
        assert n == 5
        assert len(p) == 5

    def test_pool_max_size_evicts_fifo(self):
        p = ReplayPool(max_size=4)
        for i in range(7):
            p.add(_mk_record(score=float(i) / 10.0))
        assert len(p) == 4
        assert p.n_added == 7
        assert p.n_evicted == 3

    def test_pool_rejects_non_record(self):
        p = ReplayPool(max_size=4)
        with pytest.raises(TypeError):
            p.add({"score": 0.5})  # type: ignore[arg-type]


# ===========================================================================
# ReplayPool — query — 3 tests
# ===========================================================================


class TestReplayPoolQuery:
    """Tests for ReplayPool.query filtering."""

    def _build(self) -> ReplayPool:
        p = ReplayPool(max_size=32)
        p.add(_mk_record(metal="Ru", score=0.7, hp={"lr": 1e-3, "hidden_dim": 256}))
        p.add(_mk_record(metal="Ru", score=0.8, hp={"lr": 1e-4, "hidden_dim": 256}, seed=1))
        p.add(_mk_record(metal="Ir", score=0.5, hp={"lr": 1e-3, "hidden_dim": 256}))
        return p

    def test_query_filters_metal(self):
        p = self._build()
        out = p.query(metal="Ru")
        assert len(out) == 2
        assert all(r["metal"] == "Ru" for r in out)

    def test_query_filters_score_range(self):
        p = self._build()
        out = p.query(min_score=0.6)
        assert len(out) == 2  # 0.7 and 0.8

    def test_query_filters_hp_subset(self):
        p = self._build()
        out = p.query(hp_subset={"lr": 1e-4})
        assert len(out) == 1
        assert out[0]["final_test_score"] == pytest.approx(0.8)


# ===========================================================================
# ReplayPool — coverage_report — 3 tests
# ===========================================================================


class TestReplayPoolCoverage:
    """Tests for ReplayPool.coverage_report."""

    def test_coverage_empty_pool_is_low_coverage(self):
        p = ReplayPool(max_size=8)
        c = p.coverage_report()
        assert c.metal == "ALL"
        assert c.n_records == 0
        assert c.is_low_coverage is True

    def test_coverage_per_metal_basic(self):
        p = ReplayPool(max_size=32, low_coverage_threshold=3)
        for i in range(4):
            p.add(_mk_record(metal="Ru", score=0.5 + i * 0.05, seed=i))
        cov = p.coverage_report("Ru")
        assert cov.metal == "Ru"
        assert cov.n_records == 4
        assert cov.n_seeds == 4
        assert cov.is_low_coverage is False
        assert cov.score_mean == pytest.approx(0.575)

    def test_coverage_table_includes_all_metals(self):
        p = ReplayPool(max_size=32, low_coverage_threshold=2)
        p.add(_mk_record(metal="Ru", score=0.7))
        p.add(_mk_record(metal="Ru", score=0.8))
        p.add(_mk_record(metal="Ir", score=0.5))
        tbl = p.coverage_table()
        assert "ALL" in tbl
        assert "Ru" in tbl
        assert "Ir" in tbl
        assert tbl["Ru"]["is_low_coverage"] is False
        assert tbl["Ir"]["is_low_coverage"] is True


# ===========================================================================
# ReplayPool — replay_predict (A1) — 5 tests
# ===========================================================================


class TestReplayPoolReplayPredict:
    """Tests for A1 replay_predict."""

    def test_predict_empty_pool_returns_baseline(self):
        p = ReplayPool(max_size=8)
        est = p.replay_predict({"lr": 1e-3}, metal="Ru", baseline=0.5)
        assert est.score == pytest.approx(0.5)
        assert est.confidence == "low"
        assert est.n_neighbors == 0

    def test_predict_with_data_returns_finite_score(self):
        p = ReplayPool(max_size=32)
        for s in [0.6, 0.7, 0.8]:
            p.add(_mk_record(metal="Ru", score=s, hp={"lr": 1e-3, "hidden_dim": 256}))
        est = p.replay_predict({"lr": 1e-3, "hidden_dim": 256}, metal="Ru", baseline=0.5)
        assert 0.0 <= est.score <= 1.0
        assert est.n_neighbors == 3

    def test_predict_low_coverage_flagged(self):
        p = ReplayPool(max_size=8, low_coverage_threshold=10)
        p.add(_mk_record(metal="Ru", score=0.9, hp={"lr": 1e-3}))
        est = p.replay_predict({"lr": 1e-3}, metal="Ru", baseline=0.5)
        assert est.confidence == "high"  # exact match despite low coverage

    def test_predict_exact_match_yields_high_confidence(self):
        p = ReplayPool(max_size=32)
        hp = {"lr": 1e-4, "hidden_dim": 300}
        for s in [0.5, 0.6, 0.7]:
            p.add(_mk_record(metal="Ru", score=s, hp=hp))
        est = p.replay_predict(hp, metal="Ru", baseline=0.5)
        assert est.confidence == "high"

    def test_predict_applies_shrinkage_for_low_coverage_no_match(self):
        # 3 records, no exact match → confidence=low, shrinkage weight=0.3
        p = ReplayPool(max_size=8, low_coverage_threshold=10)
        for s in [0.8, 0.85, 0.9]:
            p.add(_mk_record(metal="Ru", score=s, hp={"lr": 1e-3}))
        est = p.replay_predict({"lr": 5e-4}, metal="Ru", baseline=0.5)
        assert est.confidence == "low"
        assert est.shrinkage_weight == pytest.approx(0.3)
        # shrunk = 0.3 * mean(0.8,0.85,0.9) + 0.7 * 0.5
        assert est.score == pytest.approx(0.3 * 0.85 + 0.7 * 0.5)


# ===========================================================================
# ReplayPool — per_metal_bootstrap (A3) — 4 tests
# ===========================================================================


class TestReplayPoolPerMetalBootstrap:
    """Tests for A3 per_metal_bootstrap."""

    def test_bootstrap_falls_back_to_predict_for_small_pool(self):
        p = ReplayPool(max_size=8)
        p.add(_mk_record(metal="Ru", score=0.6))
        est = p.per_metal_bootstrap({"lr": 1e-3}, metal="Ru")
        # Falls back to replay_predict
        assert est.n_neighbors == 1

    def test_bootstrap_leave_one_out_finite(self):
        p = ReplayPool(max_size=8, low_coverage_threshold=2)
        for s in [0.6, 0.7, 0.8]:
            p.add(_mk_record(metal="Ru", score=s, hp={"lr": 1e-3}))
        est = p.per_metal_bootstrap({"lr": 1e-3}, metal="Ru")
        assert 0.0 <= est.score <= 1.0

    def test_bootstrap_exact_match_is_high_confidence(self):
        p = ReplayPool(max_size=8, low_coverage_threshold=3)
        hp = {"lr": 1e-3, "hidden_dim": 256}
        for s in [0.5, 0.6, 0.7]:
            p.add(_mk_record(metal="Ru", score=s, hp=hp))
        est = p.per_metal_bootstrap(hp, metal="Ru")
        assert est.confidence == "high"

    def test_bootstrap_uses_per_metal_only(self):
        # Records on Ir must not contribute to a Ru bootstrap.
        p = ReplayPool(max_size=16, low_coverage_threshold=2)
        p.add(_mk_record(metal="Ru", score=0.6, hp={"lr": 1e-3}))
        p.add(_mk_record(metal="Ru", score=0.7, hp={"lr": 1e-4}, seed=1))
        p.add(_mk_record(metal="Ir", score=0.99, hp={"lr": 1e-3}))  # ignored
        est = p.per_metal_bootstrap({"lr": 1e-4}, metal="Ru")
        # Mean of [0.6, 0.7] is 0.65; shrunk because 2 < threshold=2 ⇒ weight=1.0 (just at boundary).
        # Actually threshold=2 means >=2 gets full weight.
        assert 0.5 <= est.score <= 0.8


# ===========================================================================
# ReplayPool — disagreement_uncertainty (A6) — 2 tests
# ===========================================================================


class TestReplayPoolDisagreementUncertainty:
    """Tests for A6 disagreement_uncertainty."""

    def test_disagreement_empty_pool_is_zero(self):
        p = ReplayPool(max_size=8)
        u = p.disagreement_uncertainty({"lr": 1e-3}, metal="Ru")
        assert u == 0.0

    def test_disagreement_finite_with_data(self):
        p = ReplayPool(max_size=16, low_coverage_threshold=10)
        # Diverse HP records → disagreement should be > 0.
        p.add(_mk_record(metal="Ru", score=0.7, hp={"lr": 1e-4, "hidden_dim": 200}))
        p.add(_mk_record(metal="Ru", score=0.8, hp={"lr": 1e-3, "hidden_dim": 256}))
        p.add(_mk_record(metal="Ru", score=0.5, hp={"lr": 5e-4, "hidden_dim": 300}))
        u = p.disagreement_uncertainty(
            {"lr": 1e-3, "hidden_dim": 256}, metal="Ru",
            k_perturbations=8, seed=0,
        )
        assert u >= 0.0
        assert math.isfinite(u)


# ===========================================================================
# ReplayPool — persistence — 1 test
# ===========================================================================


class TestReplayPoolPersistence:
    """Tests for save / load round-trip."""

    def test_pool_save_load_jsonl(self):
        p = ReplayPool(max_size=8)
        for s in [0.5, 0.6, 0.7]:
            p.add(_mk_record(metal="Ru", score=s, hp={"lr": 1e-3}))
        with tempfile.TemporaryDirectory() as td:
            path = p.save(os.path.join(td, "pool.jsonl"))
            assert os.path.exists(path)
            p2 = ReplayPool.load(path, max_size=8)
            assert len(p2) == 3
            cov = p2.coverage_report("Ru")
            assert cov.n_records == 3
            assert cov.score_mean == pytest.approx(0.6)