"""Tests for the Counterfactual Replay Mining module (TODO-153).

Covers:
- CounterfactualRow dataclass roundtrip
- CounterfactualReport.ranked() ordering
- CounterfactualReplay.replay_counterfactual with empty pool (low confidence)
- CounterfactualReplay.replay_counterfactual with cache hits (high confidence)
- CounterfactualReplay.replay_counterfactual with novel candidates (medium/low)
- CounterfactualReplay.best_alternative top-K filtering
- CounterfactualReplay.coverage_summary
- Determinism of reports given the same seed
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable when tests run from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from molmetal.reports.dream_rsi_replay_pool.counterfactual_replay import (
    MIN_COUNTERFACTUAL_POOL,
    CounterfactualReplay,
    CounterfactualReport,
    CounterfactualRow,
)
from molmetal.reports.dream_rsi_replay_pool.vina_replay_pool import (
    ReplayPoolSurrogate,
    VinaReplayPool,
    pocket_feature_vector,
    random_dmpnn_embedding,
)


def _seed_pool(rows, td):
    """Build a VinaReplayPool from ``rows`` in the temp dir ``td``."""
    pool = VinaReplayPool(storage_dir=td)
    for r in rows:
        emb = random_dmpnn_embedding(r["smiles"], seed=r["seed"])
        pool.add_to_pool(
            smiles=r["smiles"],
            pocket_id=r["pocket_id"],
            vina_score=r["vina_score"],
            seed=r["seed"],
            dmpnn_embedding=emb,
            pocket_features=pocket_feature_vector(r["pocket_id"]),
        )
    return pool


def _make_rows():
    """Build a deterministic 80-row (smiles, pocket, score) fixture with
    unique (smiles, pocket, seed) triples."""
    smiles_pool = [
        "Cc1cnnn1-c1ccc(S(N)(=O)=O)cc1",
        "CCN(CC)CC",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "Nc1ccc(C(=O)O)cc1",
        "OC(=O)c1ccccc1O",
        "CCO",
        "CC(C)O",
        "Brc1ccccc1",
        "Fc1ccccc1",
        "CCBr",
        "CC#N",
        "COC",
        "CCOC",
        "NC=O",
        "C=C",
    ]
    pockets = ["1h36", "3eml", "3ny8", "4aaw", "4rlu"]
    rows = []
    for i in range(80):
        smi_idx = i % len(smiles_pool)
        pid_idx = (i // len(smiles_pool)) % len(pockets)
        seed = (i // (len(smiles_pool) * len(pockets))) % 4
        rows.append(
            {
                "smiles": smiles_pool[smi_idx],
                "pocket_id": pockets[pid_idx],
                "vina_score": -7.5 - 0.05 * i,
                "seed": int(seed),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestCounterfactualRow:
    def test_row_roundtrip(self):
        row = CounterfactualRow(
            smiles="CCN(CC)CC",
            predicted_score=-8.1,
            confidence="medium",
            n_pool_used=42,
            cache_hit=False,
            metadata={"method": "test"},
        )
        d = row.to_dict()
        assert d["smiles"] == "CCN(CC)CC"
        assert d["predicted_score"] == -8.1
        assert d["confidence"] == "medium"
        assert d["n_pool_used"] == 42
        assert d["cache_hit"] is False
        assert d["metadata"] == {"method": "test"}

    def test_row_cache_hit_flag(self):
        row = CounterfactualRow(
            smiles="c1ccccc1",
            predicted_score=-7.0,
            confidence="high",
            n_pool_used=42,
            cache_hit=True,
        )
        assert row.cache_hit is True
        assert row.to_dict()["cache_hit"] is True


class TestCounterfactualReport:
    def test_ranked_orders_ascending(self):
        rows = [
            CounterfactualRow("A", -7.5, "low", 5),
            CounterfactualRow("B", -9.0, "medium", 5),
            CounterfactualRow("C", -8.0, "high", 5),
        ]
        report = CounterfactualReport(
            target_smiles="T",
            pocket_id="1h36",
            seed=0,
            rows=rows,
            n_pool_used=5,
            surrogate_class="test",
        )
        ranked = report.ranked()
        assert [r.smiles for r in ranked] == ["B", "C", "A"]
        # Input order preserved on the raw rows.
        assert [r.smiles for r in report.rows] == ["A", "B", "C"]

    def test_to_dict_includes_all_fields(self):
        rows = [
            CounterfactualRow("X", -7.0, "medium", 10),
        ]
        report = CounterfactualReport(
            target_smiles="T",
            pocket_id="3eml",
            seed=2,
            rows=rows,
            n_pool_used=10,
            surrogate_class="CounterfactualReplay[ridge]",
        )
        d = report.to_dict()
        assert d["target_smiles"] == "T"
        assert d["pocket_id"] == "3eml"
        assert d["seed"] == 2
        assert d["n_pool_used"] == 10
        assert d["surrogate_class"] == "CounterfactualReplay[ridge]"
        assert len(d["rows"]) == 1
        assert d["rows"][0]["smiles"] == "X"


class TestCounterfactualReplayEmptyPool:
    def test_replay_counterfactual_empty_pool_is_low_confidence(self):
        cf = CounterfactualReplay(pool=None)
        report = cf.replay_counterfactual(
            target_smiles="T",
            alternative_smiles=["A", "B", "C"],
            pocket_id="1h36",
        )
        # No pool → surrogate raises during construction; we
        # therefore expect a surrogate to be auto-built but
        # flagged low-confidence because pool size is 0.
        assert len(report.rows) == 3
        for row in report.rows:
            assert row.confidence == "low"
            assert row.cache_hit is False

    def test_replay_counterfactual_empty_alternatives(self):
        cf = CounterfactualReplay(pool=None)
        report = cf.replay_counterfactual(
            target_smiles="T",
            alternative_smiles=[],
            pocket_id="1h36",
        )
        assert report.rows == []
        assert report.target_smiles == "T"
        assert report.surrogate_class == "CounterfactualReplay[empty]"


class TestCounterfactualReplayWithPool:
    def _build_pool(self, td):
        rows = _make_rows()
        return _seed_pool(rows, td)

    def test_cache_hit_path_returns_high_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            pool = self._build_pool(td)
            cf = CounterfactualReplay(pool=pool)
            # Pick an alternative that was added to the pool.
            cached_smi = pool._by_pocket["1h36"][0].smiles  # noqa: SLF001
            cached_score = float(pool._by_pocket["1h36"][0].vina_score)  # noqa: SLF001
            report = cf.replay_counterfactual(
                target_smiles="WHATEVER",
                alternative_smiles=[cached_smi],
                pocket_id="1h36",
            )
            assert len(report.rows) == 1
            row = report.rows[0]
            assert row.cache_hit is True
            assert row.confidence == "high"
            assert abs(row.predicted_score - cached_score) < 1e-9

    def test_novel_candidate_returns_surrogate_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            pool = self._build_pool(td)
            cf = CounterfactualReplay(pool=pool)
            novel = "CCSCC"  # not in pool (different from cache rows)
            report = cf.replay_counterfactual(
                target_smiles="T",
                alternative_smiles=[novel],
                pocket_id="1h36",
            )
            assert len(report.rows) == 1
            row = report.rows[0]
            assert row.cache_hit is False
            assert row.confidence in ("low", "medium")
            assert isinstance(row.predicted_score, float)
            assert math.isfinite(row.predicted_score)

    def test_best_alternative_top_k(self):
        with tempfile.TemporaryDirectory() as td:
            pool = self._build_pool(td)
            cf = CounterfactualReplay(pool=pool)
            alts = ["CCSC", "CCNCC", "c1ccncc1", "CC(=O)C", "Nc1ccccc1"]
            top = cf.best_alternative(
                target_smiles="T",
                alternative_smiles=alts,
                pocket_id="1h36",
                top_k=3,
                min_confidence="low",
            )
            assert len(top) <= 3
            # Verify ascending order.
            scores = [r.predicted_score for r in top]
            assert scores == sorted(scores)

    def test_best_alternative_min_confidence_filter(self):
        with tempfile.TemporaryDirectory() as td:
            pool = self._build_pool(td)
            cf = CounterfactualReplay(pool=pool)
            alts = ["CCSC", "CCNCC"]
            top = cf.best_alternative(
                target_smiles="T",
                alternative_smiles=alts,
                pocket_id="1h36",
                top_k=5,
                min_confidence="high",
            )
            # With a small per-pocket pool every row is low/medium
            # confidence, so the high-confidence filter drops them
            # all.  Result must be empty.
            assert top == []

    def test_coverage_summary(self):
        with tempfile.TemporaryDirectory() as td:
            pool = self._build_pool(td)
            cf = CounterfactualReplay(pool=pool)
            summary = cf.coverage_summary()
            assert set(summary.keys()) == {"1h36", "3eml", "3ny8", "4aaw", "4rlu"}
            for pid, info in summary.items():
                assert "n" in info
                assert "is_low_coverage" in info
                assert isinstance(info["n"], int)
                assert isinstance(info["is_low_coverage"], bool)

    def test_determinism_same_seed(self):
        with tempfile.TemporaryDirectory() as td:
            pool = self._build_pool(td)
            cf = CounterfactualReplay(pool=pool)
            alts = ["CCSC", "CCNCC", "c1ccncc1"]
            r1 = cf.replay_counterfactual(
                target_smiles="T", alternative_smiles=alts, pocket_id="1h36", seed=42
            )
            r2 = cf.replay_counterfactual(
                target_smiles="T", alternative_smiles=alts, pocket_id="1h36", seed=42
            )
            assert [row.predicted_score for row in r1.rows] == [
                row.predicted_score for row in r2.rows
            ]
            assert [row.confidence for row in r1.rows] == [
                row.confidence for row in r2.rows
            ]


class TestCounterfactualReplayWithPreFittedSurrogate:
    def test_accepts_prefitted_surrogate(self):
        with tempfile.TemporaryDirectory() as td:
            pool = _seed_pool(_make_rows(), td)
            surrogate = ReplayPoolSurrogate(pool, backend="ridge", alpha=1.0)
            surrogate.fit()
            cf = CounterfactualReplay(pool=pool, surrogate=surrogate)
            assert cf.surrogate is surrogate
            assert cf.surrogate.is_fitted() is True
            report = cf.replay_counterfactual(
                target_smiles="T",
                alternative_smiles=["CCSCC"],
                pocket_id="1h36",
            )
            assert len(report.rows) == 1
            assert math.isfinite(report.rows[0].predicted_score)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))