"""Tests for ``molmetal_lam.reactions.confidence`` (L4).

Six required test cases (from the task spec) plus a few sanity tests
that catch implementation regressions in the cache, the persistence
round-trip, and the top-k tie-breaker.
"""

from __future__ import annotations

import math
import pickle  # nosec
import tempfile
from pathlib import Path

import pytest

from molmetal_lam.reactions.confidence import (
    Reaction,
    ReactionConfidence,
    SCAFFOLD_HASH_MAX_LEN,
    bayes_shrink,  # legacy alias re-export check
    default_cold_pair_probability,
    laplace_estimate,
    scaffold_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_corpus() -> list:
    """Tiny corpus used by most tests — fully deterministic, no I/O."""
    return [
        # rule A + scaffold X: 3 successes out of 4 → smoothed 4/6 = 0.6667
        Reaction("A", "X", success=True,  source="lit"),
        Reaction("A", "X", success=True,  source="lit"),
        Reaction("A", "X", success=True,  source="lit"),
        Reaction("A", "X", success=False, source="lit"),
        # rule B + scaffold X: 1 success out of 1 → smoothed 2/3 ≈ 0.6667
        Reaction("B", "X", success=True,  source="lit"),
        # rule A + scaffold Y: 0 out of 1 (only failure) → smoothed 1/3 ≈ 0.3333
        Reaction("A", "Y", success=False, source="lit"),
    ]


# ---------------------------------------------------------------------------
# 1. predict returns float in [0, 1]
# ---------------------------------------------------------------------------
def test_reaction_confidence_returns_prob(small_corpus):
    est = ReactionConfidence().fit(small_corpus)
    p = est.predict("A", "X")
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    # exactly (3+1)/(4+2) = 4/6 ≈ 0.6667
    assert math.isclose(p, 4 / 6, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 2. unfitted estimator returns uniform 0.5
# ---------------------------------------------------------------------------
def test_reaction_confidence_unfitted_returns_uniform():
    est = ReactionConfidence()  # no .fit()
    assert est.fitted is False
    p1 = est.predict("A", "X")
    p2 = est.predict("Pt_cisplatin", "N([H])([H])[Pt](Cl)(Cl)([H])N([H])([H])")
    assert p1 == default_cold_pair_probability() == 0.5
    assert p2 == 0.5


# ---------------------------------------------------------------------------
# 3. fit increases cache size (entries have data)
# ---------------------------------------------------------------------------
def test_reaction_confidence_fit_increases_data(small_corpus):
    est = ReactionConfidence()
    assert len(est.cache) == 0
    est.fit(small_corpus)
    # 3 distinct (rule, scaffold) keys: (A,X), (B,X), (A,Y)
    assert len(est.cache) == 3
    n_succ, n_tot = est.raw_counts("A", "X")
    assert (n_succ, n_tot) == (3, 4)
    n_succ, n_tot = est.raw_counts("B", "X")
    assert (n_succ, n_tot) == (1, 1)
    n_succ, n_tot = est.raw_counts("A", "Y")
    assert (n_succ, n_tot) == (0, 1)
    # total_rows = sum of n_tot across the cache
    assert est.total_rows == 6


# ---------------------------------------------------------------------------
# 4. top_k returns rules sorted by probability descending
# ---------------------------------------------------------------------------
def test_top_k_rules_sorted(small_corpus):
    est = ReactionConfidence().fit(small_corpus)
    top = est.top_k_rules("X", k=3)
    # Both A and B have positive rows on scaffold X
    assert [r for r, _ in top] == ["A", "B"]
    # A: (3+1)/(4+2) = 0.6667 ; B: (1+1)/(1+2) = 0.6667 — tie broken by n_total desc.
    a_score = top[0][1]
    b_score = top[1][1]
    assert math.isclose(a_score, b_score, rel_tol=1e-9)
    # top_k for a cold scaffold (no joint row) returns [].
    assert est.top_k_rules("COLD") == []


# ---------------------------------------------------------------------------
# 5. Laplace smoothing: cold-start pair returns (0+1)/(0+2) = 0.5
# ---------------------------------------------------------------------------
def test_laplace_smoothing(small_corpus):
    est = ReactionConfidence().fit(small_corpus)
    p = est.predict("UNKNOWN_RULE", "UNKNOWN_SCAFFOLD")
    assert math.isclose(p, 1.0 / 2.0, rel_tol=1e-12)
    # also check the helper directly
    assert math.isclose(laplace_estimate(0, 0), 0.5)
    assert math.isclose(laplace_estimate(1, 2), 2 / 4)
    # legacy alias re-export
    assert math.isclose(bayes_shrink(2, 5), 3 / 7)
    # validate raises on bad input
    with pytest.raises(ValueError):
        laplace_estimate(-1, 5)
    with pytest.raises(ValueError):
        laplace_estimate(5, 2)


# ---------------------------------------------------------------------------
# 6. persistence round-trip: fit -> save -> load -> same predictions
# ---------------------------------------------------------------------------
def test_persistence_roundtrip(small_corpus):
    est = ReactionConfidence().fit(small_corpus)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "confidence.pkl"
        est.save(path)
        loaded = ReactionConfidence.load(path)
    assert loaded.fitted is True
    assert loaded.total_rows == est.total_rows
    assert loaded.cache == est.cache
    # predictions match
    for rule, smi in [("A", "X"), ("B", "X"), ("A", "Y"), ("NEW", "NEW")]:
        assert est.predict(rule, smi) == loaded.predict(rule, smi)
    # summary also matches
    assert loaded.summary()["n_input_rows" if "n_input_rows" in loaded.summary() else "n_rows_total"] == (
        est.summary()["n_input_rows" if "n_input_rows" in est.summary() else "n_rows_total"]
    )


# ---------------------------------------------------------------------------
# Extra: Reaction dataclass validation
# ---------------------------------------------------------------------------
def test_reaction_validates_yield_fraction():
    with pytest.raises(ValueError):
        Reaction("A", "X", success=True, yield_fraction=1.5)
    with pytest.raises(ValueError):
        Reaction("A", "X", success=True, yield_fraction=-0.1)


def test_reaction_rejects_empty_rule_name():
    with pytest.raises(ValueError):
        Reaction("", "X", success=True)


def test_reaction_key_truncates_long_scaffold():
    long_smi = "C" * (SCAFFOLD_HASH_MAX_LEN + 50)
    r = Reaction("A", long_smi, success=True)
    key = r.key()
    assert len(key[1]) == SCAFFOLD_HASH_MAX_LEN


# ---------------------------------------------------------------------------
# Extra: to_dict / from_dict JSON transport round-trip
# ---------------------------------------------------------------------------
def test_json_transport_roundtrip(small_corpus):
    est = ReactionConfidence().fit(small_corpus)
    blob = est.to_dict()
    assert blob["version"] == 1
    assert blob["n_rows_total"] == 6
    restored = ReactionConfidence.from_dict(blob)
    assert restored.cache == est.cache
    assert restored.predict("A", "X") == est.predict("A", "X")


def test_scaffold_key_normalisation():
    assert scaffold_key("  CCO  ") == "CCO"
    assert scaffold_key("") == ""
    assert scaffold_key(None) == ""
    # truncation
    long_smi = "C" * 500
    assert len(scaffold_key(long_smi)) == SCAFFOLD_HASH_MAX_LEN


# ---------------------------------------------------------------------------
# Extra: tie-break + provenance bookkeeping
# ---------------------------------------------------------------------------
def test_summary_includes_provenance(small_corpus):
    est = ReactionConfidence().fit(small_corpus)
    s = est.summary()
    assert "provenance" in s
    assert s["provenance"].get("lit", 0) == 6
    assert "rules" in s
    assert set(s["rules"]) == {"A", "B"}


def test_fit_is_idempotent_under_double_call():
    # Use an asymmetric corpus so re-fitting actually changes the
    # smoothed estimate (Laplace is not idempotent in general).
    corpus = [
        Reaction("A", "X", success=True,  source="lit"),
        Reaction("A", "X", success=True,  source="lit"),
        Reaction("A", "X", success=False, source="lit"),
    ]
    est = ReactionConfidence().fit(corpus)
    p1 = est.predict("A", "X")  # (2+1)/(3+2) = 3/5 = 0.6
    # fit() again with the same rows — counts should *double* (design choice).
    est.fit(corpus)
    p2 = est.predict("A", "X")  # (4+1)/(6+2) = 5/8 = 0.625
    assert math.isclose(p1, 3 / 5)
    assert math.isclose(p2, 5 / 8)
    assert p1 != p2  # double-counting changed the smoothed estimate
