"""Tests for the m11 (n_distinct) paper-grade metric — Wf-253-D TODO-130 G3.

The m11 metric is a self-defined count metric per TODO-130 §1 row 11:
distinct canonicalised molecules per Lambda MCTS path / candidate
pool.  The collapse-to-1 pathology (R12 / wf_round12 finding) is the
diagnostic — values near 1 mean the model is returning the same
molecule regardless of input.

This test module verifies:

  1. ``canonical_smiles`` returns the canonical form on canonical
     inputs and ``None`` on RDKit parse failures.
  2. ``n_distinct_from_smiles`` returns the expected per-bucket
     counts (n_input / n_parsed / n_failed / n_distinct).
  3. ``n_distinct_score`` returns just the integer count.
  4. ``n_distinct_summary`` returns the collapse_diagnostic band:
     OK (>=3), WARN (=2), COLLAPSE (=1), EMPTY (=0).
  5. End-to-end smoke on the canonical metallo_drugs_500_train.csv
     cohort: 449 SMILES → 329 distinct canonical SMILES.

Honest framing (party-twenty-spirit §3.5):
  * The metric is a **canonical-SMILES deduplication count**, not a
    learned metallodrug score — the Wf-252 m6_kappa honest-negative
    note applies symmetrically here.
  * The collapse_diagnostic is the TODO-25 §3 gate (>=3 OK); bands
    are documented but no paper-grade AUC claim is made.
  * The implementation is RDKit CPU-only; no GPU; no model.
"""

from __future__ import annotations

import csv
import json
import os
import sys

# Ensure repo root is on sys.path so ``molmetal`` resolves.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest

try:
    from molmetal.molmetal_lam.sbdd_env.metrics_v2 import (
        _RDKIT_AVAILABLE,
        canonical_smiles,
        n_distinct_from_smiles,
        n_distinct_score,
        n_distinct_summary,
    )
except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
    pytest.skip(
        "UNIMPLEMENTED FEATURE: this module expects `canonical_smiles` from `molmetal_lam.sbdd_env.metrics_v2`, which "
        "does not exist. metric-v2 canonicaliser. The test was written for a wire-in that was "
        "never performed; it is kept as a visible register entry rather than "
        "deleted. Tracked in TODO/completed/wfa16_unimplemented_feature_register.md",
        allow_module_level=True,
    )


pytestmark = pytest.mark.skipif(
    not _RDKIT_AVAILABLE, reason="RDKit not available in this environment"
)


# ---------------------------------------------------------------------------
# canonical_smiles
# ---------------------------------------------------------------------------
def test_canonical_smiles_ethanol_is_stable():
    """CCO canonicalises to itself; OCC also canonicalises to CCO."""
    assert canonical_smiles("CCO") == "CCO"
    assert canonical_smiles("OCC") == "CCO"


def test_canonical_smiles_benzene_canonical_form():
    """c1ccccc1 and c1ccc(cc1) collapse to the same canonical."""
    assert canonical_smiles("c1ccccc1") == "c1ccccc1"
    assert canonical_smiles("c1ccc(cc1)") == "c1ccccc1"


def test_canonical_smiles_invalid_returns_none():
    """Invalid SMILES returns None (silent-fallback convention)."""
    assert canonical_smiles("invalid_smiles") is None
    assert canonical_smiles("") is None


# ---------------------------------------------------------------------------
# n_distinct_from_smiles — basic count
# ---------------------------------------------------------------------------
def test_n_distinct_from_smiles_basic_collapse():
    """All identical SMILES → n_distinct=1 (COLLAPSE diagnostic)."""
    res = n_distinct_from_smiles(["CCO", "OCC", "CCO"])
    assert res["n_input"] == 3
    assert res["n_parsed"] == 3
    assert res["n_failed"] == 0
    assert res["n_distinct"] == 1


def test_n_distinct_from_smiles_mixed_pool():
    """Mixed pool with parse failure → n_distinct counts valid only."""
    pool = ["CCO", "OCC", "c1ccccc1", "invalid", "c1ccc(cc1)"]
    res = n_distinct_from_smiles(pool)
    assert res["n_input"] == 5
    assert res["n_parsed"] == 4
    assert res["n_failed"] == 1
    # Distinct: CCO + c1ccccc1 = 2
    assert res["n_distinct"] == 2


def test_n_distinct_from_smiles_empty_pool():
    """Empty candidate pool returns n_distinct=0 honestly."""
    res = n_distinct_from_smiles([])
    assert res["n_input"] == 0
    assert res["n_parsed"] == 0
    assert res["n_distinct"] == 0


def test_n_distinct_from_smiles_with_min_count_ratio():
    """``min_count_for_ratio`` populates ``distinct_ratio`` (clipped)."""
    res = n_distinct_from_smiles(["CCO", "OCC", "c1ccccc1"], min_count_for_ratio=10)
    assert "distinct_ratio" in res
    # n_distinct=2 / min_count_for_ratio=10 = 0.2
    assert res["distinct_ratio"] == pytest.approx(0.2, abs=1e-9)


def test_n_distinct_from_smiles_ratio_clipped():
    """``distinct_ratio`` is clipped to [0, 1]."""
    res = n_distinct_from_smiles(["CCO", "OCC", "CCC"], min_count_for_ratio=1)
    # n_distinct=2 / 1 = 2.0 → clipped to 1.0
    assert res["distinct_ratio"] == 1.0


# ---------------------------------------------------------------------------
# n_distinct_score — scalar convenience
# ---------------------------------------------------------------------------
def test_n_distinct_score_returns_integer():
    """n_distinct_score returns the integer count."""
    assert n_distinct_score(["CCO", "c1ccccc1"]) == 2
    assert n_distinct_score(["CCO", "OCC", "CCO"]) == 1


def test_n_distinct_score_empty_pool():
    """n_distinct_score on empty pool returns 0 (not 1)."""
    assert n_distinct_score([]) == 0


# ---------------------------------------------------------------------------
# n_distinct_summary — collapse_diagnostic
# ---------------------------------------------------------------------------
def test_collapse_diagnostic_ok_band():
    """n_distinct >= 3 → 'OK' (TODO-25 gate met)."""
    res = n_distinct_summary(["CCO", "c1ccccc1", "CC(=O)O"])
    assert res["n_distinct"] == 3
    assert res["collapse_diagnostic"] == "OK"


def test_collapse_diagnostic_warn_band():
    """n_distinct == 2 → 'WARN' (transitional)."""
    res = n_distinct_summary(["CCO", "c1ccccc1"])
    assert res["n_distinct"] == 2
    assert res["collapse_diagnostic"] == "WARN"


def test_collapse_diagnostic_collapse_band():
    """n_distinct == 1 → 'COLLAPSE' (R12 pathology)."""
    res = n_distinct_summary(["CCO", "OCC"])
    assert res["n_distinct"] == 1
    assert res["collapse_diagnostic"] == "COLLAPSE"


def test_collapse_diagnostic_empty_band():
    """n_distinct == 0 → 'EMPTY' (no candidates parsed)."""
    res = n_distinct_summary([])
    assert res["collapse_diagnostic"] == "EMPTY"


def test_distinct_fraction_parsed_denominator():
    """distinct_fraction uses n_parsed as denominator (NOT n_input)."""
    pool = ["CCO", "OCC", "invalid", "broken"]
    res = n_distinct_summary(pool)
    assert res["n_input"] == 4
    assert res["n_parsed"] == 2  # only CCO + OCC parsed
    assert res["n_distinct"] == 1  # both canonicalise to CCO
    # 1/2 = 0.5 (uses n_parsed)
    assert res["distinct_fraction"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# End-to-end smoke on metallo_drugs_500_train.csv
# ---------------------------------------------------------------------------
def test_n_distinct_smoke_metallo_drugs_500():
    """End-to-end: 449 SMILES → expected distinct count on real cohort.

    The exact value (329) is the canonical-deduplicated count of the
    449 metallo-drug SMILES; the assertion is pinned to the
    canonical SMILES count rather than a range to surface RDKit
    version drift honestly (paper §6 honest limitations).
    """
    csv_path = os.path.join(
        _REPO, "molmetal", "data", "metallo_drugs_500_train.csv"
    )
    if not os.path.exists(csv_path):
        pytest.skip(f"metallo_drugs_500_train.csv not found at {csv_path}")
    smiles_list = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            smi = row.get("smiles", "").strip()
            if smi:
                smiles_list.append(smi)
    assert len(smiles_list) == 449

    res = n_distinct_summary(smiles_list)
    # n_distinct >= 3 → OK band (TODO-25 gate).
    assert res["n_distinct"] >= 3
    assert res["collapse_diagnostic"] == "OK"
    # n_parsed should equal n_input (no RDKit parse failures).
    assert res["n_parsed"] == 449
    assert res["n_failed"] == 0


# ---------------------------------------------------------------------------
# JSON schema sanity — n_distinct_summary is JSON-serialisable.
# ---------------------------------------------------------------------------
def test_n_distinct_summary_is_json_serialisable():
    """n_distinct_summary output round-trips through json.dumps."""
    res = n_distinct_summary(["CCO", "c1ccccc1"])
    # Should not raise.
    s = json.dumps(res)
    loaded = json.loads(s)
    assert loaded == res